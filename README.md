# Dataframe Pipeline - A framework to build a machine-learning pipeline

[![License](https://img.shields.io/badge/License-Apache2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![CLA assistant](https://cla-assistant.io/readme/badge/IBM/dataframe-pipeline)](https://cla-assistant.io/IBM/dataframe-pipeline)

This framework provides APIs called data transformers to represent popular data transformation patterns on a pandas DataFrame object which is a 2D array consisting of rows and labeled columns. You can construct an machine-learning pipeline with the data trasformers, and then export it with a trained ML model into a file in the [ONNX](https://onnx.ai/) format which is a standard to represent a ML model and data transformations.

## How to install via Docker
The easiest way to use the dataframe pipeline is to build a docker image that includes all of the dependencies. If you want to install your native environment, please follow the steps written in docker/Dockerfile.

### 1. Set up kaggle API credentials by following the procedure in [Kaggle API](https://github.com/Kaggle/kaggle-api) 
This step is needed to download datasets used by our benchmarks. If you do not run the benchmarks, you can skip this step and comment out the lines to copy the kaggle API credentials from docker/Dockerfile. After this step, you should have a json file that includes your API key under ~/.kaggle.

### 2. Clone this repository
```
# git clone https://github.com/IBM/dataframe-pipeline.git
# cd dfpipeline
```

### 3. Build a docker image
```
# cd docker
# ./build-docker.sh
```
If you succeeded to build the image, you can find an image named **dfp** by running a docker command `docker images`. You can use the dataframe pipeline in a docker container by running `docker run -it dfp bash`.

Note that docker/Dockerfile builds the ONNX Runtime which we extended. You can export a ML pipeline in the ONNX format as shown in the following steps. However, for now, current ONNX operators are not enough to represent all of the data transformations available in the dataframe pipeline. Therefore, extended the ONNX Runtime to add some operators which are needed for the data transformations in the dataframe pipeline.

## How to use
### 1. Define your pipeline
```
import dfpipeline as dfp

pipeline = dfp.DataframePipeline(steps=[
  dfp.ComplementLabelEncoder(inputs=['emaildomain', 'card'], outputs=['emaildomain', 'card']),
  dfp.FrequencyEncoder(inputs=['emaildomain', 'card'], outputs=['emaildomain_fe', 'card_fe']),
  dfp.Aggregator(inputs=['Amt'], groupby=['card'], outputs=['Amt_card_mean'], func='mean'),
])
```

### 2. Transform a dataframe for training
```
import pandas as pd

train_df = pd.read_csv('training.csv')
train_df = pipeline.fit_transform(df)
```

### 3. Train a ML model using the transformed dataframe
```
import xgboost as xgb

clf = xgb.XGBClassifier(...)
clf.fit(train_df)
```

### 4. Convert the trained ML model into an ONNX model
```
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

initial_type = [('dense_input', FloatTensorType([None, len(pipeline.output_columns)]))]
onnx_ml_model = convert_xgboost(clf, initial_types=initial_type)
```

### 5. Export a dataframe pipeline with a trained ML model into ONNX
```
input_columns_to_onnx = pipeline.export('dense_input', [onnx_ml_model], 'pipeline.onnx')
```

### 6. Load an ONNX file and run the pipeline
```
import onnxrutime as rt

test_df = pd.read_csv('test.csv')
sess = rt.InferenceSession('pipeline.onnx')
tensors = dfp.DataframePipeline.convert_to_tensors(test_df, input_columns_to_onnx)
preds = sess.run(None, tensors)
```

## Benchmarking
We developed benchmarks to evaluate the performance of ML pipelines on Python and the [ONNX Runtime](https://github.com/microsoft/onnxruntime) referring the following use cases.
 - [XGB Fraud with Magic](https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600)
 - [Xgboost_benchmark](https://www.kaggle.com/mpearmain/xgboost-benchmark)
 - [On Hot Encoding For Categorical Encoding](https://www.kaggle.com/c7934597/on-hot-encoding-for-categorical-encoding)

### 1. Go to the benchmark directory in a docker container
```
cd /git/dataframe-pipeline/benchmarks
```

### 2. Download datasets
```
# cd benchmarks
# ./download_inputs.sh
```

### 3. Run benchmarks
```
# ./run.sh
```

# Contributing
Follow [our contribution guidelines](https://github.com/IBM/dataframe-pipeline/blob/master/CONTRIBUTING.md).
