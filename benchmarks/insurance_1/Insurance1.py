##############################################################################
# Copyright 2020 IBM Corp. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
##############################################################################

import sys
sys.path.append('../')
import MLPipelineBenchmark

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import gc
import dfpipeline as dfp
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

class Insurance1(MLPipelineBenchmark.MLPipelineBenchmark):

    def __init__(self, path):
        super().__init__(path)

    def load_data(self, frac, num_tests):
        X_train = pd.read_csv(self.path + '/../inputs/insurance_1/train.csv')

        if frac < 1.0:
            X_train = X_train.sample(frac=frac, random_state=1)

        Y_train = X_train['QuoteConversion_Flag'].copy()

        del X_train['QuoteConversion_Flag']; x = gc.collect()

        X = X_train
        Y = Y_train

        idx_train = X.index[:3*len(X_train)//4]
        idx_test = X.index[3*len(X_train)//4:]

        X_train = X.loc[idx_train]
        Y_train = Y.loc[idx_train]

        X_test = X.loc[idx_test]
        Y_test = Y.loc[idx_test]

        print('Train shape',X_train.shape,'test shape',X_test.shape)

        return X_train, Y_train, X_test, Y_test
    
    def define_pipeline(self, columns, dtypes):

        cat_columns = []
        for c in columns:
            if dtypes[c]=='object':
                cat_columns.append(c)

        self.pipeline = dfp.DataframePipeline(steps=[
            dfp.DateTransformer(column='Original_Quote_Date'),
            dfp.ComplementLabelEncoder(inputs=cat_columns, outputs=cat_columns),
            dfp.ColumnSelector(columns=['QuoteNumber', 'Original_Quote_Date', 'Original_Quote_Date_WY', 'Original_Quote_Date_DY', 'Original_Quote_Date_DM', 'Original_Quote_Date_HD'],
                               drop=True),
            ])

    def do_training(self, X, Y, create_onnx = True):
        print('Pre-processing ...')
        X = self.transform(X, do_fit=True)

        new_columns = {}
        for i, c in enumerate(X.columns):
            new_columns[c] = i
        X = X.rename(columns=new_columns)

        print('Training with ' + str(len(X.columns)) + ' columns ...')
        self.clfs = []
        clf = xgb.XGBClassifier(n_estimators=1700,
                                nthread=32,
                                max_depth=6,
                                learning_rate=0.024,
                                subsample=0.8,
                                colsample_bytree=0.65)
        xgb_model = clf.fit(X, Y, eval_metric="auc", verbose=True)
        self.clfs.append(clf)

        if create_onnx:
            print('Converting models into ONNX ...')
            onnx_ml_models = []
            for i, clf in enumerate(self.clfs):
                initial_type = [('dense_input', FloatTensorType([None, len(self.pipeline.output_columns)]))]
                onnx_ml_models.append(convert_xgboost(clf, initial_types=initial_type))

            self.create_onnx('insurance', onnx_ml_models)
