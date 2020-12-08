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

import pandas as pd
import numpy as np

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx import update_registered_converter
from lightgbm import LGBMClassifier

from .OnnxGraph import OnnxGraph

class DataframePipeline:

    def __init__(self, steps=[]):
        self.steps = steps
        self.column_info = {}
        self.new_columns = []
        update_registered_converter(LGBMClassifier, 'LightGbmLGBMClassifier',
                                    calculate_linear_classifier_output_shapes,
                                    convert_lightgbm)

    def clear(self):
        self.steps = []

    def append(self, o):
        self.steps.append(o)

    def fit(self, df, **kwargs):
        """
        Collect the information to transform a dataframe. This should be called only for training.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe for training
        """
        for i, tr in enumerate(self.steps):
            # print('fitting with ' + str(tr))
            self.steps[i] = tr.fit(df, **kwargs)
        return

    def transform(self, df):
        """
        Transform values in a dataframe. This can be called for both training and scoring.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe for training

        Returns
        ----------
        X : pandas.DataFrame
            Transformed dataframe 
        """
        X = df
        self.input_columns = X.columns
        self.input_dtypes = X.dtypes
        for tr in self.steps:
            # print('transforming with ' + str(tr))
            X = tr.transform(X)
        self.output_columns = X.columns
        return X

    def fit_transform(self, df, **kwargs):
        """
        Shortcut to call both fit and transform functions. This should be called only for training.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe for training

        Returns
        ----------
        X : pandas.DataFrame
            Transformed dataframe 
        """
        X = df
        self.input_columns = X.columns
        self.input_dtypes = X.dtypes
        for i, tr in enumerate(self.steps):
            # print('fitting with ' + str(tr))
            self.steps[i] = tr.fit(X, **kwargs)
            X = self.steps[i].transform(X)
        self.output_columns = X.columns        
        return X

    def export(self, ml_model_input_name, path, ml_models=[], with_pre_process=True, name='DataframePipeline'):
        """
        Export a dataframe pipeline with a trained model into a file in the ONNX format.

        Parameters
        ----------
        ml_model_input_name : str
            A name of the input to a trained model. This should be decided when a ML model is converted into an ONNX model.

        path : str
            A path to an ONNX file.

        ml_models : List of ONNX models (default is an empty list)
            These models are connected to ONNX transformers converted from a dataframe pipeline. If not specified, only the ONNX transformers are exported.

        with_pre_process : Boolean (default is True)
            If False, ONNX trasformers are not exported.

        name : str (default is 'DataFramePipeline')
            A name of an exported model.

        Returns
        ----------
        input_columns_to_onnx : Dict
            Each key-value pair is the name of a tensor inputted into an exported model and the dtype of the tensor.
        """
        onnx_graph = OnnxGraph(ml_model_input_name, ml_models, path, with_pre_process, name, self.steps, self.input_columns, self.input_dtypes)        
        columns = onnx_graph.create(self)
        input_columns_to_onnx = {}
        if with_pre_process:
            for c in columns:
                input_columns_to_onnx[c] = self.input_dtypes[c]
        return input_columns_to_onnx
        
    @classmethod
    def convert_to_tensors(self, df, input_columns_to_onnx):
        """
        Convert a dataframe into a set of tensors (i.e., NumPy arrays)

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe

        input_columns_to_onnx : Dict
            Each key-value pair is the name of a tensor inputted into an exported model and the dtype of the tensor.

        Returns
        ----------
        tensors : Dict
            Each key-value pair is the name of a tensor inputted into an exported model and the tensor (NumPy array)
            
        """
        tensors = {c: df[c].values for c in input_columns_to_onnx.keys()}
        for c, dtype in input_columns_to_onnx.items():
            if (str(dtype) == 'category')|(dtype=='object'):
                tensors[c] = tensors[c].astype('str')
            else:
                tensors[c] = tensors[c].astype(dtype)
            tensors[c] = tensors[c].reshape((tensors[c].shape[0], 1))
        return tensors

    def num_column_info(self, c):
        assert c != None
        i = self.column_info.get(c)
        return 0 if i is None else len(i)

    def get_last_column_info(self, c, add_new_column = False):
        assert c != None
        i = self.column_info.get(c)
        if i is None:
            if add_new_column:
                self.column_info[c] = [c]
                self.new_columns.append(c)
            return c
        return i[-1]

    def update_column_info(self, c, name):
        #print("update_column_info: ", c, name)
        assert c != None
        dic = self.column_info.get(c)
        if dic is None:
            self.column_info[c] = [name]
            self.new_columns.append(c)
        else:
            self.column_info[c].append(name)

    def print_column_info(self):
        for k, v in self.column_info.items():
            print(k, v)

