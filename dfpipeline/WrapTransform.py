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
from . import DFPBase
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
import collections

import onnx
from onnx import helper

class WrapTransformer(DFPBase):
    """
    Wrap an existing transformer.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    transformer : A transformer object
        This is an object instantiated from an existing transformer class such as sklearn.preprocessing.LabelEncoder().

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, 100, 400],
                           'Country': ['US', 'CAN', 'JP', 'JP']})
    Wrap LabelEncoder()
    >>> tf1 = WrapTransformer(inputs=['Country'],
                              outputs=['Country'],
                              transformer=LabelEncoder())
    """
    def __init__(
        self,
        inputs=DFPBase._PARM_ALL,
        outputs=DFPBase._PARM_ALL,
        transformer=None
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.transformer = transformer
        self.transformers = []
        self.dtypes = []

    def fit(self, df, **params):
        self.transformers.clear()
        self.dtypes.clear()
        self.inputs = DFPBase.replace_PARM_ALL(df, self.inputs)
        self.outputs = DFPBase.replace_PARM_ALL(df, self.outputs)
        for input in self.inputs:
            input_list = []
            if type(input) is tuple:
                for i in range(len(input)):
                    input_list.append(input[i])
            else:
                if type(self.transformer) is LabelEncoder:
                    input_list = input
                else:
                    input_list.append(input)

            tr = copy.deepcopy(self.transformer)
            tr.fit(df[input_list])
            self.transformers.append(tr)
            self.dtypes.append(df[input].dtype)
            
        return self

    def get_transformers(self):
        return self.transformers

    def get_transformer(self, c):
        return self.transformers[c]

    def transform(self, df):
        self.inputs = DFPBase.replace_PARM_ALL(df, self.inputs)
        self.outputs = DFPBase.replace_PARM_ALL(df, self.outputs)
        for input, output, tr in zip(self.inputs, self.outputs, self.transformers):
            input_list = []
            output_list = []
            if type(input) is tuple:
                assert type(output) is tuple
                assert len(input) == len(output)
                for i in range(len(input)):
                    input_list.append(input[i])
                    output_list.append(output[i])
            else:
                if type(tr) is LabelEncoder:
                    input_list = input
                    output_list = output
                else:
                    input_list.append(input)
                    output_list.append(output)

            df[output_list] = tr.transform(df[input_list])

        return df

    def to_onnx_operator(self, graph):
        assert False, 'Not implemented yet'
