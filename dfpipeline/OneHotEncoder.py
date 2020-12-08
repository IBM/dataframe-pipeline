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

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

class OneHotEncoder(DFPBase):
    """
    Perform pd.get_dummies() to given columns and concatinate them with the other columns

    Parameters
    ----------
    columns : List of strings
        Each string is a one hot column label.
    
    Examples:
    ----------
    * Input df
          sex   C2
    0    male    3
    1  female    4
    2  female    6
    3    male    9
    4  female   13
    5    male   17
    6  female   20
    7  female  100

    * Expected result of OneHotEncoder(columns=['sex'])
        C2  female  male
    0    3       0     1
    1    4       1     0
    2    6       1     0
    3    9       0     1
    4   13       1     0
    5   17       0     1
    6   20       1     0
    7  100       1     0
    """
    def __init__(
        self,
        columns=[]
    ):
        super().__init__()
        self.columns = columns

    def fit(self, df):
        self.onehot_cats = []
        for c in self.columns:
            onehot_df = pd.get_dummies(df[c])
            cats = []
            for cat in onehot_df.columns:
                cats.append(cat)
            self.onehot_cats.append(cats)
        return self

    def transform(self, df):
        # Check invalid case first
        if self.columns is None or len(self.columns) == 0:
            return df

        for c, cats in zip(self.columns, self.onehot_cats):
            for cat in cats:
                # print('Adding ' + c + '__' + str(cat))
                df[c + '__' + str(cat)] = df[c].map({cat: 1}).fillna(0)
        return df

    def to_onnx_operator(self, graph):
        ops = []
        for input_column, cats in zip(self.columns, self.onehot_cats):
            input_tensor = graph.get_current_tensor(input_column)
            for c in cats:
                ops = []

                kwargs = {}
                if graph.is_int_tensor(input_tensor.type):
                    kwargs['cats_int64s'] = [int(c)]
                elif input_tensor.type == TensorProto.STRING:
                    kwargs['cats_strings'] = [str(c)]
                else:
                    assert False, str(input_tensor.type) + ' is not supported'

                onehot_tensor = graph.get_tmp_tensor()
                output_tensor = graph.get_next_tensor(input_column + '__' + str(c), TensorProto.INT32)
                squeeze_kwargs={'axes': [2, 2]}
                ops.append(helper.make_node('OneHotEncoder', [input_tensor.name], [onehot_tensor], graph.get_node_name('OneHotEncoder'), domain='ai.onnx.ml', **kwargs))
                ops.append(helper.make_node('Squeeze', [onehot_tensor], [output_tensor.name], graph.get_node_name('Squeeze'), domain='ai.onnx', **squeeze_kwargs))
                graph.add([input_tensor], [output_tensor], ops)
