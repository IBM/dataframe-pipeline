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
from sklearn.preprocessing import LabelEncoder
import copy
import numpy as np

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

class ComplementLabelEncoder(DFPBase):
    """
    Encoder categorical (string) values into numerical values.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.
    """
    def __init__(
        self,
        inputs=DFPBase._PARM_ALL,
        outputs=DFPBase._PARM_ALL,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.maps = []
        self.vals = []

    def __fit(self, X, encoder):
        label_col = X.map(lambda x:'extra_category_' if str(x)=='nan' else x).astype('str')
        encoder.fit(label_col)
        if 'extra_category_' not in encoder.classes_:
            encoder.classes_ = list(encoder.classes_) + ['extra_category_']
        m = {encoder.classes_[i]:i for i in range(len(encoder.classes_))}
        return m, m['extra_category_']

    def fit(self, df):
        self.maps.clear()
        self.vals.clear()
        self.inputs = DFPBase.replace_PARM_ALL(df, self.inputs)
        self.outputs = DFPBase.replace_PARM_ALL(df, self.outputs)
        for input in self.inputs:
            m, v = self.__fit(df[input], LabelEncoder())
            self.maps.append(m)
            self.vals.append(v)
        return self

    def __transform(self, X, m, v):
        if str(X.dtype) == 'category':
            X = X.cat.add_categories(['extra_category_'])
        return X.fillna('extra_category_').map(m).fillna(v).astype('int32')

    def transform(self, df):
        self.inputs = DFPBase.replace_PARM_ALL(df, self.inputs)
        self.outputs = DFPBase.replace_PARM_ALL(df, self.outputs)
        for input, output, m, v in zip(self.inputs, self.outputs, self.maps, self.vals):
            df[output] = self.__transform(df[input], m, v)
        return df

    def to_onnx_operator(self, graph):
        for input_column, output_column, m, v in zip(self.inputs, self.outputs, self.maps, self.vals):
            
            input_tensor = graph.get_current_tensor(input_column)
            output_tensor = graph.get_next_tensor(output_column, TensorProto.INT64)

            assert input_tensor.type == TensorProto.STRING

            kwargs = {}
            keys = []
            for key in m.keys():
                key = key.replace('nan', 'NaN')
                keys.append(key)
            vals = list(m.values())
            kwargs['keys_strings'] = keys
            kwargs['values_int64s'] = vals
            kwargs['default_int64'] = v

            graph.add([input_tensor], [output_tensor], [helper.make_node('LabelEncoder', [input_tensor.name], [output_tensor.name], graph.get_node_name('LabelEncoder'), domain='ai.onnx.ml', **kwargs)])
