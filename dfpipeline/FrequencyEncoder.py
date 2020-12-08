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

class FrequencyEncoder(DFPBase):
    """
    Count the number of appearances for each value in a column.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    normalize : Boolean
        When this is True, the counts are normalized between 0.0 and 1.0
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        normalize=False
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.normalize = normalize
        self.maps = []
        assert len(self.inputs) == len(self.outputs)

    def __encode(self, X):
        return X.value_counts(normalize=self.normalize).to_dict()
    
    def fit(self, df):
        for input in self.inputs:
            self.maps.append(self.__encode(df[input]))
        return self
        
    def transform(self, df):
        for input, output, m in zip(self.inputs, self.outputs, self.maps):
            if self.normalize:
                df[output] = df[input].map(m).fillna(0.0)
            else:
                df[output] = df[input].map(m).fillna(1)
        return df

    def to_onnx_operator(self, graph):
        for input_column, output_column, m in zip(self.inputs, self.outputs, self.maps):

            input_tensor = graph.get_current_tensor(input_column)

            kwargs = {}
            keys = list(m.keys())
            vals = list(m.values())
            if graph.is_int_tensor(input_tensor.type):
                kwargs['keys_int64s'] = keys
            elif graph.is_float_tensor(input_tensor.type):
                kwargs['keys_floats'] = keys
            elif graph.is_string_tensor(input_tensor.type):
                kwargs['keys_strings'] = keys
            else:
                assert False, input_column + ' column has a unknow type'

            if self.normalize:
                output_tensor = graph.get_next_tensor(output_column, TensorProto.FLOAT)
                kwargs['values_floats'] = vals
                kwargs['default_float'] = 0.0
            else:
                output_tensor = graph.get_next_tensor(output_column, TensorProto.INT64)
                kwargs['values_int64s'] = vals
                kwargs['default_int64'] = 1

            graph.add([input_tensor], [output_tensor], [helper.make_node('LabelEncoder', [input_tensor.name], [output_tensor.name], graph.get_node_name('LabelEncoder'), domain='ai.onnx.ml', **kwargs)])
