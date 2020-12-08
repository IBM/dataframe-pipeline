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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import onnx
from onnx import helper

class Scaler(DFPBase):
    """
    Normalize column values based on a strategy.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    strategy : String
        minmax : This is same as MinMaxScaler of scikit-learn
        standard: This is same as StandardScaler of scikit-learn
        min: Subtract the min value in a column from the values in the column
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        strategy = None,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.strategy = strategy
        self.scaler = None
        self.mins = []
        if strategy is 'minmax':
            self.scaler = MinMaxScaler()
        elif strategy is 'standard':
            self.scaler = StandardScaler()
        elif strategy is 'min':
            pass

        assert strategy == 'min' or self.scaler != None, "Not implemented it yet"

    def fit(self, df, **params):
        if self.strategy == 'min':
            for input in self.inputs:
                self.mins.append(df[input].min())
        else:
            if len(df.index) > 0:
                self.scaler.fit(df[self.inputs])
        return self
        
    def transform(self, df):
        if self.strategy == 'min':
            for input, output, m in zip(self.inputs, self.outputs, self.mins):
                df[output] = df[input] - m
        else:
            if len(df.index) > 0:
                df[self.outputs] = self.scaler.transform(df[self.inputs])
        return df

    def __to_onnx_operator_for_min(self, graph):
        for input_column, output_column, m in zip(self.inputs, self.outputs, self.mins):

            input_tensor = graph.get_current_tensor(input_column)
            output_tensor = graph.get_next_tensor(output_column, input_tensor.type)

            kwargs = {}
            if graph.is_int_tensor(input_tensor.type):
                kwargs['value_int'] = int(m)
            elif graph.is_float_tensor(input_tensor.type):
                kwargs['value_float'] = float(m)
            else:
                assert False, input_column + ' column is not a numeric type'

            ops = []
            min_tensor = graph.get_tmp_tensor()
            ops.append(helper.make_node('Constant', [], [min_tensor], graph.get_node_name('Constant'), **kwargs))
            ops.append(helper.make_node('Sub', [input_tensor.name, min_tensor], [output_tensor.name], graph.get_node_name('Sub')))
            graph.add([input_tensor], [output_tensor], ops)
        
    def to_onnx_operator(self, graph):
        if self.strategy == 'min':
            self.__to_onnx_operator_for_min(graph)
            return

        for i, (input_column, output_column) in enumerate(zip(self.inputs, self.outputs)):

            input_tensor = graph.get_current_tensor(input_column)
            output_tensor = graph.get_next_tensor(output_column, input_tensor.type)

            kwargs = {}
            if self.strategy == 'minmax':
                kwargs['offset'] = [float(self.scaler.data_min_[i])]
            elif self.strategy == 'standard':
                kwargs['offset'] = [float(self.scaler.mean_[i])]
            else:
                assert False, 'Unsupported strategy ' + self.strategy
            kwargs['scale'] = [float(self.scaler.scale_[i])]

            graph.add([input_tensor], [output_tensor], [helper.make_node('Scaler', [input_tensor.name], [output_tensor.name], graph.get_node_name('Scaler'), domain='ai.onnx.ml', **kwargs)])
