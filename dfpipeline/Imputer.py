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
import copy
import numpy as np

import onnx
from onnx import helper

class Imputer(DFPBase):
    """
    Replace missing values with a value.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    strategy : String
        A strategy to replace missing values. You can specify 'median' or 'mean'.

    val : Any
        A constant value to replace missing values. This is ignored when strategy is specified.
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        strategy = None,
        val=-1
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.strategy = strategy
        self.val = val
        self.values = None

    def transform(self, df):
        done = False
        self.values = []
        if self.strategy != None:
            if self.strategy is "median":
                for input, output in zip(self.inputs, self.outputs):
                    self.val = df[input].median()
                    df[output] = df[input].fillna(self.val)
                    self.values.append(self.val)
                done = True
            elif self.strategy is "mean":
                for input, output in zip(self.inputs, self.outputs):
                    self.val = df[input].mean()
                    df[output] = df[input].fillna(self.val)
                    self.values.append(self.val)
                done = True
            else:
                assert False, 'Unknown strategy ' + self.strategy

        if not done:
            for input, output in zip(self.inputs, self.outputs):
                df[output] = df[input].fillna(self.val)
                self.values.append(self.val)
        return df

    def to_onnx_operator(self, graph):
        for input_column, output_column, val in zip(self.inputs, self.outputs, self.values):

            input_tensor = graph.get_current_tensor(input_column)
            output_tensor = graph.get_next_tensor(output_column, input_tensor.type)

            kwargs = {}
            if graph.is_int_tensor(input_tensor.type):
                kwargs['imputed_value_int64s'] = [int(val)]
                kwargs['replaced_value_int64'] = int(0)
            elif graph.is_float_tensor(input_tensor.type):
                kwargs['imputed_value_floats'] = [float(val)]
                kwargs['replaced_value_float'] = np.nan
            elif graph.is_string_tensor(input_tensor.type):
                kwargs['imputed_value_strings'] = [val]
                kwargs['replaced_value_string'] = "NaN"
            else:
                assert False, input_column + ' column is not a numeric type'

            graph.add([input_tensor], [output_tensor], [helper.make_node('Imputer', [input_tensor.name], [output_tensor.name], graph.get_node_name('Imputer'), domain='ai.onnx.ml', **kwargs)])
