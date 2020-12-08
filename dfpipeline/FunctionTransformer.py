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
from . import WalkBytecode, GraphFactory

import onnx
from onnx import helper

import inspect
import numpy as np

class FunctionTransformer(DFPBase):
    """
    Apply a function to input columns.

    Parameters
    ----------
    inputs : List of strings or tuples
        Each string is an input column label. Each value of the column becomes an argument for a function.
        Each tuple is a set of input column labels (strings). Values of the columns become arguments for a function.
    
    outputs : List of strings
        Each string is an output column label. Outputs of a function are written to an output column. When an output column label is same as an input column label, the column value is replaced.

    func : Function
        This function defines operations that are applied to input column values.

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, 300],
                           'EmailDomain': ['com', 'jp', 'com'],
                           'Country': ['US', 'JP', 'CAN']})
    Calculate a logarithm
    >>> tf1 = FunctionTransformer(inputs=['TransactionAmt'],
                                  outputs=['TransactionAmt'],
                                  func=lambda x: np.log(x + 0.001))
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        func=None
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.func = func

    def transform(self, df):
        # GraphFactory(self.func).run()
        return self.function_transformer_by_element(df, self.inputs, self.outputs, self.func)

    def to_onnx_operator(self, graph, pipeline=None):
        input_tensor = None
        for input_columns, output_column in zip(self.inputs, self.outputs):
            input_tensors = []
            input_tensor_names = []
            if type(input_columns) is tuple:
                for input_column in input_columns:
                    prev_input_tensor = input_tensor
                    input_tensor = graph.get_current_tensor(input_column)
                    if prev_input_tensor is not None:
                        assert input_tensor.type == prev_input_tensor.type
                    input_tensors.append(input_tensor)
                    input_tensor_names.append(input_tensor.name)
            else:
                input_tensor = graph.get_current_tensor(input_columns)
                input_tensors.append(input_tensor)
                input_tensor_names.append(input_tensor.name)

            output_tensor = graph.get_next_tensor(output_column, input_tensor.type)
            graph.add(input_tensors, [output_tensor], WalkBytecode(self.func, [input_columns], [output_column], input_tensor_names, [output_tensor.name], graph.inc_node_count, pipeline).createONNX())
