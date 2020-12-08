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

class StringConcatenator(DFPBase):
    """
    Concatenate strings in columns.
    
    Parameters
    ----------
    inputs : List of tuples
        Each tuple is a set of the labels for columns to be concatenated.
    
    outputs : List of strings
        Each string is an output column label.

    separator: string
        A string inserted between concatenated strings.

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, 300],
                           'EmailDomain': ['com', 'jp', 'com'],
                           'Country': ['US', 'JP', 'CAN']})
    Concatenate strings
    >>> StringConcatenator(inputs=[('EmailDomain', 'Country')],
                           outputs=['EmailDomain_Country'],
                           separator='_').fit_transform(df)
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        separator=''
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.separator = separator

    def transform(self, df):
        for (output, input_tuple) in zip(self.outputs, self.inputs):
            df[output] = df[input_tuple[0]]
            for i in range(1, len(input_tuple)):
                df[output] = df[output].astype(str).apply(lambda x: x.rstrip('0').rstrip('.')) + self.separator + df[input_tuple[i]].astype(str).apply(lambda x: x.rstrip('0').rstrip('.'))
        return df

    def to_onnx_operator(self, graph):
        for input_columns, output_column in zip(self.inputs, self.outputs):

            ops = []
            
            input_tensors = []
            input_tensor_names = []
            for input_column in input_columns:
                input_tensor = graph.get_current_tensor(input_column)
                input_tensors.append(input_tensor)
                # Cast to STRING
                if input_tensor.type != TensorProto.STRING:
                    cast_kwargs = {'to': TensorProto.STRING}
                    cast_tensor = graph.get_tmp_tensor()
                    ops.append(helper.make_node('Cast', [input_tensor.name], [cast_tensor], graph.get_node_name('Cast'), **cast_kwargs))
                    input_tensor_names.append(cast_tensor)
                else:
                    input_tensor_names.append(input_tensor.name)

            output_tensor = graph.get_next_tensor(output_column, TensorProto.STRING)

            kwargs = {}
            kwargs['separator'] = self.separator

            ops.append(helper.make_node('StringConcat', input_tensor_names, [output_tensor.name], graph.get_node_name('StringConcat'), domain='ai.onnx.ml', **kwargs))
            graph.add(input_tensors, [output_tensor], ops)
