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
import numpy as np

class MapTransformer(DFPBase):
    """
    Map column values to other values based on a dictionary.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label. When a column value matches with a key in dict, it is replaced with a value mapped to the key in dict.
    
    outputs : List of strings
        Each string is an output column label. Replaced values are written to an output column. If an output label is same as an input label, the colum value is replaced. If an input column value does not match with any key in dict, the original value is written to the output column.

    dict : Dictionary
        This dictionary defines mapping from input column values that match with keys to the values mapped to the keys.

    default_value: string, int, float
        Default value to be replaced when values do not match with any key in dict.

    dtype : type
        If it is specified, this routine will try converting type to dtype

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, NaN, 400],
                           'Device': ['SM-G9650', 'SM-G610M', 'SM-310F', 'SM-G610M'],
                           'Browser': ['chrome 66', 'chrome 67', 'chrome 65', 'chrome 67']})
    Replace NaN values
    >>> tf1 = MapTransformer(inputs=['TransactionAmt'],
                             outputs=['TransactionAmt'],
                             dict={np.nan: -1})
    Summarize similar strings
    >>> tf2 = MapTransformer(inputs=['Device'],
                             outputs=['Device'],
                             dict={r'.*SM.*': 'Samsung'})
    Eliminate numbers
    >>> tf3 = MapTransformer(inputs=['Browser'],
                             outputs=['Browser'],
                             dict={r'\d+': ''})
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        dict=None,
        regex=False,
        default_value=None,
        dtype = None
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.dict = dict
        self.regex = regex
        self.default_value = default_value  
        self.dtype = dtype
        self.get_typed_key_value_pair(dict) # Type checking

    def transform(self, df):
        df[self.outputs] = df[self.inputs].replace(self.dict, regex=self.regex)
        if self.default_value is not None:
            df[self.outputs] = df[self.outputs].where(df[self.outputs].isin(list(self.dict.values())), other=self.default_value)
        if self.dtype is not None:
            df[self.outputs] = df[self.outputs].astype(self.dtype)
        return df


    def to_onnx_operator(self, graph):
        if self.regex:
            assert False, "Not implemented it yet" 
        for input_column, output_column in zip(self.inputs, self.outputs):

            input_tensor = graph.get_current_tensor(input_column)

            kwargs = {}
            keys = []
            vals = []
            for k, v in self.dict.items():
                keys.append(k)
                vals.append(v)

            type_key = type(keys[0])
            ksuffix, _ = self.convert_type_string(type_key)
            kwargs['keys_' + ksuffix] = keys

            type_val = type(vals[0])
            suffix, suffix_default = self.convert_type_string(type_val)
            tensor_types = self.conv_primitive_types_TensorProto([type_val])
            output_tensor = graph.get_next_tensor(output_column, tensor_types[0])
            kwargs['values_' + suffix] = vals
            kwargs['default_' + suffix_default] = "_Unused" if suffix_default is "string" else 0

            graph.add([input_tensor], [output_tensor], [helper.make_node('LabelEncoder', [input_tensor.name], [output_tensor.name], graph.get_node_name('LabelEncoder'), domain='ai.onnx.ml', **kwargs)])

