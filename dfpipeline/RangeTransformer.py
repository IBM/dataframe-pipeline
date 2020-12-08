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

class RangeTransformer(DFPBase):
    """
    Replace a value in a range into another value.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    dict : Map from a value range to a value
        Each key is a tuple consists of an upper bound and a lower bound. For example, when an entry is {(20, 10): 1}, a value that is greater than or equal to 10 and is lower than or equal to 20 is converted into 1. When a bound is specified by None, the bound is infinite. When multiple ranges overlap, it will use the last match. You can specify three special strings: (mean, median, and most_frequent) for the values.

    Examples:
    ----------
    >>> df = pd.DataFrame({'Age': [2, 28, 70, 55]})
    Map the Age values that are greater than or equals to 60 to 60
    Map the Age values that are lower than or equals to 20 to 20
    >>> tf1 = RangeTransformer(inputs=['Age'],
                               outputs=['Age'],
                               dict={(None, 60): 60, 
                                     (20, None): 20})
    Map the Age values that are greater than or equals to 60 to the mean value of the Age column
    Map the Age values that is lower than or equals to 20 to the mean value of the Age column
    >>> tf2 = RangeTransformer(inputs=['Age'],
                               outputs=['Age'],
                               dict={(None, 60): 'mean', 
                                     (20, None): 'mean'})
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        dict=None,
        use_all_elements = False
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.dict = dict
        self.values_dict = {}
        self.use_all_elements = use_all_elements

    def range_handler_C(self, C):
        if C is None: return C
        ret = C.copy()
        for k, v in self.dict.items():
            k1, k2 = k
            if k1 is None:
                if k2 is not None:
                    s = C[C >= k2]
                else:
                    continue    # ignore (None, None)
            else:   # k1 is not None
                if k2 is not None:
                    s = C[k1 >= C][C >= k2]
                else:
                    s = C[k1 >= C]

            tgt = C if self.use_all_elements else s
            if v == 'mean':
                newv = tgt.mean()
            elif v == 'median':
                newv = tgt.median()
            elif v == 'most_frequent':
                newv = tgt.value_counts().idxmax()
            else:
                newv = v

            ret[s.index] = newv
            self.values_dict[(C.name, k1, k2)] = float(newv)
        return ret

    def transform(self, df):
        # Check invalid case first
        if self.dict is None or len(self.dict) == 0:
            return df

        return self.function_transformer_by_column(df, self.inputs, self.outputs, self.range_handler_C)

    def to_onnx_operator(self, graph):
        key_type, value_type, keys, values = self.get_typed_key_value_pair(self.dict, ignore_typecheck=True)
        ku_list = []
        kl_list = []
        for ku, kl in keys:
            ku_list.append(str(ku))
            kl_list.append(str(kl))

        kwargs = {}
        kwargs['keys_upper_strings'] = ku_list
        kwargs['keys_lower_strings'] = kl_list

        for input_column, output_column in zip(self.inputs, self.outputs):

            input_tensor = graph.get_current_tensor(input_column)
            print("Calling graph.get_next_tensor with ", input_column, output_column, input_tensor.type)
            output_tensor = graph.get_next_tensor(output_column, input_tensor.type)
            values_list = []
            for ku, kl in keys:
                values_list.append(self.values_dict[(input_column, ku, kl)])
            kwargs['values_float'] = values_list

            graph.add([input_tensor], [output_tensor], [helper.make_node('RangeTransformer', [input_tensor.name], [output_tensor.name], graph.get_node_name('RangeTransformer'), domain='ai.onnx.ml', **kwargs)])

