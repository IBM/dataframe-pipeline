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

class Aggregator(DFPBase):
    """
    Apply an aggregate operation to column values.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    groupby : List of strings
        This string is a column label for grouping. When this is not specified, an aggregate operation is applied to all of the column values.

    func : Function or string
        This represents an aggregate function such as mean and std. You can specify a function name (i.e., string) such as 'mean' and 'std' instead of function objects.

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, 100, 400],
                           'Country': ['US', 'CAN', 'JP', 'JP']})
    Calculate a mean of all of TransactionAmt
    >>> tf1 = AggregateTransformer(inputs=['TransactionAmt'],
                                   outputs=['TransactionAmt_Mean'],
                                   func=np.mean)
    Calculate a mean of TransactionAmt grouped by Country
    >>> tf2 = AggregateTransformer(inputs=['TransactionAmt'],
                                   outputs=['TransactionAmt_Mean_To_Country'],
                                   groupby=['Country'],
                                   func=np.mean)
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        groupby=[],
        func=None
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.groupby = groupby
        self.func = func
        self.vals = []
        self.maps = []
        assert len(self.inputs) == len(self.outputs)

    def __aggregate(self, X):
        return X.aggregate(self.func)

    def __groupby_aggregate(self, df, col, groupby):
        return df.groupby(groupby)[col].aggregate(self.func).to_dict()
    
    def fit(self, df):
        self.vals.clear()
        self.maps.clear()
        if not self.groupby:
            for input in self.inputs:
                self.vals.append(self.__aggregate(df[input]))
        else:
            for input, groupby in zip(self.inputs, self.groupby):
                self.maps.append(self.__groupby_aggregate(df, input, groupby))
        return self
        
    def transform(self, df):
        if not self.maps:
            for output, v in zip(self.outputs, self.vals):
                df[output] = v
        else:
            for output, groupby, m in zip(self.outputs, self.groupby, self.maps):
                df[output] = df[groupby].map(m)
        return df

    def to_onnx_operator(self, graph):
        for input_column, output_column, m in zip(self.groupby, self.outputs, self.maps):

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

            output_tensor = graph.get_next_tensor(output_column, TensorProto.FLOAT)
            kwargs['values_floats'] = vals
            kwargs['default_float'] = np.nan

            graph.add([input_tensor], [output_tensor], [helper.make_node('LabelEncoder', [input_tensor.name], [output_tensor.name], graph.get_node_name('LabelEncoder'), domain='ai.onnx.ml', **kwargs)])
