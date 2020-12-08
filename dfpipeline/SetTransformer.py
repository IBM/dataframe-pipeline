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
import types
from . import DFPBase

class SetTransformer(DFPBase):
    """
    Perform a set operation between two Tensors.

    Parameters
    ----------
    first_operand : Function, string, or List
        Function: If a function is specified, it will be called when value is necessary in fit(), transform(), or export. Its return value must be string or List.
        string: it means a column label
        List: constant array
    
    second_operand : Function, string, or List
        Function: If a function is specified, it will be called when value is necessary in fit(), transform(), or export. Its return value must be string or List.
        string: it means a column label
        List: constant array

    set_operation : String
        '*' or '&': And operation
        '+' or '|': Union operation
        '-': Subtract operation
    
    """
    def __init__(
        self,
        first_operand=None,
        second_operand=None,
        output_operand=None,
        output_func=None,
        set_operation=None
    ):
        super().__init__()
        self.first_operand = first_operand
        self.second_operand = second_operand
        self.output_operand = output_operand
        self.output_func = output_func
        self.set_operation = set_operation

    @classmethod
    def is_method(cls, m):
        return isinstance(m, types.FunctionType) or isinstance(m, types.MethodType) or isinstance(m, types.LambdaType)

    def transform(self, df):
        # Check invalid case first
        if self.first_operand is None or self.second_operand is None or self.set_operation is None or self.output_func is None:
            return df

        first_op = self.first_operand() if self.is_method(self.first_operand) else self.first_operand
        second_op = self.second_operand() if self.is_method(self.second_operand) else self.second_operand

        # Normal case
        first = df[first_op] if type(first_op) == str else first_op
        second = df[second_op] if type(second_op) == str else second_op

        result = None
        if self.set_operation == '*' or self.set_operation == '&':
            result = set(first) & set(second)
        elif self.set_operation == '+' or self.set_operation == '|':
            result = set(first) | set(second)
        elif self.set_operation == '-':
            result = set(first) - set(second)
        else:
            return df
        result = list(result)
        self.output_func(result)
        if len(result) > len(df):
            assert False, "The length of the result is longer than that of DataFrame. len(result)=" + str(len(result)) + " len(df)=" + str(len(df)) 
        elif len(result) < len(df):
            result.extend([None] * (len(df) - len(result)))
        df[self.output_operand] = result

        return df


    def to_onnx_operator(self, inputs, outputs, pipeline=None):
        assert False, 'Not implemented yet'
