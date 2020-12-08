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

class RowTransformer(DFPBase):
    """
    Drop rows that satisfy one or more conditions.

    Parameters
    ----------
    columns : List of strings
        Each string is a column label.

    drop_values: List of int, float, or strings
        When a column value matches with one of these values, the corresponding row is dropped.

    reset_index: bool
        Specify whether or not reset index after drop rows
    Examples:
    ----------
    >>> df = pd.DataFrame({'Gender': ['male', 'female', 'p'],
                           'Age': [30, 25, 44]})
    Drop the third row because the Gender column value is 'p'
    >>> tf1 = RowTransformer(columns=['Gender'], drop_values=['p'])
    """
    def __init__(
        self,
        columns=[],
        drop_values=[],
        reset_index = True
    ):
        super().__init__()
        self.columns = columns
        self.drop_values = drop_values
        self.reset_index = reset_index

    def transform(self, df):
        for col in self.columns:
            df.drop(df[df[col].isin(self.drop_values)].index, inplace=True)
        return df.reset_index(drop=True) if self.reset_index else df

    def to_onnx_operator(self, graph):
        assert False, 'Not implemented yet'
