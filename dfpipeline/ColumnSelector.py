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

class ColumnSelector(DFPBase):
    """
    Select columns.

    Parameters
    ----------
    columns : List of strings
        Each string is a column label. When drop is False (default), the specified columns are kept and the other columns are dropped. When drop is True, the specified columns are dropped.

    drop : bool
        This boolean value controls whether keep or drop the specified columns.

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, NaN, 400],
                           'Device': ['SM-G9650', 'SM-G610M', 'SM-310F', 'SM-G610M'],
                           'Browser': ['chrome 66', 'chrome 67', 'chrome 65', 'chrome 67']})
    Drop the 'Browser' column
    >>> tf1 = ColumnSelector(columns=['TransactionAmt', 'Device'])
    or
    >>> tf1 = ColumnSelector(columns=['Browser'], drop=True)
    """
    def __init__(
        self,
        columns=[],
        drop=False
    ):
        super().__init__()
        self.columns = columns
        self.drop = drop

    def fit(self, df):
        if self.drop == True:
            self.selected_columns = list(set(df.columns) - set(self.columns))
            self.drop_columns = self.columns
        else:
            self.selected_columns = self.columns
            self.drop_columns = list(set(df.columns) - set(self.columns))
        return self
    
    def transform(self, df):
        df.drop(columns=self.drop_columns, inplace=True)
        return df

    def to_onnx_operator(self, graph):
        for c in self.selected_columns:
            graph.get_current_tensor(c)
        graph.drop(self.drop_columns)
