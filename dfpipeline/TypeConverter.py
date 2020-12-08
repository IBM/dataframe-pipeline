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
import numpy as np
import onnx
from . import DFPBase

class TypeConverter(DFPBase):
    """
    Convert column types to other types based on a dicionary.

    Parameters
    ----------
    target_columns : List of strings
        Each string is a column label.
    
    type: String
        A type object converted to.

    Examples:
    ----------
    >>> df = pd.DataFrame({'TransactionAmt': [100, 200, np.nan, 400],
                           'Device': ['SM-G9650', 'SM-G610M', 'SM-310F', 'SM-G610M'],
                           'Browser': ['chrome 66', 'chrome 67', 'chrome 65', 'chrome 67']})
    Replace float64 with float32
    >>> tf1 = TypeConverter(columns=['TransactionAmt'],
                            type=np.float32)
    """
    def __init__(
        self,
        columns=[],
        type=None,
    ):
        super().__init__()
        self.columns = columns
        self.type = type

    def transform(self, df):
        # Check invalid case first
        if self.columns is None or len(self.columns) == 0:
            return df
        if self.type is None:
            return df

        # normal case
        for c in self.columns:
            df[c] = df[c].astype(self.type)

        return df


    def to_onnx_operator(self, inputs, outputs, pipeline=None):
        assert False, 'Not implemented yet'
