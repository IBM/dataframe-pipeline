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

class MinorityTransformer(DFPBase):
    """
    Replace minority values.

    Parameters
    ----------
    inputs : List of strings
        Each string is an input column label.
    
    outputs : List of strings
        Each string is an output column label.

    value_count : int
        When the number of appearances for a value is less than this threshold, the value is replaced.

    repalced_to : string, int, float
        A value replaced to.

    Examples:
    ----------
    >>> df = pd.DataFrame({'OS': ['Mac', 'iOS', 'iOS', 'Mac', 'OS2', 'Mac', 'iOS']})
    Replace 'OS2' with 'Others'
    >>> tf1 = MinorityTransformer(inputs=['OS'],
                                  outputs=['OS'],
                                  threshold=2,
                                  replaced_to='Others')
    """
    def __init__(
        self,
        inputs=[],
        outputs=[],
        threshold=None,
        replaced_to=None
    ):
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.threshold = threshold
        self.replaced_to = replaced_to

    def transform(self, df):
        for input, output in zip(self.inputs, self.outputs):
            df[output] = df[input].where(df[input].value_counts()[df[input]].values >= self.threshold, other=self.replaced_to)
        return df

    def to_onnx_operator(self, graph):
        assert False, 'Not implemented yet'
