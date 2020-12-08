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

import pytest

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

import dfpipeline as dfp

df = pd.DataFrame({
    'col1': ['A', 'B', 'C'],
    'col2': ['L', 'M', 'N'],
    'col3': ['X', 'Y', 'Z'],
})

concat1_df = pd.DataFrame({
    'col1': ['A', 'B', 'C'],
    'col2': ['L', 'M', 'N'],
    'col3': ['X', 'Y', 'Z'],
    'col4': ['A', 'B', 'C'],
})

concat2_df = pd.DataFrame({
    'col1': ['A', 'B', 'C'],
    'col2': ['L', 'M', 'N'],
    'col3': ['X', 'Y', 'Z'],
    'col4': ['A_L', 'B_M', 'C_N'],
})

concat3_df = pd.DataFrame({
    'col1': ['A', 'B', 'C'],
    'col2': ['L', 'M', 'N'],
    'col3': ['X', 'Y', 'Z'],
    'col4': ['A_L_X', 'B_M_Y', 'C_N_Z'],
})

def test_concat1():
    concat = dfp.StringConcatenator(inputs=[('col1',)], outputs=['col4'], separator='_')
    out = concat.fit_transform(df)
    assert_frame_equal(out, concat1_df)

def test_concat2():
    concat = dfp.StringConcatenator(inputs=[('col1', 'col2')], outputs=['col4'], separator='_')
    out = concat.fit_transform(df)
    assert_frame_equal(out, concat2_df)

def test_concat3():
    concat = dfp.StringConcatenator(inputs=[('col1', 'col2', 'col3')], outputs=['col4'], separator='_')
    out = concat.fit_transform(df)
    assert_frame_equal(out, concat3_df)
