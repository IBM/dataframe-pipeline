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
from numpy import *
from pandas.testing import assert_frame_equal

import dfpipeline as dfp

df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [10, 20, 30, 40],
    'col3': [1, 4, 9, 16],
})

add_constant_df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [10, 20, 30, 40],
    'col3': [1, 4, 9, 16],
    'col4': [2, 3, 4, 5],
})

add_two_columns_df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [10, 20, 30, 40],
    'col3': [1, 4, 9, 16],
    'col4': [11, 22, 33, 44],
})

np_sqrt_df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [10, 20, 30, 40],
    'col3': [1, 4, 9, 16],
    'col4': [1.0, 2.0, 3.0, 4.0],
})

def test_add_constant():
    func = dfp.FunctionTransformer(inputs=['col1'], outputs=['col4'], func=lambda x: x + 1)
    out = func.fit_transform(df.copy())
    assert_frame_equal(out, add_constant_df)

def test_add_two_columns():
    func = dfp.FunctionTransformer(inputs=[('col1', 'col2')], outputs=['col4'], func=lambda x, y: x + y)
    out = func.fit_transform(df.copy())
    assert_frame_equal(out, add_two_columns_df)

def test_np_sqrt():
    func = dfp.FunctionTransformer(inputs=['col3'], outputs=['col4'], func=np.sqrt)
    out = func.fit_transform(df.copy())
    assert_frame_equal(out, np_sqrt_df)
