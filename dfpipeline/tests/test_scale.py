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
    'col1': [0, 1, 2],
    'col2': [1, 2, 3],
})

minmax_df = pd.DataFrame({
    'col1': [0.0, 0.5, 1.0],
    'col2': [1, 2, 3],
})

standard_df = pd.DataFrame({
    'col1': [-1.224744871391589, 0.0, 1.224744871391589],
    'col2': [1, 2, 3],
})

min_df = pd.DataFrame({
    'col1': [0, 1, 2],
    'col2': [0, 1, 2],
})

def test_minmax_scale():
    scaler = dfp.Scaler(inputs=['col1'], outputs=['col1'], strategy='minmax')
    out = scaler.fit_transform(df.copy())
    assert_frame_equal(out, minmax_df)

def test_standard_scale():
    scaler = dfp.Scaler(inputs=['col1'], outputs=['col1'], strategy='standard')
    out = scaler.fit_transform(df.copy())
    assert_frame_equal(out, standard_df)

def test_min_scale():
    scaler = dfp.Scaler(inputs=['col2'], outputs=['col2'], strategy='min')
    out = scaler.fit_transform(df.copy())
    assert_frame_equal(out, min_df)
