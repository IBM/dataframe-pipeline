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
    'col1': [2, 2, 4, 4],
    'col2': ['device-1', 'device-1', 'device-2', 'device-2'],
})

mean_df = pd.DataFrame({
    'col1': [2, 2, 4, 4],
    'col2': ['device-1', 'device-1', 'device-2', 'device-2'],
    'mean': [3.0, 3.0, 3.0, 3.0]
})

mean_to_col2_df = pd.DataFrame({
    'col1': [2, 2, 4, 4],
    'col2': ['device-1', 'device-1', 'device-2', 'device-2'],
    'mean_to_col2': [2, 2, 4, 4]
})

count_df = pd.DataFrame({
    'col1': [2, 2, 4, 4],
    'col2': ['device-1', 'device-1', 'device-2', 'device-2'],
    'count': [2, 2, 2, 2],
})

def test_mean():
    agg = dfp.Aggregator(inputs=['col1'], outputs=['mean'], func='mean')
    out = agg.fit_transform(df.copy())
    assert_frame_equal(out, mean_df)

def test_groupby_mean():
    agg = dfp.Aggregator(inputs=['col1'], outputs=['mean_to_col2'], groupby=['col2'], func='mean')
    out = agg.fit_transform(df.copy())
    assert_frame_equal(out, mean_to_col2_df)

def test_groupby_count():
    agg = dfp.Aggregator(inputs=['col2'], outputs=['count'], groupby=['col2'], func='count')
    out = agg.fit_transform(df.copy())
    assert_frame_equal(out, count_df)
