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
from pandas.testing import assert_series_equal

import dfpipeline as dfp

df = pd.DataFrame({
    'col1': [86401, 106401, 206400, 3064000],
    'col2': ['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31'],
})

seconds_df = pd.DataFrame({
    'col1': [86401, 106401, 206400, 3064000],
    'col2': ['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31'],
    'col1_WY': [48, 48, 48, 1],
    'col1_DY': [335, 335, 336, 4],
    'col1_DW': [4, 4, 5, 3],
})
seconds_df['col1_WY'] = seconds_df['col1_WY'].astype(np.int64)
seconds_df['col1_DY'] = seconds_df['col1_DY'].astype(np.int64)
seconds_df['col1_DW'] = seconds_df['col1_DW'].astype(np.int64)

date_df = pd.DataFrame({
    'col1': [86401, 106401, 206400, 3064000],
    'col2': ['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31'],
    'col2_WY': [5, 18, 31, 44],
    'col2_DY': [31, 120, 212, 304],
    'col2_DW': [2, 0, 1, 2],
})
date_df['col2_WY'] = date_df['col2_WY'].astype(np.int64)
date_df['col2_DY'] = date_df['col2_DY'].astype(np.int64)
date_df['col2_DW'] = date_df['col2_DW'].astype(np.int64)

def test_seconds():
    time = dfp.DateTransformer(column='col1', origin='2017-11-30')
    out = time.fit_transform(df.copy())
    assert_series_equal(out['col1_WY'], seconds_df['col1_WY'])
    assert_series_equal(out['col1_DY'], seconds_df['col1_DY'])
    assert_series_equal(out['col1_DW'], seconds_df['col1_DW'])

def test_date():
    time = dfp.DateTransformer(column='col2')
    out = time.fit_transform(df.copy())
    assert_series_equal(out['col2_WY'], date_df['col2_WY'])
    assert_series_equal(out['col2_DY'], date_df['col2_DY'])
    assert_series_equal(out['col2_DW'], date_df['col2_DW'])
