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

nan_dict = {np.nan: 'missing'}
exact_dict = {'device-1': 'device'}
regex_dict = {r'.*device.*': 'device'}

nan_df = pd.DataFrame({
    'col1': ['device-1', np.nan, np.nan],
    'col2': ['device-2', 'device-3', np.nan],
})

df = pd.DataFrame({
    'col1': ['device-1', 'missing', 'missing'],
    'col2': ['device-2', 'device-3', 'missing'],
})

exact_df = pd.DataFrame({
    'col1': ['device', 'missing', 'missing'],
    'col2': ['device-2', 'device-3', 'missing'],
})

exact_others_df = pd.DataFrame({
    'col1': ['device', 'others', 'others'],
    'col2': ['others', 'others', 'others'],
})

regex_df = pd.DataFrame({
    'col1': ['device', 'missing', 'missing'],
    'col2': ['device', 'device', 'missing'],
})

def test_nan():
    map = dfp.MapTransformer(inputs=['col1', 'col2'], outputs=['col1', 'col2'], dict=nan_dict)
    out = map.fit_transform(nan_df.copy())
    assert_frame_equal(out, df)

def test_exact():
    map = dfp.MapTransformer(inputs=['col1', 'col2'], outputs=['col1', 'col2'], dict=exact_dict)
    out = map.fit_transform(df.copy())
    assert_frame_equal(out, exact_df)

def test_default():
    map = dfp.MapTransformer(inputs=['col1', 'col2'], outputs=['col1', 'col2'], dict=exact_dict, default_value='others')
    out = map.fit_transform(df.copy())
    assert_frame_equal(out, exact_others_df)

def test_regex():
    map = dfp.MapTransformer(inputs=['col1', 'col2'], outputs=['col1', 'col2'], dict=regex_dict, regex=True)
    out = map.fit_transform(df.copy())
    assert_frame_equal(out, regex_df)
