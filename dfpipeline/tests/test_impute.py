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
    'col1': [1, 2, 3, np.nan],
    'col2': [1, 3, 5, np.nan]
})

mean_df = pd.DataFrame({
    'col1': [1.0, 2.0, 3.0, 2.0],
    'col2': [1, 3, 5, np.nan]
})

median_df = pd.DataFrame({
    'col1': [1, 2, 3, np.nan],
    'col2': [1.0, 3.0, 5.0, 3.0]
})

const_df = pd.DataFrame({
    'col1': [1.0, 2.0, 3.0, 0.0],
    'col2': [1.0, 3.0, 5.0, 0.0]
})

def test_impute_mean():
    im = dfp.Imputer(inputs=['col1'], outputs=['col1'], strategy='mean')
    out = im.fit_transform(df.copy())
    assert_frame_equal(out, mean_df)

def test_impute_median():
    im = dfp.Imputer(inputs=['col2'], outputs=['col2'], strategy='median')
    out = im.fit_transform(df.copy())
    assert_frame_equal(out, median_df)

def test_impute_const():
    im = dfp.Imputer(inputs=['col1', 'col2'], outputs=['col1', 'col2'], val=0)
    out = im.fit_transform(df.copy())
    assert_frame_equal(out, const_df)
