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
    'Gender': ['male', 'p', 'female', 'p'],
    'Job': ['sales', 'engineer', 'N/A', 'service'],
    'Age': [30, 22, 25, 44],
})

drop_gender_df = pd.DataFrame({
    'Gender': ['male', 'female'],
    'Job': ['sales', 'N/A'],
    'Age': [30, 25],
})

drop_gender_job_df = pd.DataFrame({
    'Gender': ['male'],
    'Job': ['sales'],
    'Age': [30],
})

def test_gender():
    row = dfp.RowTransformer(columns=['Gender'], drop_values=['p'])
    out = row.fit_transform(df)
    assert_frame_equal(out, drop_gender_df)

def test_gender_job():
    row = dfp.RowTransformer(columns=['Gender', 'Job'], drop_values=['p', 'N/A'])
    out = row.fit_transform(df)
    assert_frame_equal(out, drop_gender_job_df)

