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
    'col1': ['device-1', 'device-1', 'device-2', 'device-2'],
})

freq_df = pd.DataFrame({
    'col1': ['device-1', 'device-1', 'device-2', 'device-2'],
    'col2': [2, 2, 2, 2]
})

norm_freq_df = pd.DataFrame({
    'col1': ['device-1', 'device-1', 'device-2', 'device-2'],
    'col2': [0.5, 0.5, 0.5, 0.5]
})

def test_freq():
    fe = dfp.FrequencyEncoder(inputs=['col1'], outputs=['col2'])
    out = fe.fit_transform(df.copy())
    assert_frame_equal(out, freq_df)

def test_norm_freq():
    fe = dfp.FrequencyEncoder(inputs=['col1'], outputs=['col2'], normalize=True)
    out = fe.fit_transform(df.copy())
    assert_frame_equal(out, norm_freq_df)
