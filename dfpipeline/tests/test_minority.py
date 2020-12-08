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
    'col1': ['A', 'A', 'A', 'B', 'B', 'C'],
})

less_than_3_df = pd.DataFrame({
    'col1': ['A', 'A', 'A', 'B', 'B', 'C'],
    'col2': ['A', 'A', 'A', 'others', 'others', 'others'],
})

less_than_2_df = pd.DataFrame({
    'col1': ['A', 'A', 'A', 'B', 'B', 'C'],
    'col2': ['A', 'A', 'A', 'B', 'B', 'others'],
})

def test_less_than_3():
    minority = dfp.MinorityTransformer(inputs=['col1'], outputs=['col2'], threshold=3, replaced_to='others')
    out = minority.fit_transform(df)
    assert_frame_equal(out, less_than_3_df)

def test_less_than_2():
    minority = dfp.MinorityTransformer(inputs=['col1'], outputs=['col2'], threshold=2, replaced_to='others')
    out = minority.fit_transform(df)
    assert_frame_equal(out, less_than_2_df)
