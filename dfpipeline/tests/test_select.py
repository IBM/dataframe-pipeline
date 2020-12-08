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

selected_df = pd.DataFrame({
    'col1': ['A', 'B', 'C'],
    'col2': ['L', 'M', 'N'],
})

dropped_df = pd.DataFrame({
    'col2': ['L', 'M', 'N'],
    'col3': ['X', 'Y', 'Z'],
})

def test_select():
    select = dfp.ColumnSelector(columns=['col1', 'col2'])
    out = select.fit_transform(df.copy())
    assert_frame_equal(out, selected_df)

def test_drop():
    select = dfp.ColumnSelector(columns=['col1'], drop=True)
    out = select.fit_transform(df.copy())
    assert_frame_equal(out, dropped_df)
