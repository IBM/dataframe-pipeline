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
    'col1': ['a', 'a', 'b', np.nan],
})

encoded_df = pd.DataFrame({
    'col1': [0, 0, 1, 2],
})
encoded_df['col1'] = encoded_df['col1'].astype(np.int32)

def test_le():
    le = dfp.ComplementLabelEncoder(inputs=['col1'], outputs=['col1'])
    out = le.fit_transform(df.copy())
    assert_frame_equal(out, encoded_df)
