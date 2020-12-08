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
    'sex': ["male", "female", "female", "male", "female", "male", "female", "female"],
    'C2': [3, 4, 6, 9, None, 17, 20, 100]
})
result_df = df.copy()
result_df['C2'] = result_df['C2'].astype(np.float64)


def test_typeconv():
    conv = dfp.TypeConverter(columns=['C2'], type=np.float64)
    out = conv.fit_transform(df.copy())
    assert_frame_equal(out, result_df)
