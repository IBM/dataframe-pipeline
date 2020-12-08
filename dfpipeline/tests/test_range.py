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

df = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [3, 4, 6, 6, 9, None, 17, 17, 20, 100]})
range1 = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [3, 4, 6, 6, 9, None, 17, 17, 20, 100], 'C2_norm': [-1000, -1000, -1000, -1000, 17, None, 17, 17, 17, 1000]})
range2 = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [-1000, -1000, -1000, -1000, 17, None, 17, 17, 17, 1000]})
range3 = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [-1000, -1000, -1000, -1000, 15.75, None, 15.75, 15.75, 15.75, 1000]})

df2 = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [3, 4, 6, 6, 9, None, 17, 17, 20, 100], 'C3': [300, 200, 100, 20, 17, 17, None, 9, 6, 6]})
range4 = pd.DataFrame({'sex': ["male", "female", "female", "female", "male", "female", "male", "male", "female", "female"], 'C2': [-1000, -1000, -1000, -1000, 15.75, None, 15.75, 15.75, 15.75, 1000], 'C3': [1000, 1000, 1000, 15.75, 15.75, 15.75, None, 15.75, -1000, -1000]})


def test_range1():
    r = dfp.RangeTransformer(inputs=['C2'], outputs=['C2_norm'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'median'})
    out = r.fit_transform(df.copy())
    assert_frame_equal(out, range1)

def test_range2():
    r = dfp.RangeTransformer(inputs=['C2'], outputs=['C2'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'median'})
    out = r.fit_transform(df.copy())
    assert_frame_equal(out, range2)

def test_range2a():
    r = dfp.RangeTransformer(inputs=['C2'], outputs=['C2'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'median', (None, None): 0})
    out = r.fit_transform(df.copy())
    assert_frame_equal(out, range2)

def test_range2b():
    r = dfp.RangeTransformer(inputs=['C2'], outputs=['C2'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'most_frequent'})
    out = r.fit_transform(df.copy())
    assert_frame_equal(out, range2)

def test_range3():
    r = dfp.RangeTransformer(inputs=['C2'], outputs=['C2'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'mean'})
    out = r.fit_transform(df.copy())
    assert_frame_equal(out, range3)

def test_range4():
    r = dfp.RangeTransformer(inputs=['C2', 'C3'], outputs=['C2', 'C3'], dict={(None, 20): 1000, (6, None): -1000, (20, 9): 'mean'})
    out = r.fit_transform(df2.copy())
    assert_frame_equal(out, range4)
