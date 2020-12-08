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
    'Email': ['taro.jp.com', 'alice.us.com', 'bob.us'],
    'ID': ['abcd', 'klmn', 'wxyz'],
})

prefix_df = pd.DataFrame({
    'Email': ['taro.jp.com', 'alice.us.com', 'bob.us'],
    'ID': ['abcd', 'klmn', 'wxyz'],
    'Email_prefix': ['taro', 'alice', 'bob'],
})

suffix_df = pd.DataFrame({
    'Email': ['taro.jp.com', 'alice.us.com', 'bob.us'],
    'ID': ['abcd', 'klmn', 'wxyz'],
    'Email_suffix': ['com', 'com', 'us'],
})

index_prefix_df = pd.DataFrame({
    'Email': ['taro.jp.com', 'alice.us.com', 'bob.us'],
    'ID': ['abcd', 'klmn', 'wxyz'],
    'ID_prefix': ['ab', 'kl', 'wx'],
})

index_suffix_df = pd.DataFrame({
    'Email': ['taro.jp.com', 'alice.us.com', 'bob.us'],
    'ID': ['abcd', 'klmn', 'wxyz'],
    'ID_suffix': ['cd', 'mn', 'yz'],
})

def test_split_prefix():
    split = dfp.StringSplitter(inputs=['Email'], outputs=['Email_prefix'], separator='.', keep=0)
    out = split.fit_transform(df.copy())
    assert_frame_equal(out, prefix_df)

def test_split_suffix():
    split = dfp.StringSplitter(inputs=['Email'], outputs=['Email_suffix'], separator='.', keep=-1)
    out = split.fit_transform(df.copy())
    assert_frame_equal(out, suffix_df)

def test_index_split_prefix():
    split = dfp.StringSplitter(inputs=['ID'], outputs=['ID_prefix'], index=2, keep=0)
    out = split.fit_transform(df.copy())
    assert_frame_equal(out, index_prefix_df)

def test_index_split_suffix():
    split = dfp.StringSplitter(inputs=['ID'], outputs=['ID_suffix'], index=2, keep=-1)
    out = split.fit_transform(df.copy())
    assert_frame_equal(out, index_suffix_df)
