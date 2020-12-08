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
from pandas.testing import assert_series_equal
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import dfpipeline as dfp

df = pd.DataFrame({
    'col1': ['A', 'B', 'C', 'C'],
    'col2': [1, 1, 2, 2],
    'col3': [2, 2, 2, np.nan],
    'col4': [4, 4, np.nan, 4],
})

label_df = pd.DataFrame({
    'col1': [0, 1, 2, 2],
    'col2': [1, 1, 2, 2],
    'col3': [2, 2, 2, np.nan],
    'col4': [4, 4, np.nan, 4],
})

scale_df = pd.DataFrame({
    'col1': ['A', 'B', 'C', 'C'],
    'col2': [-1.0, -1.0, 1.0, 1.0],
    'col3': [2, 2, 2, np.nan],
    'col4': [4, 4, np.nan, 4],
})

impute_df = pd.DataFrame({
    'col1': ['A', 'B', 'C', 'C'],
    'col2': [1, 1, 2, 2],
    'col3': [2.0, 2.0, 2.0, 2.0],
    'col4': [4.0, 4.0, 4.0, 4.0],
})

def test_label():
    wrap = dfp.WrapTransformer(inputs=['col1'], outputs=['col1'], transformer=LabelEncoder())
    out = wrap.fit_transform(df.copy())
    assert_frame_equal(out, label_df)

def test_scaler():
    wrap = dfp.WrapTransformer(inputs=['col2'], outputs=['col2'], transformer=StandardScaler())
    out = wrap.fit_transform(df.copy())
    assert_frame_equal(out, scale_df)

def test_scaler_for_muliticolumns():
    columns = ['col2', 'col3', 'col4']
    wrap = dfp.WrapTransformer(inputs=columns, outputs=columns, transformer=MinMaxScaler())
    out = wrap.fit_transform(df.copy())
    correct = pd.concat([df['col1'], pd.DataFrame(MinMaxScaler().fit_transform(df[columns].copy()), columns = columns)], axis = 1)
    assert_frame_equal(out, correct)

def test_impute():
    wrap = dfp.WrapTransformer(inputs=['col3', 'col4'], outputs=['col3', 'col4'], transformer=SimpleImputer(strategy='mean'))
    out = wrap.fit_transform(df.copy())
    assert_frame_equal(out, impute_df)
