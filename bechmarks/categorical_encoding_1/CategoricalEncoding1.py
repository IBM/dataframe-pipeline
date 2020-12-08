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

import sys
sys.path.append('../')
import MLPipelineBenchmark

import numpy as np
import pandas as pd
import sklearn
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from skl2onnx import convert_sklearn
from onnxmltools.convert.common.data_types import FloatTensorType
import dfpipeline as dfp

class CategoricalEncoding1(MLPipelineBenchmark.MLPipelineBenchmark):

    def __init__(self, path):
        super().__init__(path)

    def load_data(self, frac, num_tests):
        X_train = pd.read_csv(self.path + '/../inputs/categorical_encoding_1/train.csv', index_col='id')

        if frac < 1.0:
            X_train = X_train.sample(frac=frac, random_state=1)

        Y_train = X_train['target'].copy()

        del X_train['target']; x = gc.collect()

        X = X_train
        Y = Y_train

        idx_train = X.index[:3*len(X_train)//4]
        idx_test = X.index[3*len(X_train)//4:]

        X_train = X.loc[idx_train]
        Y_train = Y.loc[idx_train]

        X_test = X.loc[idx_test]
        Y_test = Y.loc[idx_test]

        print('Train shape',X_train.shape,'test shape',X_test.shape)

        return X_train, Y_train, X_test, Y_test

    def define_pipeline(self, columns, dtypes):
        binvar = ['bin_' + str(i) for i in range(1,5)]
        ordvar = ['ord_' + str(i) for i in range(6)]
        nomvar = ['nom_' + str(i) for i in range(10)]
        dmvar  = ['day', 'month']

        self.pipeline = dfp.DataframePipeline(steps=[
            dfp.MapTransformer(inputs=['bin_3'], outputs=['bin_3'], dict={'F':0, 'T':1}),
            dfp.MapTransformer(inputs=['bin_4'], outputs=['bin_4'], dict={'N':0, 'Y':1}),
            dfp.FunctionTransformer(inputs=['bin_0'], outputs=['bin_0'], func=lambda x: x - 1),
            dfp.MapTransformer(inputs=['ord_1'], outputs=['ord_1'], dict={'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4}),
            dfp.MapTransformer(inputs=['ord_2'], outputs=['ord_2'], dict={'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5}),
            dfp.ComplementLabelEncoder(inputs=['ord_3', 'ord_4', 'ord_5'], outputs=['ord_3', 'ord_4', 'ord_5']),
            dfp.Scaler(inputs=ordvar, outputs=ordvar, strategy='standard'),
            dfp.StringSplitter(inputs=['nom_5'], outputs=['nom_5'], index=8, keep=-1),
            dfp.StringSplitter(inputs=['nom_6', 'nom_7', 'nom_8', 'nom_9'],
                               outputs=['nom_6', 'nom_7', 'nom_8', 'nom_9'],
                               index=3,
                               keep=-1),
            # dfp.OneHotTransformer(onehot_columns=nomvar+dmvar),
            dfp.OneHotEncoder(columns=['nom_0','nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6']+dmvar),
            dfp.ColumnSelector(columns=['bin_0']+nomvar+dmvar, drop=True)
        ])

    def do_training(self, X, Y, create_onnx = True):
        print('Pre-processing ...')
        X = self.transform(X, do_fit=True)

        print('Training ...')
        C = 0.12
        self.clfs = []
        clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, verbose=1, n_jobs=32)
        clf.fit(X, Y)

        self.clfs.append(clf)

        if create_onnx:
            print('Converting models into ONNX ...')
            onnx_ml_models = []
            for i, clf in enumerate(self.clfs):
                initial_type = [('dense_input', FloatTensorType([None, len(self.pipeline.output_columns)]))]
                onnx_ml_models.append(convert_sklearn(clf, initial_types=initial_type, options={type(clf): {'zipmap': False}}))

            self.create_onnx('categorical-encoding', onnx_ml_models)
