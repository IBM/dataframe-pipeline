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

#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../')
import MLPipelineBenchmark

import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import dfpipeline as dfp
import xgboost as xgb
import onnx
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import datetime

class FraudDetection1(MLPipelineBenchmark.MLPipelineBenchmark):

    def __init__(self, path):
        super().__init__(path)

    def load_data(self, frac, num_tests):
        # COLUMNS WITH STRINGS
        str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
                    'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
        str_type += ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 
                     'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']

        # FIRST 53 COLUMNS
        cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
                'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
                'M5', 'M6', 'M7', 'M8', 'M9']

        # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
        # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
        v =  [1, 3, 4, 6, 8, 11]
        v += [13, 14, 17, 20, 23, 26, 27, 30]
        v += [36, 37, 40, 41, 44, 47, 48]
        v += [54, 56, 59, 62, 65, 67, 68, 70]
        v += [76, 78, 80, 82, 86, 88, 89, 91]

        #v += [96, 98, 99, 104] #relates to groups, no NAN 
        v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
        v += [124, 127, 129, 130, 136] # relates to groups, no NAN

        # LOTS OF NAN BELOW
        v += [138, 139, 142, 147, 156, 162] #b1
        v += [165, 160, 166] #b1
        v += [178, 176, 173, 182] #b2
        v += [187, 203, 205, 207, 215] #b2
        v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
        v += [218, 223, 224, 226, 228, 229, 235] #b3
        v += [240, 258, 257, 253, 252, 260, 261] #b3
        v += [264, 266, 267, 274, 277] #b3
        v += [220, 221, 234, 238, 250, 271] #b3

        v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
        v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
        v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
        #v += [332, 325, 335, 338] # b4 lots NAN

        cols += ['V'+str(x) for x in v]
        dtypes = {}
        for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]:
            dtypes[c] = 'float32'
        for c in str_type:
            dtypes[c] = 'category'

        X_train = pd.read_csv(self.path + '/../inputs/fraud_detection_1/train_transaction.csv',index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])
        train_id = pd.read_csv(self.path + '/../inputs/fraud_detection_1/train_identity.csv',index_col='TransactionID', dtype=dtypes)
        X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)

        if frac < 1.0:
            X_train = X_train.sample(frac=frac, random_state=1)

        Y_train = X_train['isFraud'].copy()
        del train_id, X_train['isFraud']; x = gc.collect()

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
        cat_columns = []
        num_columns = []
        for c in columns:
            if (np.str(dtypes[c])=='category')|(dtypes[c]=='object'):
                cat_columns.append(c)
            elif c not in ['TransactionAmt','TransactionDT']:
                num_columns.append(c)

        self.pipeline = dfp.DataframePipeline(steps=[
            dfp.FunctionTransformer(inputs=['TransactionDT'], outputs=['day'], func=lambda x: x / 86400.0),
            dfp.FunctionTransformer(inputs=[('D4', 'day'), ('D6', 'day'), ('D7', 'day'), ('D8', 'day'), ('D10', 'day'), ('D11', 'day'), ('D12', 'day'), ('D13', 'day'), ('D14', 'day'), ('D15', 'day')],
                                    outputs=['D4', 'D6', 'D7', 'D8', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15'],
                                    func=lambda x, y: x - y),
            dfp.ComplementLabelEncoder(inputs=cat_columns, outputs=cat_columns),
            dfp.Scaler(inputs=num_columns, outputs=num_columns, strategy='min'),
            # dfp.Imputer(inputs=num_columns, outputs=num_columns, val=-1),
            dfp.StringConcatenator(inputs=[('card1', 'addr1'), ('card1_addr1', 'P_emaildomain')],
                                   outputs=['card1_addr1', 'card1_addr1_P_emaildomain'],
                                   separator='_'),
            dfp.ComplementLabelEncoder(inputs=['card1_addr1', 'card1_addr1_P_emaildomain'],
                                       outputs=['card1_addr1', 'card1_addr1_P_emaildomain']),
            dfp.FrequencyEncoder(inputs=['addr1', 'card1', 'card2', 'card3', 'P_emaildomain', 'card1_addr1', 'card1_addr1_P_emaildomain'],
                                 outputs=['addr1_FE', 'card1_FE', 'card2_FE', 'card3_FE', 'P_emaildomain_FE', 'card1_addr1_FE', 'card1_addr1_P_emaildomain_FE'],
                                 normalize=True),
            dfp.Aggregator(inputs=['TransactionAmt', 'TransactionAmt', 'TransactionAmt',
                                   'D9', 'D9', 'D9',
                                   'D11', 'D11', 'D11'],
                           outputs=['TransactionAmt_card1_mean', 'TransactionAmt_card1_addr1_mean', 'TransactionAmt_card1_addr1_P_emaildomain_mean', 
                                    'D9_card1_mean', 'D9_card1_addr1_mean', 'D9_card1_addr1_P_emaildomain_mean', 
                                    'D11_card1_mean', 'D11_card1_addr1_mean', 'D11_card1_addr1_P_emaildomain_mean'],
                           groupby=['card1', 'card1_addr1', 'card1_addr1_P_emaildomain',
                                    'card1', 'card1_addr1', 'card1_addr1_P_emaildomain',
                                    'card1', 'card1_addr1', 'card1_addr1_P_emaildomain'],
                           func='mean'),
            dfp.Aggregator(inputs=['TransactionAmt', 'TransactionAmt', 'TransactionAmt',
                                   'D9', 'D9', 'D9',
                                   'D11', 'D11', 'D11'],
                           outputs=['TransactionAmt_card1_std', 'TransactionAmt_card1_addr1_std', 'TransactionAmt_card1_addr1_P_emaildomain_std', 
                                    'D9_card1_std', 'D9_card1_addr1_std', 'D9_card1_addr1_P_emaildomain_std', 
                                    'D11_card1_std', 'D11_card1_addr1_std', 'D11_card1_addr1_P_emaildomain_std'],
                           groupby=['card1', 'card1_addr1', 'card1_addr1_P_emaildomain',
                                    'card1', 'card1_addr1', 'card1_addr1_P_emaildomain',
                                    'card1', 'card1_addr1', 'card1_addr1_P_emaildomain'],
                           func='std'),
            dfp.FunctionTransformer(inputs=['TransactionAmt'],
                                    outputs=['cents'],
                                    func=lambda x: x - np.floor(x)),
            dfp.ColumnSelector(columns=['TransactionDT','D6','D7','D8','D9','D12','D13','D14','C3','M5','id_08','id_33', 'card4','id_07','id_14','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_30','id_32','id_34'],
                               drop=True),
        ])

    def do_training(self, X, Y, create_onnx = True):
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        month = X['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
        month = (month.dt.year-2017)*12 + month.dt.month

        print('Pre-processing ...')
        X = self.transform(X, do_fit=True)

        new_columns = {}
        for i, c in enumerate(X.columns):
            new_columns[c] = i
        X = X.rename(columns=new_columns)

        print('Training ...')
        self.clfs = []
        skf = GroupKFold(n_splits=5)
        for i, (idxT, idxV) in enumerate( skf.split(X, Y, groups=month) ):
            m = month.iloc[idxV].iloc[0]
            print('Fold',i,'withholding month',m)
            print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))
            clf = xgb.XGBClassifier(
                n_estimators=5000,
                max_depth=12,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.4,
                # [missing=-1,
                eval_metric='auc',
                # USE CPU
                nthread=32,
                tree_method='hist'
                # USE GPU
                #tree_method='gpu_hist' 
            )
            h = clf.fit(X.iloc[idxT], Y.iloc[idxT], 
                        eval_set=[(X.iloc[idxV],Y.iloc[idxV])],
                        verbose=100, early_stopping_rounds=200)
            self.clfs.append(clf)
        
        # idxT = X.index[:3*len(X)//4]
        # idxV = X.index[3*len(X)//4:]

        # oof = np.zeros(len(idxV))
        # clf = xgb.XGBClassifier(
        #     n_estimators=5000,
        #     max_depth=12,
        #     learning_rate=0.02,
        #     subsample=0.8,
        #     colsample_bytree=0.4,
        #     # missing=-1,
        #     eval_metric='auc',
        #     # USE CPU
        #     nthread=32,
        #     tree_method='hist'
        #     # USE GPU
        #     #tree_method='gpu_hist' 
        # )
        # h = clf.fit(X.loc[idxT], Y.loc[idxT], 
        #             eval_set=[(X.loc[idxV],Y.loc[idxV])],
        #                  verbose=100, early_stopping_rounds=200)
        # self.clfs.append(clf)

        if create_onnx:
            print('Converting models into ONNX ...')
            onnx_ml_models = []
            for i, clf in enumerate(self.clfs):
                initial_type = [('dense_input', FloatTensorType([None, len(self.pipeline.output_columns)]))]
                onnx_ml_models.append(convert_xgboost(clf, initial_types=initial_type))

            self.create_onnx('fraud-detection', onnx_ml_models)
