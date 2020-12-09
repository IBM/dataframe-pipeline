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

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import sklearn
import gc

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

X_train = pd.read_csv('train.csv', index_col='id')
Y_train = X_train['target'].copy()
del X_train['target']; x = gc.collect()

binvar = ['bin_' + str(i) for i in range(1,5)]
ordvar = ['ord_' + str(i) for i in range(6)]
nomvar = ['nom_' + str(i) for i in range(10)]
dmvar  = ['day', 'month']

X_train.drop('bin_0', inplace=True, axis=1)

X_train['bin_3'] = X_train['bin_3'].map({'F':0, 'T':1})
X_train['bin_4'] = X_train['bin_4'].map({'N':0, 'Y':1})

X_train['ord_0'] = X_train['ord_0'] - 1

ord1dict = {'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4}
X_train['ord_1'] = X_train['ord_1'].map(ord1dict)

ord2dict = {'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5}
X_train['ord_2'] = X_train['ord_2'].map(ord2dict)

oe = OrdinalEncoder(categories='auto')
X_train[ordvar[3:]] = oe.fit_transform(X_train[ordvar[3:]])
for var, cl in zip(ordvar[3:], oe.categories_):
    print(var)
    print(cl)

X_train[ordvar] = StandardScaler().fit_transform(X_train[ordvar])

X_train['nom_5'] = X_train['nom_5'].str[4:]
X_train['nom_6'] = X_train['nom_6'].str[3:]
X_train['nom_7'] = X_train['nom_7'].str[3:]
X_train['nom_8'] = X_train['nom_8'].str[3:]
X_train['nom_9'] = X_train['nom_9'].str[3:]

onehot_columns = nomvar+dmvar
print('OneHotEncoding for ' + str(onehot_columns))
for c in onehot_columns:
    onehot_df = pd.get_dummies(X_train[c])
    print('Adding ' + str(len(onehot_df.columns)) + ' columns')
    for oc in onehot_df.columns:
        X_train[c + '_' + oc] = onehot_df[oc]
X_train.drop(nomvar+dmvar, inplace=True, axis=1)
print('Done')

X = X_train
Y = Y_train

idx_train = X.index[:3*len(X_train)//4]
idx_test = X.index[3*len(X_train)//4:]

X_train = X.loc[idx_train]
Y_train = Y.loc[idx_train]

X_test = X.loc[idx_test]
Y_test = Y.loc[idx_test]

print('Train shape',X_train.shape,'test shape',X_test.shape)

C = 0.12
clf = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, verbose=0, n_jobs=-1)
clf.fit(X_train, Y_train)

preds = clf.predict_proba(X_test)[:,1]
print(preds)

targets = Y_test.to_numpy()
predictions = [round(value) for value in preds]
accuracy = accuracy_score(targets, predictions)
score = roc_auc_score(targets, preds)
print("Accuracy: %.4f" % accuracy)
print("RocAuc: %.4f" % score)
print()
