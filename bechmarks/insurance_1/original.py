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

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

X_train = pd.read_csv('train.csv')

Y_train = X_train['QuoteConversion_Flag'].copy()

X_train = X_train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)

X = X_train
Y = Y_train

idx_train = X.index[:3*len(X_train)//4]
idx_test = X.index[3*len(X_train)//4:]

X_train = X.loc[idx_train]
Y_train = Y.loc[idx_train]

X_test = X.loc[idx_test]
Y_test = Y.loc[idx_test]

print('Train shape',X_train.shape,'test shape',X_test.shape)

# Lets play with some dates
X_train['Date'] = pd.to_datetime(pd.Series(X_train['Original_Quote_Date']))
X_train = X_train.drop('Original_Quote_Date', axis=1)

X_test['Date'] = pd.to_datetime(pd.Series(X_test['Original_Quote_Date']))
X_test = X_test.drop('Original_Quote_Date', axis=1)

X_train['Year'] = X_train['Date'].apply(lambda x: int(str(x)[:4]))
X_train['Month'] = X_train['Date'].apply(lambda x: int(str(x)[5:7]))
X_train['weekday'] = X_train['Date'].dt.dayofweek


X_test['Year'] = X_test['Date'].apply(lambda x: int(str(x)[:4]))
X_test['Month'] = X_test['Date'].apply(lambda x: int(str(x)[5:7]))
X_test['weekday'] = X_test['Date'].dt.dayofweek

X_train = X_train.drop('Date', axis=1)
X_test = X_test.drop('Date', axis=1)

# X_train = X_train.fillna(-1)
# X_test = X_test.fillna(-1)

for f in X_train.columns:
    if X_train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

clf = xgb.XGBClassifier(n_estimators=1700,
                        nthread=32,
                        max_depth=6,
                        learning_rate=0.024,
                        silent=True,
                        subsample=0.8,
                        colsample_bytree=0.65)
                        
xgb_model = clf.fit(X_train, Y_train, eval_metric="auc")

preds = clf.predict_proba(X_test)[:,1]

targets = Y_test.to_numpy()
predictions = [round(value) for value in preds]
accuracy = accuracy_score(targets, predictions)
score = roc_auc_score(targets, preds)
print("Accuracy: %.4f" % accuracy)
print("RocAuc: %.4f" % score)
print()
