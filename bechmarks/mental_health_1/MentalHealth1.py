import sys
sys.path.append('../')
import MLPipelineBenchmark

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import gc
import dfpipeline as dfp
from dfpipeline import *
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []

genders = {
    "m": "male",
    "male-ish": "male",
    "maile": "male",
    "mal": "male",
    "male (cis)": "male",
    "make": "male",
    "man": "male",
    "msle": "male",
    "mail": "male",
    "malr": "male",
    "cis man": "male",
    "Cis Male": "male",
    "cis male": "male",
    "cis female": "female",
    "f": "female",
    "woman": "female",
    "femake": "female",
    "cis-female/femme": "female",
    "female (cis)": "female", 
    "femail": "female",
    "trans-female": "trans",
    "something kinda male?": "trans",
    "queer/she/they": "trans",
    "non-binary": "trans",
    "nah": "trans",
    "all": "trans",
    "enby": "trans", 
    "fluid": "trans",
    "genderqueer": "trans",
    "androgyne": "trans",
    "agender": "trans",
    "male leaning androgynous": "trans",
    "guy (-ish) ^_^": "trans",
    "trans woman": "trans",
    "neuter": "trans",
    "female (trans)": "trans",
    "queer": "trans",
    "ostensibly male, unsure what that really means": "trans"
}

feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']

class MentalHealth1(MLPipelineBenchmark.MLPipelineBenchmark):

    def __init__(self, path):
        super().__init__(path)

    def load_data(self, frac, num_tests):
        X_train = pd.read_csv(self.path + '/../inputs/mental_health_1/survey.csv')
        X_train[intFeatures] = X_train[intFeatures].astype(np.float32)

        if frac < 1.0:
            X_train = X_train.sample(frac=frac, random_state=1)

        X_train = RowTransformer(columns=['Gender'],
                            drop_values=['A little about you', 'p'],
                            reset_index = False).fit_transform(X_train)

        Y_train = X_train.treatment.copy()
        Y_train = Y_train.map(lambda x: 1 if x == 'Yes' else 0)

        del X_train['treatment']; x = gc.collect()

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
        for c in columns:
            if dtypes[c]=='object':
                cat_columns.append(c)

        self.pipeline = dfp.DataframePipeline(steps=[
            #dealing with missing data
            #Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
            # train_df = train_df.drop(['comments'], axis= 1)
            # train_df = train_df.drop(['state'], axis= 1)
            # train_df = train_df.drop(['Timestamp'], axis= 1)
            ColumnSelector(columns=['comments', 'state', 'Timestamp'],
                            drop=True),

            Imputer(inputs=intFeatures,
                    outputs=intFeatures,
                    val=0),
            Imputer(inputs=stringFeatures,
                    outputs=stringFeatures,
                    val='NaN'),

            FunctionTransformer(inputs=['Gender'],
                        outputs=['Gender'],
                        func=lambda x: str.lower(x)),
            MapTransformer(inputs=['Gender'],
                        outputs=['Gender'],
                        dict=genders),

            #Get rid of bullshit
            # stk_list = ['A little about you', 'p']
            # train_df = train_df[~train_df['Gender'].isin(stk_list)]
            RowTransformer(columns=['Gender'],
                        drop_values=['a little about you', 'p'],
                        reset_index = False),

            #complete missing age with mean
            # train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
            Imputer(inputs=['Age'],
                    outputs=['Age'],
                    strategy='median'),

            # Fill with media() values < 18 and > 120
            # s = pd.Series(train_df['Age'])
            # s[s<18] = train_df['Age'].median()
            # train_df['Age'] = s
            # s = pd.Series(train_df['Age'])
            # s[s>120] = train_df['Age'].median()
            # train_df['Age'] = s
            RangeTransformer(inputs=['Age'],
                            outputs=['Age'],
                            dict={(None, 121): 'median',
                                (17, None): 'median'}),

            #Ranges of Age
            # train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)
            #RangeTransformer(inputs=['Age'],
            #                 outputs=['age_range'],
            #                 dict={(0, 20): '0-20',
            #                       (21, 30): '21-30',
            #                       (31, 65): '31-65',
            #                       (66, 100): '66-100'}),

            #There are only 0.014% of self employed so let's change NaN to NOT self_employed
            #Replace "NaN" string from defaultString
            # train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')
            MapTransformer(inputs=['self_employed'],
                        outputs=['self_employed'],
                        dict={"NaN": 'No'}),

            #There are only 0.20% of self work_interfere so let's change NaN to "Don't know
            #Replace "NaN" string from defaultString
            # train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know' )
            MapTransformer(inputs=['work_interfere'],
                        outputs=['work_interfere'],
                        dict={"NaN": 'Don\'t know'}),

    
            ComplementLabelEncoder(inputs=stringFeatures, outputs=stringFeatures),

            #Get rid of 'Country'
            # train_df = train_df.drop(['Country'], axis= 1)
            ColumnSelector(columns=['Country'], drop=True),

            # Scaling Age
            # scaler = MinMaxScaler()
            # train_df['Age'] = scaler.fit_transform(train_df[['Age']])
            Scaler(inputs=['Age'],
                outputs=['Age'],
                strategy='minmax'),


            # define X and y
            # feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
            #SelectTransformer(columns=['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']),
            ])

    def do_training(self, X, Y, create_onnx = True):
        print('Pre-processing ...')
        X = self.transform(X, do_fit=True)

        new_columns = {}
        for i, c in enumerate(X.columns):
            new_columns[c] = i
        X = X.rename(columns=new_columns)

        print('Training with ' + str(len(X.columns)) + ' columns ...')
        self.clfs = []
        clf = xgb.XGBClassifier(n_estimators=1700,
                                nthread=32,
                                max_depth=6,
                                learning_rate=0.024,
                                subsample=0.8,
                                colsample_bytree=0.65)
        this_model = clf.fit(X, Y)
        self.clfs.append(clf)

        if create_onnx:
            print('Converting models into ONNX ...')
            onnx_ml_models = []
            for i, clf in enumerate(self.clfs):
                initial_type = [('dense_input', FloatTensorType([None, len(self.pipeline.output_columns)]))]
                onnx_ml_models.append(convert_xgboost(clf, initial_types=initial_type))

            self.create_onnx('mental_health', onnx_ml_models)
