#!/usr/bin/env python
# coding: utf-8

# # **Predictors of mental health illness** <br>
# Kairos<br>
# Created: 1/23/2018<br>
# Last update: 6/23/2018<br>
# 
# This kernel is modified from the “starter kernel” by Megan Risdal<br>
# 
# ### **Question:**
# **Can you predict whether a patient should be treated of his/her mental illness or not according to the values obtained in the dataset?**
# 
# This is my first kernel after taking several courses of ML. I'm trying to understand better every concept by practicing and Kaggle is a good place to do it. Thanks Kaggle team.
# 
# The proccess is the following:
# 1. [Library and data loading](#Library_and_data_loading)
# 2. [Data cleaning](#Data_cleaning)
# 3. [Encoding data](#Encoding_data)
# 4. [Covariance Matrix. Variability comparison between categories of variables](#Covariance_Matrix)
# 5. [Some charts to see data relationship](#Some_charts_to_see_data_relationship)
# 6. [Scaling and fitting](#Scaling_and_fitting)
# 7. [Tuning](#Tuning)
# 8. [Evaluating models](#Evaluating_models)    
#     1. [Logistic Eegression](#Logistic_regressio)
#     2. [KNeighbors Classifier](#KNeighborsClassifier)
#     3. [Decision Tree Classifier](#Decision_Tree_classifier)
#     4. [Random Forests](#Random_Forests)
#     5. [Bagging](#Bagging)
#     6. [Boosting](#Boosting)
#     7. [Stacking](#Stacking)
# 9. [Predicting with Neural Network](#Predicting_with_Neural_Network)
# 10. [Success method plot](#Success_method_plot)
# 11. [Creating predictions on test set](#Creating_predictions_on_test_set)
# 12. [Submission](#Submission)
# 13. [Conclusions](#Conclusions)
# 

# <a id='Library_and_data_loading'></a>
# ## **1. Library and data loading** ##

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import randint

# prep
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Validation libraries
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

#Neural Network
from sklearn.neural_network import MLPClassifier
#from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#Naive bayes
from sklearn.naive_bayes import GaussianNB 

#Stacking
from mlxtend.classifier import StackingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.

#reading in CSV's from a file path
train_df = pd.read_csv('survey.csv')

# Assign default values for each data type
defaultInt = 0
defaultString = 'NaN'
defaultFloat = 0.0

# Create lists by data tpe
intFeatures = ['Age']
stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
floatFeatures = []
leDict = None
scaler = None

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class ComplementLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = preprocessing.LabelEncoder()

    def fit(self, X, y=None, **fit_params):
        label_col = X.map(lambda x:'extra_category_' if str(x)=='nan' else x).astype('str')
        self.encoder.fit(label_col)
        if 'extra_category_' not in self.encoder.classes_:
            self.encoder.classes_ = list(self.encoder.classes_) + ['extra_category_']
        self.mapping_table = {self.encoder.classes_[i]:i for i in range(len(self.encoder.classes_))}
        self.default_value = self.mapping_table['extra_category_']
        return self

    def transform(self, X, **transform_params):
        if str(X.dtype) == 'category':
            X = X.cat.add_categories(['extra_category_'])
        transformed_arr = X.fillna('extra_category_').map(self.mapping_table).fillna(self.default_value).astype('int32')
        return transformed_arr

def data_preprocessing(train_df, isFit):
   global leDict, scaler
   #dealing with missing data
   #Let’s get rid of the variables "Timestamp",“comments”, “state” just to make our lives easier.
   train_df = train_df.drop(['comments'], axis= 1)
   train_df = train_df.drop(['state'], axis= 1)
   train_df = train_df.drop(['Timestamp'], axis= 1)

   # Clean the NaN's
   for feature in train_df:
       if feature in intFeatures:
           train_df[feature] = train_df[feature].fillna(defaultInt)
       elif feature in stringFeatures:
           train_df[feature] = train_df[feature].fillna(defaultString)
       elif feature in floatFeatures:
           train_df[feature] = train_df[feature].fillna(defaultFloat)
       else:
           print('Error: Feature %s not recognized.' % feature)
   #train_df.head(5)   

   #clean 'Gender'
   #Slower case all columm's elements
   #gender = train_df['Gender'].str.lower()
   #print(gender)

   #Select unique elements
   #gender = train_df['Gender'].unique()

   #Made gender groups
   male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
   trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
   female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

   for (row, col) in train_df.iterrows():

       if str.lower(col.Gender) in male_str:
           train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

       if str.lower(col.Gender) in female_str:
           train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

       if str.lower(col.Gender) in trans_str:
           train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

   #Get rid of bullshit
   stk_list = ['A little about you', 'p']
   train_df = train_df[~train_df['Gender'].isin(stk_list)]

   #complete missing age with mean
   train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

   # Fill with media() values < 18 and > 120
   s = pd.Series(train_df['Age'])
   s[s<18] = train_df['Age'].median()
   train_df['Age'] = s
   s = pd.Series(train_df['Age'])
   s[s>120] = train_df['Age'].median()
   train_df['Age'] = s

   #Ranges of Age
   #train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

   #There are only 0.014% of self employed so let's change NaN to NOT self_employed
   #Replace "NaN" string from defaultString
   train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

   train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], 'Don\'t know' )

   #Encoding data
   #labelDict = {}
   if isFit:
       leDict = {}
       for feature in stringFeatures:
           le = ComplementLabelEncoder()
           le.fit(train_df[feature])
           leDict[feature] = le
       
   for feature in stringFeatures:
       le = leDict[feature]
       #le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
       train_df[feature] = le.transform(train_df[feature])
       # Get labels
       #labelKey = 'label_' + feature
       #labelValue = [*le_name_mapping]
       #labelDict[labelKey] =labelValue
    
   #Get rid of 'Country'
   train_df = train_df.drop(['Country'], axis= 1)
   #train_df.head()

   # Scaling Age
   if isFit:
       scaler = MinMaxScaler()
       scaler.fit(train_df[['Age']])
   train_df['Age'] = scaler.transform(train_df[['Age']])
   #train_df.head()
   return train_df

train_org, test_org = train_test_split(train_df, test_size=0.30, random_state=0)
train_org = data_preprocessing(train_org, True)
# define X and y
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
X_train = train_org[feature_cols]
y_train = train_org.treatment
pd.set_option('display.max_columns', 50)
print("NORI X_train=")
print(X_train)
print("NORI y_train=")
print(y_train)

# Generate predictions with the best method
clf = AdaBoostClassifier()
clf.fit(X_train, y_train)

num = len(test_org)
predictionResults = []
# Measurement
def measure():
    for i in range(num):
        element = test_org.iloc[i:i+1]
        element = data_preprocessing(element, False)
        X_test = element[feature_cols]
        #print("NORI X_test=")
        #print(element)
        dfTestPredictions = clf.predict(X_test)
        predictionResults.extend(dfTestPredictions)
        #print(dfTestPredictions)

import timeit
result = timeit.timeit(measure, number=1)
print("whole time: ", result, " sec")
print("num =", num, " one row=", result / num * 1000, " msec")
print("predictionResults=", predictionResults)

import collections
c = collections.Counter(predictionResults)
print(c.most_common())
