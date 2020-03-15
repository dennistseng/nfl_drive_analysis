# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:02:38 2020

@author: M44427


Just some additional preprocessing
"""

import numpy as np
import pandas as pd

import sys
import csv
import math
from operator import itemgetter
import time

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale




drives = pd.read_csv("../../data/drives/drives.csv", low_memory = False)

# Check if any columns have null values
drives.columns[drives.isna().any()].tolist()

# Separate Class from Data
drives['points_scored'] = drives['points_scored'].astype('category')
target_np = drives['points_scored']
del drives['points_scored']

# Change certain columns to categorical variables
drives['month'] = drives['month'].astype('category')
drives['qtr'] = drives['qtr'].astype('category')

# Create dummies
drives['posteam_type'] = pd.get_dummies(drives['posteam_type'], drop_first = True)
drives.rename({'posteam_type': 'home_team'}, axis=1, inplace=True)


drives = pd.get_dummies(drives)

data_np = drives.copy()

# Create Dummy Variables


# Split samples before normalizing data

data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

####Classifiers####
clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=None)
clf.fit(data_train, target_train)
print('Decision Tree Acc:', clf.score(data_test, target_test))


rf = RandomForestClassifier(n_estimators = 1000, max_depth = 10, min_samples_split = 25)
rf.fit(data_train, target_train)
rfpredictions = rf.predict(data_test)
print("Train Accuracy :: ", accuracy_score(target_train, rf.predict(data_train)))
print("Test Accuracy  :: ", accuracy_score(target_test, rfpredictions))
print("\n")
print("Confusion matrix: \n", confusion_matrix(target_test, rfpredictions))

rfconfusionMatrix = confusion_matrix(target_test, rfpredictions)

print(classification_report(target_test, rfpredictions))

'''
# Scale new Drives dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
min_max_scaler = sk.preprocessing.MinMaxScaler().fit(drive_train)
normDrivesProcessed = min_max_scaler.transform(drive_train)
normDrivesProcessed = pd.DataFrame(normDrivesProcessed, columns = drive_train.columns)
normDrivesTest = pd.DataFrame(min_max_scaler.transform(drive_test), columns = drive_test.columns)

normDrivesProcessed.head()
'''