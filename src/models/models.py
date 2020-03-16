#Scikit Template
'''created by Casey Bennett 2018, www.CaseyBennett.com
   Copyright 2018

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License (Lesser GPL) as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   Modified by Dennis Tseng for DSC540 Final Project
'''

import sys
import csv
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import time

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from catboost import CatBoostClassifier, cv, Pool

from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, scale, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from collections import Counter

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

imb_class=2                                         #Control switch for type of sampling to deal with imbalanced class (0=None, 1=SMOTE, 2=NearMiss)
cross_val=1                                         #Control Switch for CV
norm_features=0                                     #Normalize features switch
feat_select=1                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=1                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
param_tuning = 0                                    #Turn on model parameter tuning


#Set global model parameters
rand_st=0                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################

data = pd.read_csv("../../data/drives/drives_analysis.csv", low_memory = False)

target=data['points_scored']
del data['points_scored']

# Split prior to any normalization/sampling. Stratify is set understanding that we have imbalanced classes and want to preserve that in the split
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, stratify = target)


#############################################################################
#
# Preprocess data
#
##########################################

# Dealing with imbalanced classes
if imb_class == 0:
    data_np = data
    target_np = target
elif imb_class == 1:
    # Oversample with SMOTE
    oversample = SMOTE(random_state = rand_st)
    data_np, target_np = oversample.fit_resample(data_train, target_train)
elif imb_class == 2:
    # Undersample using NearMiss
    undersample = NearMiss(version=1)
    data_np, target_np = undersample.fit_resample(data_train,target_train)


# Normalize features
if norm_features==1:    
    '''
    scaler = MinMaxScaler()
    min_max_scaler = MinMaxScaler().fit(data_train)
    normDrivesProcessed = min_max_scaler.transform(data_train)
    normDrivesProcessed = pd.DataFrame(normDrivesProcessed, columns = data_train.columns)
    target_np = pd.DataFrame(min_max_scaler.transform(drive_test), columns = drive_test.columns)
    '''
    data_np=pd.DataFrame(scale(data_train), columns = data_train.columns)


#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    #LV Threshold
    sel = VarianceThreshold(threshold=0.50)                                      #Removes any feature with less than 20% variance
    fit_mod=sel.fit(data_np)
    fitted=sel.transform(data_np)
    sel_idx=fit_mod.get_support()
    variance_features = list(data_np.columns[np.array(sel_idx).astype(bool)])
    
    print('Selected', variance_features)
    print('Features (total, selected):', len(data_np.columns), len(variance_features))
    print('\n')

    #Filter selected columns from original dataset
    data_np = data_np[variance_features]


#Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
        sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
        print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()      

    if fs_type==2:
        #Wrapper Select via model
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=None)
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                                                           #to select only based on max_features, set to integer value and set threshold=-np.inf
        print ('Wrapper Select - Random Forest: ')
        
        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()

    if fs_type==3:                                                                ######Only work if the Target is binned###########
        #Univariate Feature Selection - Chi-squared
        sel=SelectKBest(chi2, k=k_cnt)
        fit_mod=sel.fit(data_np, target_np)                                         #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
        print ('Univariate Feature Selection - Chi2: ')
        sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(data_np.columns)):            
            temp.append([data_np.columns[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np.columns)):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(data_np.columns[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np.columns), len(temp))
    print('\n')
                
    ##3) Filter selected columns from original dataset #########
    selected_features = list(data_np.columns[np.array(sel_idx).astype(bool)])                          #Deletes non-selected features by index)
    data_test = data_test[selected_features]
    data_np = data_np[selected_features]



#############################################################################
#
# GridSearchCV to find best models
#
##########################################

best_params = []

if param_tuning == 1:
    
    scorers = {'Accuracy': 'accuracy', 
               'F1': make_scorer(f1_score, average = 'micro'),
               'AUC' : make_scorer(roc_auc_score, needs_proba= True, average = 'macro', multi_class ='ovr')}
    
    ################
    # Decision Tree
    ################
    max_depth = [int(x) for x in np.linspace(1, 10, num = 10)]
    #max_depth.append(None)
    class_weight = ['balanced']
    clf = DecisionTreeClassifier()
    
    dt_grid = {'criterion': ['gini'],
               'max_features': ['auto', 'sqrt', 'log2', None],
               'max_depth': max_depth,
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
               'class_weight': class_weight}
    
    clf_random = GridSearchCV(estimator = clf, param_grid = dt_grid, scoring=scorers, refit='F1', cv = 5, verbose=10, n_jobs = -1)
    
    # Fit the random search model
    clf_random.fit(data_np, target_np)
    best_params.append(('Decision Tree',clf_random.best_params_, clf_random.best_score_))
    print(clf_random.best_params_)
    print(clf_random.best_score_)
    
    
    #{'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 32, 'min_samples_split': 2}
    #0.7965322930212397
    
    ################
    # Random Forest
    ################
    
    max_depth = [int(x) for x in np.linspace(1, 7, num = 7)]
    max_depth.append(None)
    class_weight = ['balanced', 'balanced_subsample', None]
    rf = RandomForestClassifier()
    
    rf_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)],
               'max_features': ['auto', 'sqrt', 'log2'],
               'max_depth': max_depth,
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [2, 4, 8, 16, 32, 64],
               'class_weight': class_weight,
               'bootstrap': [True, False]}
    
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = rf_grid, n_iter = 10000, scoring=scorers, refit='F1', cv = 5, verbose=10, n_jobs = -1)
    
    # Fit the random search model
    rf_random.fit(data_np, target_np)
    best_params.append(('Random Forest',rf_random.best_params_, rf_random.best_score_))
    print(rf_random.best_params_)
    print(rf_random.best_score_)
    
    
    #best results
    #{'n_estimators': 300, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_features': 'sqrt', 'max_depth': None, 'class_weight': None, 'bootstrap': False}
    #0.7906805374945817
    
    ################
    # AdaBoost
    ################
    
    ab = AdaBoostClassifier()
    
    ab_grid = {'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
               'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
               'algorithm': ['SAMME', 'SAMME.R']}
    
    ab_random = GridSearchCV(estimator = ab, param_grid = ab_grid, scoring=scorers, refit='F1', cv = 5, verbose=10, n_jobs = -1)
    
    # Fit the random search model
    ab_random.fit(data_np, target_np)
    best_params.append(('Ada Boost',ab_random.best_params_, ab_random.best_score_))
    print(ab_random.best_params_)
    print(ab_random.best_score_)
    
    
    ################
    # Gradient Boosting Classifier
    ################
    gb = GradientBoostingClassifier()
    
    gb_grid = {'loss': ['deviance', 'exponential'],
               'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
               'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
               'max_depth' : [3,4,5,6,7,8],
               'max_features' : ['auto','sqrt','log2',None],
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
               }
    
    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = gb_grid, n_iter = 10000, scoring=scorers, refit='F1', cv = 5, verbose=10, n_jobs = -1)
    
    # Fit the random search model
    ab_random.fit(data_np, target_np)
    best_params.append(('Gradient Boost',gb_random.best_params_, gb_random.best_score_))
    print(gb_random.best_params_)
    print(gb_random.best_score_)

    ################
    # Neural Networks
    ################    
    mlp = MLPClassifier()
    
    nn_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'hidden_layer_sizes' : [(100,), (10,), (50,50), (20,20), (100,50), (200,100), (30,30,30)],
               'solver': ['lbfgs', 'sgd', 'adam'],
               'alpha' : [0.0001, 0.001, 0.01],
               'learning_rate' : ['constant', 'invscaling', 'adaptive'],
               'max_depth' : [3,4,5,6,7,8],
               'max_features' : ['auto','sqrt','log2',None],
               'min_samples_split': [2, 5, 10, 20, 25, 50, 100],
               'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
               }
    
    nn_random = RandomizedSearchCV(estimator = mlp, param_distributions = nn_grid, n_iter = 500, scoring=scorers, refit='F1', cv = 5, verbose=10, n_jobs = -1)
    
    # Fit the random search model
    nn_random.fit(data_np, target_np)
    best_params.append(('Neural Networks',nn_random.best_params_, nn_random.best_score_))
    print(nn_random.best_params_)
    print(nn_random.best_score_)

#############################################################################
#
# Train SciKit Models
#
##########################################

print('--ML Model Output--', '\n')

# Non-Cross Validation
if cross_val == 0:
    # Decision Tree
    clf = DecisionTreeClassifier(criterion='gini', 
                                 splitter='best', 
                                 max_depth=10, 
                                 min_samples_split=2, 
                                 min_samples_leaf=32, 
                                 max_features=None,
                                 class_weight='balanced',
                                 random_state=rand_st)
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Decision Tree Train Accuracy:",clf.score(data_np, target_np))
    print('Decision Tree Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    
    # Random Forest
    clf = RandomForestClassifier(n_estimators=1000, 
                                 max_depth=None, 
                                 min_samples_split=5, 
                                 min_samples_leaf = 2, 
                                 max_features = 'auto',                                 
                                 random_state=rand_st)
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Random Forest Train Accuracy:",clf.score(data_np, target_np))
    print('Random Forest Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))

    # AdaBoost
    clf=AdaBoostClassifier(n_estimators = 950,
                           base_estimator = None,
                           learning_rate = 0.1,
                           algorithm = 'SAMME.R',
                           random_state = rand_st)
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("AdaBoost Train Accuracy:",clf.score(data_np, target_np))
    print('AdaBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))

    # Gradient Boosting
    clf=GradientBoostingClassifier(n_estimators = 850, 
                                   loss = 'deviance', 
                                   learning_rate = 0.01, 
                                   max_depth = 7, 
                                   min_samples_split = 20, 
                                   min_samples_leaf = 2,
                                   random_state = rand_st)
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("Gradient Boosting Train Accuracy:",clf.score(data_np, target_np))
    print('Gradient Boosting Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))
    
     # Neural Network
    clf=MLPClassifier(activation = 'logistic',
                      learning_rate = 'adaptive',
                      solver = 'adam',
                      alpha = 0.01,
                      hidden_layer_sizes = (200,100), 
                      random_state = rand_st)
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("NN Train Accuracy:",clf.score(data_np, target_np))
    print('NN Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))   
    
    # Catboost
    clf=CatBoostClassifier(task_type = 'GPU')
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("CatBoost Train Accuracy:",clf.score(data_np, target_np))
    print('CatBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))       
    
    # XGBoost
    clf=xgb.XGBClassifier()
    clf.fit(data_np, target_np)
    test_predictions = clf.predict(data_test)
    scores_ACC = clf.score(data_test, target_test)
    print("XGBoost Train Accuracy:",clf.score(data_np, target_np))
    print('XGBoost Test Acc:', scores_ACC)
    print(classification_report(target_test, test_predictions))   


####Cross-Val Classifiers####
if cross_val == 1:
    #Setup Crossval classifier scorers
    
    
    scorers = {'Accuracy': 'accuracy', 
               #'Precision': make_scorer(precision_score, average='micro'),
               #'Recall': make_scorer(recall_score, average = 'micro'),
               #'F1': make_scorer(f1_score, average = 'micro'),
               'AUC' : make_scorer(roc_auc_score, needs_proba= True, average = 'macro', multi_class ='ovr')}
    
    
    #SciKit Decision Tree - Cross Val
    start_ts=time.time()
    clf = DecisionTreeClassifier(criterion='gini', 
                                 splitter='best', 
                                 max_depth=10, 
                                 min_samples_split=2, 
                                 min_samples_leaf=32, 
                                 max_features=None,
                                 class_weight='balanced',
                                 random_state=rand_st)
    scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']
    print("Decision Tree Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("Decision Tree Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("Decision Tree Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("Decision Tree F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))          
    print("Random Forest AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                          
    print("CV Runtime:", time.time()-start_ts)
        
    
    #SciKit Random Forest - Cross Val
    print()
    start_ts=time.time()
    clf = RandomForestClassifier(n_estimators=1000, 
                                 max_depth=None, 
                                 min_samples_split=5, 
                                 min_samples_leaf = 2, 
                                 max_features = 'auto',                                 
                                 random_state=rand_st)
    scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']
    print("Random Forest Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("Random Forest Tree Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("Random Forest Tree Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("Random Forest F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("Random Forest AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))     
    print("CV Runtime:", time.time()-start_ts)
    
    
    #SciKit Gradient Boosting - Cross Val
    print()
    start_ts=time.time()
    clf=GradientBoostingClassifier(n_estimators = 850, 
                                   loss = 'deviance', 
                                   learning_rate = 0.01, 
                                   max_depth = 7, 
                                   min_samples_split = 20, 
                                   min_samples_leaf = 2,
                                   random_state = rand_st)
    scores = cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']                                                                                                                                 
    print("Gradient Boosting Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("Gradient Boosting Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("Gradient Boosting Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("Gradient Boosting F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("Gradient Boosting AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))     
    print("CV Runtime:", time.time()-start_ts)
    
    
    #SciKit Ada Boosting - Cross Val
    print()
    start_ts=time.time()
    clf=AdaBoostClassifier(n_estimators = 950,
                           base_estimator = None,
                           learning_rate = 0.1,
                           algorithm = 'SAMME.R',
                           random_state = rand_st)
    scores = cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC'] 
    print("Ada Boost Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("Ada Boost Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("Ada Boost Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("Ada Boost F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("Ada Boost AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                          
    print("CV Runtime:", time.time()-start_ts)
    
    
    #SciKit Neural Network - Cross Val
    print()
    start_ts=time.time()
    clf=MLPClassifier(activation = 'logistic',
                      learning_rate = 'adaptive',
                      solver = 'adam',
                      alpha = 0.01,
                      hidden_layer_sizes = (200,100), 
                      random_state = rand_st)
    scores = cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']                                                                                                                                     
    print("Neural Network Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("Neural Network Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("Neural Network Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("Neural Network F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("Neural Network AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                              
    print("CV Runtime:", time.time()-start_ts)
    
    '''
    #SciKit SVM - Cross Val
    print()
    start_ts=time.time()
    clf=SVC(kernel = 'rbf',
            gamma = 'scale',
            C = 1.0,
            probability = True,
            random_state = rand_st)
    scores=cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    scores_Pre = scores['test_Precision']
    scores_Rec = scores['test_Recall']
    scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']                                                                                                                                      
    print("SVM Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    print("SVM Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    print("SVM Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    print("SVM F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("SVM AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                                
    print("CV Runtime:", time.time()-start_ts)
    '''
    
    #Catboost Classifier - Cross Val
    print()
    start_ts=time.time()
    clf=CatBoostClassifier(task_type = 'GPU')
    scores=cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']                                                                                                                                      
    print("SVM Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                               
    #print("SVM Precision: %0.4f (+/- %0.4f)" % (scores_Pre.mean(), scores_Pre.std() * 2))                               
    #print("SVM Recall: %0.4f (+/- %0.4f)" % (scores_Rec.mean(), scores_Rec.std() * 2))                               
    #print("SVM F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("SVM AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                                
    print("CV Runtime:", time.time()-start_ts)
    
    #XGBoost Classifier - Cross Val
    print()
    start_ts=time.time()
    clf=xgb.XGBClassifier()
    scores=cross_validate(estimator = clf, X = data_np, y = target_np, scoring = scorers, cv =5 )
    scores_Acc = scores['test_Accuracy']
    #scores_Pre = scores['test_Precision']
    #scores_Rec = scores['test_Recall']
    #scores_F1 = scores['test_F1']
    scores_AUC = scores['test_AUC']                                                                                                                                      
    print("XGBoost Acc: %0.4f (+/- %0.4f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                          
    #print("XGBoost F1: %0.4f (+/- %0.4f)" % (scores_F1.mean(), scores_F1.std() * 2))                               
    print("XGBoost AUC: %0.4f (+/- %0.4f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                                
    print("CV Runtime:", time.time()-start_ts)
        