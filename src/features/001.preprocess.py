# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:02:38 2020

@author: M44427
"""

import numpy as np
import pandas as pd


drives = pd.read_csv("../../data/drives/drives.csv", low_memory = False)


# Remove columns unecessary for analysis
del drives['game_id']
del drives['drive']
del drives['posteam']
del drives['defteam']
del drives['game_half']

# Separate Class from Data
drives['PointsScored'] = drives['PointsScored'].astype('category')
drives = drives['PointsScored']
del drives['PointsScored']

# Change certain columns to categorical variables
drives['posteam_type'] = drives['posteam_type'].astype('category')
drives['GameYear'] = drives['GameYear'].astype('category')
drives['GameMonth'] = drives['GameMonth'].astype('category')
drives['qtr'] = drives['qtr'].astype('category')
drives['game_half'] = drives['game_half'].astype('category')


'''
# Create Dummy Variables
drives = pd.get_dummies(drives)

# Split samples before normalizing data
drive_train, drive_test, target_train, target_test = train_test_split(DrivesProcessed, DriveClass, test_size = .2, random_state= 50, stratify = DriveClass)
drive_train_old, drive_test_old, target_train_old, target_test_old = train_test_split(DrivesProcessed_old, DriveClass_old, test_size = .2, random_state= 50, stratify = DriveClass)

# Scale new Drives dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
min_max_scaler = sk.preprocessing.MinMaxScaler().fit(drive_train)
normDrivesProcessed = min_max_scaler.transform(drive_train)
normDrivesProcessed = pd.DataFrame(normDrivesProcessed, columns = drive_train.columns)
normDrivesTest = pd.DataFrame(min_max_scaler.transform(drive_test), columns = drive_test.columns)

# Scale old Drives dataset for comparison
min_max_scaler_old = sk.preprocessing.MinMaxScaler().fit(drive_train_old)
normDrivesProcessed_old = min_max_scaler_old.transform(drive_train_old)
normDrivesProcessed_old = pd.DataFrame(normDrivesProcessed_old, columns = drive_train_old.columns)
normDrivesTest_old = pd.DataFrame(min_max_scaler_old.transform(drive_test_old), columns = drive_test_old.columns)

normDrivesProcessed.head()
'''