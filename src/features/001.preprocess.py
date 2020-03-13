# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:02:38 2020

@author: M44427
"""

import os
import numpy as np
import pandas as pd
import math
import sklearn as sk
from sklearn.model_selection import train_test_split
import pylab as pl
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.datasets import make_imbalance



# Create a copy of the Drives dataset and remove any columns that won't become categorical attributes 
DrivesProcessed = Drives.copy()
DrivesProcessed_old = Drives_old.copy()
del DrivesProcessed['game_id']
del DrivesProcessed['drive']
del DrivesProcessed['posteam']
del DrivesProcessed['defteam']
del DrivesProcessed['DefTimePerDrive']
del DrivesProcessed['OffTimePerDrive']

del DrivesProcessed_old['game_id']
del DrivesProcessed_old['drive']
del DrivesProcessed_old['posteam']
del DrivesProcessed_old['defteam']

# Separate Class from Data
DrivesProcessed['PointsScored'] = DrivesProcessed['PointsScored'].astype('category')
DriveClass = DrivesProcessed['PointsScored']
del DrivesProcessed['PointsScored']

DrivesProcessed_old['PointsScored'] = DrivesProcessed_old['PointsScored'].astype('category')
DriveClass_old = DrivesProcessed_old['PointsScored']
del DrivesProcessed_old['PointsScored']

# Change certain columns to categorical variables
DrivesProcessed['posteam_type'] = DrivesProcessed['posteam_type'].astype('category')
DrivesProcessed['GameYear'] = DrivesProcessed['GameYear'].astype('category')
DrivesProcessed['GameMonth'] = DrivesProcessed['GameMonth'].astype('category')
DrivesProcessed['qtr'] = DrivesProcessed['qtr'].astype('category')
DrivesProcessed['game_half'] = DrivesProcessed['game_half'].astype('category')

DrivesProcessed_old['posteam_type'] = DrivesProcessed_old['posteam_type'].astype('category')
DrivesProcessed_old['GameYear'] = DrivesProcessed_old['GameYear'].astype('category')
DrivesProcessed_old['GameMonth'] = DrivesProcessed_old['GameMonth'].astype('category')
DrivesProcessed_old['qtr'] = DrivesProcessed_old['qtr'].astype('category')
DrivesProcessed_old['game_half'] = DrivesProcessed_old['game_half'].astype('category')


# Create Dummy Variables
DrivesProcessed = pd.get_dummies(DrivesProcessed)
DrivesProcessed_old = pd.get_dummies(DrivesProcessed_old)

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