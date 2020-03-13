# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:28:51 2020

@author: M44427
"""

#import libraries
import numpy as np
import pandas as pd

# %% 
#######################################################
## Load and Clean Play By Play Dataset
#######################################################

# Load dataset
plays = pd.read_csv("../../data/clean_play_by_play/plays.csv", low_memory = False)
play_outcomes = pd.read_csv("../../data/clean_play_by_play/last_plays.csv", low_memory = False)
ThirdDown = pd.read_csv("../../data/clean_play_by_play/ThirdDown.csv", low_memory = False)
FirstDown = pd.read_csv("../../data/clean_play_by_play/FirstDown.csv", low_memory = False)
PenaltiesPos = pd.read_csv("../../data/clean_play_by_play/PenaltiesPos.csv", low_memory = False)
PenaltiesDef = pd.read_csv("../../data/clean_play_by_play/PenaltiesDef.csv", low_memory = False)
Passes = pd.read_csv("../../data/clean_play_by_play/Passes.csv", low_memory = False)
Runs = pd.read_csv("../../data/clean_play_by_play/Runs.csv", low_memory = False)
FG = pd.read_csv("../../data/clean_play_by_play/FG.csv", low_memory = False)

# Team Drive Offense and Defense
teamDriveOffense = pd.read_csv("../../data/clean_play_by_play/teamDriveOffense.csv", low_memory = False)
teamDriveDefense = pd.read_csv("../../data/clean_play_by_play/teamDriveDefense.csv", low_memory = False)

# %%
#######################################################
## Create Drive Dataset
####################################################### 

# Group plays to create 'Drives' table, add additional information
Drives = plays.groupby(['game_id', 'drive', 'posteam']).agg({'posteam_type' :'min',
                                                                     'defteam': 'min',
                                                                     'GameYear' : 'min', 
                                                                     'GameMonth':'min', 
                                                                     'qtr':'min',
                                                                     'game_half' : 'min',
                                                                     'yardline_100': 'min',
                                                                     'game_seconds_remaining' : 'min',
                                                                     'drive_starting_time' : 'min',
                                                                     'drive_end_time' : 'min',
                                                                     'score_differential' : 'min',
                                                                     'RunOver10' : 'sum',
                                                                     'PassOver20' : 'sum',
                                                                     'pass_attempt' : 'sum',
                                                                     'complete_pass' : 'sum',
                                                                     'rush_attempt' : 'sum',
                                                                     'sack' : 'sum',
                                                                     'tackled_for_loss':'sum',
                                                                     'play_id':'count',
                                                                     'points_earned' : 'sum'
                                                                    })


Drives['RunPercentage'] = Drives['rush_attempt'] / (Drives['rush_attempt'] + Drives['pass_attempt'])

#Renaming Cleanup
Drives.rename({'play_id': 'Plays'}, axis=1, inplace=True)
Drives.rename({'yardline_100': 'StartingYdsToGo'}, axis=1, inplace=True)
Drives.rename({'game_seconds_remaining': 'StartingTimeLeftInGame'}, axis=1, inplace=True)
Drives.rename({'score_differential': 'StartingScoreDifferential'}, axis=1, inplace=True)
Drives.rename({'pass_attempt': 'PassAttempts'}, axis=1, inplace=True)
Drives.rename({'complete_pass': 'PassCompletions'}, axis=1, inplace=True)
Drives.rename({'rush_attempt': 'RushAttempts'}, axis=1, inplace=True)
Drives.rename({'tackled_for_loss': 'TackledForLossPlays'}, axis=1, inplace=True)
Drives.rename({'points_earned': 'PointsScored'}, axis=1, inplace=True)


Drives = pd.merge(Drives, play_outcomes, how = 'left', on=['game_id', 'drive', 'posteam'])

#%%
# Incorporate 3rd Down, 1st Down, Penalties, Yardage stats
Drives = pd.merge(Drives, ThirdDown, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, FirstDown, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, PenaltiesPos, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, PenaltiesDef, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, Passes, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, Runs, how = 'left', on=['game_id', 'drive', 'posteam'])
Drives = pd.merge(Drives, FG, how = 'left', on=['game_id', 'drive', 'posteam'])

# Drive Dataset Cleanup
Drives['points_earned'].fillna(0, inplace=True)
Drives['PointsScored'] = np.where((Drives['PointsScored'] == 0), Drives['points_earned'], Drives['PointsScored'])
del Drives['points_earned']
Drives['penalty_x'].fillna(0, inplace=True)
Drives['penalty_y'].fillna(0, inplace=True)
Drives['NetPenalties'] = Drives['penalty_x'] - Drives['penalty_y']
Drives['penalty_yards_x'].fillna(0, inplace=True)
Drives['penalty_yards_y'].fillna(0, inplace=True)
Drives['NetPenaltyYardage'] = Drives['penalty_yards_x'] + Drives['penalty_yards_y']
Drives['posteam'].replace('JAX', 'JAC', inplace=True)
Drives['defteam'].replace('JAX', 'JAC', inplace=True)
del Drives['penalty_x']
del Drives['penalty_y']
del Drives['penalty_yards_x']
del Drives['penalty_yards_y']

Drives['PointsScored'].fillna(0, inplace=True)
Drives['PassYardage'].fillna(0, inplace=True)
Drives['RunYardage'].fillna(0, inplace=True)
Drives['NetPenaltyYardage'].fillna(0, inplace=True)
Drives['NetPenalties'].fillna(0, inplace=True)
Drives['ydstogo'].fillna(0, inplace=True)
Drives['third_down_converted'].fillna(0, inplace=True)
Drives['yards_gained'].fillna(0, inplace=True)

Drives.rename({'ydstogo': 'AvgYdsToGo3rd'}, axis=1, inplace=True)
Drives.rename({'third_down_converted': 'ThirdDownConversions'}, axis=1, inplace=True)
Drives.rename({'yards_gained': 'Avg1stDownGain'}, axis=1, inplace=True)

#Finally, remove all drives that had data inconsistencies
Drives = Drives[~((Drives['PointsScored'] == -12) | (Drives['PointsScored'] == 12) | (Drives['PointsScored'] == -4) | (Drives['PointsScored'] == -6) | (Drives['PointsScored'] == -2))]
Drives = Drives[~(Drives['Plays'] <= 2)]
Drives = Drives[~(Drives['PassYardage'] + Drives['RunYardage'] + Drives['NetPenaltyYardage'] > 100)]

# Rename teams that moved from one city to another for consistency
Drives['posteam'].replace('STL', 'LA', inplace=True)
Drives['posteam'].replace('SD', 'LAC', inplace=True)
Drives['defteam'].replace('STL', 'LA', inplace=True)
Drives['defteam'].replace('SD', 'LAC', inplace=True)



# Incorporate defense and offense team data
Drives_old = Drives.copy()
Drives = pd.merge(Drives, teamDriveOffense, how = 'left', on=['posteam', 'GameYear'])
Drives = pd.merge(Drives, teamDriveDefense, how = 'left', on=['defteam', 'GameYear'])
