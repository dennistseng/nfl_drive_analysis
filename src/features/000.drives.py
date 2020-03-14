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
plays = pd.read_csv("../../data/clean/plays.csv", low_memory = False)
play_outcomes = pd.read_csv("../../data/clean/last_plays.csv", low_memory = False)
ThirdDown = pd.read_csv("../../data/clean/ThirdDown.csv", low_memory = False)
FirstDown = pd.read_csv("../../data/clean/FirstDown.csv", low_memory = False)
PenaltiesPos = pd.read_csv("../../data/clean/PenaltiesPos.csv", low_memory = False)
PenaltiesDef = pd.read_csv("../../data/clean/PenaltiesDef.csv", low_memory = False)
Passes = pd.read_csv("../../data/clean/Passes.csv", low_memory = False)
Runs = pd.read_csv("../../data/clean/Runs.csv", low_memory = False)
FG = pd.read_csv("../../data/clean/FG.csv", low_memory = False)

# Team Drive Offense and Defense
#teamDriveOffense = pd.read_csv("../../data/clean/teamDriveOffense.csv", low_memory = False)
#teamDriveDefense = pd.read_csv("../../data/clean/teamDriveDefense.csv", low_memory = False)

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
                                                                     'game_seconds_remaining' : 'max',
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
Drives['drive_length'] = Drives['drive_starting_time'] - Drives['drive_end_time']

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

#%% Create two columns for previous drive of opposing and poss team

# This was intended to account for if a team was able to possess the ball more than 2 drives in a row, but surprisingly this has never happened in our dataset.
for l in [1,2]:
    col_string_outcome = 'previous_drive_outcome_' + str(l)  
    col_string_plays = 'previous_drive_plays_' + str(l)
    col_string_drive_length = 'previous_drive_drive_length_' + str(l)
    col_string_expl_run = 'previous_drive_RunOver10_' + str(l)
    col_string_expl_pass = 'previous_drive_PassOver20_' + str(l)
    col_string_yds_to_go = 'previous_drive_StartingYdsToGo_' + str(l)
    col_string_team = 'previous_drive_team_' + str(l)
    
    prev_string_outcome = 'previous_drive_outcome_' + str(l-1)
    prev_string_plays = 'previous_drive_plays_' + str(l-1)
    prev_string_drive_length = 'previous_drive_drive_length_' + str(l-1)
    prev_string_expl_run = 'previous_drive_RunOver10_' + str(l-1)
    prev_string_expl_pass = 'previous_drive_PassOver20_' + str(l-1)
    prev_string_yds_to_go = 'previous_drive_StartingYdsToGo_' + str(l-1)
    prev_string_team = 'previous_drive_team_' + str(l-1)
    
    if l == 1:
        Drives[col_string_outcome] = Drives.groupby(['game_id'])['drive_outcome'].shift(fill_value = 'no_previous_drive')
        
        
        
        Drives[col_string_team] = Drives.groupby(['game_id'])['posteam'].shift(fill_value = 'n/a')
    else:
        Drives[col_string_outcome] = Drives.groupby(['game_id'])[prev_string_outcome].shift(fill_value = 'no_previous_drive')
        
        
        
        Drives[col_string_team] = Drives.groupby(['game_id'])[prev_string_team].shift(fill_value = 'n/a')

#Drive Logic. We make an assumption that halfs are sufficient enough breaks to alter 'momentum,' but this may not always be the case
Drives['previous_drive'] = np.where((Drives['previous_drive_team_2'] == Drives['posteam']), Drives['previous_drive_outcome_2'], 'no_previous_drive')
Drives['previous_drive'] = np.where((Drives['previous_drive_team_1'] == Drives['posteam']), Drives['previous_drive_outcome_1'], Drives['previous_drive'])
Drives['previous_oppose_drive'] = np.where((Drives['previous_drive_team_2'] == Drives['defteam']), Drives['previous_drive_outcome_2'], 'no_previous_drive')
Drives['previous_oppose_drive'] = np.where((Drives['previous_drive_team_1'] == Drives['defteam']), Drives['previous_drive_outcome_1'], Drives['previous_oppose_drive'] )

# Remove temporary staging columns
Drives.drop(['previous_drive_team_1', 'previous_drive_team_2', 'previous_drive_outcome_1', 'previous_drive_outcome_2', 'drive_outcome'], axis=1, inplace=True)



# Incorporate defense and offense team data
#Drives_old = Drives.copy()
#Drives = pd.merge(Drives, teamDriveOffense, how = 'left', on=['posteam', 'GameYear'])
#Drives = pd.merge(Drives, teamDriveDefense, how = 'left', on=['defteam', 'GameYear'])

#%% 

Drives.to_csv('../../data/drives/drives.csv', index = False)