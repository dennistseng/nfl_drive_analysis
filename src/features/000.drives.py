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
                                                                     'points_earned' : 'sum',
                                                                     'pass_touchdown' : 'max',
                                                                     'interception' : 'max'
                                                                    })


Drives['RunPercentage'] = Drives['rush_attempt'] / (Drives['rush_attempt'] + Drives['pass_attempt'])
Drives['drive_length'] = Drives['drive_starting_time'] - Drives['drive_end_time']
Drives.drop(['drive_starting_time', 'drive_end_time'], axis=1, inplace=True)

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
    col_string_team = 'previous_drive_team_' + str(l)
    
    prev_string_outcome = 'previous_drive_outcome_' + str(l-1)
    prev_string_team = 'previous_drive_team_' + str(l-1)
    
    if l == 1:
        Drives[col_string_outcome] = Drives.groupby(['game_id'])['drive_outcome'].shift(fill_value = 'no_previous_drive')
        Drives[col_string_team] = Drives.groupby(['game_id'])['posteam'].shift(fill_value = 'n/a')
    else:
        Drives[col_string_outcome] = Drives.groupby(['game_id'])[prev_string_outcome].shift(fill_value = 'no_previous_drive')     
        Drives[col_string_team] = Drives.groupby(['game_id'])[prev_string_team].shift(fill_value = 'n/a')

# Drive Logic. We make an assumption that halfs are sufficient enough breaks to alter 'momentum,' but this may not always be the case
Drives['previous_drive'] = np.where((Drives['previous_drive_team_2'] == Drives['posteam']), Drives['previous_drive_outcome_2'], 'no_previous_drive')
Drives['previous_drive'] = np.where((Drives['previous_drive_team_1'] == Drives['posteam']), Drives['previous_drive_outcome_1'], Drives['previous_drive'])
Drives['previous_oppose_drive'] = np.where((Drives['previous_drive_team_2'] == Drives['defteam']), Drives['previous_drive_outcome_2'], 'no_previous_drive')
Drives['previous_oppose_drive'] = np.where((Drives['previous_drive_team_1'] == Drives['defteam']), Drives['previous_drive_outcome_1'], Drives['previous_oppose_drive'] )

# Look at select previous drive stats for the offense only
Drives['previous_drive_plays'] = Drives.groupby(['game_id', 'posteam'])['Plays'].shift(fill_value = 0)
Drives['previous_drive_drive_length'] = Drives.groupby(['game_id', 'posteam'])['drive_length'].shift(fill_value = 0)
Drives['previous_drive_RunOver10'] = Drives.groupby(['game_id', 'posteam'])['RunOver10'].shift(fill_value = 0)
Drives['previous_drive_PassOver20'] = Drives.groupby(['game_id', 'posteam'])['PassOver20'].shift(fill_value = 0)
Drives['previous_drive_StartingYdsToGo'] = Drives.groupby(['game_id', 'posteam'])['StartingYdsToGo'].shift(fill_value = 0)


# Remove temporary staging columns
Drives.drop(['previous_drive_team_1', 'previous_drive_team_2', 'previous_drive_outcome_1', 'previous_drive_outcome_2', 'drive_outcome'], axis=1, inplace=True)

# Incorporate defense and offense team data
#Drives_old = Drives.copy()
#Drives = pd.merge(Drives, teamDriveOffense, how = 'left', on=['posteam', 'GameYear'])
#Drives = pd.merge(Drives, teamDriveDefense, how = 'left', on=['defteam', 'GameYear'])

#%% 
# For each game and team, create cumulative counts, sums, and averages for particular stats

# Averages
explosive_runs = Drives.groupby(['game_id', 'posteam'])['RunOver10'].expanding().mean().reset_index([0,1])
del explosive_runs['game_id']
del explosive_runs['posteam']

explosive_passes = Drives.groupby(['game_id', 'posteam'])['PassOver20'].expanding().mean().reset_index([0,1])
del explosive_passes['game_id']
del explosive_passes['posteam']

average_points = Drives.groupby(['game_id', 'posteam'])['PointsScored'].expanding().mean().reset_index([0,1])
del average_points['game_id']
del average_points['posteam']

average_plays = Drives.groupby(['game_id', 'posteam'])['Plays'].expanding().mean().reset_index([0,1])
del average_plays['game_id']
del average_plays['posteam']

average_top = Drives.groupby(['game_id', 'posteam'])['drive_length'].expanding().mean().reset_index([0,1])
del average_top['game_id']
del average_top['posteam']

average_sacks = Drives.groupby(['game_id', 'posteam'])['sack'].expanding().mean().reset_index([0,1])
del average_sacks['game_id']
del average_sacks['posteam']

average_tfl = Drives.groupby(['game_id', 'posteam'])['TackledForLossPlays'].expanding().mean().reset_index([0,1])
del average_tfl['game_id']
del average_tfl['posteam']

# Sums to calculate percentages

# Y/A passing
pass_attempts = Drives.groupby(['game_id', 'posteam'])['PassAttempts'].expanding().sum().reset_index([0,1])
del pass_attempts['game_id']
del pass_attempts['posteam']

pass_completions = Drives.groupby(['game_id', 'posteam'])['PassCompletions'].expanding().sum().reset_index([0,1])
del pass_completions['game_id']
del pass_completions['posteam']

pass_yardage = Drives.groupby(['game_id', 'posteam'])['PassYardage'].expanding().sum().reset_index([0,1])
del pass_yardage['game_id']
del pass_yardage['posteam']

# Y/A rushing
rush_attempts = Drives.groupby(['game_id', 'posteam'])['RushAttempts'].expanding().sum().reset_index([0,1])
del rush_attempts['game_id']
del rush_attempts['posteam']

rush_yardage = Drives.groupby(['game_id', 'posteam'])['RunYardage'].expanding().sum().reset_index([0,1])
del rush_yardage['game_id']
del rush_yardage['posteam']

# Interceptions and Passing Touchdowns for passer rating calculations
interceptions = Drives.groupby(['game_id', 'posteam'])['interception'].expanding().sum().reset_index([0,1])
del interceptions['game_id']
del interceptions['posteam']

pass_touchdown = Drives.groupby(['game_id', 'posteam'])['pass_touchdown'].expanding().sum().reset_index([0,1])
del pass_touchdown['game_id']
del pass_touchdown['posteam']


#%%

# Merge above columns
Drives = pd.merge(Drives, explosive_runs, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, explosive_passes, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, average_points, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, average_plays, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, average_top, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, average_sacks, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, average_tfl, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, pass_attempts, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, pass_completions, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, pass_yardage, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, rush_attempts, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, rush_yardage, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, interceptions, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))
Drives = pd.merge(Drives, pass_touchdown, how = 'left', left_index=True, right_index=True, suffixes = ('','_prior_drives'))


Drives['prior_drives_expl_pass'] = Drives.groupby(['game_id','posteam'])['PassOver20_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_expl_run'] = Drives.groupby(['game_id','posteam'])['RunOver10_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_average_points'] = Drives.groupby(['game_id','posteam'])['PointsScored_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_average_plays'] = Drives.groupby(['game_id','posteam'])['Plays_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_average_top'] = Drives.groupby(['game_id','posteam'])['drive_length_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_average_sacks'] = Drives.groupby(['game_id','posteam'])['sack_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_average_tfl'] = Drives.groupby(['game_id','posteam'])['TackledForLossPlays_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_pass_attempts'] = Drives.groupby(['game_id','posteam'])['PassAttempts_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_completions'] = Drives.groupby(['game_id','posteam'])['PassCompletions_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_pass_yardage'] = Drives.groupby(['game_id','posteam'])['PassYardage_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_rush_attempts'] = Drives.groupby(['game_id','posteam'])['RushAttempts_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_rush_yardage'] = Drives.groupby(['game_id','posteam'])['RunYardage_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_interceptions'] = Drives.groupby(['game_id','posteam'])['interception_prior_drives'].shift(fill_value = 0)
Drives['prior_drives_passing_tds'] = Drives.groupby(['game_id','posteam'])['pass_touchdown_prior_drives'].shift(fill_value = 0)


#%% 

# Calculate cumulative Passer Rating
Drives['a'] = (Drives['prior_drives_completions']/Drives['prior_drives_pass_attempts'] - .3) * 5
Drives['b'] = (Drives['prior_drives_pass_yardage']/Drives['prior_drives_pass_attempts'] - 3) * .25
Drives['c'] = (Drives['prior_drives_passing_tds']/Drives['prior_drives_pass_attempts']) * 20
Drives['d'] = (2.375 - (Drives['prior_drives_interceptions']/Drives['prior_drives_pass_attempts'] * 25))

# Cap values between 0 and 2.375 
Drives['a'].clip(lower = 0, upper = 2.375, inplace = True)
Drives['b'].clip(lower = 0, upper = 2.375, inplace = True)
Drives['c'].clip(lower = 0, upper = 2.375, inplace = True)
Drives['d'].clip(lower = 0, upper = 2.375, inplace = True)

# Finally, calculate the actual passer rating
Drives['prior_passer_rating'] = (Drives['a'] + Drives['b'] + Drives['c'] + Drives['d'])/6 * 100

# Instead of replacing null values with 0, we put the average across all years ~ 81
Drives['prior_passer_rating'].fillna(Drives['prior_passer_rating'].mean(), inplace = True)

#%% 

#Create column




#%%
# Remove temporary staging columns
Drives.drop(['RunOver10_prior_drives', 'PassOver20_prior_drives', 'PointsScored_prior_drives', 'Plays_prior_drives', 'drive_length_prior_drives', 
             'sack_prior_drives', 'TackledForLossPlays_prior_drives', 'PassAttempts_prior_drives', 'PassCompletions_prior_drives', 'PassYardage_prior_drives',
             'RushAttempts_prior_drives', 'RunYardage_prior_drives', 'pass_touchdown','interception', 'a', 'b','c','d'], axis=1, inplace=True)

#%%
Drives.to_csv('../../data/drives/drives.csv', index = False)