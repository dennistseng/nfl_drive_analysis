# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:39:57 2020

@author: halfc
"""

#import libraries
import os
import numpy as np
import pandas as pd
import math
import sklearn as sk
from sklearn.model_selection import train_test_split
import pylab as pl
import matplotlib.pyplot as plt
import datetime
from collections import Counter
#from imblearn.datasets import make_imbalance

# %% 
#######################################################
## Load and Clean Play By Play Dataset
#######################################################

# Load dataset
rawPlayByPlay = pd.read_csv("../../data/raw/reg_pbp_2018.csv", low_memory = False)

# Remove any colums deemed unecessary for both pre-processing and analysis 
del rawPlayByPlay['fumble_recovery_1_player_id']
del rawPlayByPlay['fumble_recovery_1_team']
del rawPlayByPlay['fumble_recovery_1_yards']
del rawPlayByPlay['fumble_recovery_1_player_name']
del rawPlayByPlay['fumble_recovery_2_player_id']
del rawPlayByPlay['fumble_recovery_2_player_name']
del rawPlayByPlay['fumble_recovery_2_team']
del rawPlayByPlay['fumble_recovery_2_yards']
del rawPlayByPlay['fumbled_1_team']
del rawPlayByPlay['fumbled_1_player_name']
del rawPlayByPlay['fumbled_1_player_id']
del rawPlayByPlay['fumbled_2_player_id']
del rawPlayByPlay['fumbled_2_player_name']
del rawPlayByPlay['fumbled_2_team']
del rawPlayByPlay['pass_defense_1_player_id']
del rawPlayByPlay['pass_defense_1_player_name']
del rawPlayByPlay['pass_defense_2_player_id']
del rawPlayByPlay['pass_defense_2_player_name']
del rawPlayByPlay['assist_tackle_4_player_id']
del rawPlayByPlay['assist_tackle_4_player_name']
del rawPlayByPlay['assist_tackle_4_team']
del rawPlayByPlay['assist_tackle_3_team']
del rawPlayByPlay['assist_tackle_3_player_name']
del rawPlayByPlay['assist_tackle_3_player_id']
del rawPlayByPlay['assist_tackle_2_player_id']
del rawPlayByPlay['assist_tackle_2_player_name']
del rawPlayByPlay['assist_tackle_2_team']
del rawPlayByPlay['assist_tackle_1_player_id']
del rawPlayByPlay['assist_tackle_1_player_name']
del rawPlayByPlay['assist_tackle_1_team']
del rawPlayByPlay['solo_tackle_2_player_name']
del rawPlayByPlay['solo_tackle_1_player_name']
del rawPlayByPlay['solo_tackle_2_team']
del rawPlayByPlay['solo_tackle_1_team']
del rawPlayByPlay['solo_tackle_1_player_id']
del rawPlayByPlay['solo_tackle_2_player_id']
del rawPlayByPlay['forced_fumble_player_2_player_name']
del rawPlayByPlay['forced_fumble_player_2_player_id']
del rawPlayByPlay['forced_fumble_player_2_team']
del rawPlayByPlay['forced_fumble_player_1_player_name']
del rawPlayByPlay['forced_fumble_player_1_player_id']
del rawPlayByPlay['forced_fumble_player_1_team']
del rawPlayByPlay['qb_hit_2_player_name']
del rawPlayByPlay['qb_hit_2_player_id']
del rawPlayByPlay['qb_hit_1_player_id']
del rawPlayByPlay['qb_hit_1_player_name']
del rawPlayByPlay['tackle_for_loss_2_player_name']
del rawPlayByPlay['tackle_for_loss_2_player_id']
del rawPlayByPlay['tackle_for_loss_1_player_id']
del rawPlayByPlay['tackle_for_loss_1_player_name']
del rawPlayByPlay['penalty_player_name']
del rawPlayByPlay['penalty_player_id']
del rawPlayByPlay['blocked_player_id']
del rawPlayByPlay['blocked_player_name']
del rawPlayByPlay['own_kickoff_recovery_player_id']
del rawPlayByPlay['own_kickoff_recovery_player_name']
del rawPlayByPlay['own_kickoff_recovery_td']
del rawPlayByPlay['kicker_player_name']
del rawPlayByPlay['kicker_player_id']
del rawPlayByPlay['punter_player_name']
del rawPlayByPlay['punter_player_id']
del rawPlayByPlay['lateral_kickoff_returner_player_id']
del rawPlayByPlay['lateral_kickoff_returner_player_name']
del rawPlayByPlay['kickoff_returner_player_id']
del rawPlayByPlay['kickoff_returner_player_name']
del rawPlayByPlay['lateral_punt_returner_player_id']
del rawPlayByPlay['lateral_punt_returner_player_name']
del rawPlayByPlay['punt_returner_player_id']
del rawPlayByPlay['punt_returner_player_name']
del rawPlayByPlay['lateral_interception_player_id']
del rawPlayByPlay['lateral_interception_player_name']
del rawPlayByPlay['interception_player_id']
del rawPlayByPlay['interception_player_name']
del rawPlayByPlay['lateral_sack_player_id']
del rawPlayByPlay['lateral_sack_player_name']
del rawPlayByPlay['lateral_rusher_player_name']
del rawPlayByPlay['lateral_rusher_player_id']
del rawPlayByPlay['rusher_player_name']
del rawPlayByPlay['rusher_player_id']
del rawPlayByPlay['lateral_receiver_player_id']
del rawPlayByPlay['lateral_receiver_player_name']
del rawPlayByPlay['passer_player_id']
del rawPlayByPlay['passer_player_name']
del rawPlayByPlay['receiver_player_id']
del rawPlayByPlay['receiver_player_name']
del rawPlayByPlay['total_away_raw_yac_wpa']
del rawPlayByPlay['total_home_raw_yac_wpa']
del rawPlayByPlay['total_away_raw_air_wpa']
del rawPlayByPlay['total_home_comp_air_wpa']
del rawPlayByPlay['total_away_comp_air_epa']
del rawPlayByPlay['total_away_comp_air_wpa']
del rawPlayByPlay['total_away_comp_yac_epa']
del rawPlayByPlay['total_away_comp_yac_wpa']
del rawPlayByPlay['total_home_epa']
del rawPlayByPlay['total_away_pass_epa']
del rawPlayByPlay['total_away_pass_wpa']
del rawPlayByPlay['total_away_raw_air_epa']
del rawPlayByPlay['total_away_raw_yac_epa']
del rawPlayByPlay['total_away_epa']
del rawPlayByPlay['total_away_rush_epa']
del rawPlayByPlay['total_away_rush_wpa']
del rawPlayByPlay['total_home_comp_air_epa']
del rawPlayByPlay['total_home_comp_yac_epa']
del rawPlayByPlay['total_home_rush_wpa']
del rawPlayByPlay['total_home_pass_epa']
del rawPlayByPlay['total_home_pass_wpa']
del rawPlayByPlay['total_home_comp_yac_wpa']
del rawPlayByPlay['total_home_raw_air_epa']
del rawPlayByPlay['total_home_raw_yac_epa']
del rawPlayByPlay['total_home_rush_epa']
del rawPlayByPlay['comp_yac_wpa']
del rawPlayByPlay['comp_yac_epa']
del rawPlayByPlay['comp_air_epa']
del rawPlayByPlay['comp_air_wpa']
del rawPlayByPlay['no_score_prob']
del rawPlayByPlay['opp_fg_prob']
del rawPlayByPlay['opp_safety_prob']
del rawPlayByPlay['opp_td_prob']
del rawPlayByPlay['air_epa']
del rawPlayByPlay['yac_epa']
del rawPlayByPlay['air_wpa']
del rawPlayByPlay['epa']
del rawPlayByPlay['ep']
del rawPlayByPlay['wp']
del rawPlayByPlay['wpa']
del rawPlayByPlay['fg_prob']
del rawPlayByPlay['safety_prob']
del rawPlayByPlay['td_prob']
del rawPlayByPlay['extra_point_prob']
del rawPlayByPlay['two_point_conversion_prob']
del rawPlayByPlay['def_wp']
del rawPlayByPlay['home_wp']
del rawPlayByPlay['away_wp']
del rawPlayByPlay['home_wp_post']
del rawPlayByPlay['yac_wpa']
del rawPlayByPlay['away_wp_post']
del rawPlayByPlay['kick_distance']
del rawPlayByPlay['kickoff_inside_twenty']
del rawPlayByPlay['kickoff_attempt']
del rawPlayByPlay['kickoff_downed']
del rawPlayByPlay['kickoff_in_endzone']
del rawPlayByPlay['kickoff_fair_catch']
del rawPlayByPlay['kickoff_out_of_bounds']
del rawPlayByPlay['own_kickoff_recovery']
del rawPlayByPlay['punt_attempt']
del rawPlayByPlay['punt_inside_twenty']
del rawPlayByPlay['punt_downed']
del rawPlayByPlay['punt_in_endzone']
del rawPlayByPlay['punt_out_of_bounds']
del rawPlayByPlay['first_down_pass']
del rawPlayByPlay['first_down_penalty']
del rawPlayByPlay['first_down_rush']
del rawPlayByPlay['field_goal_result']
del rawPlayByPlay['third_down_failed']
del rawPlayByPlay['fourth_down_converted']
del rawPlayByPlay['fourth_down_failed']
del rawPlayByPlay['home_timeouts_remaining']
del rawPlayByPlay['away_timeouts_remaining']
del rawPlayByPlay['solo_tackle']
del rawPlayByPlay['posteam_timeouts_remaining']
del rawPlayByPlay['defteam_timeouts_remaining']
del rawPlayByPlay['return_team']
del rawPlayByPlay['return_yards']
del rawPlayByPlay['return_touchdown']
del rawPlayByPlay['air_yards']
del rawPlayByPlay['yards_after_catch']
del rawPlayByPlay['punt_fair_catch']
del rawPlayByPlay['home_team']
del rawPlayByPlay['away_team']
del rawPlayByPlay['side_of_field']
del rawPlayByPlay['quarter_seconds_remaining']
del rawPlayByPlay['goal_to_go']
del rawPlayByPlay['quarter_end']
del rawPlayByPlay['time']
del rawPlayByPlay['total_home_score']
del rawPlayByPlay['total_away_score']
del rawPlayByPlay['posteam_score']
del rawPlayByPlay['defteam_score']
del rawPlayByPlay['posteam_score_post']
del rawPlayByPlay['defteam_score_post']
del rawPlayByPlay['assist_tackle']
del rawPlayByPlay['replay_or_challenge']
del rawPlayByPlay['replay_or_challenge_result']
del rawPlayByPlay['defensive_extra_point_attempt']
del rawPlayByPlay['defensive_extra_point_conv']
del rawPlayByPlay['lateral_recovery']
del rawPlayByPlay['lateral_reception']
del rawPlayByPlay['lateral_rush']
del rawPlayByPlay['lateral_return']
del rawPlayByPlay['timeout_team']
del rawPlayByPlay['td_team']
del rawPlayByPlay['yrdln']
del rawPlayByPlay['half_seconds_remaining']
del rawPlayByPlay['total_home_raw_air_wpa']

# Remove any duplicate rows
rawPlayByPlay.drop_duplicates(keep='first', inplace=True)
# %% Clean values within columns

# Replace any null/None/NaN values
rawPlayByPlay['desc'].fillna('Other' , inplace=True)
rawPlayByPlay['play_type'].fillna('Other' , inplace=True)
rawPlayByPlay['down'].fillna(0, inplace=True)
rawPlayByPlay['yards_gained'].fillna(0, inplace=True)
rawPlayByPlay['penalty_yards'].fillna(0, inplace=True)
rawPlayByPlay['game_seconds_remaining'].fillna(-1, inplace=True)
rawPlayByPlay['two_point_attempt'].fillna(0, inplace=True)
rawPlayByPlay['defensive_two_point_conv'].fillna(0, inplace=True)
rawPlayByPlay['score_differential'].fillna(0, inplace=True)
rawPlayByPlay['score_differential_post'].fillna(0, inplace=True)
rawPlayByPlay['incomplete_pass'].fillna(0, inplace=True)
rawPlayByPlay['timeout'].fillna(0, inplace=True)

# Set description strings to lowercase
rawPlayByPlay['desc'] = rawPlayByPlay['desc'].str.lower()


# %%

# Calculate/determine if a play resulted in points
# Fixes errors within data (defensive 2 point scores)
rawPlayByPlay['points_earned'] = rawPlayByPlay['score_differential_post'] - rawPlayByPlay['score_differential']
rawPlayByPlay['points_earned'] = np.where(rawPlayByPlay['defensive_two_point_conv'] == 1, -2, rawPlayByPlay['points_earned'])
rawPlayByPlay['points_earned'].fillna(0, inplace=True)

del rawPlayByPlay['score_differential_post']

rawPlayByPlay['play_type'] = np.where(rawPlayByPlay['two_point_attempt'] == 1, "2pt Attempt", rawPlayByPlay['play_type'])
rawPlayByPlay['points_earned'] = np.where(rawPlayByPlay['incomplete_pass'] == 1, 0, rawPlayByPlay['points_earned'])


# %% Time Stoppages

# Separate time stoppages (end of quarters, end of games, timeouts)
timeouts = rawPlayByPlay[(rawPlayByPlay['timeout'] == 1) & (rawPlayByPlay['desc'].str.contains('timeout') & rawPlayByPlay['down'] == 0)]
timeouts = timeouts[['game_id','drive','play_id']]
timeouts = timeouts.groupby(['game_id', 'drive']).agg(['count'])

playStoppages = rawPlayByPlay[(rawPlayByPlay['desc'].str.contains('end of quarter')) | (rawPlayByPlay['desc'].str.contains('end quarter')) | (rawPlayByPlay['desc'].str.contains('two-minute warning'))]
playStoppages = playStoppages[['game_id','drive','play_id']]
playStoppages = playStoppages.groupby(['game_id', 'drive']).agg(['count'])

endOfGames = rawPlayByPlay[(rawPlayByPlay['desc'].str.contains('end game')) | (rawPlayByPlay['desc'].str.contains('end of game'))]


# %%
# Additional Cleanup for offsetting penalties (no plays), substitutions, other non-play tuples
misc = rawPlayByPlay[(rawPlayByPlay['play_type'] == "Other") & (rawPlayByPlay['game_seconds_remaining'] == -1)]
offset = rawPlayByPlay[(rawPlayByPlay['desc'].str.contains('offsetting')) & (rawPlayByPlay['play_type'] == "Other")]
misc2 = rawPlayByPlay[(rawPlayByPlay['play_type'] == "Other")]
misc = pd.concat([misc,misc2])
del misc2

# %%
# Remove Timeouts and Play Stoppages (Non Penalties) from Play Set
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['timeout'] == 1) & (rawPlayByPlay['play_type'] == "no_play") & (rawPlayByPlay['down'] == 0))]
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['down'] == 0) & (rawPlayByPlay['desc'].str.contains('timeout')))]
rawPlayByPlay = rawPlayByPlay[~rawPlayByPlay['desc'].str.contains('two-minute warning', na=False)]
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['desc'].str.contains('end of quarter')) | (rawPlayByPlay['desc'].str.contains('end quarter'))| (rawPlayByPlay['desc'].str.contains('end of half')))]
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['desc'].str.contains('end game')) | (rawPlayByPlay['desc'].str.contains('end of game')))]
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['play_type'] == "Other") & (rawPlayByPlay['game_seconds_remaining'] == -1))]
rawPlayByPlay = rawPlayByPlay[~((rawPlayByPlay['play_type'] == "Other") & (rawPlayByPlay['desc'].str.contains('offsetting')))]
rawPlayByPlay = rawPlayByPlay[~(rawPlayByPlay['play_type'] == "Other")]
rawPlayByPlay = rawPlayByPlay[~(rawPlayByPlay['play_type'] == "extra_point")]
rawPlayByPlay = rawPlayByPlay[~(rawPlayByPlay['play_type'] == "2pt Attempt")]

# %%
# Remove Kickoffs, as they don't count as the first play of a drive and would affect
# starting field position calculations
rawPlayByPlay = rawPlayByPlay[~(rawPlayByPlay['play_type'] == "kickoff")]

# %%
# Convert dates to usable months and year attributes
rawPlayByPlay['GameMonth'] = pd.DatetimeIndex(rawPlayByPlay['game_date']).month
rawPlayByPlay['GameYear'] = pd.DatetimeIndex(rawPlayByPlay['game_date']).year
del rawPlayByPlay['game_date']

# %%
# Create new columns for analysis
rawPlayByPlay['RunOver10'] = np.where((rawPlayByPlay['play_type'] == "run") & (rawPlayByPlay['yards_gained'] >= 10), 1, 0)
rawPlayByPlay['PassOver20'] = np.where((rawPlayByPlay['play_type'] == "pass") & (rawPlayByPlay['yards_gained'] >= 20), 1, 0)
rawPlayByPlay['penalty_yards'] = np.where((rawPlayByPlay['play_type'] == "pass") & (rawPlayByPlay['yards_gained'] >= 20), 1, 0)

# Build tables to join back to Drive dataset later

ThirdDown = rawPlayByPlay[rawPlayByPlay['down'] == 3]
ThirdDown = ThirdDown[['game_id', 'drive', 'posteam', 'ydstogo','third_down_converted']]
ThirdDown = ThirdDown.groupby(['game_id', 'drive', 'posteam']).agg({'ydstogo' : np.mean, 'third_down_converted' : np.sum})

FirstDown = rawPlayByPlay[rawPlayByPlay['down'] == 1]
FirstDown = FirstDown[['game_id', 'drive', 'posteam', 'yards_gained']]
FirstDown = FirstDown.groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained' : np.mean})

PenaltiesPos = rawPlayByPlay[((rawPlayByPlay['penalty'] == 1) & (rawPlayByPlay['posteam'] == rawPlayByPlay['penalty_team']))]
PenaltiesPos = PenaltiesPos[['game_id', 'drive', 'posteam', 'penalty', 'penalty_yards']]
PenaltiesPos = PenaltiesPos.groupby(['game_id', 'drive', 'posteam']).agg({'penalty_yards' : np.sum, 'penalty' : np.sum})

PenaltiesDef = rawPlayByPlay[((rawPlayByPlay['penalty'] == 1) & (rawPlayByPlay['posteam'] != rawPlayByPlay['penalty_team']))]
PenaltiesDef = PenaltiesDef[['game_id', 'drive', 'posteam', 'penalty', 'penalty_yards']]
PenaltiesDef = PenaltiesDef.groupby(['game_id', 'drive', 'posteam']).agg({'penalty_yards' : np.sum, 'penalty' : np.sum})


# %%
# Take out plays labeled qb_kneels that are at the begining of the drive (touchbacks)
idx = rawPlayByPlay.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmax()
Touchbacks = rawPlayByPlay.loc[idx]
Touchbacks = Touchbacks[(Touchbacks['play_type'] == 'qb_kneel')]

rawPlayByPlay = rawPlayByPlay[~rawPlayByPlay.index.isin(Touchbacks.index)]


# %%
# Remove 'no play' plays that were needed in case they were penalties, but not for actual drives
rawPlayByPlay = rawPlayByPlay[~(rawPlayByPlay['play_type'] == "no_play")]

# Take out punts and field goals only if they're the last play in the drive
idx = rawPlayByPlay.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmin()
PuntsAndFG = rawPlayByPlay.loc[idx]
PuntsAndFG = PuntsAndFG[((PuntsAndFG['play_type'] == 'field_goal') | (PuntsAndFG['play_type'] == 'punt'))]

FG = PuntsAndFG[PuntsAndFG['play_type'] == "field_goal"]
FG = FG[['game_id','drive','posteam','points_earned']]
FG['points_earned'] = 3
FG.drop_duplicates(keep='first')


# %%

# Create first play table to determine starting field position/yards to go
idx = rawPlayByPlay.groupby(['game_id','drive', 'posteam'])['game_seconds_remaining'].idxmax()
FirstPlays = rawPlayByPlay.loc[idx, ['game_id', 'drive', 'posteam', 'game_seconds_remaining', 'yardline_100']]
rawPlayByPlay = pd.merge(rawPlayByPlay, FirstPlays, how='left', on=['game_id', 'drive', 'posteam'])
rawPlayByPlay.rename(columns={'game_seconds_remaining_x':'game_seconds_remaining', 'game_seconds_remaining_y':'drive_starting_time', 'yardline_100_x':'yardline_100', 'yardline_100_y':'drive_starting_location'}, inplace=True)


# Create last play table to determine time elapsed per drive
l_idx = rawPlayByPlay.groupby(['game_id','drive', 'posteam'])['game_seconds_remaining'].idxmin()
LastPlays = rawPlayByPlay.loc[l_idx, ['game_id', 'drive', 'posteam', 'game_seconds_remaining', 'yardline_100']]
rawPlayByPlay = pd.merge(rawPlayByPlay, LastPlays, how='left', on=['game_id', 'drive', 'posteam'])
rawPlayByPlay.rename(columns={'game_seconds_remaining_x':'game_seconds_remaining', 'game_seconds_remaining_y':'drive_end_time', 'yardline_100_x':'yardline_100', 'yardline_100_y':'drive_end_location'}, inplace = True)


# %%

#########
#This outputs play by play outcomes for use in creating historical features
#########
prior_play_by_play = rawPlayByPlay[['play_id','game_id', 'posteam', 'defteam', 'drive', 'sp', 'qtr', 'down', 'ydstogo', 'ydsnet','game_seconds_remaining', 'play_type', 'yards_gained', 'punt_blocked', 'interception', 'fumble_lost', 'points_earned']]
last_play_idx = prior_play_by_play.groupby(['game_id', 'drive', 'posteam'])['game_seconds_remaining'].idxmin()
last_plays = prior_play_by_play.loc[last_play_idx]

# set up 'Outcome' column
last_plays['drive_outcome'] = str(0)
last_plays['drive_outcome'] = np.where(last_plays['punt_blocked'] == 1, 'punt_blocked', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['fumble_lost'] == 1, 'fumble_lost', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['interception'] == 1, 'interception', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['play_type'] == 'punt', 'punt', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['play_type'] == 'field_goal', 'field_goal', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['points_earned'] == 6, 'touchdown', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(((last_plays['down'] == 4) & (last_plays['drive_outcome'] == '0')), 'turnover_on_downs', last_plays['drive_outcome'])
last_plays['drive_outcome'] = np.where(last_plays['drive_outcome'] == '0', 'end_of_half', last_plays['drive_outcome'])

last_plays = last_plays[['game_id','posteam','drive','drive_outcome']]

# %%
# Removes punts and field goals from drives for 'clean' play-by-play

rawPlayByPlay = rawPlayByPlay[~rawPlayByPlay.index.isin(PuntsAndFG.index)]

# %%
# YPA for Rushes and Passes
Passes = rawPlayByPlay[(rawPlayByPlay['play_type']== "pass")].groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained': np.sum})
Passes.rename({'yards_gained': 'PassYardage'}, axis=1, inplace=True)

Runs = rawPlayByPlay[(rawPlayByPlay['play_type']== "run")].groupby(['game_id', 'drive', 'posteam']).agg({'yards_gained': np.sum})
Runs.rename({'yards_gained': 'RunYardage'}, axis=1, inplace=True)

DrivePlays = rawPlayByPlay[['game_id', 'drive','posteam','play_id']].groupby(['game_id', 'drive', 'posteam']).count()
DrivePlays.rename({'play_id': 'Count'}, axis=1, inplace=True)

# %%
#########
#This outputs final play by play data for analysis
#########
rawPlayByPlay.to_csv('../../data/clean_play_by_play/plays.csv', index = False)
last_plays.to_csv('../../data/clean_play_by_play/last_plays.csv', index = False)
FG.to_csv('../../data/clean_play_by_play/FG.csv', index = False)
PenaltiesDef.to_csv('../../data/clean_play_by_play/PenaltiesDef.csv')
PenaltiesPos.to_csv('../../data/clean_play_by_play/PenaltiesPos.csv')
FirstDown.to_csv('../../data/clean_play_by_play/FirstDown.csv')
ThirdDown.to_csv('../../data/clean_play_by_play/ThirdDown.csv')
Runs.to_csv('../../data/clean_play_by_play/Runs.csv')
Passes.to_csv('../../data/clean_play_by_play/Passes.csv')