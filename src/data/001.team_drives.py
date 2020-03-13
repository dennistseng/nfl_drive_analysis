# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:56:15 2020

@author: M44427
"""

import pandas as pd
import numpy as np


# Load team defense on drives against - extracted from Pro-Football-Reference.com
# https://www.pro-football-reference.com/years/2008/opp.htm
# Replace team names so we can incorporate it back into the Drives Data grouped using rawPlayByPlay

# Team Defense
teamDriveDefense = pd.read_csv("../../data/raw/TeamDefenseYearlyDrivesAgainst.csv", low_memory = False)
teamDriveDefense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamDriveDefense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamDriveDefense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamDriveDefense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamDriveDefense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamDriveDefense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamDriveDefense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamDriveDefense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamDriveDefense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamDriveDefense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamDriveDefense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamDriveDefense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamDriveDefense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamDriveDefense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamDriveDefense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamDriveDefense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamDriveDefense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamDriveDefense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamDriveDefense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamDriveDefense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamDriveDefense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamDriveDefense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamDriveDefense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamDriveDefense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamDriveDefense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamDriveDefense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamDriveDefense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamDriveDefense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamDriveDefense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamDriveDefense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamDriveDefense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamDriveDefense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamDriveDefense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamDriveDefense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamDriveDefense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
teamDriveDefense['DefStartingYdsToGo'] = 100- teamDriveDefense['Start']
teamDriveDefense.rename({'Plays.1': 'DefAvgPlaysPerDrive'}, axis=1, inplace=True)

# Load team drive defense - 3rd down conversion stats
teamDriveDefenseConversion = pd.read_csv("../../data/raw/TeamDefenseConversionsAgainst.csv", low_memory = False)
teamDriveDefenseConversion['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamDriveDefenseConversion['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamDriveDefenseConversion['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamDriveDefenseConversion['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamDriveDefenseConversion['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamDriveDefenseConversion['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamDriveDefenseConversion['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamDriveDefenseConversion['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamDriveDefenseConversion['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
del teamDriveDefenseConversion['3D%']
del teamDriveDefenseConversion['G']
del teamDriveDefenseConversion['4D%']
del teamDriveDefenseConversion['4DConv']
del teamDriveDefenseConversion['4DAtt']
del teamDriveDefenseConversion['RZPct']
del teamDriveDefenseConversion['Rk']

# Load team rushing defense
teamRushingDefense = pd.read_csv("../../data/raw/TeamRushingDefense.csv", low_memory = False)
teamRushingDefense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamRushingDefense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamRushingDefense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamRushingDefense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamRushingDefense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamRushingDefense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamRushingDefense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamRushingDefense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamRushingDefense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamRushingDefense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamRushingDefense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamRushingDefense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamRushingDefense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamRushingDefense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamRushingDefense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamRushingDefense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamRushingDefense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamRushingDefense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamRushingDefense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamRushingDefense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamRushingDefense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamRushingDefense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamRushingDefense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamRushingDefense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamRushingDefense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamRushingDefense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamRushingDefense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamRushingDefense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamRushingDefense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamRushingDefense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamRushingDefense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamRushingDefense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamRushingDefense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamRushingDefense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamRushingDefense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
teamRushingDefense.rename({'Y/A': 'RushYards'}, axis=1, inplace=True)
teamRushingDefense.rename({'Y/G': 'RushYardsPerGame'}, axis=1, inplace=True)
teamRushingDefense.rename({'Att': 'RushAtts'}, axis=1, inplace=True)
teamRushingDefense.rename({'Yds': 'RushYds'}, axis=1, inplace=True)
teamRushingDefense.rename({'TD': 'RushTDs'}, axis=1, inplace=True)
del teamRushingDefense['EXP']
del teamRushingDefense['Rk']
del teamRushingDefense['G']

# Passing team passing defense stats
teamPassingDefense = pd.read_csv("../../data/raw/TeamPassingDefense.csv", low_memory = False)
teamPassingDefense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamPassingDefense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamPassingDefense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamPassingDefense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamPassingDefense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamPassingDefense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamPassingDefense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamPassingDefense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamPassingDefense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamPassingDefense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamPassingDefense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamPassingDefense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamPassingDefense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamPassingDefense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamPassingDefense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamPassingDefense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamPassingDefense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamPassingDefense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamPassingDefense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamPassingDefense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamPassingDefense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamPassingDefense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamPassingDefense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamPassingDefense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamPassingDefense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamPassingDefense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamPassingDefense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamPassingDefense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamPassingDefense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamPassingDefense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamPassingDefense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamPassingDefense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamPassingDefense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamPassingDefense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamPassingDefense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
del teamPassingDefense['G']
del teamPassingDefense['Rk']
del teamPassingDefense['QBHits']
del teamPassingDefense['TD%']
del teamPassingDefense['Y/G']
del teamPassingDefense['Y/C']
del teamPassingDefense['Yds.1']
del teamPassingDefense['EXP']
del teamPassingDefense['Sk%']
del teamPassingDefense['Int%']
teamPassingDefense.rename({'Att': 'PassAtts'}, axis=1, inplace=True)
teamPassingDefense.rename({'Cmp': 'PassCmp'}, axis=1, inplace=True)
teamPassingDefense.rename({'Cmp%': 'PassCmp%'}, axis=1, inplace=True)
teamPassingDefense.rename({'Yds': 'PassYds'}, axis=1, inplace=True)
teamPassingDefense.rename({'TD': 'PassTD'}, axis=1, inplace=True)
teamPassingDefense.rename({'Y/A': 'PassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingDefense.rename({'AY/A': 'AdjustedPassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingDefense.rename({'Rate': 'PasserRating'}, axis=1, inplace=True)
teamPassingDefense.rename({'NY/A': 'NetPassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingDefense.rename({'ANY/A': 'AdjustedNetPassYardsPerAttempt'}, axis=1, inplace=True)

#%%
# Merge all defensive team stats
teamDriveDefense = pd.merge(teamDriveDefense, teamDriveDefenseConversion, how = 'left', on=['Year', 'Tm'])
teamDriveDefense = pd.merge(teamDriveDefense, teamRushingDefense, how = 'left', on=['Year', 'Tm'])
teamDriveDefense = pd.merge(teamDriveDefense, teamPassingDefense, how = 'left', on=['Year', 'Tm'])
teamDriveDefense.rename({'Tm': 'defteam'}, axis=1, inplace=True)
teamDriveDefense.rename({'Year': 'GameYear'}, axis=1, inplace=True)
teamDriveDefense.rename({'Time': 'DefTimePerDrive'}, axis=1, inplace=True)
teamDriveDefense.rename({'PassCmp%': 'DefPassCmp%'}, axis=1, inplace=True)
teamDriveDefense.rename({'PasserRating': 'DefPasserRating'}, axis=1, inplace=True)
teamDriveDefense.rename({'TO%': 'DefTO%'}, axis=1, inplace=True)
teamDriveDefense.rename({'Pts': 'DefPtsPerDrive'}, axis=1, inplace=True)
teamDriveDefense.head()

#%%
# Create Defensive Team/Drive relevant stats and remove unecessary ones
teamDriveDefense['DefPassTDsPerDrive'] = teamDriveDefense['PassTD'] / teamDriveDefense['#Dr']
teamDriveDefense['DefRushTDsPerDrive'] = teamDriveDefense['RushTDs'] / teamDriveDefense['#Dr']
teamDriveDefense['DefPassYardsPerDrive'] = teamDriveDefense['PassYds'] / teamDriveDefense['#Dr']
teamDriveDefense['DefRushYardsPerDrive'] = teamDriveDefense['RushYds'] / teamDriveDefense['#Dr']
teamDriveDefense['DefRushAttsPerDrive'] = teamDriveDefense['RushAtts'] / teamDriveDefense['#Dr']
teamDriveDefense['Def3DConvPerDrive'] = teamDriveDefense['3DConv'] / teamDriveDefense['#Dr']
teamDriveDefense['Def3D%'] = teamDriveDefense['3DConv'] / teamDriveDefense['3DAtt']
teamDriveDefense['DefSacksPerDrive'] = teamDriveDefense['Sk'] / teamDriveDefense['#Dr']
teamDriveDefense['DefTFLPerDrive'] = teamDriveDefense['TFL'] / teamDriveDefense['#Dr']
teamDriveDefense['DefRunPct'] = teamDriveDefense['RushAtts'] / (teamDriveDefense['RushAtts'] + teamDriveDefense['PassAtts'])
teamDriveDefense['DefRunPct'] = teamDriveDefense['RushAtts'] / (teamDriveDefense['RushAtts'] + teamDriveDefense['PassAtts'])

del teamDriveDefense['RushYardsPerGame']
del teamDriveDefense['RZTD']
del teamDriveDefense['RZAtt']
del teamDriveDefense['Rk']
del teamDriveDefense['G']
del teamDriveDefense['3DAtt']
del teamDriveDefense['3DConv']
del teamDriveDefense['#Dr']
del teamDriveDefense['Plays']
del teamDriveDefense['Yds']
del teamDriveDefense['Start']
del teamDriveDefense['Sk']
del teamDriveDefense['TFL']
del teamDriveDefense['Sc%']
del teamDriveDefense['Int']
del teamDriveDefense['PD']
del teamDriveDefense['RushAtts']
del teamDriveDefense['RushYds']
del teamDriveDefense['PassYds']
del teamDriveDefense['PassTD']
del teamDriveDefense['RushTDs']
del teamDriveDefense['RushYards']
del teamDriveDefense['PassAtts']
del teamDriveDefense['PassCmp']
del teamDriveDefense['AdjustedNetPassYardsPerAttempt']
del teamDriveDefense['AdjustedPassYardsPerAttempt']
del teamDriveDefense['NetPassYardsPerAttempt']
del teamDriveDefense['PassYardsPerAttempt']

teamDefensePlays = pd.crosstab(teamDriveDefense['GameYear'], teamDriveDefense['defteam'], values = teamDriveDefense['DefAvgPlaysPerDrive'], aggfunc = np.sum)

#%%

# Load team offense on drives - extracted from Pro-Football-Reference.com
# https://www.pro-football-reference.com/years/2017/
# Replace team names so we can incorporate it back into the Drives Data grouped using rawPlayByPlay

# Team Offense
teamDriveOffense = pd.read_csv("../../data/raw/TeamOffenseDrives.csv", low_memory = False)
teamDriveOffense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamDriveOffense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamDriveOffense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamDriveOffense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamDriveOffense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamDriveOffense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamDriveOffense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamDriveOffense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamDriveOffense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamDriveOffense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamDriveOffense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamDriveOffense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamDriveOffense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamDriveOffense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamDriveOffense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamDriveOffense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamDriveOffense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamDriveOffense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamDriveOffense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamDriveOffense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamDriveOffense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamDriveOffense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamDriveOffense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamDriveOffense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamDriveOffense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamDriveOffense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamDriveOffense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamDriveOffense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamDriveOffense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamDriveOffense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamDriveOffense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamDriveOffense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamDriveOffense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamDriveOffense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamDriveOffense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
teamDriveOffense['OffStartingYdsToGo'] = 100- teamDriveOffense['Start']
teamDriveOffense.rename({'Plays.1': 'OffAvgPlaysPerDrive'}, axis=1, inplace=True)

# Load team drive offense - 3rd down conversion stats
teamDriveOffenseConversion = pd.read_csv("../../data/raw/TeamOffenseConversions.csv", low_memory = False)
teamDriveOffenseConversion['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamDriveOffenseConversion['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamDriveOffenseConversion['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamDriveOffenseConversion['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamDriveOffenseConversion['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamDriveOffenseConversion['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamDriveOffenseConversion['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamDriveOffenseConversion['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamDriveOffenseConversion['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
del teamDriveOffenseConversion['3D%']
del teamDriveOffenseConversion['G']
del teamDriveOffenseConversion['4D%']
del teamDriveOffenseConversion['4DConv']
del teamDriveOffenseConversion['4DAtt']
del teamDriveOffenseConversion['RZPct']
del teamDriveOffenseConversion['Rk']

# Load team rushing offense
teamRushingOffense = pd.read_csv("../../data/raw/TeamRushingOffense.csv", low_memory = False)
teamRushingOffense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamRushingOffense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamRushingOffense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamRushingOffense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamRushingOffense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamRushingOffense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamRushingOffense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamRushingOffense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamRushingOffense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamRushingOffense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamRushingOffense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamRushingOffense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamRushingOffense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamRushingOffense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamRushingOffense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamRushingOffense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamRushingOffense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamRushingOffense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamRushingOffense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamRushingOffense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamRushingOffense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamRushingOffense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamRushingOffense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamRushingOffense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamRushingOffense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamRushingOffense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamRushingOffense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamRushingOffense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamRushingOffense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamRushingOffense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamRushingOffense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamRushingOffense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamRushingOffense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamRushingOffense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamRushingOffense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
teamRushingOffense.rename({'Y/A': 'RushYardsPerAttempt'}, axis=1, inplace=True)
teamRushingOffense.rename({'Y/G': 'RushYardsPerGame'}, axis=1, inplace=True)
teamRushingOffense.rename({'Att': 'RushAtts'}, axis=1, inplace=True)
teamRushingOffense.rename({'Yds': 'RushYds'}, axis=1, inplace=True)
teamRushingOffense.rename({'TD': 'RushTDs'}, axis=1, inplace=True)
del teamRushingOffense['EXP']
del teamRushingOffense['Rk']
del teamRushingOffense['G']

# Passing team passing offense stats
teamPassingOffense = pd.read_csv("../../data/raw/TeamPassingOffense.csv", low_memory = False)
teamPassingOffense['Tm'].replace('Houston Texans', 'HOU', inplace=True)
teamPassingOffense['Tm'].replace('New York Giants', 'NYG', inplace=True)
teamPassingOffense['Tm'].replace('Jacksonville Jaguars', 'JAC', inplace=True)
teamPassingOffense['Tm'].replace('Philadelphia Eagles', 'PHI', inplace=True)
teamPassingOffense['Tm'].replace('Denver Broncos', 'DEN', inplace=True)
teamPassingOffense['Tm'].replace('Cleveland Browns', 'CLE', inplace=True)
teamPassingOffense['Tm'].replace('Detroit Lions', 'DET', inplace=True)
teamPassingOffense['Tm'].replace('San Diego Chargers', 'SD', inplace=True)
teamPassingOffense['Tm'].replace('St. Louis Rams', 'STL', inplace=True)
teamPassingOffense['Tm'].replace('Kansas City Chiefs', 'KC', inplace=True)
teamPassingOffense['Tm'].replace('Carolina Panthers', 'CAR', inplace=True)
teamPassingOffense['Tm'].replace('Dallas Cowboys', 'DAL', inplace=True)
teamPassingOffense['Tm'].replace('San Francisco 49ers', 'SF', inplace=True)
teamPassingOffense['Tm'].replace('Los Angeles Chargers', 'LAC', inplace=True)
teamPassingOffense['Tm'].replace('Seattle Seahawks', 'SEA', inplace=True)
teamPassingOffense['Tm'].replace('Buffalo Bills', 'BUF', inplace=True)
teamPassingOffense['Tm'].replace('Jacksonville Jaguars', 'JAX', inplace=True)
teamPassingOffense['Tm'].replace('Washington Redskins', 'WAS', inplace=True)
teamPassingOffense['Tm'].replace('New England Patriots', 'NE', inplace=True)
teamPassingOffense['Tm'].replace('New Orleans Saints', 'NO', inplace=True)
teamPassingOffense['Tm'].replace('Los Angeles Rams', 'LA', inplace=True)
teamPassingOffense['Tm'].replace('Indianapolis Colts', 'IND', inplace=True)
teamPassingOffense['Tm'].replace('Baltimore Ravens', 'BAL', inplace=True)
teamPassingOffense['Tm'].replace('Pittsburgh Steelers', 'PIT', inplace=True)
teamPassingOffense['Tm'].replace('Green Bay Packers', 'GB', inplace=True)
teamPassingOffense['Tm'].replace('Tampa Bay Buccaneers', 'TB', inplace=True)
teamPassingOffense['Tm'].replace('Chicago Bears', 'CHI', inplace=True)
teamPassingOffense['Tm'].replace('Arizona Cardinals', 'ARI', inplace=True)
teamPassingOffense['Tm'].replace('Tennessee Titans', 'TEN', inplace=True)
teamPassingOffense['Tm'].replace('New York Jets', 'NYJ', inplace=True)
teamPassingOffense['Tm'].replace('Oakland Raiders', 'OAK', inplace=True)
teamPassingOffense['Tm'].replace('Atlanta Falcons', 'ATL', inplace=True)
teamPassingOffense['Tm'].replace('Cincinnati Bengals', 'CIN', inplace=True)
teamPassingOffense['Tm'].replace('Minnesota Vikings', 'MIN', inplace=True)
teamPassingOffense['Tm'].replace('Miami Dolphins', 'MIA', inplace=True)
del teamPassingOffense['G']
del teamPassingOffense['Rk']
del teamPassingOffense['TD%']
del teamPassingOffense['Y/G']
del teamPassingOffense['Y/C']
del teamPassingOffense['Yds.1']
del teamPassingOffense['EXP']
del teamPassingOffense['Sk%']
del teamPassingOffense['Int%']
del teamPassingOffense['4QC']
del teamPassingOffense['GWD']
teamPassingOffense.rename({'Att': 'PassAtts'}, axis=1, inplace=True)
teamPassingOffense.rename({'Cmp': 'PassCmp'}, axis=1, inplace=True)
teamPassingOffense.rename({'Cmp%': 'OffPassCmp%'}, axis=1, inplace=True)
teamPassingOffense.rename({'Yds': 'PassYds'}, axis=1, inplace=True)
teamPassingOffense.rename({'TD': 'PassTD'}, axis=1, inplace=True)
teamPassingOffense.rename({'Y/A': 'PassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingOffense.rename({'AY/A': 'AdjustedPassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingOffense.rename({'Rate': 'PasserRating'}, axis=1, inplace=True)
teamPassingOffense.rename({'NY/A': 'NetPassYardsPerAttempt'}, axis=1, inplace=True)
teamPassingOffense.rename({'ANY/A': 'AdjustedNetPassYardsPerAttempt'}, axis=1, inplace=True)

#%%
# Merge all offensive team stats
teamDriveOffense = pd.merge(teamDriveOffense, teamDriveOffenseConversion, how = 'left', on=['Year', 'Tm'])
teamDriveOffense = pd.merge(teamDriveOffense, teamRushingOffense, how = 'left', on=['Year', 'Tm'])
teamDriveOffense = pd.merge(teamDriveOffense, teamPassingOffense, how = 'left', on=['Year', 'Tm'])
teamDriveOffense.rename({'Year': 'GameYear'}, axis=1, inplace=True)
teamDriveOffense.rename({'Tm': 'posteam'}, axis=1, inplace=True)
teamDriveOffense.rename({'PasserRating': 'OffPasserRating'}, axis=1, inplace=True)
teamDriveOffense.rename({'Time': 'OffTimePerDrive'}, axis=1, inplace=True)
teamDriveOffense.rename({'TO%': 'OffTO%'}, axis=1, inplace=True)
teamDriveOffense.head()

#%% 

# Cleanup and creating offensive drive stats
teamDriveOffense['OffPassTDsPerDrive'] = teamDriveOffense['PassTD'] / teamDriveOffense['#Dr']
teamDriveOffense['OffRushTDsPerDrive'] = teamDriveOffense['RushTDs'] / teamDriveOffense['#Dr']
teamDriveOffense['OffPassYardsPerDrive'] = teamDriveOffense['PassYds'] / teamDriveOffense['#Dr']
teamDriveOffense['OffRushYardsPerDrive'] = teamDriveOffense['RushYds'] / teamDriveOffense['#Dr']
teamDriveOffense['Off3DConvPerDrive'] = teamDriveOffense['3DConv'] / teamDriveOffense['#Dr']
teamDriveOffense['Off3D%'] = teamDriveOffense['3DConv'] / teamDriveOffense['3DAtt']
teamDriveOffense['OffPtsPerDrive'] = teamDriveOffense['Pts'] / teamDriveOffense['#Dr']
teamDriveOffense['OffSacksPerDrive'] = teamDriveOffense['Sk'] / teamDriveOffense['#Dr']
teamDriveOffense['OffRunPct'] = teamDriveOffense['RushAtts'] / (teamDriveOffense['RushAtts'] + teamDriveOffense['PassAtts'])
teamDriveOffense['OffRunPct'] = teamDriveOffense['RushAtts'] / (teamDriveOffense['RushAtts'] + teamDriveOffense['PassAtts'])
teamDriveOffense['OffRushAttsPerDrive'] = teamDriveOffense['RushAtts'] / teamDriveOffense['#Dr']

del teamDriveOffense['RushYardsPerGame']
del teamDriveOffense['RZTD']
del teamDriveOffense['RZAtt']
del teamDriveOffense['Rk']
del teamDriveOffense['G']
del teamDriveOffense['3DAtt']
del teamDriveOffense['3DConv']
del teamDriveOffense['#Dr']
del teamDriveOffense['Plays']
del teamDriveOffense['Yds']
del teamDriveOffense['Start']
del teamDriveOffense['Pts']
del teamDriveOffense['Sk']
del teamDriveOffense['Sc%']
del teamDriveOffense['Int']
del teamDriveOffense['PassYds']
del teamDriveOffense['PassTD']
del teamDriveOffense['RushTDs']
del teamDriveOffense['RushYds']
del teamDriveOffense['PassAtts']
del teamDriveOffense['RushAtts']
del teamDriveOffense['PassCmp']
del teamDriveOffense['Lng_x']
del teamDriveOffense['Fmb']
del teamDriveOffense['Lng_y']
del teamDriveOffense['RushYardsPerAttempt']
del teamDriveOffense['AdjustedNetPassYardsPerAttempt']
del teamDriveOffense['AdjustedPassYardsPerAttempt']
del teamDriveOffense['NetPassYardsPerAttempt']
del teamDriveOffense['PassYardsPerAttempt']


teamDriveOffense['posteam'].replace('STL', 'LA', inplace=True)
teamDriveOffense['posteam'].replace('SD', 'LAC', inplace=True)

teamDriveDefense['defteam'].replace('STL', 'LA', inplace=True)
teamDriveDefense['defteam'].replace('SD', 'LAC', inplace=True)

teamDriveDefense.to_csv('../../data/clean_play_by_play/teamDriveDefense.csv', index = False)
teamDriveOffense.to_csv('../../data/clean_play_by_play/teamDriveOffense.csv', index = False)