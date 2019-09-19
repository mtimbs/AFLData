import random
import pandas as pd


def calculate_margin(row):
    '''
    Calculates margin for the match and assigns it to the player based on which team they are on
    '''
    if (row['Team'] == row['Home_Team']):
        return row['Home_Score'] - row['Away_Score']
    else:
        return row['Away_Score'] - row['Home_Score']


def sample_no_votes(row, sample_probability=10):
    '''
    The number of players receiving 0 votes vastly outnumbers the number getting 1,2 or 3.
    This means that models can be fairly accurate predicting 0 votes for every player.
    Some models perform poorly if one label outweighs the others by too much.
    '''
    if row['Brownlow_Votes'] > 0:
        return True
    else:
        return True if random.randint(0, 100) < sample_probability else False


def all_data_without_contested_possession_breakdown(sample_probability=None):
    df = pd.read_csv("Brownlow Full Database.csv")
    df = df.iloc[:, :-4]   # removing some empty cells at the end of the data-frame that seem to be caused by the csv
    df['Winning_Margin'] = df.apply(lambda x: calculate_margin(x), axis=1)

    if sample_probability is not None:
        df['keep'] = df.apply(lambda x: sample_no_votes(x, sample_probability), axis=1)
        df = df.drop(df[df.keep == False].index)
        df = df.drop(['keep'], axis=1)

    # Remove columns that we won't use as features in the training data
    df = df.drop([
        'Unique_Game_ID',
        'Year',
        'Round',
        'Game_ID',
        'Home_Team',
        'Home_Score',
        'Away_Team',
        'Away_Score',
        'Player',
        'Team',
        'Contested_Possessions',  # values not recorded in 2012 data
        'Uncontested_Possessions',  # values not recorded in 2012 data
    ], axis=1)


    features = df.drop([
        'Brownlow_Votes'  # this is the target
    ], axis=1).values

    targets = df['Brownlow_Votes'].fillna(0).values  # in 2018 data 0 votes are just blank cells

    return features, targets
