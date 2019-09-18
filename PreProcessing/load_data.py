import random
import numpy as np
import pandas as pd


def calculate_margin(row):
    if (row['Team'] == row['Home_Team']):
        return row['Home_Score'] - row['Away_Score']
    else:
        return row['Away_Score'] - row['Home_Score']


def sample_no_votes(row, sample_probability=10):
    if row['Brownlow_Votes'] > 0:
        return True
    else:
        return True if random.randint(0, 100) < sample_probability else False


def all_data_without_contested_possession_breakdown(sample_probability=None):
    df = pd.read_csv("Brownlow Full Database.csv")
    df = df.iloc[:, :-4]
    df['Winning_Margin'] = df.apply(lambda x: calculate_margin(x), axis=1)
    df = df.iloc[:, 10:]

    if sample_probability is not None:
        df['keep'] = df.apply(lambda x: sample_no_votes(x, sample_probability), axis=1)
        df = df.drop(df[df.keep is False].index)
        df = df.drop(['keep'])

    features = df.drop([
        'Contested_Possessions',  # values not recorded in 2012 data
        'Uncontested_Possessions',  # values not recorded in 2012 data
        'Brownlow_Votes'  # this is the target
    ], axis=1).values

    targets = df['Brownlow_Votes'].fillna(0).values  # in 2018 data 0 votes are just blank cells

    return features, targets
