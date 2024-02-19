#!/usr/bin/env python
import pandas as pd
import numpy as np

def clean(filename):
    """
    Cleans data.
    -------------------------
    INPUTS:
        filename: (str) File path (local)

    OUTPUTS:
        (pandas DataFrame) Cleaned data frame.
    """
    df = pd.read_csv(r"./trainingT2HAR/har_train.csv", header=None, ).copy()
    df.rename(columns=({328: "activity"}), inplace=True)
    df["activity"] = df["acitivty"].str.lower() # Lowercase the Series
    return df

# deleteme2.py so far is my best bet...


df = clean(r"./trainingT2HAR/har_train.csv")
labels = pd.Series(df['activity'].unique())



breakpoint()
