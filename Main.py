

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV



pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21","sensor measurement 22","sensor measurement 23"]
    df = pd.read_csv("AML\Data\{}" .format(set), header=None, delimiter=" ")
    df.columns = Header
    return df

def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit') 
    max_time = train_grouped_by_unit['time'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit',right_index=True)
    merged["RUL"] = merged["max_time"] - merged['time']
    merged = merged.drop("max_time", axis=1)
    return merged

df = importData("train_FD003.txt")
df.drop(["sensor measurement 22","sensor measurement 23"], axis='columns', inplace=True)

# print(df.describe().T)
plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap = sns.diverging_palette(230, 20, as_cmap=True),
            cbar_kws={'label': 'Missing Data'})

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

df.drop(["operational setting 1", "operational setting 2", "operational setting 3", "sensor measurement 1", "sensor measurement 5", "sensor measurement 16", "sensor measurement 18", "sensor measurement 19", "sensor measurement 21"],
         axis='columns', inplace=True)
# target = pd.read_csv("AML\Data\RUL_FD003.txt")








