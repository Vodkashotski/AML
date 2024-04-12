import pandas as pd
import numpy as np
from matplotlib import pyplot

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21","sensor measurement 22","sensor measurement 23"]
    df = pd.read_csv("Data/{}" .format(set), header=None, delimiter=" ")
    df.columns = Header
    return df

def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit number') 
    max_time = train_grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    merged["RUL"] = merged["max_time"] - merged['time, in cycles']
    merged = merged.drop("max_time", axis=1)
    return merged

df = importData("train_FD003.txt")
train=add_RUL_column(df)
constants=['operational setting 3', 'sensor measurement 1', 'sensor measurement 5', 'sensor measurement 16', 'sensor measurement 18', 'sensor measurement 19', 'sensor measurement 22', 'sensor measurement 23']
train=train.drop(constants, axis=1)

corr = train.corr()
fig=pyplot.figure(figsize=(10,5))
ax=fig.add_subplot(111)
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(np.arange(21))
ax.set_xticklabels(train.columns, rotation=90, fontsize=15)
ax.set_yticks(np.arange(21))
ax.set_yticklabels(train.columns, fontsize=15)
pyplot.show()

print(corr.iloc[:,-1])



