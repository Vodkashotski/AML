import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import tree

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21"]
    df = pd.read_csv("AML/Data/{}" .format(set), header=None, delim_whitespace=True)
    df.columns = Header
    return df

def get_RUL_column(df):
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

df = importData("train_FD003.txt")
RUL=get_RUL_column(df)
remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10',
       'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing

df = df[remaining]

X_train, X_test, y_train, y_test=train_test_split(df, RUL, test_size=0.2, random_state=42)
feature_range = range(1, X_train.shape[1])
n_estimator_range = [10, 50, 100, 250, 300, 350, 400, 430, 450, 470]
depth_range = range(10, 14)

rf = RandomForestRegressor(random_state=42)
param_grid={'n_estimators': n_estimator_range,
            'max_depth': depth_range,
            'max_features': feature_range
            }
grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)

grid.fit(X_train, y_train)
scores = pd.DataFrame(grid.cv_results_)

print(grid.best_params_)
print(grid.best_score_)