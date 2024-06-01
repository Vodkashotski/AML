import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
import time
#import winsound

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.neighbors import KNeighborsRegressor

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21"]
    data = pd.read_csv("{}" .format(set), header=None, delim_whitespace=True)
    data.columns = Header
    return data

def get_RUL_column(df):
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

def get_RUL_column_test(df,RUL_np):
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    for i in range(len(max_time)):
        max_time.iloc[i]=max_time.iloc[i]+RUL_np[i]
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

def score_func(y_true, y_pred, setName): 
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    score_list = [round(mae, 2), round(rmse, 2), round(r2, 2), round(mape, 2)]
    # printing metrics
    print('{}\n' .format(setName))
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print(f' Mean Absolute Percentage Error: {score_list[3]}')
    print("<)-------------X-------------(>")

data = importData("Data/train_FD003.txt")
test = importData("Data/test_FD003.txt")
end = pd.read_csv("Data/RUL_FD003.txt", header=None, delim_whitespace=True).to_numpy() #Importing the RUL values for the test set at ended trajectory

RUL = get_RUL_column(data)
RUL_test = get_RUL_column_test(test,end)

remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10', 'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing

train = data[remaining]
test = test[remaining]

new_scaler = MinMaxScaler()

scaled_data = new_scaler.fit_transform(data)
test_scaled = new_scaler.transform(test)

model = KNeighborsRegressor(n_neighbors=37, n_jobs=-1)

model.fit(scaled_data, RUL)

train_predict = model.predict(scaled_data)

test_predict = model.predict(test_scaled)

print(score_func(RUL, train_predict, 'TRAIN SET'))

print(score_func(RUL_test, test_predict, 'TEST SET'))