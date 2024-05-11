

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def score_func(y_true,y_pred):
    """
    !might be flawed
    model evaluation function
    
    Args:
        y_true = true target RUL value
        y_pred = predicted target RUL value
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    score_list = [round(mae, 2), round(rmse, 2), round(r2, 2)]
    # printing metrics
    print("Classification Report:\n", report)
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print("<)-------------X-------------(>")

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21"]
    data = pd.read_csv("AML/Data/{}" .format(set), header=None, delim_whitespace=True)
    data.columns = Header
    return data

def get_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit number') 
    max_time = train_grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

data = importData("train_FD003.txt")
RUL = get_RUL_column(data)

data = data.drop(["unit number"], axis=1) #Effectively just a name so can't enter into the regression

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.show()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data) #scaling the data to use for variance

var_thresh = VarianceThreshold(threshold=0.01)
var_thresh.fit(data_scaled)
#at threshold 0.01 removes: operational setting 3, sensor measurement 1,5,8,9,13,14,16,18,19

data = data.loc[:, var_thresh.get_support()] #removing the columns with low variance for both the unscaled and scaled set

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.show()

relation_to_RUL=correlation.iloc[:,-1] #correlation to target

to_drop = []
for index, value in relation_to_RUL.items(): #loop which finds the columns with low correlation to the target
    if abs(value) < 0.1:
        to_drop.append(index)
data=data.drop(to_drop, axis=1) #removing the columns with low correlation to the target

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.show()

high_corr_indices = [] #finding the columns with high correlation to each other
for i in range(len(correlation)):
    for j in range(i+1, len(correlation)):
        if abs(correlation.iloc[i, j]) > 0.9:
            high_corr_indices.append((i, j))
            
print(high_corr_indices) #inspecting to find out which ones to remove
data=data.drop("sensor measurement 7", axis=1) #removing the columns with low correlation to the target


dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.show()
print(data.columns)#the columns left that can be copied into tuning codes



