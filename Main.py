

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import time


from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.svm import SVR
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
    data = pd.read_csv("Data\{}" .format(set), header=None, delim_whitespace=True)
    data.columns = Header
    return data

def get_RUL_column(df): #function which makes a RUL column based on the time variate series
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

def get_RUL_column_test(df,RUL_np): #function which makes a RUL column based on the time variate series and remaining time
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    for i in range(len(max_time)):
        max_time.iloc[i]=max_time.iloc[i]+RUL_np[i]
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL


data = importData("train_FD003.txt") #importing the main data
test = importData("test_FD003.txt")  #importing the other data
end = pd.read_csv("Data\RUL_FD003.txt", header=None, delim_whitespace=True).to_numpy() #Importing the RUL values for the test set at ended trajectory¢

RUL = get_RUL_column(data) #making the RL columns
RUL_test = get_RUL_column_test(test,end)

data = data.drop(["unit number"], axis=1) #Effectively just a name so can't enter into the regression

dummy_set1 = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making dummy set so RUL can be in the corr matrix
correlation1 = dummy_set1.corr() #correlation matrix
labels1 =["Time","Setting 1", "Setting 2", "Setting 3", "Sensor 1","Sensor 2", "Sensor 3", "Sensor 4","Sensor 5", "Sensor 6","Sensor 7","Sensor 8","Sensor 9", "Sensor 10", "Sensor 11", "Sensor 12", "Sensor 13", "Sensor 14", "Sensor 15", "Sensor 16", "Sensor 17", "Sensor 18", "Sensor 19", "Sensor 20",  "Sensor 21", "RUL"]
plt.figure(figsize=(10,6))
sns.heatmap(correlation1,xticklabels=labels1, yticklabels=labels1)
plt.show()

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data) #scaling the data to use for variance

var_thresh = VarianceThreshold(threshold=0.01)
var_thresh.fit(data_scaled)
#at threshold 0.01 removes: operational setting 3, sensor measurement 1,5,8,9,13,14,16,18,19

data = data.loc[:, var_thresh.get_support()] #removing the columns with low variance for both the unscaled and scaled set

relation_to_RUL=correlation1.iloc[:,-1] #correlation to target

to_drop = []
for index, value in relation_to_RUL.items(): #loop which finds the columns with low correlation to the target
    if abs(value) < 0.1:
        to_drop.append(index)
data=data.drop(to_drop, axis=1) #removing the columns with low correlation to the target

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix

high_corr_indices = [] #finding the columns with high correlation to each other
for i in range(len(correlation)):
    for j in range(i+1, len(correlation)):
        if abs(correlation.iloc[i, j]) > 0.9:
            high_corr_indices.append((i, j))
            
print(high_corr_indices) #inspecting to find out which ones to remove

data=data.drop("sensor measurement 7", axis=1) #removing the columns with low correlation to the target
new_scaler = MinMaxScaler()

scaled_data = new_scaler.fit_transform(data)

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
labels =["Time", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 6", "Sensor 10", "Sensor 11", "Sensor 12", "Sensor 17", "RUL"]

fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.set(font_scale=0.9)
sns.heatmap(correlation1, xticklabels=labels1, yticklabels=labels1, ax=ax[0])
ax[0].set_title("Full data set", fontsize=14)

sns.heatmap(correlation, annot=True, xticklabels=labels, yticklabels=labels, ax=ax[1])
ax[1].set_title("Tailored data set", fontsize=14)
plt.tight_layout()
plt.show()

test = test.loc[:,data.columns]
test_scaled = new_scaler.transform(test)

print(data.columns)#the columns left that can be copied into tuning codes


#Evaluation of the optimised methods, by plotting and duration
rf=RandomForestRegressor(n_estimators=490, max_features=1, max_depth=13, random_state=42)
start = time.time()
rf.fit(data, RUL)
predictions_rf = rf.predict(test)
end = time.time()
rf_time = round(end-start,2)

svr = SVR(kernel='rbf', C=1000000, gamma="scale", epsilon=50)
start = time.time()
svr.fit(data, RUL)
predictions_svr = svr.predict(test)
end = time.time()
svr_time = round(end-start,2)

clf = tree.DecisionTreeRegressor(min_samples_split= 21, max_depth= 10) #set params to the best peforming
start = time.time()
clf.fit(data, RUL)
predictions_clf = clf.predict(test)
end = time.time()
clf_time = round(end-start,2)

kn = KNeighborsRegressor(n_neighbors=37)
start = time.time()
kn.fit(data, RUL)
predictions_kn = kn.predict(test)
end = time.time()
kn_time = round(end-start,2)


fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[1,1].scatter(RUL_test, predictions_rf, alpha=0.1)
ax[1,1].plot(predictions_rf,predictions_rf, linestyle='--', color='red')
ax[1,1].set_title(f"RF model\n Fitted and predicted in {rf_time} secs")
ax[1,1].set_xlabel('Actual RUL')
ax[1,1].set_ylabel('Predicted RUL')

ax[1,0].scatter(RUL_test, predictions_clf, alpha=0.1)
ax[1,0].plot(predictions_clf,predictions_clf, linestyle='--', color='red')
ax[1,0].set_title(f"Decision tree model\n Fitted and predicted in {clf_time} secs")
ax[1,0].set_xlabel('Actual RUL')
ax[1,0].set_ylabel('Predicted RUL')

ax[0,1].scatter(RUL_test, predictions_svr, alpha=0.1)
ax[0,1].plot(predictions_svr,predictions_svr, linestyle='--', color='red')
ax[0,1].set_title(f"SVR model\n Fitted and predicted in {svr_time} secs")
ax[0,1].set_xlabel('Actual RUL')
ax[0,1].set_ylabel('Predicted RUL')

ax[0,0].scatter(RUL_test, predictions_kn, alpha=0.1)
ax[0,0].plot(predictions_kn,predictions_kn, linestyle='--', color='red')
ax[0,0].set_title(f"Linear K-neighbours model\n Fitted and predicted in {kn_time} secs")
ax[0,0].set_xlabel('Actual RUL')
ax[0,0].set_ylabel('Predicted RUL')

plt.tight_layout()
plt.show()
