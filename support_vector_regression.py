import pandas as pd
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt
import time
import winsound

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVR

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def score_func_and_save(y_true, y_pred, setName): 
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    score_list = [round(mae, 2), round(rmse, 2), round(r2, 2), round(mape, 2)]
    # printing metrics
    print('{}\n' .format(setName))
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print(f' Mean Absolute Percentage Error: {score_list[3]}')
    print("<)-------------X-------------(>")
    print("Scores written to SVR_data/svr_scores_{}.txt".format(setName))

    with open('SVR_data/svr_scores_{}.txt'.format(setName),'w') as f:
        f.write('{}\n' .format(setName))
        f.write(f'Mean Absolute Error (MAE): {round(mae, 2)}\n')
        f.write(f'Root Mean Squared Error (RMSE): {round(rmse, 2)}\n')
        f.write(f'R2 Score: {round(r2, 2)}\n')
        f.write(f'Mean Absolute Percentage Error: {round(mape, 2)}\n')
        f.write("<)-------------X-------------(>")


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

def plot_heatmap(cv_results):
    # Filter the DataFrame based on the gamma type
    auto = cv_results[cv_results['param_gamma'] == 'auto']
    scale = cv_results[cv_results['param_gamma'] == 'scale']

    # Pivot the DataFrame
    pivot_auto = auto.pivot('param_C', 'param_epsilon', 'mean_test_score')
    pivot_scale = scale.pivot('param_C', 'param_epsilon', 'mean_test_score')

    # Create the heatmap
    fig, axs = plt.subplots(1, 2,figsize=(20, 10))
        
    # Plot the first heatmap
    sns.heatmap(pivot_auto, annot=True, fmt=".3f", linewidths=.5, ax=axs[0], cmap='viridis')
    axs[0].set_title(f'$\gamma$ = "auto"')
    axs[0].set_xlabel(r'$\epsilon$-value')
    axs[0].set_ylabel('C')

    # Plot the second heatmap
    sns.heatmap(pivot_scale, annot=True, fmt=".3f", linewidths=.5, ax=axs[1], cmap='viridis')
    axs[1].set_title(f'$\gamma$ = "scale"')
    axs[1].set_xlabel(r'$\epsilon$-value')
    axs[1].set_ylabel('C')

    # Adjust layout to prevent overlap
    fig.suptitle(f'GridSearchCV mean $R^2$ score')
    plt.tight_layout()
    plt.show()


data = importData("Data/train_FD003.txt")
test = importData("Data/test_FD003.txt")
end = pd.read_csv("Data/RUL_FD003.txt", header=None, delim_whitespace=True).to_numpy() #Importing the RUL values for the test set at ended trajectory

RUL = get_RUL_column(data)
RUL_test = get_RUL_column_test(test,end)

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
new_scaler = MinMaxScaler()

scaled_data = new_scaler.fit_transform(data)

dummy_set = data.merge(RUL.to_frame(name='RUL'), left_on=data.columns[0],right_index=True) #making set so RUL can be in the corr matrix
correlation = dummy_set.corr() #correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.show()

test = test.loc[:,data.columns]
test_scaled = new_scaler.transform(test)

print(data.columns)#the columns left that can be copied into tuning codes

# Splitting train data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(data, RUL, test_size=0.25, random_state=42)

# Defining the model using typical parameters.
#model = SVR(kernel='rbf', C=1000, gamma="scale", epsilon=0.5)

# Defining model using optimized hyperparameters
model = SVR(kernel='rbf', C=1000000, gamma="scale", epsilon=50)

print("Training the model\n")

model.fit(X_train, y_train)

# Using model to predict RUL
train_predict = model.predict(X_train)

#val_predict = model.predict(X_val)

test_predict = model.predict(test)

print("Target predicted\n")

print(score_func_and_save(y_train, train_predict, 'Training scores'))

print(score_func_and_save(RUL_test, test_predict, 'Test scores'))

## Performing GridSearchCV on SVR model
#param_grid = {
#    'C': [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000],
#    'epsilon': [0.1, 0.5, 1, 5, 10, 50, 100],
#    'kernel': ['rbf'],
#    'gamma': ['scale', 'auto']
#}
#svr = SVR()
#
#print("Performing GridSearchCV")
#
#grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='r2', n_jobs=14, verbose=2)
#
#grid_search.fit(X_train, y_train)
#
#print("GridSearchCV finished")
#
#best_model = grid_search.best_estimator_
#
#gscvTrain_predict = best_model.predict(X_train)
#
#gscvVal_predict = best_model.predict(X_val)
#
#gscvTest_predict = best_model.predict(test)

#print("Grid search training score")
#print(score_func(y_train, gscvTrain_predict))
#
#
#print("Grid search test score")
#print(score_func(RUL_test, gscvTest_predict))

#Notification sound for finished GridSearchCV
#winsound.Beep(2000, 1000)

# Saves the best hyperparameters of the GridSearchCV to a .txt file.
#best_params = grid_search.best_params_
#file_path = os.path.join('SVR_data', 'best_params.txt')
#with open(file_path, 'w') as f:
#    for param, value in best_params.items():
#        f.write(f"{param}: {value}\n")


# Saves the results of the GridSearchCV to a csv file.
#results = pd.DataFrame(grid_search.cv_results_)
#results = results.sort_values("rank_test_score")
#results = results.to_csv("SVR_data/cv_results.csv", sep=',', index=False)

cv_results = pd.read_csv("SVR_data/cv_results.csv")
plot_heatmap(cv_results)