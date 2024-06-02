import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
import time

import warnings
warnings.filterwarnings('ignore') #Warnings have been disables due to the delim_whitespace parameter being used. This decreases output time and declutters.

def importData(set):
    #Standard import function. header have been written manually.
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
    #making RUL column
    grouped_by_unit = df.groupby(by='unit number') #group the dataset by the different engines.
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

df = importData("train_FD003.txt") #import the data of the third jet engine and get the RUL column for the data
test = importData("test_FD003.txt")
end = pd.read_csv("AML\Data\RUL_FD003.txt", header=None, delim_whitespace=True).to_numpy() #Importing the RUL values for the test set at ended trajectory¢

RUL=get_RUL_column(df)
RUL_test = get_RUL_column_test(test, end)
remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10',
       'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing script in Main.py

df = df[remaining] #constrict the dataframe

# df.drop('unit number', axis='columns', inplace=True) #used for testing purposes

X_train, X_test, y_train, y_test=train_test_split(df, RUL, test_size=0.2, random_state=42) #make a test train split using random state 42 for consistency

print(f"Dimension of feature matrix : {X_train.shape}\n Dimension of target vector: {y_train.shape}") #Check shapes for good measure

param_grid = {'max_depth':range(1, 12, 1),'min_samples_split':range(1, 121, 10)} #parameter grid for the descision tree

#using grid search to find best params 
grid = GridSearchCV(estimator= tree.DecisionTreeRegressor(random_state=42), param_grid=param_grid, cv=None, return_train_score=True, n_jobs=-1)
grid.fit(X_train, y_train)
scores = pd.DataFrame(grid.cv_results_)
print(" Results from {} " .format(grid.__class__))
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_, "\n")

#grid search plot
plt.figure(0)
scores.plot(x='param_max_depth', y='mean_train_score', yerr='std_train_score', ax=plt.gca(), figsize=(20,8))
scores.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score', ax=plt.gca(), figsize=(20,8))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel('max_depth with subsplits of min_samp_split', fontsize=26)
plt.ylabel('Score', fontsize=26)
plt.legend(fontsize=18)
plt.show()

dtr = tree.DecisionTreeRegressor(min_samples_split= 91, max_depth= 9) #set params to the best peforming

dtr.fit(X_train, y_train)

feat_importance = dtr.feature_importances_

#make a bar plot showing feature importance
plt.figure(1)
plt.barh(range(np.size(feat_importance)), feat_importance)
plt.yticks(range(np.size(feat_importance)),df.columns.tolist()[0:np.size(feat_importance)])
plt.title("Feature importance for Decision tree")
plt.show()

print('Cross val score:\n',cross_val_score(dtr, df, RUL, cv=5, n_jobs=-1), "\n") #doing cross val to see how well the tree works with the dataframe



#Evaluation of the optimised methods, by plotting and duration

test = test.loc[:,df.columns]
y_pred = dtr.predict(test)
print("Test scores:")
print("R2 Score: {:.2f}" .format(r2_score(y_pred, RUL_test)))
print("RMS Error: {:.2f}".format(sklearn.metrics.mean_squared_error(y_pred, RUL_test)))
print("Mean Absolute Error: {:.2f}".format(sklearn.metrics.mean_absolute_error(y_pred, RUL_test)))
print("Mean Absolute Percentage Error: {:.2f} \n".format(sklearn.metrics.mean_absolute_percentage_error(y_pred,RUL_test) *100 ))

y_pred_train = dtr.predict(df)
print("Train scores:")
print("R2 Score: {:.2f}" .format(r2_score(y_pred_train, RUL)))
print("RMS Error: {:.2f}".format(sklearn.metrics.mean_squared_error(y_pred_train, RUL)))
print("Mean Absolute Error: {:.2f}".format(sklearn.metrics.mean_absolute_error(y_pred_train, RUL)))
print("Mean Absolute Percentage Error: {:.2f}".format(sklearn.metrics.mean_absolute_percentage_error(y_pred_train,RUL) * 100 ))

plt.figure(2)
start = time.time()
dtr.fit(df, RUL)
predictions_dtr = dtr.predict(test)
end = time.time()
dtr_time = round(end-start,2)
plt.scatter(RUL_test, predictions_dtr, alpha=0.1)
plt.plot(predictions_dtr,predictions_dtr, linestyle='--', color='red')
plt.title(f"Decision Tree model\n Fitted and predicted in {dtr_time} secs")
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.show()
