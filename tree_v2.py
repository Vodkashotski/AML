from matplotlib.pylab import rand
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree


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

df = importData("train_FD003.txt") #import the data of the third jet engine and get the RUL column for the data
RUL=get_RUL_column(df)
remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10',
       'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing script in Main.py

df = df[remaining] #constrict the dataframe
X_train, X_test, y_train, y_test=train_test_split(df, RUL, test_size=0.2, random_state=42) #make a test train split using random state 42 for consistency

print(f"Dimension of feature matrix : {X_train.shape}\n Dimension of target vector: {y_train.shape}") #Check shapes for good measure

param_grid = {'max_depth':range(1, 12, 1),'min_samples_split':range(1, 121, 10)} #parameter grid for the descision tree

#using grid search to find best params 
grid = GridSearchCV(estimator= tree.DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=None, return_train_score=True, n_jobs=-1)
grid.fit(X_train, y_train)
scores = pd.DataFrame(grid.cv_results_)
print(" Results from {} " .format(grid.__class__))
print("\n The best estimator across ALL searched params:\n",grid.best_estimator_)
print("\n The best score across ALL searched params:\n",grid.best_score_)
print("\n The best parameters across ALL searched params:\n",grid.best_params_)

scores.plot(x='param_max_depth', y='mean_train_score', yerr='std_train_score', ax=plt.gca(), figsize=(20,8))
scores.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score', ax=plt.gca(), figsize=(20,8))
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel('max_depth with subsplits of min_samp_split', fontsize=26)
plt.legend(fontsize=18)

plt.show()

clf = tree.DecisionTreeClassifier(min_samples_split= 21, max_depth= 10) #set params to the best peforming

clf.fit(X_train, y_train)

feat_importance = clf.feature_importances_

#make a bar plot showing feature importance
plt.barh(range(np.size(feat_importance)), feat_importance)
plt.yticks(range(np.size(feat_importance)),df.columns.tolist()[0:np.size(feat_importance)])

# plt.show()

print('Cross val score:\n',cross_val_score(clf, df, RUL, cv=5, n_jobs=-1)) #doing cross val to see how well the tree works with the dataframe

print("Test score: {:.3f}" .format(clf.score(X_test, y_test))) 
print("Train score: {:.3f}".format(clf.score(X_train, y_train)))


