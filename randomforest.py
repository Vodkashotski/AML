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
#constants have been found using an initial correlation matrix
train=train.drop(constants, axis=1)

corr = train.corr()

relation_to_RUL=corr.iloc[:,-1] #correlation to target
print(relation_to_RUL)

to_drop = []
for index, value in relation_to_RUL.items():
    if abs(value) < 0.1:
        to_drop.append(index)

train=train.drop(to_drop, axis=1)

corr=train.corr()

#fig=pyplot.figure(figsize=(10,5))
#ax=fig.add_subplot(111)
#cax = ax.matshow(corr, vmin=-1, vmax=1)
#fig.colorbar(cax)
#ax.set_xticks(np.arange(15))
#ax.set_xticklabels(train.columns, rotation=90, fontsize=15)
#ax.set_yticks(np.arange(15))
#ax.set_yticklabels(train.columns, fontsize=15)
#pyplot.show()

#by examination some seem to have a too close correlation, so we will drop some of them

high_corr_indices = []
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr_indices.append((i, j))
            
print(high_corr_indices)
#code above shows for which values the correlation is too high
too_much_corr = [train.columns[5], train.columns[6], train.columns[12]] #dropped the ones with the lowest correlation to RUL
train=train.drop(too_much_corr, axis=1)

target_set = train.iloc[:,-1]
data_set = train.iloc[:,1:-1]

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test=train_test_split(data_set, target_set, test_size=0.3, random_state=42)

#print(y_train.shape)

rf = RandomForestRegressor(random_state=42)
param_grid={'n_estimators': [200, 225, 250], 'max_depth': [9, 10, 11]}
grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)

grid.fit(X_train, y_train)
scores = pd.DataFrame(grid.cv_results_)

print(grid.best_params_)
print(grid.best_score_)

best_rf = grid.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(test_score)