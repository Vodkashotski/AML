import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
#from sklearn.metrics import root_mean_squared_error

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21"]
    df = pd.read_csv("Data/{}" .format(set), header=None, delim_whitespace=True)
    df.columns = Header
    return df

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


df = importData("train_FD003.txt")
test = importData("test_FD003.txt")
end = pd.read_csv("Data/RUL_FD003.txt", header=None, delim_whitespace=True).to_numpy() #Importing the RUL values for the test set at ended trajectory¢

RUL = get_RUL_column(df)
RUL_test = get_RUL_column_test(test,end)

remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10', 'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing

train = df[remaining]
test = test[remaining]

#defining paramters for grid search
feature_range = np.arange(1, train.shape[1]+1)
n_estimator_range = [10, 50, 100, 250, 350, 450, 470, 490, 510, 540, 580]
depth_range = np.arange(1, 15)

#rf = RandomForestRegressor(random_state=42)

#param_grid={'n_estimators': n_estimator_range,
            #'max_depth': depth_range,
            #'max_features': feature_range,
#             }
# grid = GridSearchCV(rf, param_grid, cv=5) #grid search

# grid.fit(train, RUL)
# scores = grid.cv_results_

# print(grid.best_params_)
# print(grid.best_score_)

#Finding scores for plots/narrowing

# scores_cross = []
# std_cross = []
# scores_train =[]
# std_train = []

# for j in depth_range:
#             rf = RandomForestRegressor(max_depth=j, n_estimators=490, max_features=1, random_state=42)
#             rf.fit(train,RUL)
#             score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#             scores_cross.append(np.array(score['test_score']).mean())
#             std_cross.append(np.array(score['test_score']).std())
#             scores_train.append(np.array(score['train_score']).mean())
#             std_train.append(np.array(score['train_score']).std())


# Saving the data to avoid having to run the loop every time
# np.save("cross_val_scores_d.npy", scores_cross)
# np.save("cross_val_std_d.npy", std_cross)
# np.save("train_std_d.npy", std_train)
# np.save("scores_train_d.npy", scores_train)

# scores_cross = []
# std_cross = []
# scores_train =[]
# std_train = []

# for j in feature_range:
#             rf = RandomForestRegressor(n_estimators=490, max_features=j, max_depth=13, random_state=42)
#             rf.fit(train,RUL)
#             score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#             scores_cross.append(np.array(score['test_score']).mean())
#             std_cross.append(np.array(score['test_score']).std())
#             scores_train.append(np.array(score['train_score']).mean())
#             std_train.append(np.array(score['train_score']).std())

# #Saving the data to avoid having to run the loop every time
# np.save("cross_val_scores.npy", scores_cross)
# np.save("cross_val_std.npy", std_cross)
# np.save("train_std.npy", std_train)
# np.save("scores_test.npy", scores_test)
# np.save("scores_train.npy", scores_train)

# scores_cross = []
# std_cross = []
# scores_train =[]
# std_train = []

# for j in  n_estimator_range:
#            rf = RandomForestRegressor(n_estimators=j, max_depth=13, max_features=1, random_state=42)
#            rf.fit(train,RUL)
#            score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#            scores_cross.append(np.array(score['test_score']).mean())
#            std_cross.append(np.array(score['test_score']).std())
#            scores_train.append(np.array(score['train_score']).mean())
#            std_train.append(np.array(score['train_score']).std())
          

# np.save("cross_val_scores_n.npy", scores_cross)
# np.save("cross_val_std_n.npy", std_cross)
# np.save("train_std_n.npy", std_train)
# np.save("scores_train_n.npy", scores_train)

# X_train, X_test, y_train, y_test=train_test_split(train, RUL, test_size=0.2, random_state=42)
# rf = RandomForestRegressor(random_state=42)
# rf.fit(train,RUL)
# score = rf.score(train, RUL)
# print("Untuned train score", score.mean())
# predictions = rf.predict(test)
# print("Untuned test set score",r2_score(RUL_test, predictions))


# rf = RandomForestRegressor(n_estimators=490, max_features=1, max_depth=13, random_state=42)
# rf.fit(train,RUL)
# predictions_train = rf.predict(train)
# predictions_test = rf.predict(test)
# r2_train = r2_score(RUL, predictions_train)
# RSME_train = root_mean_squared_error(RUL, predictions_train)

# r2_test = r2_score(RUL_test, predictions_test)
# RSME_test = root_mean_squared_error(RUL_test, predictions_test)

# print("Tuned train R2 score: ", r2_train)
# print("Tuned train RSME score: ", RSME_train)

# print("Tuned test R2 score: ",r2_test)
# print("Tuned test RSME score: ", RSME_test)

# #loading data for plotting
score_cross = np.load("cross_val_scores.npy", allow_pickle=True)
std_cross = np.load("cross_val_std.npy", allow_pickle=True)
std_train = np.load("train_std.npy", allow_pickle=True)
scores_train = np.load("scores_train.npy", allow_pickle=True)

score_cross_n = np.load("cross_val_scores_n.npy", allow_pickle=True)
std_cross_n = np.load("cross_val_std_n.npy", allow_pickle=True)
std_train_n = np.load("train_std_n.npy", allow_pickle=True)
scores_train_n = np.load("scores_train_n.npy", allow_pickle=True)

score_cross_d = np.load("cross_val_scores_d.npy", allow_pickle=True)
std_cross_d = np.load("cross_val_std_d.npy", allow_pickle=True)
std_train_d = np.load("train_std_d.npy", allow_pickle=True)
scores_train_d = np.load("scores_train_d.npy", allow_pickle=True)

#plotting
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].errorbar(feature_range, score_cross, yerr=std_cross, label = 'Cross validation score')
ax[0].errorbar(feature_range, scores_train, yerr=std_train, label = 'Train score')
ax[0].set_xticks(feature_range)
ax[0].set_xticklabels(feature_range, fontsize=12)
ax[0].set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
ax[0].set_yticklabels([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],fontsize=12)
ax[0].set_xlabel('Max Features',fontsize=14)
ax[0].set_ylabel('R2 score',fontsize=14)

ax[1].errorbar(n_estimator_range, score_cross_n, yerr=std_cross_n, label = 'Cross validation score')
ax[1].errorbar(n_estimator_range, scores_train_n, yerr=std_train_n, label = 'Train score')
ax[1].set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
ax[1].set_yticklabels([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],fontsize=12)
ax[1].set_xticks([0, 100, 200, 300, 400, 500, 600])
ax[1].set_xticklabels([0, 100, 200, 300, 400, 500, 600], fontsize=12)
ax[1].set_xlabel('N estimators',fontsize=14)
ax[1].set_ylabel('R2 score',fontsize=14)

ax[2].errorbar(depth_range, score_cross_d, yerr=std_cross_d, label = 'Cross validation score')
ax[2].errorbar(depth_range, scores_train_d, yerr=std_train_d, label = 'Train score')
ax[2].set_xticks([2, 4, 6, 8,10, 12 ,14])
ax[2].set_xticklabels([2, 4, 6, 8,10, 12 ,14], fontsize=12)
ax[2].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax[2].set_yticklabels([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],fontsize=12)
ax[2].set_xlabel('Max depth', fontsize=14)
ax[2].set_ylabel('R2 score', fontsize=14)

plt.legend(fontsize=12, loc=4)
fig.suptitle('Optimisation parameters for Random Forest Regresssion', fontsize=16)
plt.tight_layout()
plt.show()