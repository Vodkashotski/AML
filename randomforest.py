import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score


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
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10',
        'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing

train = df[remaining]
test = test[remaining]

#defining paramters for grid search
feature_range = np.arange(1, train.shape[1]+1)
n_estimator_range = [10, 50, 100, 250, 350, 400, 430, 450, 470]
depth_range = np.arange(1, 15)
rf = RandomForestRegressor(random_state=42)

param_grid={'n_estimators': n_estimator_range,
           'max_depth': depth_range,
           'max_features': feature_range
           }
grid = GridSearchCV(rf, param_grid, cv=5) #grid search

grid.fit(train, RUL)
scores = grid.cv_results_

print(grid.best_params_)
print(grid.best_score_)

#Finding scores for plots/narrowing

# scores_cross = []
# std_cross = []
# scores_train =[]
# std_train = []
# scores_test = []

# for j in depth_range:
#             rf = RandomForestRegressor(max_depth=j, random_state=42)
#             rf.fit(train,RUL)
#             score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#             scores_cross.append(np.array(score['test_score']).mean())
#             std_cross.append(np.array(score['test_score']).std())
#             scores_train.append(np.array(score['train_score']).mean())
#             std_train.append(np.array(score['train_score']).std())

#             predictions = rf.predict(test)
#             scores_test.append(r2_score(RUL_test,predictions))


# # Saving the data to avoid having to run the loop every time
# np.save("cross_val_scores_d.npy", scores_cross)
# np.save("cross_val_std_d.npy", std_cross)
# np.save("train_std_d.npy", std_train)
# np.save("scores_test_d.npy", scores_test)
# np.save("scores_train_d.npy", scores_train)

# scores_cross = []
# std_cross = []
# scores_train =[]
# std_train = []
# scores_test = []

# for j in feature_range:
#             rf = RandomForestRegressor(n_estimators=490, max_features=j, max_depth=13, random_state=42)
#             rf.fit(train,RUL)
#             score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#             scores_cross.append(np.array(score['test_score']).mean())
#             std_cross.append(np.array(score['test_score']).std())
#             scores_train.append(np.array(score['train_score']).mean())
#             std_train.append(np.array(score['train_score']).std())

#             predictions = rf.predict(test)
#             scores_test.append(r2_score(RUL_test,predictions))


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
# scores_test = []

# for j in  n_estimator_range:
#            rf = RandomForestRegressor(n_estimators=j, random_state=42)
#            rf.fit(train,RUL)
#            score = cross_validate(rf, train, RUL, cv=5, scoring='r2', return_train_score=True)
#            scores_cross.append(np.array(score['test_score']).mean())
#            std_cross.append(np.array(score['test_score']).std())
#            scores_train.append(np.array(score['train_score']).mean())
#            std_train.append(np.array(score['train_score']).std())
           
#            predictions = rf.predict(test)
#            scores_test.append(r2_score(RUL_test,predictions))
          

# np.save("cross_val_scores_n.npy", scores_cross)
# np.save("cross_val_std_n.npy", std_cross)
# np.save("train_std_n.npy", std_train)
# np.save("scores_test_n.npy", scores_test)
# np.save("scores_train_n.npy", scores_train)

#X_train, X_test, y_train, y_test=train_test_split(train, RUL, test_size=0.2, random_state=42)
#rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train,y_train)
# score = rf.score(X_train, y_train)
# predictions = rf.predict(X_test)
# print("Untuned train score", score.mean())
# print("Untuned validation score",r2_score(y_test, predictions))
# predictions = rf.predict(test)
# print("Untuned test set score",r2_score(RUL_test, predictions))


# rf = RandomForestRegressor(n_estimators=450, max_features=1, max_depth=13, random_state=42)
# rf.fit(X_train,y_train)
# score = rf.score(X_train, y_train)
# predictions = rf.predict(X_test)
# print("Tuned train score", score.mean())
# print("Tuned validation score",r2_score(y_test, predictions))
# predictions = rf.predict(test)
# print("Tuned test set score",r2_score(RUL_test, predictions))

# #loading data for plotting
# score_cross = np.load("cross_val_scores.npy", allow_pickle=True)
# std_cross = np.load("cross_val_std.npy", allow_pickle=True)
# std_train = np.load("train_std.npy", allow_pickle=True)
# scores_test = np.load("scores_test.npy", allow_pickle=True)
# scores_train = np.load("scores_train.npy", allow_pickle=True)

# score_cross_n = np.load("cross_val_scores_n.npy", allow_pickle=True)
# std_cross_n = np.load("cross_val_std_n.npy", allow_pickle=True)
# std_train_n = np.load("train_std_n.npy", allow_pickle=True)
# scores_test_n = np.load("scores_test_n.npy", allow_pickle=True)
# scores_train_n = np.load("scores_train_n.npy", allow_pickle=True)

# score_cross_d = np.load("cross_val_scores_d.npy", allow_pickle=True)
# std_cross_d = np.load("cross_val_std_d.npy", allow_pickle=True)
# std_train_d = np.load("train_std_d.npy", allow_pickle=True)
# scores_test_d = np.load("scores_test_d.npy", allow_pickle=True)
# scores_train_d = np.load("scores_train_d.npy", allow_pickle=True)

#plotting
# fig, ax = plt.subplots(1,2)
# ax[0].errorbar(feature_range, score_cross, yerr=std_cross, label = 'Cross validation score')
# ax[0].errorbar(feature_range, scores_train, yerr=std_train, label = 'Train score')
# ax[0].set_xlabel('Max Features')
# ax[0].set_ylabel('R2 score')

# ax[1].errorbar(n_estimator_range, score_cross_n, yerr=std_cross_n, label = 'Cross validation score')
# ax[1].errorbar(n_estimator_range, scores_train_n, yerr=std_train_n, label = 'Train score')
# ax[1].set_xlabel('N estimators')
# ax[1].set_ylabel('R2 score')

# ax[2].errorbar(depth_range, score_cross_d, yerr=std_cross_d, label = 'Cross validation score')
# ax[2].errorbar(depth_range, scores_train_d, yerr=std_train_d, label = 'Train score')
# ax[2].set_xlabel('Max_depth')
# ax[2].set_ylabel('R2 score')

# plt.legend()
# plt.show()