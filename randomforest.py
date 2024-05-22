import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
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

X_train, X_test, y_train, y_test=train_test_split(train, RUL, test_size=0.2, random_state=42)
feature_range = np.arange(1, X_train.shape[1]+1)
n_estimator_range = [10, 50, 100, 250, 350, 400, 450, 470, 480, 490, 500, 550]
#n_estimator_range = [400, 450, 470, 480, 490, 500, 550]
depth_range = np.arange(10, 14)

scores_cross = []
std_cross = []
scores_train =[]
scores_valid = []
scores_test = []

#for j in feature_range:
#            rf = RandomForestRegressor(n_estimators=490, max_features=j, max_depth=13, random_state=42)
#            rf.fit(X_train,y_train)
#            score = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
#            scores_cross.append(score.mean())
#            std_cross.append(score.std())
#            scores_train.append(rf.score(X_train, y_train))
#            
#            predictions = rf.predict(X_test)
#            scores_valid.append(r2_score(y_test, predictions))
#            
#            predictions2 = rf.predict(test)
#            scores_test.append(r2_score(RUL_test,predictions2))

#Saving the data to avoid having to run the loop every time
#np.save("cross_val_scores.npy", scores_train)
#np.save("cross_val_std.npy", std_cross)
#np.save("scores_valid.npy", scores_valid)
#np.save("scores_test.npy", scores_test)
#np.save("scores_train.npy", scores_train)

#Loading the saved data
#scores_cross = np.load("cross_val_scores.npy")
#std_cross = np.load("cross_val_std.npy")
#scores_train = np.load("scores_train.npy")
#scores_test = np.load("scores_test.npy")

#plt.errorbar(x=feature_range,y=scores_cross, yerr=std_cross, label="Cross-validation")
#plt.plot(feature_range,  scores_train, label="Traning set")
#plt.plot(feature_range, scores_test, label="Validation set")
#plt.xlabel("Max features")
#plt.ylabel("R2 score")
#plt.legend()
#plt.show()

for j in  n_estimator_range:
            rf = RandomForestRegressor(n_estimators=j, max_features=7, max_depth=13, random_state=42)
            rf.fit(train,RUL)
            score = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
            scores_cross.append(score.mean())
            std_cross.append(score.std())
            scores_train.append(rf.score(X_train, y_train))
            
            predictions = rf.predict(X_test)
            scores_valid.append(r2_score(y_test, predictions))
            predictions2 = rf.predict(test)
            scores_test.append(r2_score(RUL_test,predictions2))
          

np.save("cross_val_scores_n.npy", scores_train)
np.save("cross_val_std_n.npy", std_cross)
np.save("scores_test_n.npy", scores_test)
np.save("scores_train_n.npy", scores_train)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)
score = rf.score(X_train, y_train)
predictions = rf.predict(X_test)
print("Untuned train score", score.mean())
print("Untuned validation score",r2_score(y_test, predictions))
predictions = rf.predict(test)
print("Untuned test set score",r2_score(RUL_test, predictions))

#param_grid={'n_estimators': n_estimator_range,
#            'max_depth': depth_range,
#            'max_features': feature_range
#            }
#grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)

#grid.fit(X_train, y_train)
#scores = grid.cv_results_

#print(grid.best_params_)
#print(grid.best_score_)

rf = RandomForestRegressor(n_estimators=490, max_depth=13, max_features=7, random_state=42)
rf.fit(X_train,y_train)
score = rf.score(X_train, y_train)
predictions = rf.predict(X_test)
print("Tuned train score", score.mean())
print("Tuned validation score",r2_score(y_test, predictions))
predictions = rf.predict(test)
print("Tuned test set score",r2_score(RUL_test, predictions))