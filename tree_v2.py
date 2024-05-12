import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def importData(set):
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
    grouped_by_unit = df.groupby(by='unit number') 
    max_time = grouped_by_unit['time, in cycles'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit number',right_index=True)
    RUL = merged["max_time"] - merged['time, in cycles']
    return RUL

df = importData("train_FD003.txt")
RUL=get_RUL_column(df)
remaining = ['time, in cycles', 'sensor measurement 2', 'sensor measurement 3',
       'sensor measurement 4', 'sensor measurement 6', 'sensor measurement 10',
       'sensor measurement 11', 'sensor measurement 12',
       'sensor measurement 17'] #taken from preprocessing

df = df[remaining]
scaler = MinMaxScaler()
X_train, X_test, y_train, y_test=train_test_split(df, RUL, test_size=0.2, random_state=42)
# feature_range = range(1, X_train.shape[1])
# X_train = scaler.fit_transform(X_train)

print(f"Dimension of feature matrix : {X_train.shape}\n Dimension of target vector: {y_train.shape}")

param_grid = {'max_depth':range(1, 12, 1),'min_samples_split':range(1, 150, 5), 'min_samples_leaf':range(1, 50,5)}

rand_search = RandomizedSearchCV(estimator= tree.DecisionTreeClassifier(random_state=42), param_distributions=param_grid, cv=None, return_train_score=True, n_jobs=-1, random_state=42)
rand_search.fit(X_train, y_train)

scores = pd.DataFrame(rand_search.cv_results_)
print(" Results from {} " .format(rand_search.__class__))
print("\n The best estimator across ALL searched params:\n",rand_search.best_estimator_)
print("\n The best score across ALL searched params:\n",rand_search.best_score_)
print("\n The best parameters across ALL searched params:\n",rand_search.best_params_)

clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

feat_importance = clf.feature_importances_

plt.barh(range(np.size(feat_importance)), feat_importance)
plt.yticks(range(np.size(feat_importance)),df.columns.tolist()[0:np.size(feat_importance)])

# plt.show()

print('Cross val score:\n',cross_val_score(clf, X_test, y_test, cv=5, n_jobs=-1))

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
# print("Accuracy:",accuracy_score(y_test, y_pred))


# print("Test score: {:.2f}" .format(clf.score(X_test, y_test)))
# print("Train score: {:.2f}".format(clf.score(X_train, y_train)))


