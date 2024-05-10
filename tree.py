
from textwrap import fill
from matplotlib.pylab import RandomState
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import sklearn.preprocessing

#Display option for outputting big datasets
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


def transform_df(file_path, returndf=False):
    #!This function generelises the data processing done in main.py, the specified columns in coloumns_to_drop is
    #!      determined by prior data analysis
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.drop([26,27], axis=1)
    column_names = ["unit_number","time_in_cycles","setting1","setting2","setting3",
                "s01","s02","s03","s04","s05","s06","s07","s08","s09","s10","s11","s12","s13",
                "s14","s15","s16","s17","s18","s19","s20","s21"]

    df.columns = column_names
    # Drop specified columns
    columns_to_drop = ["s19", "s18", "s16", "s05", "s01", "setting3",
                       "setting1", "setting2", "s21",'s07', 's15', 's12', 's10']
    df = df.drop(columns=columns_to_drop, axis=1)

    # Making an array which contains EOL of all the Ids
    EOL = []
    for i in df['unit_number']:
        EOL.append(((df[df['unit_number'] == i]['time_in_cycles']).values)[-1])

    df["EOL"] = EOL

    # Calculate "LR"
    df["LR"] = df["time_in_cycles"].div(df["EOL"])

    # Create 'label' column
    bins = [0, 0.6, 0.8, np.inf]
    labels = [0, 1, 2]
    df['label'] = pd.cut(df['LR'], bins=bins, labels=labels, right=False)

    # Drop unnecessary columns
    df.drop(columns=['unit_number', 'EOL', 'LR'], inplace=True)

    X_train = df.drop(["label"], axis=1).values
    y_train = df["label"] 

    scaler = sklearn.preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)

    if returndf == True:
        return df, X_train, y_train

    return X_train, y_train

trainset_path = r"C:\Users\Simon\Documents\Code projects\AML\Data\train_FD003.txt"
df, X_train, y_train = transform_df(trainset_path, returndf= True)
# print(f"Dimension of feature matrix : {X_train.shape}\ndimension of target vector: {y_train.shape}")

testset_path =r"C:\Users\Simon\Documents\Code projects\AML\Data\test_FD003.txt"
df_test, X_test, y_test = transform_df(testset_path, returndf=True)

'''
# !Grid search CV code to find the best parameters. Code takes a while to run so it is commented out.

print(f"Dimension of feature matrix : {X_train.shape}\n Dimension of target vector: {y_train.shape}")

param_grid = {'max_depth':range(1, 12, 1),'min_samples_split':range(1, 150, 5), 'min_samples_leaf':range(1, 50,5)}

# grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid=param_grid, cv=None, return_train_score=True)
# grid.fit(X_train, y_train)

rand_search = RandomizedSearchCV(estimator= tree.DecisionTreeClassifier(random_state=0), param_distributions=param_grid, cv=None, return_train_score=True, n_jobs=-1)
rand_search.fit(X_train, y_train)

scores = pd.DataFrame(rand_search.cv_results_)

# print(scores)

# scores.plot(x='param_max_depth', y='mean_train_score', yerr='std_train_score', ax=plt.gca(), figsize=(20,5))
# scores.plot(x='param_max_depth', y='mean_test_score', yerr='std_test_score', ax=plt.gca(), figsize=(20,5))
# plt.tick_params(axis='x', labelsize=20)
# plt.tick_params(axis='y', labelsize=20)
# plt.xlabel('max_depth', fontsize=26)
# plt.legend(fontsize=18)

# plt.show()

# scores.plot(x='param_min_samples_split', y='mean_train_score', yerr='std_train_score', ax=plt.gca(), figsize=(20,5))
# scores.plot(x='param_min_samples_split', y='mean_test_score', yerr='std_test_score', ax=plt.gca(), figsize=(20,5))
# plt.tick_params(axis='x', labelsize=20)
# plt.tick_params(axis='y', labelsize=20)
# plt.xlabel('min_samples_split', fontsize=26)
# plt.legend(fontsize=18)

# plt.show()



print(" Results from {} " .format(rand_search.__class__))
print("\n The best estimator across ALL searched params:\n",rand_search.best_estimator_)
print("\n The best score across ALL searched params:\n",rand_search.best_score_)
print("\n The best parameters across ALL searched params:\n",rand_search.best_params_)
# '''




clf = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=146,min_samples_leaf=31 ,random_state=1)


# plt.figure(figsize=(10,8), dpi=300)


clf.fit(X_train, y_train)

feat_importance = clf.feature_importances_
# feat_importance = np.sort(feat_importance) #! this should never be sorted unless the coloumns can be sorted with it



plt.barh(range(np.size(feat_importance)), feat_importance)
plt.yticks(range(np.size(feat_importance)),df.columns.tolist()[0:np.size(feat_importance)])

nonimp_feats = ['s20', 's17', 's14', 's13', 's04', 's03', 's02']

df.drop(nonimp_feats, axis='columns', inplace=True)
df_test.drop(nonimp_feats, axis='columns', inplace=True)

x_train_select = df.drop(['label'], axis='columns')
y_train_select = df['label']

x_test_select = df_test.drop(['label'], axis='columns')
y_test_select = df_test['label']



clf.fit(x_train_select, y_train_select)

# y_pred = clf.predict(X_test_scaled)

# tree.plot_tree(clf, filled=True, fontsize=5)

print("Test score: {:.2f}" .format(clf.score(x_test_select, y_test_select)))
print("Train score: {:.2f}".format(clf.score(x_train_select, y_train_select)))


plt.show()

