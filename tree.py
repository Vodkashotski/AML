from textwrap import fill
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import tree

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

def score_func(y_true,y_pred):
    """
    !might be flawed
    model evaluation function
    
    Args:
        y_true = true target RUL value
        y_pred = predicted target RUL value
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    score_list = [round(mae, 2), round(rmse, 2), round(r2, 2)]
    # printing metrics
    print("Classification Report:\n", report)
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print("<)-------------X-------------(>")

def transform_df(file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(file_path, sep=" ", header=None)
    df = df.drop([26,27], axis=1)
    column_names = ["unit_number","time_in_cycles","setting1","setting2","setting3",
                "s01","s02","s03","s04","s05","s06","s07","s08","s09","s10","s11","s12","s13",
                "s14","s15","s16","s17","s18","s19","s20","s21"]

    df.columns = column_names
    # Drop specified columns
    columns_to_drop = ["s19", "s18", "s16", "s05", "s01", "setting3",
                       "setting1", "setting2", "s21"]
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

    return X_train, y_train

trainset_path = r"C:\Users\Simon\Documents\Code projects\AML\Data\train_FD003.txt"
X_train, y_train = transform_df(trainset_path)
# print(f"Dimension of feature matrix : {X_train.shape}\ndimension of target vector: {y_train.shape}")

testset_path =r"C:\Users\Simon\Documents\Code projects\AML\Data\test_FD003.txt"
X_test, y_test = transform_df(testset_path)
# print(f"Dimension of feature matrix : {X_train.shape}\ndimension of target vector: {y_train.shape}")

clf = tree.DecisionTreeClassifier(max_depth=14, min_samples_split=50)
plt.figure(figsize=(50,45), dpi=300)

clf.fit(X_train, y_train)

tree.plot_tree(clf, filled=True, fontsize=2)
print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))
# plt.show()