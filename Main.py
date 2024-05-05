

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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

def importData(set):
    Header = ["unit_number","time_in_cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21","sensor measurement 22","sensor measurement 23"]
    data = pd.read_csv("AML\Data\{}" .format(set), header=None, delimiter=" ")
    data.columns = Header
    return data

def add_RUL_column(df):
    train_grouped_by_unit = df.groupby(by='unit') 
    max_time = train_grouped_by_unit['time'].max()
    merged = df.merge(max_time.to_frame(name='max_time'), left_on='unit',right_index=True)
    merged["RUL"] = merged["max_time"] - merged['time']
    merged = merged.drop("max_time", axis=1)
    return merged

data = importData("train_FD003.txt")

#dropped because of missing values
data.drop(["sensor measurement 22","sensor measurement 23"], axis='columns', inplace=True)

#dropped because of correlation
data.drop(["operational setting 1", "operational setting 2", "operational setting 3", "sensor measurement 1", "sensor measurement 5", "sensor measurement 16", "sensor measurement 18", "sensor measurement 19", "sensor measurement 21"],
         axis='columns', inplace=True)

# print(data.describe().T)
plt.figure(figsize=(10,6))
sns.heatmap(data.isna().transpose(),
            cmap = sns.diverging_palette(230, 20, as_cmap=True),
            cbar_kws={'label': 'Missing Data'})

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(10, 6))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()


EOL=[]
for i in data['unit_number']:
        EOL.append( ((data[data['unit_number'] == i]["time_in_cycles"]).values)[-1])
data["EOL"]=EOL
# Calculate "LR"
data["LR"] = data["time_in_cycles"].div(data["EOL"])
data['label'] = pd.cut(data['LR'], bins=[0, 0.6, 0.8, np.inf], labels=[0, 1, 2], right=False)

#copy to keep old data set
df = data.copy()

print(df.head())

#drop unnecessary features
df.drop(columns=['unit_number', 'EOL', 'LR'], inplace=True)

X_train = df.drop(["label"], axis=1).values
y_train = df["label"] 

print(f"Dimension of feature matrix : {X_train.shape}\ndimension of target vector: {y_train.shape}")









