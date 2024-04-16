import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVR

def transformData(data):
    df = pd.read_csv(data, header=None, delim_whitespace=' ')
    header = ['unit_number','time_in_cycles','setting1','setting2','setting3',
                's01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13',
                's14','s15','s16','s17','s18','s19','s20','s21']
    df.columns = header

    corr = df.corr()

    #Drop columns with low or NaN correlation
    # Set threshold for highly correlated features
    threshold = 0.8

    # Find pairs of highly correlated features
    highly_correlated = (corr.abs() > threshold) & (corr.abs() < 1.0)

    # Get indices of highly correlated features
    correlated_indices = pd.DataFrame(highly_correlated.unstack())
    correlated_indices = correlated_indices[correlated_indices[0]].index.tolist()

    # Remove duplicates and self-correlations
    correlated_indices = set([(i[0], i[1]) if i[0] < i[1] 
                            else 
                            (i[1], i[0]) for i in correlated_indices])

    # Identify features to remove (optional)
    #features_to_remove = [feat[1] for feat in correlated_indices]

    # Remove features from the dataset (optional)
    #df = df.drop(columns=features_to_remove)
    # Removes settings and NaN correlated features
    df = df.drop(columns=['setting1', 'setting2', 'setting3', 's01', 
                     's05', 's10', 's16', 's18', 's19'])

    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Finding the end of life of each engine unit
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

def score_func(y_true,y_pred):
    """
    model evaluation function
    
    Args:
        y_true = true target RUL value
        y_pred = predicted target RUL value
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    score_list = [round(mae, 2), round(rmse, 2), round(r2, 2)]
    # printing metrics
    print(f' Mean Absolute Error (MAE): {score_list[0]}')
    print(f' Root Mean Squared Error (RMSE): {score_list[1]}')
    print(f' R2 Score: {score_list[2]}')
    print("<)-------------X-------------(>")

def importRUL(rul):
    RUL = pd.read_csv(rul, header=None, delim_whitespace=' ')
    header = ['remaining_useful_lifetime']
    RUL.columns = header
    RUL = RUL.values
    return RUL

set_number = 'FD001'

# Define the filenames and function calls using string formatting
rul_filename = f'RUL_{set_number}.txt'
train_filename = f'train_{set_number}.txt'
test_filename = f'test_{set_number}.txt'

# Call importRUL function with dynamically constructed filename
y_true = importRUL(rul_filename)

# Call transformData function with dynamically constructed filenames
X, y = transformData(train_filename)
X_test, y_test = transformData(test_filename)

X_train_def, X_val_def, y_train_def, y_val_def = train_test_split(X, y, test_size=0.3, random_state=7) 

#scaler = StandardScaler()
#X_train_def_scaled = scaler.fit_transform(X_train_def)
#X_val_def_scaled = scaler.transform(X_val_def)

# Initialize the SVR model
model = SVR(kernel='rbf')  # You can specify other kernels like 'linear', 'poly', 'sigmoid', etc.

# Fit the SVR model to the training data
model.fit(X_train_def, y_train_def)

# Predict the target variable on the validation set
y_val_pred = model.predict(X_val_def)

print('Validation score: \n')
score_func(y_val_def, y_val_pred)

# Predict the target variable on the test set
y_pred = model.predict(X_test)

print('Test score: \n')
score_func(y_pred, y_test)