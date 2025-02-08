# import all necessary libraries here

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

## These are all the necessary libraries and packages needed for this assignment. They have been installed and imported. 

# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    
    """
    Fill missing values in the dataset.
    :param clean_data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    #First, check to ensure that 'strategy' exist/valid.
    
    if strategy not in ['mean', 'median', 'mode']:
        raise ValueError("Strategy needs to be 'mean', 'median' or 'mode' ")
    
    #Next, to prevent the original copy of the data from being modified, make a copy. 
    imputed_data = data.copy()
    
    #Then, iterate over each column that has numerial data. 
    #Only need to look for missing values, so only the columns that have missing values would be processed
    for column in imputed_data.select_dtypes(include=['number']).columns:
        if imputed_data[column].isnull().sum() > 0:
            if strategy == 'mean':
                fill_value = imputed_data[column].mean()
            elif strategy == 'median':
                fill_value = imputed_data[column].median()   
            elif strategy == 'mode':
                fill_value = imputed_data[column].mode()[0]  
                
             #As per the instructions, missing values within the dataset would be filled    
            imputed_data[column] = imputed_data[column].fillna(fill_value)
            
    return imputed_data
    

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    #Remove duplicate rows
    return data.drop_duplicates()

    

# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    
    #To avoid modification of original data let's make a copy
    norm_data = data.copy()
    numeric_columns = norm_data.select_dtypes(include=['number']).columns
    
    #Normalize numerical data using Min-Max or Standard scaling by using te MinMaxScaler
    if method == 'minmax':
        scaler = MinMaxScaler()
        
    elif method == 'standard':
        scaler = StandardScaler()
        
    else:
        raise ValueError("Must use 'minmax' or 'standard'")
    norm_data[numeric_columns] = scaler.fit_transform(norm_data[numeric_columns])
    return norm_data

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    #Remove redundant features based on the correlation threshold (HINT: you can use the corr() method)
    #Calculate the Correlation matrix
    
    Corr_Matrix = data.corr()
    redundant_colms = set()
    
    #Iterate over the Corr matrix to identify any redundant features/columns 
    for i in range(len(Corr_Matrix.columns)):
        for j in range(i):
            if abs(Corr_Matrix.iloc[i, j]) > threshold:
                colname = Corr_Matrix.columns[i]
                redundant_colms.add(colname)
    
    return data.drop(columns=redundant_colms)
    
    
### Tested code to ensure it all worked!!###

# ---------------------------------------------------
#Define the simple_model function 

def simple_model(input_data,split_data=True, scale_data=False, print_report=False):
    
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """
    
    #Tells the amount of rows and columns we had before preprocessing
    initial_rows, initial_columns = input_data.shape

    # if there's any missing data, remove the columns
    input_data = input_data.dropna(inplace=True)
    
    
    #Tracks the rows left after preprocessing 
    rows_after, columns_after = input_data.shape
    
    removed_rows = initial_rows - rows_after
    removed_columns = initial_columns - columns_after
    
    print(f"Rows removed: {removed_rows}")
    print(f"Columns removed: {removed_columns}")

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    #Split the data into training and testing sets
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None