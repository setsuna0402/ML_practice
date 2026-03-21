'''
The py file records func used to load data from csv file, preprocess data for NN and split data into train and validate sets
Date: 21/3/2026
Author: Dr. Edward
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# load data func:
def load_file(file_path: str) -> pd.DataFrame:
    file = pd.read_csv(file_path)
    return file

# this func checks the data properties, such as data types, missing values, and basic statistics.
def check_data(file: pd.DataFrame) -> None:
    print("First 5 rows of the data:")
    print(file.head()) # first 5 rows of the data
    print("Shape of the table:")
    print(file.shape)  # number of rows and columns
    print("Data types and non-null counts:")
    print(file.info()) # data types and non-null counts
    print("Statistical summary of numerical columns:")
    print(file.describe()) # statistical summary of numerical columns
    # print(file["Age"].describe()) # distribution of age values
    # print(file["Age"].value_counts()) # count of each unique age value (histogram of age distribution)
    print("Column names:")
    print(file.columns) # list of column names
    print("Count of missing values in each column:")
    print(file.isnull().sum()) # count of missing values in each column


# split target feature and input features:
def split_target_and_input(file: pd.DataFrame, target_column: str = "EngagementLevel"):
    feature_data = file.drop(columns=[target_column], errors='ignore') # input features
    target = file[target_column].copy() # target feature
    return feature_data, target

# drop unwanted columns:
def drop_columns(file: pd.DataFrame, columns_to_drop: str) -> pd.DataFrame:
    # drop columns that are not needed for analysis or modeling, such as "PlayerID"
    file = file.drop(columns=[columns_to_drop], errors='ignore')
    return file

# build preprocessing pipeline:
def build_preprocessor() -> ColumnTransformer:
    """
    Build a preprocessing transformer:
    - one-hot encode categorical columns
    - scale numeric columns
    """
    # define which columns are categorical and which are numerical
    # We choose to one-hot encode all categorical features, including "GameDifficulty".
    categorical = ["Gender", "Location", "GameGenre", "InGamePurchases", "GameDifficulty"]
    numerical = ["Age", "PlayTimeHours", "SessionsPerWeek", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
    # numerical = ["Age", "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return preprocessor

# this func is used to get the feature names after preprocessing, which is useful for understanding the transformed data and for feature importance analysis later on.
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(columns)

        elif name == "cat":
            encoded_names = transformer.get_feature_names_out(columns)
            feature_names.extend(encoded_names)

    return feature_names

def prepare_ml_data(data_table: pd.DataFrame, target_column: str = "EngagementLevel", valid_fraction: float = 0.2, random_seed: int = 42) -> tuple:
    '''
    pipeline order:
    1. drop unwanted columns (e.g., "PlayerID")
    2. split target and input features
    3. split data into train and validate sets
    4. build preprocessor based on training data
    5. fit preprocessor on training data and transform both train and validate data
    6. encode target labels into integers (for multi-class classification)
    '''
    # drop unwanted columns
    data_table = drop_columns(data_table, "PlayerID")
    # bypassing data which may be directly related to the target variable "EngagementLevel" to prevent data leakage.
    # data_table = drop_columns(data_table, "PlayTimeHours")
    # data_table = drop_columns(data_table, "SessionsPerWeek")
    # data_table = drop_columns(data_table, "AvgSessionDurationMinutes")
    # split target and input features
    data, label = split_target_and_input(data_table, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=valid_fraction, random_state=random_seed, stratify=label
    )
    # build preprocessor
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    # print(preprocessor.transformers_) # The collection of fitted transformers as tuples of (name, fitted_transformer, column)
    # you might ask why we don't do one-hot encoding for the target variable "EngagementLevel". 
    # That is because it is a multi-class classification problem, and many machine learning algorithms can handle integer-encoded target labels directly.
    # pytorch cross-entropy loss function, for example, expects class labels to be provided as integers, not one-hot encoded vectors.
    # target_encoder = LabelEncoder() # encode target labels into integers
    # y_train_encoded = target_encoder.fit_transform(y_train)
    # y_test_encoded = target_encoder.transform(y_test)
    # We choose to do explicit mapping for the target variable "EngagementLevel" 
    # to ensure that the classes are encoded in a specific order (Low=0, Medium=1, High=2)
    mapping_dict = {"Low": 0, "Medium": 1, "High": 2}
    y_train_encoded = y_train.map(mapping_dict).to_numpy()
    y_test_encoded = y_test.map(mapping_dict).to_numpy()
    return X_train_processed, X_test_processed, y_train_encoded, y_test_encoded, preprocessor, mapping_dict, get_feature_names(preprocessor)



# data = pd.read_csv("./online_gaming_behavior_dataset.csv")
# check_data(data)
# prepare_ml_data(data)
# split_data(data)