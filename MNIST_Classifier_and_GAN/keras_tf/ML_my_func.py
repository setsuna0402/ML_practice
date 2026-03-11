import numpy as np  
import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def one_hot_encoder(label):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(label)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def normalisation(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)