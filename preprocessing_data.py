import pandas as pd
import numpy as np
pd.options.display.max_rows=1000
pd.options.display.max_columns=1000

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def preprocessing_one_floor(s, nan_value):
    if type(s) == str:
        s = s.lower()
        if s.find('подвал') != -1 or s.find('цоколь') != -1:
            return -1
        if s.find('антресоль') != -1 or s.find('чердак') != -1 or s.find('мансарда') != -1:
            return 5
        if s.find('(6)') != -1:
            return 6
    try:
        return int(float(s))
    except:
        return nan_value


def preprocessing_floor(data, nan_value):
    new_data = data.copy()
    new_data['floor'] = new_data['floor'].apply(lambda s: preprocessing_one_floor(s, nan_value))
    return new_data


def preprocessing_data(data):
    new_data = data.copy()
    new_data = preprocessing_floor(new_data, 100500)

    def ord_enc(col):
        enc = OrdinalEncoder()
        X = new_data[[col]]
        enc.fit(X)
        return enc.transform(X)

    new_data['floor'] = ord_enc('floor')
    new_data['city'] = ord_enc('city')
    new_data['osm_city_nearest_name'] = ord_enc('osm_city_nearest_name')
    new_data['region'] = ord_enc('region')
    return new_data


train = pd.read_csv('data/train.csv')
train_preprocessing = preprocessing_data(train)
train_preprocessing.to_csv('new_data')
print(train_preprocessing.head())
