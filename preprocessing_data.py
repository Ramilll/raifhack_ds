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


def preprocessing_data(train, test):
    new_train = preprocessing_floor(train.copy(), 100500)
    new_test = preprocessing_floor(test.copy(), 100500)

    def ord_enc(col):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X = new_train[[col]]
        # X = np.concatenate((new_train[[col]].to_numpy(), new_test[[col]].to_numpy()))
        enc.fit(X)
        return enc.transform(new_train[[col]]), enc.transform(new_test[[col]])

    new_train['floor'], new_test['floor'] = ord_enc('floor')
    new_train['city'], new_test['city'] = ord_enc('city')
    new_train['osm_city_nearest_name'], new_test['osm_city_nearest_name'] = ord_enc('osm_city_nearest_name')
    new_train['region'], new_test['region'] = ord_enc('region')
    return new_train, new_test


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_preprocessing, test_preprocessing = preprocessing_data(train_data, test_data)
train_preprocessing.to_csv('new_train_data')
test_preprocessing.to_csv('new_test_data')
print(train_preprocessing.head())
print(test_preprocessing.head())
