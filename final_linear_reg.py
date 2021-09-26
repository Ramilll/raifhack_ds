import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import json


def predict(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('lr_coeffs.json', 'w') as f:
        json.dump(model.coef_.tolist(), f)


    return model.predict(X_test)


train = pd.read_csv('data_split/train.csv', index_col='id')
train['realty_type_0'] = (train['realty_type'] == 10).astype(int)
train['realty_type_1'] = (train['realty_type'] == 100).astype(int)
train['realty_type_2'] = (train['realty_type'] == 110).astype(int)
train = train.drop(columns=['realty_type'])

features = ['total_square', 'osm_city_nearest_population', 'realty_type_0', 'realty_type_1', 'realty_type_2']

train['osm_city_nearest_population'] = train['osm_city_nearest_population'].fillna(0)

train1 = train[train.price_type == 1]
y_train = train1['per_square_meter_price']
train1 = train1[features]

train1knn01 = pd.read_csv('generated_features/train1knn01.csv', index_col='id')

train1lgbm = pd.read_csv('generated_features/train1lgbm.csv', index_col='id')

train_ttl = pd.concat([train1, train1knn01, train1lgbm], axis=1, join="inner")



X_train = train_ttl.to_numpy()



test = pd.read_csv('data/test.csv', index_col='id')
test['realty_type_0'] = (test['realty_type'] == 10).astype(int)
test['realty_type_1'] = (test['realty_type'] == 100).astype(int)
test['realty_type_2'] = (test['realty_type'] == 110).astype(int)
test = test.drop(columns=['realty_type'])

test['osm_city_nearest_population'] = test['osm_city_nearest_population'].fillna(0)


test = test[features]

test1knn01 = pd.read_csv('generated_features/test1knn01.csv', index_col='id')

test1lgbm = pd.read_csv('generated_features/test1lgbm.csv', index_col='id')

test_ttl = pd.concat([test, test1knn01, test1lgbm], axis=1, join="inner")

X_test = test_ttl.to_numpy()



# print(test_ttl.isna().sum())

test['per_square_meter_price'] = predict(X_train, y_train, X_test)
test[['per_square_meter_price']].to_csv('output_final.csv')
