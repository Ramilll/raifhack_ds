import pandas as pd
from sklearn.linear_model import Ridge


args = {
    'tr'      : 'data_split/train.csv',    # Путь до обучающего датасета
    'tst'     : 'data_split/val.csv',      # Путь до отложенной выборки
    'o_tst'   : 'data_split/val_pred',       # Путь до предсказания отложенной выборки
    'lgbm_tr' : 'generated_features/train1_lgbm.csv',    # LGBMRegressor for train
    'lgbm_tst': 'generated_features/val_lgbm.csv',    # LGBMRegressor for train
    'knn_tr'  : 'generated_features/train1_knn.csv',    # KNN for train
    'knn_tst' : 'generated_features/val_knn.csv',    # KNN for train
}


def predict(X_train, y_train, X_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    return model.predict(X_test)


train = pd.read_csv(args['tr'], index_col='id')
train['realty_type_0'] = (train['realty_type'] == 10).astype(int)
train['realty_type_1'] = (train['realty_type'] == 100).astype(int)
train['realty_type_2'] = (train['realty_type'] == 110).astype(int)
train = train.drop(columns=['realty_type'])

features = ['total_square', 'osm_city_nearest_population', 'realty_type_0', 'realty_type_1', 'realty_type_2']

train['osm_city_nearest_population'] = train['osm_city_nearest_population'].fillna(0)

train1 = train[train.price_type == 1]
y_train = train1['per_square_meter_price']
train1 = train1[features]

train1knn01 = pd.read_csv(args['knn_tr'], index_col='id')
train1lgbm = pd.read_csv(args['knn_tst'], index_col='id')

train_ttl = pd.concat([train1, train1knn01, train1lgbm], axis=1, join="inner")



X_train = train_ttl.to_numpy()



test = pd.read_csv('data/test.csv', index_col='id')
test['realty_type_0'] = (test['realty_type'] == 10).astype(int)
test['realty_type_1'] = (test['realty_type'] == 100).astype(int)
test['realty_type_2'] = (test['realty_type'] == 110).astype(int)
test = test.drop(columns=['realty_type'])

test['osm_city_nearest_population'] = test['osm_city_nearest_population'].fillna(0)


test = test[features]

test1knn01 = pd.read_csv(args['knn_tst'], index_col='id')
test1lgbm = pd.read_csv(args['lgbm_tst'], index_col='id')

test_ttl = pd.concat([test, test1knn01, test1lgbm], axis=1, join="inner")

X_test = test_ttl.to_numpy()



# print(test_ttl.isna().sum())

test['per_square_meter_price'] = predict(X_train, y_train, X_test)
test[['per_square_meter_price']].to_csv(args['o_tst'])
