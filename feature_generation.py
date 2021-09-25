import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from os import makedirs

def gen_knn_geo_date_features():

    val = pd.read_csv('data_split/val.csv', index_col='id')
    train = pd.read_csv('data_split/train.csv', index_col='id')
    train['date'] = (pd.to_datetime(train['date']) - pd.Timestamp('2020-01-01')).dt.days
    val['date'] = (pd.to_datetime(val['date']) - pd.Timestamp('2020-01-01')).dt.days

    y_val = pd.read_csv('data_split/val_true.csv', index_col='id').to_numpy()

    knn_features = ['lng', 'lat', 'date']

    X_train = train[knn_features].to_numpy()
    y_train = train['per_square_meter_price'].to_numpy()
    X_val = val[knn_features].to_numpy()


    neigh = KNeighborsRegressor(n_neighbors=5)
    neigh.fit(X_train, y_train)

    y_pred_val = neigh.predict(X_val)

    val['knn_feature'] = y_pred_val


    new_df = val[['knn_feature']]


    makedirs('generated_features', exist_ok=True)

    new_df.to_csv('generated_features/knn_geo_date.csv')


if __name__ == '__main__':
    gen_knn_geo_date_features()