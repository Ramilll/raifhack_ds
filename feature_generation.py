import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from os import makedirs, path
from sklearn.preprocessing import StandardScaler
from preprocessing_data import preprocessing_floor


def gen_knn_geo_date_features(train_path, val_path):
    val = pd.read_csv(val_path, index_col='id')
    train = pd.read_csv(train_path, index_col='id')

    train['osm_city_nearest_population'] = train['osm_city_nearest_population'].fillna(0)
    val['osm_city_nearest_population'] = val['osm_city_nearest_population'].fillna(0)

    # print(train['osm_city_nearest_population'].isna().sum())

    # date to int
    train['date'] = (pd.to_datetime(train['date']) - pd.Timestamp('2020-01-01')).dt.days
    val['date'] = (pd.to_datetime(val['date']) - pd.Timestamp('2020-01-01')).dt.days

    # process One Hot Encoding on realty_type
    train['realty_type_0'] = (train['realty_type'] == 10).astype(int)
    train['realty_type_1'] = (train['realty_type'] == 100).astype(int)
    train['realty_type_2'] = (train['realty_type'] == 110).astype(int)
    train = train.drop(columns=['realty_type'])

    val['realty_type_0'] = (val['realty_type'] == 10).astype(int)
    val['realty_type_1'] = (val['realty_type'] == 100).astype(int)
    val['realty_type_2'] = (val['realty_type'] == 110).astype(int)
    val = val.drop(columns=['realty_type'])

    y_val = pd.read_csv('data_split/val_true.csv', index_col='id').to_numpy()

    knn_features = ['lng', 'lat', 'total_square', 'realty_type_0', 'realty_type_1', 'realty_type_2', 'osm_city_nearest_population']

    
    train0 = train[train['price_type'] == 0]
    train1 = train[train['price_type'] == 1]

    X_train0 = train0[knn_features].to_numpy()
    y_train0 = train0['per_square_meter_price'].to_numpy()
    X_train1 = train1[knn_features].to_numpy()
    y_train1 = train1['per_square_meter_price'].to_numpy()
    X_val = val[knn_features].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X_train0)

    X_train0 = scaler.transform(X_train0)
    X_train1 = scaler.transform(X_train1)
    X_val = scaler.transform(X_val)

    neigh0 = KNeighborsRegressor(n_neighbors=10)
    neigh0.fit(X_train0, y_train0)

    neigh1 = KNeighborsRegressor(n_neighbors=10)
    neigh1.fit(X_train1, y_train1)

    y_pred_train0 = neigh0.predict(X_train1)
    y_pred_train1 = neigh1.predict(X_train1)

    train1['knn0'] = y_pred_train0
    train1['knn1'] = y_pred_train1

    new_df = train1[['knn0', 'knn1']]

    makedirs('generated_features', exist_ok=True)

    new_df.to_csv('generated_features/train1knn01.csv')

    y_pred_val0 = neigh0.predict(X_val)
    y_pred_val1 = neigh1.predict(X_val)

    val['knn0'] = y_pred_val0
    val['knn1'] = y_pred_val1

    new_df = val[['knn0', 'knn1']]
    filename = val_path.split('.csv')[0].split('/')[1]
    new_df.to_csv(f'generated_features/{filename}1knn01.csv')


def _get_new_features(data: pd.DataFrame) -> pd.DataFrame:
    data = preprocessing_floor(data, 2)
    data['floor'][data.floor <= 0] = 0
    data['floor'][(1 <= data.floor) & (data.floor <= 1)] = 1
    data['floor'][(2 <= data.floor) & (data.floor <= 2)] = 2
    data['floor'][(3 <= data.floor) & (data.floor <= 5)] = 3
    data['floor'][(6 <= data.floor) & (data.floor <= 10)] = 4
    data['floor'][(11 <= data.floor)] = 10

    def need_log(f_name):
        for s in ['osm_building_points', 'osm_city_closest_dist', 'osm_crossing_closest_dist', 'total_square', 'osm_culture_points', 'osm_hotels_points', 'osm_subway_closest_dist', 'osm_train_stop']:
            if f_name.startswith(s):
                return True
        return False

    for feature in data.columns:
        if need_log(feature):
            print(f'{feature} -> log_{feature}')
            try:
                data[f'log_{feature}'] = np.log(1 + data[feature])
                data.drop(feature, axis=1, inplace=True)
            except:
                print('Failed!')
    return data


def gen_new_features():
    val = pd.read_csv('data_split/val.csv', index_col='id')
    train = pd.read_csv('data_split/train.csv', index_col='id')
    res_dir = 'generated_features'
    if not path.exists(res_dir):
        makedirs(res_dir)
    gen_train = _get_new_features(train)
    gen_val = _get_new_features(val)
    gen_train.to_csv(path.join(res_dir, 'train.csv'))
    gen_val.to_csv(path.join(res_dir, 'val.csv'))


if __name__ == '__main__':
    gen_knn_geo_date_features(train_path='data_split/train.csv', val_path='data_split/val.csv')
    gen_knn_geo_date_features(train_path='data_split/train.csv', val_path='data/test.csv')
