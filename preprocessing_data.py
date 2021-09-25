import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from tqdm import tqdm
from os import makedirs

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


def preprocessing_v1():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    train_preprocessing, test_preprocessing = preprocessing_data(train_data, test_data)
    train_preprocessing.to_csv('new_train_data')
    test_preprocessing.to_csv('new_test_data')
    print(train_preprocessing.head())
    print(test_preprocessing.head())
    
def transform_numeric_osm_point_features(df):
    df_copy = df.copy(deep=True)
    columns_to_transform = sorted([c for c in df.select_dtypes(np.number).columns if (c.find("points") != -1) and (c[:3] == "osm")])
    column_prefixes = set([column[:10] for column in columns_to_transform ])
    for column_prefix in column_prefixes:
        columns_starts_with_prefix = sorted([c for c in columns_to_transform if c[:10] == column_prefix])
        for i in range(1, len(columns_starts_with_prefix)):
            old_column_name = str(columns_starts_with_prefix[i])
            new_column_name = str(columns_starts_with_prefix[i] + "-" + columns_starts_with_prefix[i-1])    
            df_copy[new_column_name] = df_copy[columns_starts_with_prefix[i]] - df_copy[columns_starts_with_prefix[i-1]]
        df_copy.drop(columns_starts_with_prefix[1:], axis = 1, inplace=True)
    return df_copy



def preprocessing_v2(path_to_train: str, path_to_test: str, path_to_save_train: str, path_to_save_test: str) -> None:

    # read data
    train = pd.read_csv(path_to_train, index_col='id')
    test = pd.read_csv(path_to_test, index_col='id')

    # process One Hot Encoding on realty_type
    train['realty_type_0'] = (train['realty_type'] == 10).astype(int)
    train['realty_type_1'] = (train['realty_type'] == 100).astype(int)
    train['realty_type_2'] = (train['realty_type'] == 110).astype(int)
    train = train.drop(columns=['realty_type'])

    test['realty_type_0'] = (test['realty_type'] == 10).astype(int)
    test['realty_type_1'] = (test['realty_type'] == 100).astype(int)
    test['realty_type_2'] = (test['realty_type'] == 110).astype(int)
    test = test.drop(columns=['realty_type'])

    # process floor
    train = preprocessing_floor(train, 2)
    test = preprocessing_floor(test, 2)

    # del city, street, osm_city_nearest_name, id
    train = train.drop(columns=['city', 'street', 'osm_city_nearest_name'])
    test = test.drop(columns=['city', 'street', 'osm_city_nearest_name'])
    
    # get diff of features (osm_points)
    train = transform_numeric_osm_point_features(train)
    test = transform_numeric_osm_point_features(test)

    # fill nans with mean
    columns_with_nans = ['osm_city_nearest_population', 'reform_house_population_1000', 'reform_house_population_500', 'reform_mean_floor_count_1000', 'reform_mean_floor_count_500', 'reform_mean_year_building_1000', 'reform_mean_year_building_500']
    for column in columns_with_nans:
        train[column].fillna((train[column].mean()), inplace=True)
        test[column].fillna((test[column].mean()), inplace=True)

    # process all prices to price_type=1
    avg_price = [{}, {}]
    for i, row in tqdm(train.iterrows()):
        if row['region'] in avg_price[row['price_type']].keys():
            avg_price[row['price_type']][row['region']]['sum_prices'] += row['per_square_meter_price']
            avg_price[row['price_type']][row['region']]['num_objects'] += 1
        else:
            avg_price[row['price_type']][row['region']] = {'sum_prices': row['per_square_meter_price'], 'num_objects': 1}

    mul_const = {}

    for city_name in train.region.unique():

        if city_name in avg_price[0].keys():
            mean_price_0 = avg_price[0][city_name]['sum_prices'] / avg_price[0][city_name]['num_objects']
        else:
            mean_price_0 = avg_price[1][city_name]['sum_prices'] / avg_price[1][city_name]['num_objects']

        if city_name in avg_price[1].keys():
            mean_price_1 = avg_price[1][city_name]['sum_prices'] / avg_price[1][city_name]['num_objects']
        else:
            mean_price_1 = mean_price_0

        mul_const[city_name] = mean_price_1 / mean_price_0

    def mul_price(df):
        df['per_square_meter_price'] *= mul_const[df.name]
        return df
        
    train = train.groupby('region').apply(mul_price)

    # del price_type
    train = train.drop(columns=['price_type'])
    test = test.drop(columns=['price_type'])

    # process region to MTE
    reg_mte = train.groupby('region')['per_square_meter_price'].mean()
    train['region'] = train['region'].apply(lambda x: reg_mte[x])
    test['region'] = test['region'].apply(lambda x: reg_mte[x])

    # process date
    train['date'] = (pd.to_datetime(train['date']) - pd.Timestamp('2020-01-01')).dt.days
    test['date'] = (pd.to_datetime(test['date']) - pd.Timestamp('2020-01-01')).dt.days

    # sort by date
    train = train.sort_values(by=['date'])
    test = test.sort_values(by=['date'])

    train.to_csv(path_to_save_train)
    test.to_csv(path_to_save_test)



if __name__ == '__main__':

    
    PATH_TO_INITIAL_DATA = 'data_split/'
    PATH_TO_SAVE_PREPROCESSED_DATA = 'data_preprocessed/'

    makedirs(PATH_TO_SAVE_PREPROCESSED_DATA, exist_ok=True)

    preprocessing_v2(PATH_TO_INITIAL_DATA + 'train.csv', PATH_TO_INITIAL_DATA + 'val.csv', PATH_TO_SAVE_PREPROCESSED_DATA + 'train_v2.csv', PATH_TO_SAVE_PREPROCESSED_DATA + 'val_v2.csv')