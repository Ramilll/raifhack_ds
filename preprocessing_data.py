import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from tqdm import tqdm

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
        enc = OrdinalEncoder()
        X = np.concatenate((new_train[[col]].to_numpy(), new_test[[col]].to_numpy()))
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


def preprocessing_v2(path_to_train: str, path_to_test: str, path_to_save_train: str, path_to_save_test: str) -> None:

    # read data
    train = pd.read_csv(path_to_train, index_col='id')
    test = pd.read_csv(path_to_test, index_col='id')

    # process floor
    train = preprocessing_floor(train, 2)
    test = preprocessing_floor(test, 2)

    # del region, street
    train = train.drop(columns=['city', 'street'])
    test = test.drop(columns=['city', 'street'])

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

    PATH_TO_INITIAL_DATA = '/Users/olegmelnikov/Desktop/data/'
    PATH_TO_SAVE_PREPROCESSED_DATA = '/Users/olegmelnikov/Desktop/data/'

    preprocessing_v2(PATH_TO_INITIAL_DATA + 'train.csv', PATH_TO_INITIAL_DATA + 'test.csv', PATH_TO_INITIAL_DATA + 'train_v2.csv', PATH_TO_INITIAL_DATA + 'test_v2.csv')