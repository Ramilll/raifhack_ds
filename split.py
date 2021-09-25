import pandas as pd
from os import makedirs

data_dir = 'data/'
split_dir = 'data_split/'

makedirs(split_dir, exist_ok=True)

train = pd.read_csv(data_dir + 'train.csv')
# test = pd.read_csv(data_dir + 'test.csv')
# test_submission = pd.read_csv(data_dir + 'test_submission.csv')

(_, train0), (_, train1) = tuple(train.groupby('price_type'))

assert(train0.date.is_monotonic_increasing)

VALIDATION_PART = 0.25
first_validation_date = train0.date.iloc[round(len(train0) * (1 - VALIDATION_PART))]

train_train = train[train.date < first_validation_date]
validation = train1[train1.date >= first_validation_date]
# validation0 = train0[train0.date >= first_validation_date]

train_train.to_csv(split_dir + 'train.csv', index=False)

validation.drop('per_square_meter_price', axis=1).to_csv(split_dir + 'val.csv', index=False)
validation[['id', 'per_square_meter_price']].to_csv(split_dir + 'val_true.csv', index=False)
