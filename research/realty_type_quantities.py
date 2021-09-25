import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '../data/'

train = pd.read_csv(data_dir + 'train.csv')

rt = train.realty_type

counts = rt.groupby(rt).count().rename('count').reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=counts, x='realty_type', y='count', ax=ax)
fig.savefig('plot/realty_type_distribution.jpg')