import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

data_dir = '../data/'

train = pd.read_csv(data_dir + 'train.csv')

sns.set_theme(style="whitegrid")
sobj = sns.displot(data=train, x='per_square_meter_price', palette="bright", hue='realty_type', kind="kde", height=8, aspect=1.5)
ax = sobj.axes[0,0]
ax.set_xlim([0, 5e5])
ax.ticklabel_format(style = 'plain')
sobj.fig.savefig('plot/realty_type_price_distribution.jpg')

