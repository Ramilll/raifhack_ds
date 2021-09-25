import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

data_dir = '../data/'

train = pd.read_csv(data_dir + 'train.csv')

msc = train[train.city == 'Москва']

(_, msc0), (_, msc1) = msc.groupby('price_type')

counts = msc.groupby('price_type').count()['city'].to_dict()
ratio = counts[0] // counts[1]

df = pd.concat([msc0] + [msc1] * ratio)

# sns.set(rc={'figure.figsize':(60,30)})
sobj = sns.displot(data=df, x='per_square_meter_price', hue='price_type', stat='probability', binwidth=20000, height=8, aspect=1.5)
ax = sobj.axes[0,0]
ax.set_xlim([0, 6e5])
ax.ticklabel_format(style = 'plain')
sobj.fig.savefig('plot/moscow_price_distribution.jpg')

