#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['AX','AY','C','FX','FY','K','Strides','??']
df = pd.read_csv('NeuroSIM/NetWork_VGG8.csv', names=cols)

from IPython.display import display 

display(df)

layer_params = df['C']*df['FX']*df['FY']*df['K']
print(f'VGG8 Parameters: {layer_params.sum()}')

# 12973440

df2 = pd.read_csv('NeuroSIM/NetWork_ResNet18.csv', names=cols)
display(df2)

layer_params = df2['C']*df2['FX']*df2['FY']*df2['K']
print(f'ResNet18 Parameters: {layer_params.sum()}')

#%%

from matplotlib.ticker import EngFormatter 

sns.set_theme()

titles = ['NeuroSim NM','RectPack','RectPack with Hierarchy']
values = [142606333,13041644,13631488]

fig,ax = plt.subplots()
plt.bar(titles,values)

ax.yaxis.set_major_formatter(EngFormatter(unit=''))