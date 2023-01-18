# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

result_path = '/home/lborne/result/chime/dnsmos'
dnsmos_eval = pd.read_csv(f'{result_path}/dnsmos_eval_1.csv')
dnsmos_eval['split'] = 'eval'
dnsmos_dev = pd.read_csv(f'{result_path}/dnsmos_dev_1.csv')
dnsmos_dev['split'] = 'dev's
dnsmos = pd.concat([dnsmos_eval, dnsmos_dev])
dnsmos.rename(columns={"Unnamed: 0": "file"}, inplace=True)
dnsmos = dnsmos.melt(['file', 'split'], value_vars=['sig_mos', 'bak_mos', 'ovr_mos'], value_name='score')

# %%
sns.boxplot(data=dnsmos, x='score', y='variable', hue='split')
plt.show()

# %% merge results for github

result_path = '/home/lborne/result/chime/dnsmos'
dnsmos_eval = pd.read_csv(f'{result_path}/dnsmos_eval_1.csv', index_col=0)
dnsmos_dev = pd.read_csv(f'{result_path}/dnsmos_dev_1.csv', index_col=0)
dnsmos = pd.concat([dnsmos_eval, dnsmos_dev])
dnsmos.index = [idx[6:] for idx in dnsmos.index]

dnsmos.to_csv(f'{result_path}/dnsmos.csv')
