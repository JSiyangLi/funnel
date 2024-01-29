import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

V = 0.01


def true_lnz(dim, v=V):
    return (dim / 2) * (np.log(v) - np.log(1 + v))


ns_data = pd.read_csv("combined_results.csv")
ns_data = ns_data.sort_values(by="dim")
ns_data = ns_data[['dim', 'logZ']]
ns_data = ns_data.rename(columns={ns_data.columns[1]: 'NS'})
ns_data = ns_data.reset_index(drop=True)

larti_csvs = glob.glob("Lartillot0*.csv")
data = []
for c in larti_csvs:
    dim = int(c.split(",")[1])
    print(dim)
    df = pd.read_csv(c)
    df['dim'] = int(dim)
    df = df.sample(100)
    df = df[['dim', 'FI', 'IDR', 'SS', 'TI']]
    ns_d = ns_data[ns_data['dim'] == dim].sample(100)
    df['NS'] = ns_d['NS'].values
    df.to_csv('d{dim}.csv', index=False)
    data.append(df)

data = pd.concat(data)
data = data[['dim', 'FI', 'IDR', 'SS', 'TI', 'NS']]
data.to_csv('combined_larti_results.csv', index=False)