import pandas as pd
import numpy as np
import yfinance as yf


stocks=pd.read_csv('dija.csv')['Symbol']
print(stocks)

data = np.array([yf.Ticker(s).history(period='1y', interval='1d').reset_index()[['Close']] for s in stocks])
print(data[:, :, 0])
#shape = (50, 4460)
#data = np.random.normal(size=shape)

#data[:, 1000] += data[:, 2000]

df = pd.DataFrame(data[:, :, 0])

c = df.corr()
so = c.unstack().sort_values(kind="quicksort")
print(so.head())
print()
print(so.tail())

sol = (c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().sort_values(ascending=False))
print(sol.head())

for i in sol.head().keys():
    print(df[i[0]]['Symbol'])
print()
print(sol.tail())

print("Correlation Matrix")
print(c)
print()


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 3))
