#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import yfinance as yf


stocks=pd.read_csv('dija.csv')['Symbol']
print(stocks)

dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()[['Date','Close']] for s in stocks]
print(dfs)
from functools import reduce
data = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs).iloc[:, 1:]

returns = data.pct_change()
mean_daily_returns = returns.mean()
volatilities = returns.std()

mean_daily_returns * 252
volatilities * 252
combine = pd.DataFrame({'returns': mean_daily_returns * 252,
                       'volatility': volatilities * 252})

g = sns.jointplot(x="volatility", y="returns",data=combine, kind="reg",height=7)

for i in range(combine.shape[0]):
    plt.annotate(stocks[i].replace('.csv',''), (combine.iloc[i, 1], combine.iloc[i, 0]))

plt.text(0, -1.5, 'SELL', fontsize=25)
plt.text(0, 1.0, 'BUY', fontsize=25)

plt.show()
