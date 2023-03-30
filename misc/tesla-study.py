#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)
import yfinance as yf

stock='TSLA'
tesla = yf.download(stock, start='2021-1-01', end='2021-12-31').reset_index()
#tesla = pd.read_csv('TSLA.csv')
tesla = tesla[['Date','Open','High','Low','Close']]
print(tesla.shape)
tesla.head()

tesla_2011 = yf.download(stock, start='2011-1-01', end='2011-12-31').reset_index()
#tesla_2011 = pd.read_csv('TSLA-2011.csv')
tesla_2011 = tesla_2011[['Date','Open','High','Low','Close']]
print(tesla_2011.shape)
tesla_2011.head()

import matplotlib.ticker as mticker
from mplfinance.original_flavor import candlestick_ohlc
from datetime import date
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

df_cp = tesla.copy()
df_cp.Date = date2num(pd.to_datetime(tesla.Date).dt.to_pydatetime())
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((1,1), (0,0))
candlestick_ohlc(ax1,df_cp.values, width=0.4, colorup='#77d879', colordown='#db3f3f',alpha=1)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(stock+' OHLC')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot2grid((1,1), (0,0))
ret=candlestick_ohlc(ax1,df_cp.iloc[:100,:].values, width=0.4, colorup='#77d879', colordown='#db3f3f',alpha=1)
x_range = np.arange(df_cp.shape[0])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
ax1.grid(True)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(stock+' OHLC')
plt.show()

pd.set_option('use_inf_as_na', True)
tesla = tesla.dropna()
tesla_2011 = tesla_2011.dropna()
tesla.Close.astype(int)
tesla_2011.Close.astype(int)
print(tesla.Close)
print()
for i in tesla.Close:
    if(i=='nan' or i=='inf' or i=='-inf'):
        print(i)
print()

tesla.Close.plot(figsize=(8, 6))
plt.title(stock+' Close')
plt.show()

tesla_2011.Close.plot(figsize=(8, 6))
plt.title(stock+' Close')
plt.show()


tesla.plot(kind = "line", y = ['Open', 'High', 'Low','Close'],figsize=(8, 6))
plt.show()

tesla_2011.plot(kind = "line", y = ['Open', 'High', 'Low','Close'],figsize=(8, 6))
plt.show()

tesla_2011['months'] = pd.DatetimeIndex(tesla_2011['Date']).month
tesla_2011['year'] = pd.DatetimeIndex(tesla_2011['Date']).year
tesla_2011.head()

teslaPivot = pd.pivot_table(tesla_2011, values = "Close", columns = "year", index = "months")

print(teslaPivot.head())

teslaPivot.plot(figsize=(8, 6))
plt.show()

teslaPivot.plot(subplots = True, figsize=(8, 6), sharey=True)
plt.show()

tesla.Close.plot(kind = "hist", bins = 30, figsize=(8, 6))
plt.show()

tesla['Closelog'] = np.log(tesla.Close)
print(tesla.head())

tesla.Closelog.plot(kind = "hist", bins = 30, figsize=(8, 6))
plt.show()

tesla.Closelog.plot(figsize=(8, 6))
plt.show()

model_mean_pred = tesla.Closelog.mean()
# reverse log e
tesla["Closemean"] = np.exp(model_mean_pred)
tesla.plot(kind="line", x="Date", y = ["Close", "Closemean"],figsize=(8, 6))
plt.show()

from sklearn import linear_model
x = np.arange(tesla.shape[0]).reshape((-1,1))
y = tesla.Close.values.reshape((-1,1))
reg = linear_model.LinearRegression()
pred = reg.fit(x, y).predict(x)

tesla['linear'] = pred
tesla.plot(kind="line", x="Date", y = ["Close", "Closemean", "linear"],figsize=(8, 6))
plt.show()

#tesla.Date = pd.DatetimeIndex(tesla.Date)
#tesla.index = pd.PeriodIndex(tesla.Date, freq='D')
tesla = tesla.sort_values(by = "Date")
print(tesla.head())

tesla['timeIndex']= tesla.Date - tesla.Date.min()
tesla["timeIndex"] =tesla["timeIndex"] / np.timedelta64(1, 'D')
print(tesla.head())

tesla["timeIndex"] = tesla["timeIndex"].round(0).astype(int)
print(tesla.tail())

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

model_linear = smf.ols('Closelog ~ timeIndex', data = tesla).fit()
print(model_linear.summary())

model_linear.params

model_linear_pred = model_linear.predict()
model_linear_pred.shape

tesla['linear_stats'] = model_linear_pred
print(tesla.head())

tesla.plot(kind="line", x="timeIndex", y = ["Closelog", 'linear_stats'],figsize=(8, 6))
plt.show()

model_linear.resid.plot(kind = "bar",figsize=(8, 6)).get_xaxis().set_visible(False)
plt.show()

model_linear_forecast_auto = model_linear.predict(exog = pd.DataFrame(dict(timeIndex=252), index=[0]))
print(model_linear_forecast_auto)

tesla['pricelinear'] = np.exp(model_linear_pred)
print(tesla.head())

tesla.plot(kind="line", x="timeIndex", y = ["Close", "Closemean", "pricelinear"],figsize=(8, 6))
plt.show()

tesla["CloselogShift1"] = tesla.Closelog.shift()
tesla.head()

tesla.plot(kind= "scatter", y = "Closelog", x = "CloselogShift1", s = 50,figsize=(8, 6))
plt.show()

tesla["CloselogDiff"] = tesla.Closelog - tesla.CloselogShift1
tesla.CloselogDiff.plot(figsize=(8, 6))
plt.show()

tesla["CloseRandom"] = np.exp(tesla.CloselogShift1)
print(tesla.head())

def adf(ts):
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()

    fig = plt.figure(figsize=(8, 6))
    orig = plt.plot(ts.values, color='blue',label='Original')
    mean = plt.plot(rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput

tesla['CloselogMA12'] = tesla.Closelog.rolling(window=12).mean()
tesla.plot(kind ="line", y=["CloselogMA12", "Closelog"],figsize=(8, 6))
plt.show()

ts = tesla.Closelog - tesla.CloselogMA12
ts.dropna(inplace = True)
adf(ts)
# if test statistic < critical value (any), we can assume this data is stationary.
half_life = 12
tesla['CloselogExp12'] = tesla.Closelog.ewm(halflife=half_life).mean()
1 - np.exp(np.log(0.5)/half_life)

tesla.plot(kind ="line", y=["CloselogExp12", "Closelog"],figsize=(8, 6))
plt.show()


tesla["CloseExp12"] = np.exp(tesla.CloselogExp12)
print(tesla.tail())

tesla.plot(kind="line", x="timeIndex", y = ["Close", "Closemean", "pricelinear",
                                             "CloseRandom", "CloseExp12"],figsize=(8, 6))
plt.show()

ts = tesla.Closelog - tesla.CloselogExp12
ts.dropna(inplace = True)
adf(ts)

from statsmodels.tsa.seasonal import seasonal_decompose
#tesla.index = tesla.Date.to_datetime()

decomposition = seasonal_decompose(tesla.Closelog,freq=31)

decomposition.plot(figsize=(8, 6))
plt.show()

ts = tesla.Closelog
ts_diff = tesla.CloselogDiff
ts_diff.dropna(inplace = True)

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)

ACF = pd.Series(lag_acf)

ACF.plot(kind = "bar",figsize=(8, 6))
plt.show()
