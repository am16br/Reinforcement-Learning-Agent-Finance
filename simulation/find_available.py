import pandas as pd
import numpy as np
import yfinance as yf

'''
stocks=pd.read_csv('syms.csv')['Symbol']
a=pd.read_csv('syms.csv')['Symbol']
dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()['Close'] for s in stocks]

b=pd.concat([dfs, a, a]).drop_duplicates(keep=False)

df = pd.DataFrame(b, columns=["Symbol"])
df.to_csv('available2.csv')
'''
import yahoo_fin.stock_info as si


market_list = si.tickers_dow()
name="Dow Jones"
#dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()['Symbol'] for s in market_list]
df = pd.DataFrame(market_list, columns=["Symbol"])
df.to_csv('dija.csv')

market_list = si.tickers_nasdaq()
name="NASDAQ"
#dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()['Close'] for s in market_list]
df = pd.DataFrame(market_list, columns=["Symbol"])
df.to_csv('NASDAQ.csv')

market_list = si.tickers_sp500()
name="S&P500"
#dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()['Close'] for s in market_list]
df = pd.DataFrame(market_list, columns=["Symbol"])
df.to_csv('S&P500.csv')

market_list = si.tickers_other()
#dfs = [yf.Ticker(s).history(period='1y', interval='1d').reset_index()['Close'] for s in market_list]
df = pd.DataFrame(market_list, columns=["Symbol"])
df.to_csv('other.csv')
