#https://github.com/huseinzol05/Stock-Prediction-Models#simulations
#https://huseinhouse.com/stock-forecasting-js/
#models
import helper
import importlib
#libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

agents=['lstm','bidirectional_lstm','lstm_2path','gru','bidirectional_gru',
        'gru_2path','vanilla','bidirectional_vanilla','vanilla_2path',
        'lstm_seq2seq','bidirectional_lstm_seq2seq','lstm_seq2seq_vae',
        'gru_seq2seq','bidirectional_gru_seq2seq','gru_seq2seq_vae',
        'attention_is_all_you_need','cnn_seq2seq','dilated_cnn_seq2seq']
m = importlib.import_module(agents[0])
print(m.__name__) # Printing the name of module

ms=['turtle_agent', 'moving_average_agent','signal_rolling_agent','neuro_evolution_agent','neuro_evolution_novelty_search_agent','abcd_strategy_agent']
#m=ms[5]
#m=""


stock='GOOG'
start_date = '2020-1-01'
end_date = '2021-12-31'
df = yf.download(stock,start_date, end_date)
print(df.head())

minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = pd.DataFrame(df_log)
df_log.head()
# ## Split train and test
# I will cut the dataset to train and test datasets,
# 1. Train dataset derived from starting timestamp until last 30 days
# 2. Test dataset derived from last 30 days until end of the dataset
# So we will let the model do forecasting based on last 30 days, and we will going to repeat the experiment for 10 times. You can increase it locally if you want, and tuning parameters will help you by a lot.
test_size = 30
simulation_size = 2

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]
df.shape, df_train.shape, df_test.shape
#parameters
num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 0.8
future_day = test_size
learning_rate = 0.01


results = []
for i in range(simulation_size):
    print('simulation %d'%(i + 1))
    results.append(m.forecast(learning_rate, num_layers, df_log, size_layer, dropout_rate,df,minmax,
                    epoch,future_day,timestamp,test_size,simulation_size,df_train,df_test))

accuracies = [helper.calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

plt.figure(figsize = (15, 5))
for no, r in enumerate(results):
    plt.plot(r, label = 'forecast %d'%(no + 1))
plt.plot(df['Close'].iloc[-test_size:].values, label = 'true trend', c = 'black')
plt.legend()
plt.title('average accuracy: %.4f'%(np.mean(accuracies)))
plt.show()
