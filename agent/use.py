#https://github.com/huseinzol05/Stock-Prediction-Models#simulations
#models
import importlib
import turtle_agent,moving_average_agent,signal_rolling_agent
import neuro_evolution_agent,neuro_evolution_novelty_search_agent, abcd_strategy_agent
#libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

agents=['policy_gradient_agent','q_learning_agent','evolution_strategy_agent',
        'double_q_learning_agent','recurrent_q_learning_agent','duel_q_learning_agent',
        'double_duel_q_learning_agent','duel_recurrent_q_learning_agent',
        'double_duel_recurrent_q_learning_agent','actor_critic_agent',
        'actor_critic_duel_agent','actor_critic_recurrent_agent',
        'actor_critic_duel_recurrent_agent','curiosity_q_learning_agent',
        'recurrent_curiosity_q_learning_agent','duel_curiosity_q_learning_agent']
m = importlib.import_module(agents[15])
print(m.__name__) # Printing the name of module

ms=['turtle_agent', 'moving_average_agent','signal_rolling_agent','neuro_evolution_agent','neuro_evolution_novelty_search_agent','abcd_strategy_agent']
#m=ms[5]
#m=""


stock='GOOG'
start_date = '2021-12-01'
end_date = '2021-12-31'
df = yf.download(stock,start_date, end_date)

df.head()

close = df.Close.values.tolist()
initial_money = 10000
#Agent parameters
window_size = 30
skip = 1
batch_size = 32
#Neural Network parameters
population_size = 100
generations = 100
mutation_rate = 0.1

if(m=="turtle_agent"):
    count = int(np.ceil(len(df) * 0.1))
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['trend'] = df['Close']
    signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
    signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
    signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
    signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1

    states_buy, states_sell, total_gains, invest = turtle_agent.buy_stock(df.Close, signals['signal'])
elif(m=="moving_average_agent"):
    short_window = int(0.025 * len(df))
    long_window = int(0.05 * len(df))
    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0.0
    signals['short_ma'] = df['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_ma'] = df['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:]> signals['long_ma'][short_window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    states_buy, states_sell, total_gains, invest = moving_average_agent.buy_stock(df.Close, signals['positions'])
elif(m=="signal_rolling_agent"):
    states_buy, states_sell, total_gains, invest = signal_rolling_agent.buy_stock(df.Close, initial_state = 1, delay = 4, initial_money = 10000)
elif(m=="neuro_evolution_agent"):
    neural_evolve = neuro_evolution_agent.NeuroEvolution(population_size, mutation_rate, window_size, window_size, close, skip, initial_money)
    fittest_nets = neural_evolve.evolve(50)
    states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)
elif(m=="neuro_evolution_novelty_search_agent"):
    novelty_search_threshold = 6
    novelty_log_maxlen = 1000
    backlog_maxsize = 500
    novelty_log_add_amount = 3

    neural_evolve = neuro_evolution_novelty_search_agent.NeuroEvolution(population_size, mutation_rate,
                                  window_size, window_size, close, skip, initial_money)
    fittest_nets = neural_evolve.evolve(100)
    states_buy, states_sell, total_gains, invest = neural_evolve.buy(fittest_nets)
elif(m=="abcd_strategy_agent"):
    states_buy, states_sell, total_gains, invest, states_money = abcd_strategy_agent.buy_stock(df.Close, signal)

    fig = plt.figure(figsize = (15,5))
    plt.plot(states_money, color='r', lw=2.)
    plt.plot(states_money, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(states_money, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.legend()
    plt.show()
else:
    agent = mod.Agent(state_size = window_size, window_size = window_size, trend = close, skip = skip)
    agent.train(iterations = 200, checkpoint = 10, initial_money = initial_money)
    states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)

close = df['Close']
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title(stock+' '+m+' total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
plt.show()
