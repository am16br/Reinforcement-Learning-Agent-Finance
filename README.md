<p align="center">
    <a href="#readme">
        <img alt="logo" width="50%" src="output/evolution-strategy.png">
    </a>
</p>
<p align="center">
  <a href="https://github.com/am16br/Reinforcement-Learning-Agent-Finance/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-Apache--License--2.0-yellow.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/deeplearning-30--models-success.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/agent-23--models-success.svg"></a>
</p>

---

**Stock-Prediction-Models**, Gathers machine learning and deep learning models for Stock forecasting, included trading bots and simulations.

## Table of contents
  * [Models](#models)
  * [Agents](#agents)
  * [Realtime Agent](realtime-agent)
  * [Data Explorations](#data-explorations)
  * [Simulations](#simulations)
  * [Tensorflow-js](#tensorflow-js)
  * [Misc](#misc)
  * [Results](#results)
    * [Results Agent](#results-agent)
    * [Results signal prediction](#results-signal-prediction)
    * [Results analysis](#results-analysis)
    * [Results simulation](#results-simulation)

## Contents

### Models

#### [Deep-learning models](deep-learning)
 1. LSTM
 2. LSTM Bidirectional
 3. LSTM 2-Path
 4. GRU
 5. GRU Bidirectional
 6. GRU 2-Path
 7. Vanilla
 8. Vanilla Bidirectional
 9. Vanilla 2-Path
 10. LSTM Seq2seq
 11. LSTM Bidirectional Seq2seq
 12. LSTM Seq2seq VAE
 13. GRU Seq2seq
 14. GRU Bidirectional Seq2seq
 15. GRU Seq2seq VAE
 16. Attention-is-all-you-Need
 17. CNN-Seq2seq
 18. Dilated-CNN-Seq2seq

**Bonus**

1. How to use one of the model to forecast `t + N`, [how-to-forecast.py](deep-learning/how-to-forecast.py)
2. Consensus, how to use sentiment data to forecast `t + N`, [sentiment-consensus.py](deep-learning/sentiment-consensus.py)

#### [Stacking models](stacking)
 1. Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor
 2. Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB

### [Agents](agent)

1. Turtle-trading agent
2. Moving-average agent
3. Signal rolling agent
4. Policy-gradient agent
5. Q-learning agent
6. Evolution-strategy agent
7. Double Q-learning agent
8. Recurrent Q-learning agent
9. Double Recurrent Q-learning agent
10. Duel Q-learning agent
11. Double Duel Q-learning agent
12. Duel Recurrent Q-learning agent
13. Double Duel Recurrent Q-learning agent
14. Actor-critic agent
15. Actor-critic Duel agent
16. Actor-critic Recurrent agent
17. Actor-critic Duel Recurrent agent
18. Curiosity Q-learning agent
19. Recurrent Curiosity Q-learning agent
20. Duel Curiosity Q-learning agent
21. Neuro-evolution agent
22. Neuro-evolution with Novelty search agent
23. ABCD strategy agent

### [Data Explorations](misc)

1. stock market study on TESLA stock, [tesla-study.py](misc/tesla-study.py)
2. Outliers study using K-means, SVM, and Gaussian on TESLA stock, [outliers.py](misc/outliers.py)
3. Overbought-Oversold study on TESLA stock, [overbought-oversold.py](misc/overbought-oversold.py)
4. Which stock you need to buy? [which-stock.py](misc/which-stock.py)

### [Simulations](simulation)

1. Simple Monte Carlo, [monte-carlo-drift.py](simulation/monte-carlo-drift.py)
2. Dynamic volatility Monte Carlo, [monte-carlo-dynamic-volatility.py](simulation/monte-carlo-dynamic-volatility.py)
3. Drift Monte Carlo, [monte-carlo-drift.py](simulation/monte-carlo-drift.py)
4. Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment, [multivariate-drift-monte-carlo.py](simulation/multivariate-drift-monte-carlo.py)
5. Portfolio optimization, [portfolio-optimization.py](simulation/portfolio-optimization.py), inspired from https://pythonforfinance.net/2017/01/21/investment-portfolio-optimisation-with-python/

### [Tensorflow-js](stock-forecasting-js)

I code [LSTM Recurrent Neural Network](deep-learning/lstm.py) and [Simple signal rolling agent](agent/simple-agent.py) inside Tensorflow JS, you can try it here, [stock-forecasting-js](stock-forecasting-js/), you can download any historical CSV and upload dynamically.

### [Misc](misc)

1. fashion trending prediction with cross-validation, [fashion-forecasting.py](misc/fashion-forecasting.py)
2. Bitcoin analysis with LSTM prediction, [bitcoin-analysis-lstm.py](misc/bitcoin-analysis-lstm.py)
3. Kijang Emas Bank Negara, [kijang-emas-bank-negara.py](misc/kijang-emas-bank-negara.py)

## Results

### Results Agent

**This agent only able to buy or sell 1 unit per transaction.**

1. Turtle-trading agent, [turtle-agent.py](agent/turtle-agent.py)

<img src="output-agent/turtle-agent.png" width="70%" align="">

2. Moving-average agent, [moving-average-agent.py](agent/moving-average-agent.py)

<img src="output-agent/moving-average-agent.png" width="70%" align="">

3. Signal rolling agent, [signal-rolling-agent.py](agent/signal-rolling-agent.py)

<img src="output-agent/signal-rolling-agent.png" width="70%" align="">

4. Policy-gradient agent, [policy-gradient-agent.py](agent/policy-gradient-agent.py)

<img src="output-agent/policy-gradient-agent.png" width="70%" align="">

5. Q-learning agent, [q-learning-agent.py](agent/q-learning-agent.py)

<img src="output-agent/q-learning-agent.png" width="70%" align="">

6. Evolution-strategy agent, [evolution-strategy-agent.py](agent/evolution-strategy-agent.py)

<img src="output-agent/evolution-strategy-agent.png" width="70%" align="">

7. Double Q-learning agent, [double-q-learning-agent.py](agent/double-q-learning-agent.py)

<img src="output-agent/double-q-learning.png" width="70%" align="">

8. Recurrent Q-learning agent, [recurrent-q-learning-agent.py](agent/recurrent-q-learning-agent.py)

<img src="output-agent/recurrent-q-learning.png" width="70%" align="">

9. Double Recurrent Q-learning agent, [double-recurrent-q-learning-agent.py](agent/double-recurrent-q-learning-agent.py)

<img src="output-agent/double-recurrent-q-learning.png" width="70%" align="">

10. Duel Q-learning agent, [duel-q-learning-agent.py](agent/duel-q-learning-agent.py)

<img src="output-agent/double-q-learning.png" width="70%" align="">

11. Double Duel Q-learning agent, [double-duel-q-learning-agent.py](agent/double-duel-q-learning-agent.py)

<img src="output-agent/double-duel-q-learning.png" width="70%" align="">

12. Duel Recurrent Q-learning agent, [duel-recurrent-q-learning-agent.py](agent/duel-recurrent-q-learning-agent.py)

<img src="output-agent/duel-recurrent-q-learning.png" width="70%" align="">

13. Double Duel Recurrent Q-learning agent, [double-duel-recurrent-q-learning-agent.py](agent/double-duel-recurrent-q-learning-agent.py)

<img src="output-agent/double-duel-recurrent-q-learning.png" width="70%" align="">

14. Actor-critic agent, [actor-critic-agent.py](agent/14.actor-critic-agent.py)

<img src="output-agent/actor-critic.png" width="70%" align="">

15. Actor-critic Duel agent, [actor-critic-duel-agent.py](agent/actor-critic-duel-agent.py)

<img src="output-agent/actor-critic-duel.png" width="70%" align="">

16. Actor-critic Recurrent agent, [actor-critic-recurrent-agent.py](agent/actor-critic-recurrent-agent.py)

<img src="output-agent/actor-critic-recurrent.png" width="70%" align="">

17. Actor-critic Duel Recurrent agent, [actor-critic-duel-recurrent-agent.py](agent/actor-critic-duel-recurrent-agent.py)

<img src="output-agent/actor-critic-duel-recurrent.png" width="70%" align="">

18. Curiosity Q-learning agent, [curiosity-q-learning-agent.py](agent/curiosity-q-learning-agent.py)

<img src="output-agent/curiosity-q-learning.png" width="70%" align="">

19. Recurrent Curiosity Q-learning agent, [recurrent-curiosity-q-learning.py](agent/recurrent-curiosity-q-learning-agent.py)

<img src="output-agent/recurrent-curiosity-q-learning.png" width="70%" align="">

20. Duel Curiosity Q-learning agent, [duel-curiosity-q-learning-agent.py](agent/duel-curiosity-q-learning-agent.py)

<img src="output-agent/duel-curiosity-q-learning.png" width="70%" align="">

21. Neuro-evolution agent, [neuro-evolution.py](agent/neuro-evolution-agent.py)

<img src="output-agent/neuro-evolution.png" width="70%" align="">

22. Neuro-evolution with Novelty search agent, [neuro-evolution-novelty-search.py](agent/neuro-evolution-novelty-search-agent.py)

<img src="output-agent/neuro-evolution-novelty-search.png" width="70%" align="">

23. ABCD strategy agent, [abcd-strategy.py](agent/abcd-strategy-agent.py)

<img src="output-agent/abcd-strategy.png" width="70%" align="">

### Results signal prediction

I will cut the dataset to train and test datasets,

1. Train dataset derived from starting timestamp until last 30 days
2. Test dataset derived from last 30 days until end of the dataset

So we will let the model do forecasting based on last 30 days, and we will going to repeat the experiment for 10 times. You can increase it locally if you want, and tuning parameters will help you by a lot.

1. LSTM, accuracy 95.693%, time taken for 1 epoch 01:09

<img src="output/lstm.png" width="70%" align="">

2. LSTM Bidirectional, accuracy 93.8%, time taken for 1 epoch 01:40

<img src="output/bidirectional-lstm.png" width="70%" align="">

3. LSTM 2-Path, accuracy 94.63%, time taken for 1 epoch 01:39

<img src="output/lstm-2path.png" width="70%" align="">

4. GRU, accuracy 94.63%, time taken for 1 epoch 02:10

<img src="output/gru.png" width="70%" align="">

5. GRU Bidirectional, accuracy 92.5673%, time taken for 1 epoch 01:40

<img src="output/bidirectional-gru.png" width="70%" align="">

6. GRU 2-Path, accuracy 93.2117%, time taken for 1 epoch 01:39

<img src="output/gru-2path.png" width="70%" align="">

7. Vanilla, accuracy 91.4686%, time taken for 1 epoch 00:52

<img src="output/vanilla.png" width="70%" align="">

8. Vanilla Bidirectional, accuracy 88.9927%, time taken for 1 epoch 01:06

<img src="output/bidirectional-vanilla.png" width="70%" align="">

9. Vanilla 2-Path, accuracy 91.5406%, time taken for 1 epoch 01:08

<img src="output/vanilla-2path.png" width="70%" align="">

10. LSTM Seq2seq, accuracy 94.9817%, time taken for 1 epoch 01:36

<img src="output/lstm-seq2seq.png" width="70%" align="">

11. LSTM Bidirectional Seq2seq, accuracy 94.517%, time taken for 1 epoch 02:30

<img src="output/bidirectional-lstm-seq2seq.png" width="70%" align="">

12. LSTM Seq2seq VAE, accuracy 95.4190%, time taken for 1 epoch 01:48

<img src="output/lstm-seq2seq-vae.png" width="70%" align="">

13. GRU Seq2seq, accuracy 90.8854%, time taken for 1 epoch 01:34

<img src="output/gru-seq2seq.png" width="70%" align="">

14. GRU Bidirectional Seq2seq, accuracy 67.9915%, time taken for 1 epoch 02:30

<img src="output/bidirectional-gru-seq2seq.png" width="70%" align="">

15. GRU Seq2seq VAE, accuracy 89.1321%, time taken for 1 epoch 01:48

<img src="output/gru-seq2seq-vae.png" width="70%" align="">

16. Attention-is-all-you-Need, accuracy 94.2482%, time taken for 1 epoch 01:41

<img src="output/attention-is-all-you-need.png" width="70%" align="">

17. CNN-Seq2seq, accuracy 90.74%, time taken for 1 epoch 00:43

<img src="output/cnn-seq2seq.png" width="70%" align="">

18. Dilated-CNN-Seq2seq, accuracy 95.86%, time taken for 1 epoch 00:14

<img src="output/dilated-cnn-seq2seq.png" width="70%" align="">

**Bonus**

1. How to forecast,

<img src="output/how-to-forecast.png" width="70%" align="">

2. Sentiment consensus,

<img src="output/sentiment-consensus.png" width="70%" align="">

### Results analysis

1. Outliers study using K-means, SVM, and Gaussian on TESLA stock

<img src="misc/outliers.png" width="70%" align="">

2. Overbought-Oversold study on TESLA stock

<img src="misc/overbought-oversold.png" width="70%" align="">

3. Which stock you need to buy?

<img src="misc/which-stock.png" width="40%" align="">

### Results simulation

1. Simple Monte Carlo

<img src="simulation/monte-carlo-simple.png" width="70%" align="">

2. Dynamic volatity Monte Carlo

<img src="simulation/monte-carlo-dynamic-volatility.png" width="70%" align="">

3. Drift Monte Carlo

<img src="simulation/monte-carlo-drift.png" width="70%" align="">

4. Multivariate Drift Monte Carlo BTC/USDT with Bitcurate sentiment

<img src="simulation/multivariate-drift-monte-carlo.png" width="70%" align="">

5. Portfolio optimization

<img src="simulation/portfolio-optimization.png" width="40%" align="">
