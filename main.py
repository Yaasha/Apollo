# %%
from __future__ import annotations

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from evaluation.performance import alpaca_history, DIA_history

from trading.alpaca import AlpacaPaperTrading

from train import train
from test import test

from config import (
    API_KEY,
    API_SECRET,
    API_BASE_URL,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    PERFORMANCE_CHECK_START,
    PERFORMANCE_CHECK_END,
)


# %%
ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)

print(ticker_list)
print(INDICATORS)

# %%
# amount + (turbulence, turbulence_bool) + (price, shares, cd (holding time)) * stock_dim + tech_dim
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim

print(state_dim)

# %%
# DP = DataProcessor(data_source = 'alpaca',
#                  API_KEY = API_KEY,
#                  API_SECRET = API_SECRET,
#                  API_BASE_URL = API_BASE_URL
#                  )

# data['timestamp'].nunique()

# data = DP.clean_data(data)
# data = DP.add_technical_indicator(data, INDICATORS)
# data = DP.add_vix(data)

# data.shape

# price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix=True)

# price_array

# %%
ERL_PARAMS = {
    "learning_rate": 3e-6,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": [128, 64],
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}
env = StockTradingEnv
# if you want to use larger datasets (change to longer period), and it raises error,
# please try to increase "target_step". It should be larger than the episode steps.

# %%
train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl",  # current_working_dir
    break_step=1e5,
)


# %%
account_value_erl = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    cwd="./papertrading_erl",
    net_dimension=ERL_PARAMS["net_dimension"],
)


# %%
train(
    start_date=TRAIN_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=ticker_list,
    data_source="alpaca",
    time_interval="1Min",
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    if_vix=True,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    erl_params=ERL_PARAMS,
    cwd="./papertrading_erl_retrain",
    break_step=2e5,
)


# %%
print(DOW_30_TICKER)


# %%
state_dim


# %%
action_dim


# %%
paper_trading_erl = AlpacaPaperTrading(
    ticker_list=DOW_30_TICKER,
    time_interval="1Min",
    drl_lib="elegantrl",
    agent="ppo",
    cwd="./papertrading_erl_retrain",
    net_dim=ERL_PARAMS["net_dimension"],
    state_dim=state_dim,
    action_dim=action_dim,
    API_KEY=API_KEY,
    API_SECRET=API_SECRET,
    API_BASE_URL=API_BASE_URL,
    tech_indicator_list=INDICATORS,
    turbulence_thresh=30,
    max_stock=1e2,
    fractional_shares=True,
)
paper_trading_erl.run()


# %%
df_erl, cumu_erl = alpaca_history(
    key=API_KEY,
    secret=API_SECRET,
    url=API_BASE_URL,
    start=PERFORMANCE_CHECK_START,  # must be within 1 month
    end=PERFORMANCE_CHECK_END,
)  # change the date if error occurs


# %%
df_djia, cumu_djia = DIA_history(start=PERFORMANCE_CHECK_START)


# %%
df_erl.tail()


# %%
returns_erl = cumu_erl - 1
returns_dia = cumu_djia - 1
returns_dia = returns_dia[: returns_erl.shape[0]]
print("len of erl return: ", returns_erl.shape[0])
print("len of dia return: ", returns_dia.shape[0])


# %%
returns_erl


# %%
plt.figure(dpi=200)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("Stock Trading (Paper trading)", fontsize=20)
plt.plot(returns_erl, label="ElegantRL Agent", color="red")
# plt.plot(returns_sb3, label = 'Stable-Baselines3 Agent', color = 'blue' )
# plt.plot(returns_rllib, label = 'RLlib Agent', color = 'green')
plt.plot(returns_dia, label="DJIA", color="grey")
plt.ylabel("Return", fontsize=16)
plt.xlabel("Time", fontsize=16)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(78))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(6))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
x_labels = df_erl.index.map(lambda x: x.strftime("%y-%m-%d %H:%M")).tolist()
ax.xaxis.set_major_formatter(ticker.FixedFormatter([""] + x_labels[::78]))

plt.legend(fontsize=10.5)
plt.savefig("charts/papertrading_stock.png")

# %%
