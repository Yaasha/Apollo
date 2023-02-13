#!/usr/bin/env python3

from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")

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

import state
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

import fire
from datetime import date, datetime, timezone
import alpaca_trade_api as tradeapi
import time
from notifications.discord import send_daily_performance_notification
import os


class Apollo(object):
    """Apollo controller class"""

    def __init__(self):
        self.ticker_list = DOW_30_TICKER
        self.action_dim = len(DOW_30_TICKER)

        self.state_dim = 1 + 2 + 3 * self.action_dim + len(INDICATORS) * self.action_dim

        self.ERL_PARAMS = {
            "learning_rate": 3e-6,
            "batch_size": 2048,
            "gamma": 0.985,
            "seed": 312,
            "net_dimension": [128, 64],
            "target_step": 5000,
            "eval_gap": 30,
            "eval_times": 1,
        }
        self.env = StockTradingEnv

        self.last_train_date = state.get("last_train_date")

    def train(self):
        # train on partial data
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=self.ticker_list,
            data_source="alpaca",
            time_interval="1Min",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=self.env,
            model_name="ppo",
            if_vix=True,
            API_KEY=API_KEY,
            API_SECRET=API_SECRET,
            API_BASE_URL=API_BASE_URL,
            erl_params=self.ERL_PARAMS,
            cwd="./papertrading_erl",  # current_working_dir
            break_step=1e5,
        )

        # test performance on the rest of the data
        account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=self.ticker_list,
            data_source="alpaca",
            time_interval="1Min",
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=self.env,
            model_name="ppo",
            if_vix=True,
            API_KEY=API_KEY,
            API_SECRET=API_SECRET,
            API_BASE_URL=API_BASE_URL,
            cwd="./papertrading_erl",
            net_dimension=self.ERL_PARAMS["net_dimension"],
        )
        cummulative_return_erl = (account_value_erl[-1] / account_value_erl[0]) - 1

        if cummulative_return_erl > 0 or self.last_train_date is None:
            print(
                "Trained model is profitable or model is missing. Retraining model..."
            )

            # train on all data
            train(
                start_date=TRAIN_START_DATE,
                end_date=TEST_END_DATE,
                ticker_list=self.ticker_list,
                data_source="alpaca",
                time_interval="1Min",
                technical_indicator_list=INDICATORS,
                drl_lib="elegantrl",
                env=self.env,
                model_name="ppo",
                if_vix=True,
                API_KEY=API_KEY,
                API_SECRET=API_SECRET,
                API_BASE_URL=API_BASE_URL,
                erl_params=self.ERL_PARAMS,
                cwd="./papertrading_erl_retrain",
                break_step=2e5,
            )
            self.last_train_date = state.set(
                "last_train_date", date.today().strftime("%Y-%m-%d")
            )
        else:
            print("Trained model is not profitable. Keeping old model...")

    def trade(self):
        paper_trading_erl = AlpacaPaperTrading(
            ticker_list=DOW_30_TICKER,
            time_interval="1Min",
            drl_lib="elegantrl",
            agent="ppo",
            cwd="./papertrading_erl_retrain",
            net_dim=self.ERL_PARAMS["net_dimension"],
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            API_KEY=API_KEY,
            API_SECRET=API_SECRET,
            API_BASE_URL=API_BASE_URL,
            tech_indicator_list=INDICATORS,
            turbulence_thresh=30,
            max_stock=1e2,
            fractional_shares=True,
        )
        paper_trading_erl.run()

    def plot_performance(self):
        df_erl, cumu_erl = alpaca_history(
            key=API_KEY,
            secret=API_SECRET,
            url=API_BASE_URL,
            start=PERFORMANCE_CHECK_START,  # must be within 1 month
            end=PERFORMANCE_CHECK_END,
        )  # change the date if error occurs

        df_djia, cumu_djia = DIA_history(start=PERFORMANCE_CHECK_START)
        df_erl.tail()

        returns_erl = cumu_erl - 1
        returns_dia = cumu_djia - 1
        returns_dia = returns_dia[: returns_erl.shape[0]]

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
        if not os.path.isdir("charts"):
            os.makedirs("charts")
        plt.savefig("charts/papertrading_stock.png")

    def run(self):
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )

        while True:
            # Check if market is open or opening in 2 hours or less. Wait 5 mins if not.
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(tzinfo=timezone.utc).timestamp()
            currTime = clock.timestamp.replace(tzinfo=timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            isOpen = self.alpaca.get_clock().is_open
            while not isOpen and timeToOpen > 120:
                print(f"Market is closed. {timeToOpen} mins till open. Waiting 5 mins.")
                time.sleep(60 * 5)

                clock = self.alpaca.get_clock()
                openingTime = clock.next_open.replace(tzinfo=timezone.utc).timestamp()
                currTime = clock.timestamp.replace(tzinfo=timezone.utc).timestamp()
                timeToOpen = int((openingTime - currTime) / 60)
                isOpen = self.alpaca.get_clock().is_open

            # Check if the last model was trained today. If not, train a new model.
            if (
                self.last_train_date is None
                or (
                    datetime.now() - datetime.strptime(self.last_train_date, "%Y-%m-%d")
                ).days
                >= 1
            ):
                print("Trained model, is out of date. Retraining model...")
                self.train()
            # Run the trading algorithm and plot performance
            print("Start trading...")
            self.trade()
            self.plot_performance()
            send_daily_performance_notification()


if __name__ == "__main__":
    fire.Fire(Apollo)
