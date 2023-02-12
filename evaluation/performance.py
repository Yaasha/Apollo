import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import yfinance as yf


def get_trading_days(start, end):
    nyse = tc.get_calendar("NYSE")
    df = nyse.sessions_in_range(
        pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
    )
    trading_days = []
    for day in df:
        trading_days.append(str(day)[:10])

    return trading_days


def alpaca_history(key, secret, url, start, end):
    api = tradeapi.REST(key, secret, url, "v2")
    trading_days = get_trading_days(start, end)
    df = pd.DataFrame()
    for day in trading_days:
        df = df.append(
            api.get_portfolio_history(date_start=day, timeframe="5Min").df.iloc[:78]
        )
    equities = df.equity.values
    cumu_returns = equities / equities[0]
    cumu_returns = cumu_returns[~np.isnan(cumu_returns)]

    return df, cumu_returns


def DIA_history(start):
    data_df = yf.download(["^DJI"], start=start, interval="5m")
    data_df = data_df.iloc[:]
    baseline_returns = data_df["Adj Close"].values / data_df["Adj Close"].values[0]
    return data_df, baseline_returns
