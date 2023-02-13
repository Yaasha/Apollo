import requests
import json
from config import (
    DISCORD_WEBHOOK_URL,
    API_KEY,
    API_SECRET,
    API_BASE_URL,
    PERFORMANCE_CHECK_START,
    PERFORMANCE_CHECK_END,
)
import alpaca_trade_api as tradeapi
from datetime import datetime


def get_daily_performance(from_date=None):
    if from_date is None:
        from_date = PERFORMANCE_CHECK_END
    alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
    performance = alpaca.get_portfolio_history(
        date_start=PERFORMANCE_CHECK_START, date_end=from_date, timeframe="1D"
    )

    return {
        "equity": round(performance.equity[-1]),
        "day_profit_loss": round(performance.profit_loss[-1]),
        "day_profit_loss_pct": round(performance.profit_loss_pct[-1], 2),
        "overall_profit_loss": round(sum(performance.profit_loss)),
        "overall_profit_loss_pct": round(sum(performance.profit_loss_pct), 2),
    }


def send_daily_performance_notification(date=None):
    if DISCORD_WEBHOOK_URL:
        performance = get_daily_performance(from_date=date)
        day_profit = performance["day_profit_loss"] >= 0
        overall_profit = performance["day_profit_loss"] >= 0

        data = {
            "content": None,
            "embeds": [
                {
                    "title": "Portfolio performance",
                    "color": 4502544 if day_profit else 16711680,
                    "fields": [
                        {
                            "name": "Today",
                            "value": f"${performance['day_profit_loss']} ({performance['day_profit_loss_pct']}%) :chart_with_{'upwards' if day_profit else 'downwards'}_trend:",
                            "inline": True,
                        },
                        {
                            "name": "Overall",
                            "value": f"${performance['overall_profit_loss']} ({performance['overall_profit_loss_pct']}%) :chart_with_{'upwards' if overall_profit else 'downwards'}_trend:",
                            "inline": True,
                        },
                        {
                            "name": "Portfolio value",
                            "value": f"${performance['equity']} :moneybag:",
                            "inline": False,
                        },
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "image": {"url": "attachment://performance.png"},
                }
            ],
            "username": "Apollo",
            "attachments": [],
        }
        data = {"payload_json": json.dumps(data)}
        files = {
            "files[0]": (
                "performance.png",
                open("charts/papertrading_stock.png", "rb").read(),
            )
        }

        result = requests.post(DISCORD_WEBHOOK_URL, files=files, data=data)

        try:
            result.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(err)
        else:
            print("Payload delivered successfully, code {}.".format(result.status_code))
