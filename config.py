from datetime import date, timedelta
from pandas.tseries.offsets import BDay
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
API_BASE_URL = os.getenv("API_BASE_URL")
DATA_URL = os.getenv("DATA_URL")

last_business_day = date.today() - BDay(1)
days_ago = lambda x: (last_business_day - timedelta(days=x)).strftime("%Y-%m-%d")

TRAIN_START_DATE = days_ago(18)
TRAIN_END_DATE = days_ago(3)
TEST_START_DATE = days_ago(2)
TEST_END_DATE = days_ago(1)

PERFORMANCE_CHECK_START = "2023-02-10"  # must be within 1 month
PERFORMANCE_CHECK_END = date.today().strftime("%Y-%m-%d")
