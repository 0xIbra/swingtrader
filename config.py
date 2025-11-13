"""
Configuration module for the swing trading system.
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data Provider Selection
# Options: 'oanda', 'alpha_vantage', 'yahoo', 'auto' (tries in order)
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "auto")

# Broker Selection
# Options: 'oanda', 'simulated' (paper trading without external API)
BROKER = os.getenv("BROKER", "simulated")

# OANDA Configuration
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_ENVIRONMENT = os.getenv("OANDA_ENVIRONMENT", "practice")

# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Trading pairs
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

# Timeframes
TIMEFRAMES = {
    "1H": "H1",
    "4H": "H4",
    "Daily": "D"
}

# News Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_SOURCES = [
    "https://www.forexlive.com/feed/news",
    "https://www.fxstreet.com/rss/news",
    "https://www.investing.com/rss/news.rss"
]

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Trading Parameters
BASE_RISK_PERCENT = float(os.getenv("BASE_RISK_PERCENT", 1.0))
MAX_RISK_PERCENT = float(os.getenv("MAX_RISK_PERCENT", 2.0))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))

# Model Parameters
BOUNCE_MODEL_THRESHOLD = 0.65
DIRECTION_MODEL_THRESHOLD = 0.55
TIMEFRAME_ALIGNMENT_THRESHOLD = 0.66

# Risk Management
STOP_LOSS_ATR_MULTIPLIER = 1.5
TAKE_PROFIT_ATR_MULTIPLIER = 2.5

# Feature Engineering Parameters
LOOKBACK_BARS = 100
SUPPORT_RESISTANCE_TOLERANCE = 0.02  # 2%
PATTERN_MIN_TOUCHES = 2

# Database
DB_PATH = DATA_DIR / "trade_history.db"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "swingtrader.log"

