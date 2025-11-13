"""
Yahoo Finance client using yfinance (completely free, no API key needed).
Good for backtesting and development. No live trading support.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Client for Yahoo Finance data (via yfinance)."""

    def __init__(self):
        """Initialize Yahoo Finance client."""
        pass

    def _convert_instrument(self, instrument: str) -> str:
        """
        Convert OANDA format to Yahoo Finance format.

        EUR_USD -> EURUSD=X
        """
        base, quote = instrument.split('_')
        return f"{base}{quote}=X"

    def _convert_granularity(self, granularity: str) -> str:
        """Convert OANDA granularity to yfinance interval."""
        mapping = {
            'M1': '1m',
            'M5': '5m',
            'M15': '15m',
            'M30': '30m',
            'H1': '1h',
            'H4': '1h',  # Will aggregate
            'D': '1d'
        }
        return mapping.get(granularity, '1h')

    def get_candles(self, instrument: str, granularity: str = "H4",
                    count: int = 500, from_time: Optional[datetime] = None,
                    to_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch forex candles from Yahoo Finance.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Timeframe
            count: Number of candles
            from_time: Start time
            to_time: End time

        Returns:
            DataFrame with OHLCV data
        """
        yahoo_symbol = self._convert_instrument(instrument)
        interval = self._convert_granularity(granularity)

        # Calculate period
        if from_time and to_time:
            start = from_time
            end = to_time
        elif count:
            # Estimate how much data to fetch based on count
            if granularity == 'D':
                days = count
            elif granularity == 'H4':
                days = count // 6  # ~6 4H bars per day
            elif granularity == 'H1':
                days = count // 24
            else:
                days = count // 100

            end = datetime.now()
            start = end - timedelta(days=days + 10)  # Add buffer
        else:
            end = datetime.now()
            start = end - timedelta(days=365)

        try:
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {instrument}")
                return pd.DataFrame()

            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Aggregate to H4 if needed
            if granularity == 'H4' and interval == '1h':
                df = self._aggregate_to_h4(df)

            # Limit to requested count
            df = df.tail(count)

            # Reset index name
            df.index.name = 'time'

            logger.info(f"Fetched {len(df)} bars for {instrument} from Yahoo Finance")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()

    def _aggregate_to_h4(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1H bars to 4H bars."""
        df_4h = pd.DataFrame({
            'open': df['open'].resample('4H').first(),
            'high': df['high'].resample('4H').max(),
            'low': df['low'].resample('4H').min(),
            'close': df['close'].resample('4H').last(),
            'volume': df['volume'].resample('4H').sum()
        })
        return df_4h.dropna()

    def get_historical_data(self, instrument: str, granularity: str = "H4",
                           days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical data.

        Args:
            instrument: Currency pair
            granularity: Timeframe
            days_back: Number of days

        Returns:
            DataFrame with OHLCV data
        """
        yahoo_symbol = self._convert_instrument(instrument)
        interval = self._convert_granularity(granularity)

        end = datetime.now()
        start = end - timedelta(days=days_back)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                return df

            # Rename columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Aggregate if needed
            if granularity == 'H4' and interval == '1h':
                df = self._aggregate_to_h4(df)

            df.index.name = 'time'

            logger.info(f"Fetched {len(df)} bars for {instrument}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current price.

        Args:
            instrument: Currency pair

        Returns:
            Dictionary with price info
        """
        yahoo_symbol = self._convert_instrument(instrument)

        try:
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period='1d', interval='1m')

            if not data.empty:
                price = data['Close'].iloc[-1]
                return {
                    'bid': price,
                    'ask': price,
                    'mid': price,
                    'time': data.index[-1]
                }
            else:
                return {'bid': 0, 'ask': 0, 'mid': 0, 'time': datetime.now()}

        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return {'bid': 0, 'ask': 0, 'mid': 0, 'time': datetime.now()}

