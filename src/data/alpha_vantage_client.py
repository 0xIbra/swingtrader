"""
Alpha Vantage API client for forex data (data only, no trading).
Free tier: 25 requests/day, 5 requests/minute
Get API key: https://www.alphavantage.co/support/#api-key
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
import time
import config
import logging

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    """Client for Alpha Vantage Forex API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY must be set")

    def _convert_instrument(self, instrument: str) -> tuple:
        """Convert OANDA format to Alpha Vantage format."""
        # EUR_USD -> EUR, USD
        from_currency, to_currency = instrument.split('_')
        return from_currency, to_currency

    def _convert_granularity(self, granularity: str) -> str:
        """Convert OANDA granularity to Alpha Vantage interval."""
        mapping = {
            'M1': '1min',
            'M5': '5min',
            'M15': '15min',
            'M30': '30min',
            'H1': '60min',
            'H4': '60min',  # Will aggregate
            'D': 'daily'
        }
        return mapping.get(granularity, '60min')

    def get_candles(self, instrument: str, granularity: str = "H4",
                    count: int = 500) -> pd.DataFrame:
        """
        Fetch forex candles from Alpha Vantage.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Timeframe
            count: Number of candles (Alpha Vantage returns full dataset)

        Returns:
            DataFrame with OHLCV data
        """
        from_currency, to_currency = self._convert_instrument(instrument)
        interval = self._convert_granularity(granularity)

        # Choose function based on interval
        if interval == 'daily':
            function = 'FX_DAILY'
            params = {
                'function': function,
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
        else:
            function = 'FX_INTRADAY'
            params = {
                'function': function,
                'from_symbol': from_currency,
                'to_symbol': to_currency,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'full'
            }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            # Check for errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")

            # Parse time series data
            if 'Time Series FX (Daily)' in data:
                time_series = data['Time Series FX (Daily)']
            elif f'Time Series FX ({interval})' in data:
                time_series = data[f'Time Series FX ({interval})']
            else:
                raise ValueError(f"Unexpected response format: {list(data.keys())}")

            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'time': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 0  # Alpha Vantage doesn't provide forex volume
                })

            df = pd.DataFrame(df_data)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

            # Aggregate to H4 if needed
            if granularity == 'H4' and interval == '60min':
                df = self._aggregate_to_h4(df)

            # Limit to requested count
            df = df.tail(count)

            logger.info(f"Fetched {len(df)} bars for {instrument}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage: {e}")
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

        Note: Alpha Vantage returns all available data, so we just filter by date.

        Args:
            instrument: Currency pair
            granularity: Timeframe
            days_back: Number of days

        Returns:
            DataFrame with OHLCV data
        """
        df = self.get_candles(instrument, granularity, count=10000)

        if df.empty:
            return df

        # Filter to requested time period
        cutoff_date = datetime.now() - timedelta(days=days_back)
        df = df[df.index >= cutoff_date]

        return df

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current price (real-time quote).

        Args:
            instrument: Currency pair

        Returns:
            Dictionary with price info
        """
        from_currency, to_currency = self._convert_instrument(instrument)

        params = {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'from_currency': from_currency,
            'to_currency': to_currency,
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Realtime Currency Exchange Rate' in data:
                rate_data = data['Realtime Currency Exchange Rate']
                price = float(rate_data['5. Exchange Rate'])

                return {
                    'bid': price,
                    'ask': price,
                    'mid': price,
                    'time': pd.to_datetime(rate_data['6. Last Refreshed'])
                }
            else:
                raise ValueError("Unexpected response format")

        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            return {'bid': 0, 'ask': 0, 'mid': 0, 'time': datetime.now()}


# Note: Alpha Vantage doesn't support trading, only data
# For execution, you would need to use another broker

