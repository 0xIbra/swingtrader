"""
OANDA API client for fetching price data and placing orders.
"""
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import config


class OANDAClient:
    """Client for interacting with OANDA API."""

    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None,
                 environment: Optional[str] = None):
        """
        Initialize OANDA client.

        Args:
            api_key: OANDA API key (defaults to config)
            account_id: OANDA account ID (defaults to config)
            environment: 'practice' or 'live' (defaults to config)
        """
        self.api_key = api_key or config.OANDA_API_KEY
        self.account_id = account_id or config.OANDA_ACCOUNT_ID
        self.environment = environment or config.OANDA_ENVIRONMENT

        if not self.api_key or not self.account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")

        self.client = oandapyV20.API(
            access_token=self.api_key,
            environment=self.environment
        )

    def get_candles(self, instrument: str, granularity: str = "H4",
                    count: int = 500, from_time: Optional[datetime] = None,
                    to_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch historical candles for an instrument.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            granularity: Timeframe (e.g., "H1", "H4", "D")
            count: Number of candles to fetch (max 5000)
            from_time: Start time for historical data
            to_time: End time for historical data

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        params = {
            "granularity": granularity,
            "count": count
        }

        if from_time:
            params["from"] = from_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        if to_time:
            params["to"] = to_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        request = instruments.InstrumentsCandles(instrument=instrument, params=params)
        response = self.client.request(request)

        # Parse response into DataFrame
        candles_data = []
        for candle in response['candles']:
            if candle['complete']:  # Only use completed candles
                candles_data.append({
                    'time': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })

        df = pd.DataFrame(candles_data)
        if not df.empty:
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

        return df

    def get_historical_data(self, instrument: str, granularity: str = "H4",
                           days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical data by making multiple API calls if needed.

        Args:
            instrument: Currency pair
            granularity: Timeframe
            days_back: Number of days of historical data to fetch

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        to_time = datetime.utcnow()
        from_time = to_time - timedelta(days=days_back)

        # Determine how many candles we can get per request
        candles_per_request = 5000

        while from_time < to_time:
            df = self.get_candles(
                instrument=instrument,
                granularity=granularity,
                count=candles_per_request,
                from_time=from_time,
                to_time=to_time
            )

            if df.empty:
                break

            all_data.append(df)
            # Move the from_time forward
            from_time = df.index[-1]

            # Break if we got fewer candles than requested (reached the end)
            if len(df) < candles_per_request:
                break

        if all_data:
            combined_df = pd.concat(all_data)
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.sort_index(inplace=True)
            return combined_df
        else:
            return pd.DataFrame()

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current bid/ask prices for an instrument.

        Args:
            instrument: Currency pair

        Returns:
            Dictionary with 'bid', 'ask', 'mid' prices
        """
        params = {"instruments": instrument}
        request = instruments.InstrumentsPricing(accountID=self.account_id, params=params)
        response = self.client.request(request)

        price_data = response['prices'][0]
        return {
            'bid': float(price_data['bids'][0]['price']),
            'ask': float(price_data['asks'][0]['price']),
            'mid': (float(price_data['bids'][0]['price']) + float(price_data['asks'][0]['price'])) / 2,
            'time': pd.to_datetime(price_data['time'])
        }

    def get_account_summary(self) -> Dict:
        """
        Get account summary information.

        Returns:
            Dictionary with account balance, equity, margin, etc.
        """
        request = accounts.AccountSummary(accountID=self.account_id)
        response = self.client.request(request)

        account = response['account']
        return {
            'balance': float(account['balance']),
            'nav': float(account['NAV']),
            'unrealized_pl': float(account.get('unrealizedPL', 0)),
            'margin_used': float(account.get('marginUsed', 0)),
            'margin_available': float(account.get('marginAvailable', 0)),
            'open_trade_count': int(account.get('openTradeCount', 0))
        }

    def place_market_order(self, instrument: str, units: int,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """
        Place a market order.

        Args:
            instrument: Currency pair
            units: Number of units (positive for long, negative for short)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order response dictionary
        """
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",  # Fill or Kill
                "positionFill": "DEFAULT"
            }
        }

        # Add stop loss if provided
        if stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(round(stop_loss, 5))
            }

        # Add take profit if provided
        if take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(round(take_profit, 5))
            }

        request = orders.OrderCreate(accountID=self.account_id, data=order_data)
        response = self.client.request(request)

        return response

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        request = accounts.AccountSummary(accountID=self.account_id)
        response = self.client.request(request)

        positions = []
        if 'positions' in response['account']:
            for pos in response['account']['positions']:
                if float(pos['long']['units']) != 0 or float(pos['short']['units']) != 0:
                    positions.append({
                        'instrument': pos['instrument'],
                        'long_units': float(pos['long']['units']),
                        'short_units': float(pos['short']['units']),
                        'unrealized_pl': float(pos['unrealizedPL'])
                    })

        return positions

    def close_position(self, instrument: str, long_units: str = "ALL",
                       short_units: str = "ALL") -> Dict:
        """
        Close an open position.

        Args:
            instrument: Currency pair
            long_units: Number of long units to close ("ALL" for all)
            short_units: Number of short units to close ("ALL" for all)

        Returns:
            Response dictionary
        """
        data = {
            "longUnits": long_units,
            "shortUnits": short_units
        }

        request = orders.PositionClose(accountID=self.account_id,
                                       instrument=instrument,
                                       data=data)
        response = self.client.request(request)

        return response

