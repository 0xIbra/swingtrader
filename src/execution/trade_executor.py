"""
Trade executor - places and manages trades via OANDA.
"""
from typing import Dict, Optional
from .broker_factory import get_broker
from ..data.database import TradeDatabase
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades and manages positions."""

    def __init__(self, broker=None):
        """
        Initialize trade executor.

        Args:
            broker: Broker client or None for default from config
        """
        self.broker = broker or get_broker()
        self.database = TradeDatabase()

        logger.info(f"TradeExecutor using: {type(self.broker).__name__}")

    def calculate_position_size(self, account_balance: float, risk_percent: float,
                               entry_price: float, stop_loss: float,
                               instrument: str) -> int:
        """
        Calculate position size based on risk parameters.

        Args:
            account_balance: Account balance in base currency
            risk_percent: Percentage of account to risk
            entry_price: Entry price
            stop_loss: Stop loss price
            instrument: Currency pair

        Returns:
            Number of units to trade
        """
        # Amount to risk
        risk_amount = account_balance * (risk_percent / 100)

        # Stop loss distance in price
        stop_distance = abs(entry_price - stop_loss)

        # Calculate units
        # For forex, 1 unit = 1 unit of base currency
        units = int(risk_amount / stop_distance)

        # Apply minimum/maximum limits
        units = max(units, 1000)  # Minimum 1000 units
        units = min(units, 1000000)  # Maximum 1M units

        logger.info(f"Position size: {units} units (risking {risk_amount:.2f})")

        return units

    def execute_trade(self, signal: Dict, dry_run: bool = False) -> Optional[Dict]:
        """
        Execute a trade signal.

        Args:
            signal: Trade signal dictionary from decision engine
            dry_run: If True, don't actually place the trade (paper trading)

        Returns:
            Trade execution result or None if failed
        """
        instrument = signal['instrument']
        direction = signal['direction']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        risk_percent = signal['risk_percent']

        logger.info(f"\n{'='*60}")
        logger.info(f"EXECUTING TRADE: {direction} {instrument}")
        logger.info(f"{'='*60}")

        try:
            # Get account info
            account_info = self.broker.get_account_summary()
            account_balance = account_info['balance']

            logger.info(f"Account balance: {account_balance:.2f}")

            # Calculate position size
            units = self.calculate_position_size(
                account_balance=account_balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss=stop_loss,
                instrument=instrument
            )

            # Negative units for SHORT positions
            if direction == 'SHORT':
                units = -units

            if dry_run:
                logger.info("DRY RUN - Trade not actually placed")
                order_response = {
                    'orderFillTransaction': {
                        'id': f'DRY_RUN_{datetime.now().timestamp()}',
                        'price': entry_price
                    }
                }
            else:
                # Place market order with SL and TP
                logger.info(f"Placing market order: {units} units")
                order_response = self.broker.place_market_order(
                    instrument=instrument,
                    units=units,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

            # Log trade to database
            trade_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'instrument': instrument,
                'direction': direction.lower(),
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'units': abs(units),
                'risk_percent': risk_percent,
                'confidence': signal['confidence'],
                'entry_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                # Pattern features
                'pattern_type': signal['details'].get('pattern_type', 'none'),
                'bounce_probability': signal['details'].get('bounce_prob', 0),
                'direction_probability': signal['details'].get('direction_prob', 0),

                # Market context
                'news_sentiment': signal['details'].get('news_sentiment', 0),
                'market_regime': signal['details'].get('market_regime', 'neutral'),

                # Technical features
                'atr': signal['details'].get('atr', 0),

                # Multi-timeframe
                'timeframe_alignment': signal['details'].get('timeframe_alignment', 0),
            }

            trade_id = self.database.log_trade_entry(trade_data)

            logger.info(f"✅ Trade executed successfully. Trade ID: {trade_id}")

            return {
                'success': True,
                'trade_id': trade_id,
                'order_id': order_response.get('orderFillTransaction', {}).get('id'),
                'filled_price': order_response.get('orderFillTransaction', {}).get('price', entry_price)
            }

        except Exception as e:
            logger.error(f"❌ Error executing trade: {e}")
            return None

    def close_trade(self, instrument: str, trade_id: int = None,
                   dry_run: bool = False) -> bool:
        """
        Close an open position.

        Args:
            instrument: Currency pair
            trade_id: Database trade ID (optional)
            dry_run: If True, don't actually close the position

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Closing position: {instrument}")

        try:
            # Get current position
            positions = self.broker.get_open_positions()

            position = None
            for pos in positions:
                if pos['instrument'] == instrument:
                    position = pos
                    break

            if not position:
                logger.warning(f"No open position for {instrument}")
                return False

            if dry_run:
                logger.info("DRY RUN - Position not actually closed")
                exit_price = 0.0
            else:
                # Close position
                close_response = self.broker.close_position(instrument)

                # Get exit price from response
                exit_price = close_response.get('longOrderFillTransaction', {}).get('price') or \
                           close_response.get('shortOrderFillTransaction', {}).get('price')

            # Update database if trade_id provided
            if trade_id:
                # Calculate outcome
                # This is simplified - in production you'd get actual P&L from broker
                exit_data = {
                    'exit_price': exit_price,
                    'exit_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'outcome': 'closed',  # Would determine win/loss based on P&L
                }

                self.database.update_trade_exit(trade_id, exit_data)

            logger.info(f"✅ Position closed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            return False

    def check_open_positions(self) -> list:
        """
        Check all open positions.

        Returns:
            List of open positions
        """
        try:
            positions = self.broker.get_open_positions()
            return positions
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return []

    def update_stop_loss(self, instrument: str, new_stop_loss: float) -> bool:
        """
        Update stop loss for an open position (e.g., trailing stop).

        Args:
            instrument: Currency pair
            new_stop_loss: New stop loss price

        Returns:
            True if successful
        """
        # This would require additional OANDA API calls
        # Left as placeholder for future implementation
        logger.info(f"Updating stop loss for {instrument} to {new_stop_loss}")
        return True

