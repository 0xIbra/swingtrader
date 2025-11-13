"""
Simulated broker for paper trading without external APIs.
Acts like a real broker but runs locally with real market data.
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from .data_provider_factory import get_data_client
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """
    Paper trading broker simulation with realistic execution.

    Features:
    - Uses real market data
    - Simulates realistic slippage and spreads
    - Tracks positions and P&L
    - Persists state between runs
    - No external API needed
    """

    def __init__(self, initial_balance: float = 10000,
                 spread_pips: float = 2.0,
                 slippage_pips: float = 1.0,
                 state_file: str = None):
        """
        Initialize simulated broker.

        Args:
            initial_balance: Starting account balance
            spread_pips: Bid-ask spread in pips
            slippage_pips: Execution slippage in pips
            state_file: File to persist broker state
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips

        self.positions = {}  # {instrument: position_data}
        self.orders = []  # Order history
        self.trades = []  # Trade history

        # State persistence
        from pathlib import Path
        import config
        self.state_file = state_file or str(config.DATA_DIR / "simulated_broker_state.json")

        # Data provider for real prices
        self.data_client = get_data_client()

        # Load previous state if exists
        self._load_state()

        logger.info(f"SimulatedBroker initialized: Balance=${self.balance:.2f}")

    def _pip_value(self, instrument: str, price: float) -> float:
        """Calculate pip value for an instrument."""
        # For most forex pairs, 1 pip = 0.0001
        # For JPY pairs, 1 pip = 0.01
        if 'JPY' in instrument:
            return 0.01
        return 0.0001

    def _calculate_spread(self, mid_price: float, instrument: str) -> tuple:
        """Calculate bid/ask prices with spread."""
        pip = self._pip_value(instrument, mid_price)
        spread_amount = pip * self.spread_pips / 2

        bid = mid_price - spread_amount
        ask = mid_price + spread_amount

        return bid, ask

    def _apply_slippage(self, price: float, instrument: str, direction: str) -> float:
        """Apply realistic slippage to execution price."""
        pip = self._pip_value(instrument, price)
        slippage_amount = pip * self.slippage_pips

        # Slippage works against you
        if direction == 'buy':
            return price + slippage_amount
        else:  # sell
            return price - slippage_amount

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """
        Get current market price for an instrument.

        Args:
            instrument: Currency pair

        Returns:
            Dictionary with bid, ask, mid prices
        """
        try:
            # Get real market data
            price_data = self.data_client.get_current_price(instrument)
            mid_price = price_data['mid']

            # Calculate realistic bid/ask with spread
            bid, ask = self._calculate_spread(mid_price, instrument)

            return {
                'bid': bid,
                'ask': ask,
                'mid': mid_price,
                'time': price_data['time']
            }
        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
            return {'bid': 0, 'ask': 0, 'mid': 0, 'time': datetime.now()}

    def get_account_summary(self) -> Dict:
        """
        Get account summary.

        Returns:
            Dictionary with balance, equity, margin info
        """
        # Calculate unrealized P&L from open positions
        unrealized_pl = 0
        for instrument, position in self.positions.items():
            try:
                current_price = self.get_current_price(instrument)

                if position['side'] == 'long':
                    price_diff = current_price['bid'] - position['entry_price']
                else:  # short
                    price_diff = position['entry_price'] - current_price['ask']

                unrealized_pl += price_diff * position['units']
            except:
                pass

        equity = self.balance + unrealized_pl

        return {
            'balance': self.balance,
            'nav': equity,
            'unrealized_pl': unrealized_pl,
            'margin_used': 0,  # Simplified
            'margin_available': equity,
            'open_trade_count': len(self.positions)
        }

    def place_market_order(self, instrument: str, units: int,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Dict:
        """
        Place a market order.

        Args:
            instrument: Currency pair
            units: Number of units (positive=long, negative=short)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order response dictionary
        """
        side = 'long' if units > 0 else 'short'
        abs_units = abs(units)

        # Get current price
        price_data = self.get_current_price(instrument)

        # Execute at bid (sell) or ask (buy)
        if side == 'long':
            execution_price = price_data['ask']
        else:
            execution_price = price_data['bid']

        # Apply slippage
        execution_price = self._apply_slippage(
            execution_price, instrument, 'buy' if side == 'long' else 'sell'
        )

        # Create order
        order_id = f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.orders)}"

        order = {
            'id': order_id,
            'instrument': instrument,
            'side': side,
            'units': abs_units,
            'entry_price': execution_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now().isoformat(),
            'status': 'filled'
        }

        self.orders.append(order)

        # Add to positions or close existing
        if instrument in self.positions:
            # Close or modify existing position
            existing = self.positions[instrument]
            if existing['side'] != side:
                # Closing position
                self._close_position_internal(instrument, execution_price)
            else:
                # Adding to position (average price)
                total_units = existing['units'] + abs_units
                avg_price = (existing['entry_price'] * existing['units'] +
                           execution_price * abs_units) / total_units
                existing['units'] = total_units
                existing['entry_price'] = avg_price
        else:
            # New position
            self.positions[instrument] = {
                'side': side,
                'units': abs_units,
                'entry_price': execution_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'opened_at': datetime.now().isoformat(),
                'order_id': order_id
            }

        self._save_state()

        logger.info(f"Order filled: {side.upper()} {abs_units} {instrument} @ {execution_price:.5f}")

        return {
            'orderFillTransaction': {
                'id': order_id,
                'price': execution_price,
                'units': units,
                'instrument': instrument,
                'time': datetime.now().isoformat()
            }
        }

    def _close_position_internal(self, instrument: str, exit_price: float) -> Dict:
        """Internal method to close a position."""
        if instrument not in self.positions:
            return {}

        position = self.positions[instrument]

        # Calculate P&L
        if position['side'] == 'long':
            pl = (exit_price - position['entry_price']) * position['units']
        else:
            pl = (position['entry_price'] - exit_price) * position['units']

        # Update balance
        self.balance += pl

        # Record trade
        trade = {
            'instrument': instrument,
            'side': position['side'],
            'units': position['units'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'profit': pl,
            'opened_at': position['opened_at'],
            'closed_at': datetime.now().isoformat()
        }

        self.trades.append(trade)

        # Remove position
        del self.positions[instrument]

        logger.info(f"Position closed: {instrument} P&L=${pl:.2f}")

        return trade

    def close_position(self, instrument: str, long_units: str = "ALL",
                      short_units: str = "ALL") -> Dict:
        """
        Close an open position.

        Args:
            instrument: Currency pair
            long_units: "ALL" or specific amount
            short_units: "ALL" or specific amount

        Returns:
            Close response
        """
        if instrument not in self.positions:
            logger.warning(f"No open position for {instrument}")
            return {}

        position = self.positions[instrument]

        # Get current price
        price_data = self.get_current_price(instrument)

        # Close at bid (for long) or ask (for short)
        if position['side'] == 'long':
            exit_price = price_data['bid']
        else:
            exit_price = price_data['ask']

        # Apply slippage
        exit_price = self._apply_slippage(
            exit_price, instrument, 'sell' if position['side'] == 'long' else 'buy'
        )

        trade = self._close_position_internal(instrument, exit_price)

        self._save_state()

        return {
            'longOrderFillTransaction' if position['side'] == 'long' else 'shortOrderFillTransaction': {
                'price': exit_price,
                'pl': trade.get('profit', 0)
            }
        }

    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of position dictionaries
        """
        positions = []

        for instrument, pos in self.positions.items():
            # Calculate unrealized P&L
            current_price = self.get_current_price(instrument)

            if pos['side'] == 'long':
                unrealized_pl = (current_price['bid'] - pos['entry_price']) * pos['units']
                long_units = pos['units']
                short_units = 0
            else:
                unrealized_pl = (pos['entry_price'] - current_price['ask']) * pos['units']
                long_units = 0
                short_units = pos['units']

            positions.append({
                'instrument': instrument,
                'long_units': long_units,
                'short_units': short_units,
                'unrealized_pl': unrealized_pl
            })

        return positions

    def check_stops_and_targets(self):
        """
        Check if any positions hit stop loss or take profit.
        Should be called periodically.
        """
        to_close = []

        for instrument, position in self.positions.items():
            try:
                current_price = self.get_current_price(instrument)

                if position['side'] == 'long':
                    current = current_price['bid']

                    # Check stop loss
                    if position['stop_loss'] and current <= position['stop_loss']:
                        logger.info(f"{instrument} hit stop loss: {current:.5f} <= {position['stop_loss']:.5f}")
                        to_close.append((instrument, position['stop_loss'], 'stop_loss'))

                    # Check take profit
                    elif position['take_profit'] and current >= position['take_profit']:
                        logger.info(f"{instrument} hit take profit: {current:.5f} >= {position['take_profit']:.5f}")
                        to_close.append((instrument, position['take_profit'], 'take_profit'))

                else:  # short
                    current = current_price['ask']

                    # Check stop loss
                    if position['stop_loss'] and current >= position['stop_loss']:
                        logger.info(f"{instrument} hit stop loss: {current:.5f} >= {position['stop_loss']:.5f}")
                        to_close.append((instrument, position['stop_loss'], 'stop_loss'))

                    # Check take profit
                    elif position['take_profit'] and current <= position['take_profit']:
                        logger.info(f"{instrument} hit take profit: {current:.5f} <= {position['take_profit']:.5f}")
                        to_close.append((instrument, position['take_profit'], 'take_profit'))

            except Exception as e:
                logger.error(f"Error checking stops for {instrument}: {e}")

        # Close positions that hit stops/targets
        for instrument, exit_price, reason in to_close:
            self._close_position_internal(instrument, exit_price)
            self._save_state()

    def _save_state(self):
        """Save broker state to disk."""
        state = {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'positions': self.positions,
            'orders': self.orders[-100:],  # Keep last 100
            'trades': self.trades[-100:],   # Keep last 100
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self):
        """Load broker state from disk."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)

                self.balance = state.get('balance', self.initial_balance)
                self.positions = state.get('positions', {})
                self.orders = state.get('orders', [])
                self.trades = state.get('trades', [])

                logger.info(f"Loaded previous state: Balance=${self.balance:.2f}, "
                          f"{len(self.positions)} open positions")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def reset(self):
        """Reset broker to initial state."""
        self.balance = self.initial_balance
        self.positions = {}
        self.orders = []
        self.trades = []
        self._save_state()
        logger.info("Broker reset to initial state")

    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)
        df['closed_at'] = pd.to_datetime(df['closed_at'])

        # Filter by date
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = df[df['closed_at'] >= cutoff]

        return df

