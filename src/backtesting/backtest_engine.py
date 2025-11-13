"""
Backtesting engine for validating trading strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from ..models.bounce_predictor import BouncePredictor
from ..models.direction_predictor import DirectionPredictor
from ..features.feature_builder import FeatureBuilder
from ..data.database import TradeDatabase
import config
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtests trading strategy on historical data."""

    def __init__(self, bounce_model_path: str = None, direction_model_path: str = None):
        """
        Initialize backtest engine.

        Args:
            bounce_model_path: Path to bounce predictor model
            direction_model_path: Path to direction predictor model
        """
        bounce_path = bounce_model_path or str(config.MODELS_DIR / "bounce_predictor.joblib")
        direction_path = direction_model_path or str(config.MODELS_DIR / "direction_predictor.joblib")

        self.bounce_predictor = BouncePredictor(bounce_path)
        self.direction_predictor = DirectionPredictor(direction_path)
        self.feature_builder = FeatureBuilder()

        # Backtest state
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 10000  # Default starting balance
        self.current_balance = self.initial_balance

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data with features for backtesting.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        df = self.feature_builder.build_training_features(df)
        return df

    def evaluate_bar(self, df: pd.DataFrame, idx: int) -> Dict:
        """
        Evaluate a single bar for trade signals.

        Args:
            df: DataFrame with features
            idx: Index of current bar

        Returns:
            Trade signal or None
        """
        # Need enough history
        if idx < 100:
            return None

        # Get data up to current bar (walk-forward)
        df_subset = df.iloc[:idx+1]
        current = df_subset.iloc[-1]

        # Check if near support/resistance
        distance_to_support = current.get('distance_to_support_pct', float('inf'))
        distance_to_resistance = current.get('distance_to_resistance_pct', float('inf'))

        near_support = distance_to_support < 0.5
        near_resistance = distance_to_resistance < 0.5

        if not (near_support or near_resistance):
            return None

        # Extract features
        features = {
            'support_level_strength': current.get('support_strength', 0),
            'resistance_level_strength': current.get('resistance_strength', 0),
            'distance_to_support_pct': distance_to_support if not pd.isna(distance_to_support) else 0,
            'distance_to_resistance_pct': distance_to_resistance if not pd.isna(distance_to_resistance) else 0,
            'price_volatility': current.get('price_volatility', 0),
            'rsi_14': current.get('rsi_14', 50),
            'momentum_20': current.get('momentum_20', 0),
            'macd_histogram': current.get('macd_histogram', 0),
            'volume_ratio': current.get('volume_ratio', 1),
            'volume_trend': current.get('volume_trend', 0),
            'trend': current.get('trend', 0)
        }

        # Get predictions
        bounce_prob = self.bounce_predictor.predict_single(features)
        direction_probs = self.direction_predictor.predict_single(features)

        # Apply thresholds
        if bounce_prob < config.BOUNCE_MODEL_THRESHOLD:
            return None

        # Determine direction
        if near_support:
            direction = 'LONG'
            direction_prob = direction_probs['prob_up']
        else:
            direction = 'SHORT'
            direction_prob = direction_probs['prob_down']

        if direction_prob < config.DIRECTION_MODEL_THRESHOLD:
            return None

        # Check trend alignment
        trend = current.get('trend', 0)
        if direction == 'LONG' and trend == -1:
            return None
        if direction == 'SHORT' and trend == 1:
            return None

        # Generate signal
        entry_price = current['close']
        atr = current['atr']

        if direction == 'LONG':
            stop_loss = entry_price - (config.STOP_LOSS_ATR_MULTIPLIER * atr)
            take_profit = entry_price + (config.TAKE_PROFIT_ATR_MULTIPLIER * atr)
        else:
            stop_loss = entry_price + (config.STOP_LOSS_ATR_MULTIPLIER * atr)
            take_profit = entry_price - (config.TAKE_PROFIT_ATR_MULTIPLIER * atr)

        return {
            'entry_idx': idx,
            'entry_time': current.name,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'bounce_prob': bounce_prob,
            'direction_prob': direction_prob
        }

    def simulate_trade(self, trade: Dict, df: pd.DataFrame) -> Dict:
        """
        Simulate trade execution and outcome.

        Args:
            trade: Trade signal dictionary
            df: Full DataFrame

        Returns:
            Trade outcome
        """
        entry_idx = trade['entry_idx']
        direction = trade['direction']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']

        # Look forward to see if SL or TP hit
        max_bars_forward = 50  # Maximum bars to hold trade

        exit_price = None
        exit_idx = None
        exit_reason = None

        for i in range(entry_idx + 1, min(entry_idx + max_bars_forward + 1, len(df))):
            bar = df.iloc[i]

            if direction == 'LONG':
                # Check if stop loss hit
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_idx = i
                    exit_reason = 'stop_loss'
                    break
                # Check if take profit hit
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break
            else:  # SHORT
                # Check if stop loss hit
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_idx = i
                    exit_reason = 'stop_loss'
                    break
                # Check if take profit hit
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    exit_idx = i
                    exit_reason = 'take_profit'
                    break

        # If no exit, close at market after max bars
        if exit_price is None:
            exit_idx = min(entry_idx + max_bars_forward, len(df) - 1)
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'timeout'

        # Calculate profit
        if direction == 'LONG':
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - exit_price) / entry_price) * 100

        outcome = 'win' if profit_pct > 0 else 'loss'

        # Calculate position size based on risk
        risk_amount = self.current_balance * (config.BASE_RISK_PERCENT / 100)
        stop_distance = abs(entry_price - stop_loss)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0

        # Calculate actual P&L
        profit_amount = (profit_pct / 100) * (position_size * entry_price)

        # Update balance
        self.current_balance += profit_amount

        return {
            **trade,
            'exit_idx': exit_idx,
            'exit_time': df.iloc[exit_idx].name,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'outcome': outcome,
            'balance_after': self.current_balance
        }

    def run_backtest(self, df: pd.DataFrame, start_date: str = None,
                    end_date: str = None) -> Dict:
        """
        Run full backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")

        # Prepare data
        df = self.prepare_data(df)

        # Filter date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        logger.info(f"Backtesting on {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        # Reset state
        self.trades = []
        self.equity_curve = []
        self.current_balance = self.initial_balance

        # Track active trades (for preventing multiple simultaneous trades)
        active_trade = None

        # Iterate through bars
        for idx in range(len(df)):
            # Update equity curve
            self.equity_curve.append({
                'time': df.iloc[idx].name,
                'balance': self.current_balance
            })

            # Skip if we have an active trade
            if active_trade and idx <= active_trade['exit_idx']:
                continue

            # Look for new signal
            signal = self.evaluate_bar(df, idx)

            if signal:
                # Simulate trade
                trade_result = self.simulate_trade(signal, df)
                self.trades.append(trade_result)
                active_trade = trade_result

                logger.info(f"Trade {len(self.trades)}: {trade_result['direction']} "
                          f"at {trade_result['entry_price']:.5f}, "
                          f"exit {trade_result['exit_price']:.5f}, "
                          f"profit: {trade_result['profit_pct']:+.2f}%")

        # Calculate statistics
        results = self.calculate_statistics()

        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Average Profit: {results['avg_profit']:.2f}%")
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info("=" * 60)

        return results

    def calculate_statistics(self) -> Dict:
        """
        Calculate backtest statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(trades_df)
        wins = (trades_df['outcome'] == 'win').sum()
        win_rate = wins / total_trades
        avg_profit = trades_df['profit_pct'].mean()

        # Returns
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100

        # Max drawdown
        equity_curve_df = pd.DataFrame(self.equity_curve)
        equity_curve_df['peak'] = equity_curve_df['balance'].cummax()
        equity_curve_df['drawdown'] = ((equity_curve_df['balance'] - equity_curve_df['peak']) /
                                       equity_curve_df['peak']) * 100
        max_drawdown = equity_curve_df['drawdown'].min()

        # Sharpe ratio (simplified)
        returns = trades_df['profit_pct'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': total_trades - wins,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': trades_df['profit_pct'].max(),
            'worst_trade': trades_df['profit_pct'].min(),
            'trades_df': trades_df,
            'equity_curve_df': pd.DataFrame(self.equity_curve)
        }

