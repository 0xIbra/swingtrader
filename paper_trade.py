"""
Paper trading script using simulated broker.
This runs real-time paper trading with live market data but simulated execution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.simulated_broker import SimulatedBroker
from src.engine.decision_engine import DecisionEngine
from src.execution.trade_executor import TradeExecutor
from src.monitoring.logger_config import setup_logging
from src.monitoring.telegram_bot import TelegramNotifier
import config
import logging
import time
import schedule

logger = setup_logging()


class PaperTradingBot:
    """Paper trading bot using simulated broker."""

    def __init__(self, initial_balance: float = 10000):
        """
        Initialize paper trading bot.

        Args:
            initial_balance: Starting balance for simulation
        """
        self.broker = SimulatedBroker(initial_balance=initial_balance)
        self.decision_engine = DecisionEngine()
        self.trade_executor = TradeExecutor(broker=self.broker)
        self.telegram = TelegramNotifier()

        logger.info("="*70)
        logger.info("PAPER TRADING BOT (SIMULATED BROKER)")
        logger.info(f"Initial Balance: ${initial_balance:,.2f}")
        logger.info(f"Data Provider: {type(self.decision_engine.oanda_client).__name__}")
        logger.info("="*70)

    def scan_and_trade(self):
        """Scan for setups and execute paper trades."""
        logger.info("\n" + "="*70)
        logger.info(f"SCANNING FOR SETUPS - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)

        # Check stops and targets on existing positions
        self.broker.check_stops_and_targets()

        try:
            # Scan for signals
            signals = self.decision_engine.scan_for_setups()

            if not signals:
                logger.info("No trade signals found")
                return

            logger.info(f"\n✅ Found {len(signals)} trade signal(s)")

            # Execute each signal
            for signal in signals:
                # Send Telegram notification
                self.telegram.send_trade_signal(signal)

                # Execute trade (dry_run=False since it's already simulated)
                result = self.trade_executor.execute_trade(signal, dry_run=False)

                if result and result['success']:
                    self.telegram.send_trade_executed(
                        result['trade_id'],
                        signal['instrument'],
                        signal['direction']
                    )

        except Exception as e:
            logger.error(f"Error in scan_and_trade: {e}", exc_info=True)
            self.telegram.send_error(f"Error in paper trading bot: {str(e)}")

    def show_status(self):
        """Show current account status and positions."""
        logger.info("\n" + "="*70)
        logger.info("ACCOUNT STATUS")
        logger.info("="*70)

        account = self.broker.get_account_summary()
        logger.info(f"Balance: ${account['balance']:,.2f}")
        logger.info(f"Equity: ${account['nav']:,.2f}")
        logger.info(f"Unrealized P&L: ${account['unrealized_pl']:+,.2f}")
        logger.info(f"Open Positions: {account['open_trade_count']}")

        # Show positions
        if account['open_trade_count'] > 0:
            logger.info("\nOpen Positions:")
            positions = self.broker.get_open_positions()
            for pos in positions:
                logger.info(f"  {pos['instrument']}: "
                          f"Long={pos['long_units']}, Short={pos['short_units']}, "
                          f"P&L=${pos['unrealized_pl']:+,.2f}")

        # Show recent trades
        trades_df = self.broker.get_trade_history(days=7)
        if not trades_df.empty:
            logger.info(f"\nRecent Trades (last 7 days): {len(trades_df)}")
            total_pl = trades_df['profit'].sum()
            wins = (trades_df['profit'] > 0).sum()
            win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
            logger.info(f"  Total P&L: ${total_pl:+,.2f}")
            logger.info(f"  Win Rate: {win_rate:.1%}")

        logger.info("="*70)

    def daily_summary(self):
        """Send daily performance summary."""
        logger.info("Generating daily summary...")

        try:
            trades_df = self.broker.get_trade_history(days=1)

            if not trades_df.empty:
                total_trades = len(trades_df)
                wins = (trades_df['profit'] > 0).sum()
                losses = total_trades - wins
                total_profit = trades_df['profit'].sum()
                win_rate = wins / total_trades if total_trades > 0 else 0

                summary = {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'best_trade': trades_df['profit'].max(),
                    'worst_trade': trades_df['profit'].min()
                }

                self.telegram.send_daily_summary(summary)
                logger.info("Daily summary sent")
            else:
                logger.info("No trades today")

        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")

    def run_scheduled(self):
        """Run bot on a schedule."""
        # Scan for setups every 4 hours
        schedule.every(4).hours.do(self.scan_and_trade)

        # Check status every hour
        schedule.every(1).hours.do(self.show_status)

        # Check stops/targets every 15 minutes
        schedule.every(15).minutes.do(self.broker.check_stops_and_targets)

        # Daily summary at 00:00
        schedule.every().day.at("00:00").do(self.daily_summary)

        logger.info("Paper trading scheduler started. Press Ctrl+C to stop.")

        # Run once immediately
        self.show_status()
        self.scan_and_trade()

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_once(self):
        """Run bot once and exit."""
        self.show_status()
        self.scan_and_trade()
        self.show_status()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trading Bot (Simulated)')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial balance (default: 10000)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (default is scheduled)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset broker to initial state')

    args = parser.parse_args()

    # Initialize bot
    bot = PaperTradingBot(initial_balance=args.balance)

    # Reset if requested
    if args.reset:
        bot.broker.reset()
        logger.info("Broker state reset")
        return

    try:
        if args.once:
            bot.run_once()
        else:
            bot.run_scheduled()
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
        bot.show_status()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

