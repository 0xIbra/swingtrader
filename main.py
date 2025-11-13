"""
Main entry point for the swing trading system.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.engine.decision_engine import DecisionEngine
from src.execution.trade_executor import TradeExecutor
from src.monitoring.logger_config import setup_logging
from src.monitoring.telegram_bot import TelegramNotifier
from src.data.oanda_client import OANDAClient
import config
import logging
import time
import schedule

# Setup logging
logger = setup_logging()


class SwingTradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, dry_run: bool = True):
        """
        Initialize trading bot.

        Args:
            dry_run: If True, don't actually place trades
        """
        self.dry_run = dry_run
        self.decision_engine = DecisionEngine()
        self.trade_executor = TradeExecutor()
        self.telegram = TelegramNotifier()
        self.oanda_client = OANDAClient()

        logger.info("="*60)
        logger.info("SWING TRADING BOT INITIALIZED")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
        logger.info("="*60)

    def scan_and_trade(self):
        """Scan for setups and execute trades."""
        logger.info("\n" + "="*60)
        logger.info(f"SCANNING FOR SETUPS - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

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

                # Execute trade
                result = self.trade_executor.execute_trade(signal, dry_run=self.dry_run)

                if result and result['success']:
                    self.telegram.send_trade_executed(
                        result['trade_id'],
                        signal['instrument'],
                        signal['direction']
                    )
                else:
                    self.telegram.send_error(f"Failed to execute trade for {signal['instrument']}")

        except Exception as e:
            logger.error(f"Error in scan_and_trade: {e}", exc_info=True)
            self.telegram.send_error(f"Error in trading bot: {str(e)}")

    def check_positions(self):
        """Check and manage open positions."""
        logger.info("Checking open positions...")

        try:
            positions = self.trade_executor.check_open_positions()

            if positions:
                logger.info(f"Open positions: {len(positions)}")
                for pos in positions:
                    logger.info(f"  {pos['instrument']}: "
                              f"Long={pos['long_units']}, Short={pos['short_units']}, "
                              f"P&L={pos['unrealized_pl']:.2f}")
            else:
                logger.info("No open positions")

        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    def daily_summary(self):
        """Send daily performance summary."""
        logger.info("Generating daily summary...")

        try:
            summary = self.decision_engine.database.get_performance_summary(days_back=1)

            if summary['total_trades'] > 0:
                self.telegram.send_daily_summary(summary)
                logger.info("Daily summary sent")
            else:
                logger.info("No trades today")

        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")

    def run_scheduled(self):
        """Run bot on a schedule."""
        # Scan for setups every 4 hours (for 4H timeframe)
        schedule.every(4).hours.do(self.scan_and_trade)

        # Check positions every hour
        schedule.every(1).hours.do(self.check_positions)

        # Daily summary at 00:00
        schedule.every().day.at("00:00").do(self.daily_summary)

        logger.info("Scheduler started. Press Ctrl+C to stop.")

        # Run once immediately
        self.scan_and_trade()

        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def run_once(self):
        """Run bot once and exit."""
        self.scan_and_trade()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Swing Trading Bot')
    parser.add_argument('--live', action='store_true',
                       help='Run in live mode (default is dry run)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (default is scheduled)')

    args = parser.parse_args()

    # Initialize bot
    bot = SwingTradingBot(dry_run=not args.live)

    try:
        if args.once:
            bot.run_once()
        else:
            bot.run_scheduled()
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

