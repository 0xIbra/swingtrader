"""
Script to run backtests.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.backtesting.backtest_engine import BacktestEngine
from src.data.data_provider_factory import get_data_client
from src.monitoring.logger_config import setup_logging
from src.monitoring.visualizer import TradingVisualizer
import config

# Setup logging
logger = setup_logging()


def main():
    """Run backtest."""
    import argparse

    parser = argparse.ArgumentParser(description='Backtest swing trading strategy')
    parser.add_argument('--instrument', type=str, default='EUR_USD',
                       help='Instrument to backtest (default: EUR_USD)')
    parser.add_argument('--granularity', type=str, default='H4',
                       help='Timeframe (default: H4)')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of historical data (default: 365)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("BACKTESTING SWING TRADING STRATEGY")
    logger.info("="*70)
    logger.info(f"Instrument: {args.instrument}")
    logger.info(f"Timeframe: {args.granularity}")
    logger.info(f"Period: {args.days} days")
    logger.info("="*70 + "\n")

    # Fetch historical data
    logger.info("Fetching historical data...")
    data_client = get_data_client()
    logger.info(f"Using data provider: {type(data_client).__name__}")
    df = data_client.get_historical_data(
        instrument=args.instrument,
        granularity=args.granularity,
        days_back=args.days
    )

    if df.empty:
        logger.error("No data fetched. Check your data provider configuration.")
        logger.error("Try: DATA_PROVIDER=yahoo in .env (no API key needed)")
        return

    logger.info(f"Fetched {len(df)} bars\n")

    # Run backtest
    backtest_engine = BacktestEngine()
    results = backtest_engine.run_backtest(
        df,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Generate visualizations
    if results['total_trades'] > 0:
        logger.info("\nGenerating visualizations...")
        visualizer = TradingVisualizer()

        trades_df = results['trades_df']

        visualizer.plot_equity_curve(trades_df)
        visualizer.plot_pattern_performance(trades_df)

        logger.info("Visualizations saved to logs directory")
    else:
        logger.warning("No trades generated in backtest")


if __name__ == "__main__":
    main()

