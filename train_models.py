"""
Script to train the ML models.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.training_pipeline import TrainingPipeline
from src.monitoring.logger_config import setup_logging
import config

# Setup logging
logger = setup_logging()


def main():
    """Train models for all instruments."""
    import argparse

    parser = argparse.ArgumentParser(description='Train swing trading models')
    parser.add_argument('--instrument', type=str, default=None,
                       help='Specific instrument to train (default: all)')
    parser.add_argument('--granularity', type=str, default='H4',
                       help='Timeframe (default: H4)')
    parser.add_argument('--days', type=int, default=1825,
                       help='Days of historical data (default: 1825 = 5 years)')

    args = parser.parse_args()

    pipeline = TrainingPipeline()

    # Determine which instruments to train
    instruments = [args.instrument] if args.instrument else config.INSTRUMENTS

    for instrument in instruments:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"TRAINING MODELS FOR {instrument}")
            logger.info(f"{'='*70}\n")

            results = pipeline.train_all_models(
                instrument=instrument,
                granularity=args.granularity,
                days_back=args.days
            )

            logger.info(f"\n✅ Training complete for {instrument}")

        except Exception as e:
            logger.error(f"❌ Error training {instrument}: {e}", exc_info=True)
            continue

    logger.info("\n" + "="*70)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()

