#!/bin/bash
# Run the complete 4H feature engineering pipeline

set -e  # Exit on error

echo "================================================================================"
echo "Running Complete 4H Feature Engineering Pipeline"
echo "================================================================================"

cd "$(dirname "$0")/.."

# Task 3: Economic Events
echo ""
echo "Task 3: Economic Calendar Features"
python scripts/task_3_economic_events/economic_calendar_pipeline.py

# Task 4: News Sentiment
echo ""
echo "Task 4: News Sentiment Features"
python scripts/task_4_news_sentiment/news_sentiment_pipeline.py

# Task 5: Macro Indicators
echo ""
echo "Task 5: Macro Regime Features"
python scripts/task_5_macro_indicators/macro_regime_pipeline.py

# Task 6: Price Features
echo ""
echo "Task 6: Price-Derived Features"
python scripts/task_6_price_features/price_features_pipeline.py

# Task 7: MFE/MAE Labels
echo ""
echo "Task 7: MFE/MAE Labels"
python scripts/task_7_mfe_mae_labels/mfe_mae_labels_pipeline.py

# Task 8: Direction Labels
echo ""
echo "Task 8: Direction Labels"
python scripts/task_8_direction_labels/direction_labels_pipeline.py

# Task 9: Sequence Builder
echo ""
echo "Task 9: Sequence Windows"
python scripts/task_9_sequences/sequence_window_builder.py

# Task 10: Split & Normalize
echo ""
echo "Task 10: Dataset Split & Normalization"
python scripts/task_10_split_normalize/dataset_split_normalize.py

echo ""
echo "================================================================================"
echo "✅ Pipeline Complete!"
echo "================================================================================"
