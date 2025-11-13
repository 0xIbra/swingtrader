"""
XGBoost bounce predictor - predicts if price will bounce off support/resistance.
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from typing import Dict, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class BouncePredictor:
    """Predicts whether price will bounce or break through a level."""

    def __init__(self, model_path: str = None):
        """
        Initialize bounce predictor.

        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.feature_names = None
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

    def prepare_training_data(self, df: pd.DataFrame,
                            lookforward_bars: int = 12,
                            atr_threshold: float = 1.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by labeling support/resistance touches.

        Args:
            df: DataFrame with OHLCV and features
            lookforward_bars: How many bars to look ahead for outcome
            atr_threshold: ATR multiplier to determine bounce/break

        Returns:
            Tuple of (features_df, labels)
        """
        # Identify points where price is near support or resistance
        # This is a simplified version - in production you'd be more sophisticated

        labels = []
        valid_indices = []

        for i in range(len(df) - lookforward_bars):
            current = df.iloc[i]
            future = df.iloc[i:i+lookforward_bars+1]

            # Check if we're near a level (support or resistance)
            if pd.isna(current.get('nearest_support')) and pd.isna(current.get('nearest_resistance')):
                continue

            distance_to_support = current.get('distance_to_support_pct', float('inf'))
            distance_to_resistance = current.get('distance_to_resistance_pct', float('inf'))

            # Only include if we're near a level (within 0.5%)
            near_support = distance_to_support < 0.5
            near_resistance = distance_to_resistance < 0.5

            if not (near_support or near_resistance):
                continue

            # Look forward to see if it bounced or broke
            atr = current['atr']
            current_price = current['close']

            if near_support:
                # Check if price bounced up or broke down
                max_future_price = future['high'].max()
                min_future_price = future['low'].min()

                bounce = (max_future_price - current_price) > (atr_threshold * atr)
                break_through = (current_price - min_future_price) > (0.5 * atr)

                if bounce and not break_through:
                    labels.append(1)  # Bounce
                    valid_indices.append(i)
                elif break_through and not bounce:
                    labels.append(0)  # Break
                    valid_indices.append(i)
                # Else: inconclusive, skip

            elif near_resistance:
                # Check if price bounced down or broke up
                max_future_price = future['high'].max()
                min_future_price = future['low'].min()

                bounce = (current_price - min_future_price) > (atr_threshold * atr)
                break_through = (max_future_price - current_price) > (0.5 * atr)

                if bounce and not break_through:
                    labels.append(1)  # Bounce
                    valid_indices.append(i)
                elif break_through and not bounce:
                    labels.append(0)  # Break
                    valid_indices.append(i)

        # Extract features for valid indices
        feature_columns = [
            'support_level_strength', 'resistance_level_strength',
            'distance_to_support_pct', 'distance_to_resistance_pct',
            'price_volatility', 'rsi_14', 'momentum_20', 'macd_histogram',
            'volume_ratio', 'volume_trend', 'trend'
        ]

        # Filter to only columns that exist
        available_features = [col for col in feature_columns if col in df.columns]

        features_df = df.iloc[valid_indices][available_features].copy()
        labels_series = pd.Series(labels, index=features_df.index)

        # Fill NaN values
        features_df = features_df.fillna(0)

        logger.info(f"Prepared {len(labels)} training samples")
        logger.info(f"Bounce rate: {np.mean(labels):.2%}")

        return features_df, labels_series

    def train(self, X: pd.DataFrame, y: pd.Series,
             params: Dict = None) -> Dict:
        """
        Train bounce predictor model.

        Args:
            X: Features DataFrame
            y: Labels Series
            params: XGBoost parameters (optional)

        Returns:
            Dictionary with training metrics
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42
            }

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5, gap=100)

        metrics = {
            'precision': [],
            'recall': [],
            'roc_auc': []
        }

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))

        # Train final model on all data
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()

        # Calculate average metrics
        avg_metrics = {
            'avg_precision': np.mean(metrics['precision']),
            'avg_recall': np.mean(metrics['recall']),
            'avg_roc_auc': np.mean(metrics['roc_auc'])
        }

        logger.info(f"Training complete: {avg_metrics}")

        return avg_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict bounce probability.

        Args:
            X: Features DataFrame

        Returns:
            Array of bounce probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a model.")

        # Ensure features match training
        if self.feature_names:
            # Reorder and select only training features
            X = X[self.feature_names]

        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, features: Dict) -> float:
        """
        Predict bounce probability for a single observation.

        Args:
            features: Dictionary of feature values

        Returns:
            Bounce probability
        """
        # Convert to DataFrame
        X = pd.DataFrame([features])
        return self.predict(X)[0]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save_model(self, path: str):
        """
        Save trained model to disk.

        Args:
            path: File path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load trained model from disk.

        Args:
            path: File path to load model from
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        logger.info(f"Model loaded from {path}")

