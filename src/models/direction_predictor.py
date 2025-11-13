"""
XGBoost direction predictor - predicts if price will go up, down, or sideways.
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Tuple
import joblib
import logging

logger = logging.getLogger(__name__)


class DirectionPredictor:
    """Predicts price direction (up, down, or sideways)."""

    def __init__(self, model_path: str = None):
        """
        Initialize direction predictor.

        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.feature_names = None
        self.model_path = model_path
        self.classes = ['DOWN', 'SIDEWAYS', 'UP']  # 0, 1, 2

        if model_path:
            self.load_model(model_path)

    def prepare_training_data(self, df: pd.DataFrame,
                            lookforward_bars: int = 3,
                            atr_multiplier: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by labeling price direction.

        Args:
            df: DataFrame with OHLCV and features
            lookforward_bars: How many bars to look ahead
            atr_multiplier: ATR multiplier to determine significant move

        Returns:
            Tuple of (features_df, labels)
        """
        labels = []
        valid_indices = []

        for i in range(len(df) - lookforward_bars):
            current = df.iloc[i]
            future_price = df.iloc[i + lookforward_bars]['close']
            current_price = current['close']
            atr = current['atr']

            # Calculate price change
            price_change = future_price - current_price
            threshold = atr * atr_multiplier

            # Label based on price change
            if price_change > threshold:
                labels.append(2)  # UP
            elif price_change < -threshold:
                labels.append(0)  # DOWN
            else:
                labels.append(1)  # SIDEWAYS

            valid_indices.append(i)

        # Extract all 20 features
        feature_columns = [
            # Price structure
            'support_level_strength', 'resistance_level_strength',
            'distance_to_support_pct', 'distance_to_resistance_pct',
            'price_volatility',
            # Momentum
            'rsi_14', 'momentum_20', 'macd_histogram',
            # Volume
            'volume_ratio', 'volume_trend',
            # Trend
            'trend',
            # Pattern (would add if available)
            # 'pattern_type',
            # Context (would add if available)
            # 'news_sentiment', 'market_regime', 'high_impact_event_24h'
        ]

        # Filter to only columns that exist
        available_features = [col for col in feature_columns if col in df.columns]

        features_df = df.iloc[valid_indices][available_features].copy()
        labels_series = pd.Series(labels, index=features_df.index)

        # Fill NaN values
        features_df = features_df.fillna(0)

        logger.info(f"Prepared {len(labels)} training samples")
        logger.info(f"Class distribution: UP={np.sum(np.array(labels)==2)}, "
                   f"SIDEWAYS={np.sum(np.array(labels)==1)}, "
                   f"DOWN={np.sum(np.array(labels)==0)}")

        return features_df, labels_series

    def train(self, X: pd.DataFrame, y: pd.Series,
             params: Dict = None) -> Dict:
        """
        Train direction predictor model.

        Args:
            X: Features DataFrame
            y: Labels Series (0=DOWN, 1=SIDEWAYS, 2=UP)
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
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42
            }

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5, gap=100)

        accuracies = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Calculate accuracy
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)

            logger.info(f"Fold {fold+1} accuracy: {acc:.3f}")

        # Train final model on all data
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        self.feature_names = X.columns.tolist()

        avg_accuracy = np.mean(accuracies)

        logger.info(f"Training complete. Average accuracy: {avg_accuracy:.3f}")

        return {
            'avg_accuracy': avg_accuracy,
            'fold_accuracies': accuracies
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict direction probabilities.

        Args:
            X: Features DataFrame

        Returns:
            Array of shape (n_samples, 3) with [prob_down, prob_sideways, prob_up]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a model.")

        # Ensure features match training
        if self.feature_names:
            X = X[self.feature_names]

        return self.model.predict_proba(X)

    def predict_single(self, features: Dict) -> Dict[str, float]:
        """
        Predict direction probabilities for a single observation.

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with probabilities for each direction
        """
        X = pd.DataFrame([features])
        probs = self.predict(X)[0]

        return {
            'prob_down': probs[0],
            'prob_sideways': probs[1],
            'prob_up': probs[2]
        }

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict direction class (0=DOWN, 1=SIDEWAYS, 2=UP).

        Args:
            X: Features DataFrame

        Returns:
            Array of predicted classes
        """
        if self.model is None:
            raise ValueError("Model not trained")

        if self.feature_names:
            X = X[self.feature_names]

        return self.model.predict(X)

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

