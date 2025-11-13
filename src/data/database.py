"""
SQLite database for storing trade history and building experience memory.
"""
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import config


class TradeDatabase:
    """Manages trade history database."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (defaults to config)
        """
        self.db_path = db_path or str(config.DB_PATH)
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create database tables if they don't exist."""

        # Trade history table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                units INTEGER NOT NULL,
                outcome TEXT,
                profit_pct REAL,
                profit_amount REAL,

                -- Pattern features
                pattern_type TEXT,
                bounce_probability REAL,
                direction_probability REAL,

                -- Market context
                news_sentiment REAL,
                market_regime TEXT,
                high_impact_event_24h INTEGER,

                -- Technical features
                rsi_14 REAL,
                atr REAL,
                support_level REAL,
                resistance_level REAL,
                distance_to_support_pct REAL,
                distance_to_resistance_pct REAL,

                -- Multi-timeframe
                trend_1h INTEGER,
                trend_4h INTEGER,
                trend_daily INTEGER,
                timeframe_alignment REAL,

                -- Risk management
                risk_percent REAL,
                confidence REAL,

                -- Trade management
                entry_timestamp DATETIME,
                exit_timestamp DATETIME,
                duration_hours REAL,

                -- Notes
                notes TEXT
            )
        """)

        # Index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON trades(timestamp DESC)
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_regime
            ON trades(pattern_type, market_regime, timeframe_alignment)
        """)

        self.conn.commit()

    def log_trade_entry(self, trade_data: Dict) -> int:
        """
        Log a new trade entry.

        Args:
            trade_data: Dictionary containing trade information

        Returns:
            Trade ID
        """
        columns = ', '.join(trade_data.keys())
        placeholders = ', '.join(['?' for _ in trade_data])

        query = f"""
            INSERT INTO trades ({columns})
            VALUES ({placeholders})
        """

        self.cursor.execute(query, list(trade_data.values()))
        self.conn.commit()

        return self.cursor.lastrowid

    def update_trade_exit(self, trade_id: int, exit_data: Dict):
        """
        Update trade with exit information.

        Args:
            trade_id: ID of the trade to update
            exit_data: Dictionary with exit information
        """
        set_clause = ', '.join([f"{key} = ?" for key in exit_data.keys()])
        query = f"UPDATE trades SET {set_clause} WHERE id = ?"

        values = list(exit_data.values()) + [trade_id]
        self.cursor.execute(query, values)
        self.conn.commit()

    def get_similar_setups(self, pattern_type: str, market_regime: str,
                          timeframe_alignment: float, sentiment_tolerance: float = 0.2,
                          current_sentiment: float = 0.0, days_back: int = 90) -> pd.DataFrame:
        """
        Query for similar historical setups.

        Args:
            pattern_type: Type of pattern (e.g., 'double_bottom')
            market_regime: Current market regime ('risk_on', 'risk_off', 'neutral')
            timeframe_alignment: Current timeframe alignment score
            sentiment_tolerance: How much sentiment can differ
            current_sentiment: Current news sentiment
            days_back: How many days of history to query

        Returns:
            DataFrame with similar trades
        """
        query = """
            SELECT *
            FROM trades
            WHERE pattern_type = ?
              AND market_regime = ?
              AND ABS(timeframe_alignment - ?) < 0.15
              AND ABS(news_sentiment - ?) < ?
              AND timestamp > datetime('now', ? || ' days')
              AND outcome IS NOT NULL
            ORDER BY timestamp DESC
        """

        params = (
            pattern_type,
            market_regime,
            timeframe_alignment,
            current_sentiment,
            sentiment_tolerance,
            f'-{days_back}'
        )

        df = pd.read_sql_query(query, self.conn, params=params)
        return df

    def calculate_confidence(self, pattern_type: str, market_regime: str,
                           timeframe_alignment: float, current_sentiment: float = 0.0,
                           min_sample_size: int = 10) -> Dict:
        """
        Calculate confidence score based on similar historical setups.

        Args:
            pattern_type: Type of pattern
            market_regime: Current market regime
            timeframe_alignment: Current timeframe alignment
            current_sentiment: Current news sentiment
            min_sample_size: Minimum number of samples to trust the win rate

        Returns:
            Dictionary with confidence metrics
        """
        similar_trades = self.get_similar_setups(
            pattern_type=pattern_type,
            market_regime=market_regime,
            timeframe_alignment=timeframe_alignment,
            current_sentiment=current_sentiment
        )

        if len(similar_trades) == 0:
            return {
                'confidence': 0.5,  # Neutral confidence
                'sample_size': 0,
                'win_rate': None,
                'avg_profit': None,
                'reliable': False
            }

        # Calculate win rate
        wins = (similar_trades['outcome'] == 'win').sum()
        total = len(similar_trades)
        win_rate = wins / total if total > 0 else 0.5

        # Calculate average profit
        avg_profit = similar_trades['profit_pct'].mean()

        # Adjust confidence based on sample size
        if total < min_sample_size:
            # Blend with neutral (0.5) when we have few samples
            weight = total / min_sample_size
            confidence = (win_rate * weight) + (0.5 * (1 - weight))
            reliable = False
        else:
            confidence = win_rate
            reliable = True

        return {
            'confidence': confidence,
            'sample_size': total,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'reliable': reliable
        }

    def get_performance_summary(self, days_back: int = 90) -> Dict:
        """
        Get overall performance summary.

        Args:
            days_back: How many days to include

        Returns:
            Dictionary with performance metrics
        """
        query = f"""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                AVG(profit_pct) as avg_profit_pct,
                SUM(profit_amount) as total_profit,
                MAX(profit_pct) as best_trade,
                MIN(profit_pct) as worst_trade
            FROM trades
            WHERE timestamp > datetime('now', '-{days_back} days')
              AND outcome IS NOT NULL
        """

        self.cursor.execute(query)
        result = self.cursor.fetchone()

        total_trades, wins, losses, avg_profit, total_profit, best, worst = result

        win_rate = wins / total_trades if total_trades > 0 else 0

        return {
            'total_trades': total_trades or 0,
            'wins': wins or 0,
            'losses': losses or 0,
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit or 0,
            'total_profit': total_profit or 0,
            'best_trade': best or 0,
            'worst_trade': worst or 0
        }

    def get_pattern_performance(self, days_back: int = 90) -> pd.DataFrame:
        """
        Get performance breakdown by pattern type.

        Args:
            days_back: How many days to include

        Returns:
            DataFrame with pattern performance metrics
        """
        query = f"""
            SELECT
                pattern_type,
                COUNT(*) as trade_count,
                AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(profit_pct) as avg_profit_pct
            FROM trades
            WHERE timestamp > datetime('now', '-{days_back} days')
              AND outcome IS NOT NULL
              AND pattern_type IS NOT NULL
            GROUP BY pattern_type
            ORDER BY trade_count DESC
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

