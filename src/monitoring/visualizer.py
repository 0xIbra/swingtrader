"""
Visualization tools for equity curve and trade analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import config


class TradingVisualizer:
    """Creates visualizations for trading performance."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save charts (defaults to logs dir)
        """
        self.output_dir = output_dir or config.LOGS_DIR
        self.output_dir.mkdir(exist_ok=True)

        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def plot_equity_curve(self, trades_df: pd.DataFrame, save: bool = True) -> None:
        """
        Plot cumulative equity curve.

        Args:
            trades_df: DataFrame with trade history
            save: Whether to save the plot
        """
        if trades_df.empty:
            print("No trades to plot")
            return

        # Calculate cumulative profit
        trades_df = trades_df.sort_values('timestamp')
        trades_df['cumulative_profit'] = trades_df['profit_pct'].cumsum()

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Equity curve
        ax1.plot(trades_df['timestamp'], trades_df['cumulative_profit'],
                linewidth=2, color='#2E86AB')
        ax1.fill_between(trades_df['timestamp'], trades_df['cumulative_profit'],
                         alpha=0.3, color='#2E86AB')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Profit (%)')
        ax1.grid(True, alpha=0.3)

        # Individual trades
        colors = ['green' if x == 'win' else 'red' for x in trades_df['outcome']]
        ax2.bar(range(len(trades_df)), trades_df['profit_pct'], color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Individual Trade Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Profit (%)')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'equity_curve.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Equity curve saved to {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_pattern_performance(self, trades_df: pd.DataFrame,
                                save: bool = True) -> None:
        """
        Plot performance breakdown by pattern type.

        Args:
            trades_df: DataFrame with trade history
            save: Whether to save the plot
        """
        if trades_df.empty or 'pattern_type' not in trades_df.columns:
            print("No pattern data to plot")
            return

        # Calculate stats by pattern
        pattern_stats = trades_df.groupby('pattern_type').agg({
            'profit_pct': ['count', 'mean', 'sum'],
            'outcome': lambda x: (x == 'win').sum() / len(x) if len(x) > 0 else 0
        }).reset_index()

        pattern_stats.columns = ['pattern', 'count', 'avg_profit', 'total_profit', 'win_rate']

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Trade count by pattern
        axes[0, 0].bar(pattern_stats['pattern'], pattern_stats['count'],
                      color='#A23B72', alpha=0.7)
        axes[0, 0].set_title('Trade Count by Pattern', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Trades')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Win rate by pattern
        axes[0, 1].bar(pattern_stats['pattern'], pattern_stats['win_rate'] * 100,
                      color='#18A558', alpha=0.7)
        axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Win Rate by Pattern', fontweight='bold')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Average profit by pattern
        axes[1, 0].bar(pattern_stats['pattern'], pattern_stats['avg_profit'],
                      color='#F18F01', alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('Average Profit by Pattern', fontweight='bold')
        axes[1, 0].set_ylabel('Average Profit (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Total profit by pattern
        axes[1, 1].bar(pattern_stats['pattern'], pattern_stats['total_profit'],
                      color='#2E86AB', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_title('Total Profit by Pattern', fontweight='bold')
        axes[1, 1].set_ylabel('Total Profit (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'pattern_performance.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Pattern performance saved to {filepath}")
        else:
            plt.show()

        plt.close()

    def plot_monthly_returns(self, trades_df: pd.DataFrame,
                           save: bool = True) -> None:
        """
        Plot monthly returns heatmap.

        Args:
            trades_df: DataFrame with trade history
            save: Whether to save the plot
        """
        if trades_df.empty:
            print("No trades to plot")
            return

        # Convert timestamp to datetime if not already
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

        # Extract year and month
        trades_df['year'] = trades_df['timestamp'].dt.year
        trades_df['month'] = trades_df['timestamp'].dt.month

        # Calculate monthly returns
        monthly_returns = trades_df.groupby(['year', 'month'])['profit_pct'].sum().reset_index()
        monthly_returns_pivot = monthly_returns.pivot(index='year', columns='month', values='profit_pct')

        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Monthly Return (%)'})
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Year')

        if save:
            filepath = self.output_dir / 'monthly_returns.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Monthly returns saved to {filepath}")
        else:
            plt.show()

        plt.close()

