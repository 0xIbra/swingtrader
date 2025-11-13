"""
Telegram bot for sending trading alerts.
"""
from telegram import Bot
from telegram.error import TelegramError
from typing import Dict
import asyncio
import config
import logging

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends trading notifications via Telegram."""

    def __init__(self, token: str = None, chat_id: str = None):
        """
        Initialize Telegram notifier.

        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.token = token or config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.bot = None

        if self.token and self.chat_id:
            self.bot = Bot(token=self.token)
            logger.info("Telegram bot initialized")
        else:
            logger.warning("Telegram credentials not configured")

    async def send_message_async(self, message: str):
        """Send message asynchronously."""
        if not self.bot:
            logger.warning("Telegram bot not configured, skipping notification")
            return

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.debug("Telegram message sent successfully")
        except TelegramError as e:
            logger.error(f"Error sending Telegram message: {e}")

    def send_message(self, message: str):
        """
        Send message synchronously.

        Args:
            message: Message text to send
        """
        if not self.bot:
            return

        try:
            # Run async function in event loop
            asyncio.run(self.send_message_async(message))
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def send_trade_signal(self, signal: Dict):
        """
        Send trade signal notification.

        Args:
            signal: Trade signal dictionary
        """
        message = f"""
🚨 <b>TRADE SIGNAL</b> 🚨

<b>Instrument:</b> {signal['instrument']}
<b>Direction:</b> {signal['direction']}
<b>Entry:</b> {signal['entry_price']:.5f}
<b>Stop Loss:</b> {signal['stop_loss']:.5f}
<b>Take Profit:</b> {signal['take_profit']:.5f}

<b>Confidence:</b> {signal['confidence']:.1%}
<b>Risk:</b> {signal['risk_percent']:.1f}%

<b>Details:</b>
• Bounce Prob: {signal['details']['bounce_prob']:.2f}
• Direction Prob: {signal['details']['direction_prob']:.2f}
• Pattern: {signal['details']['pattern_type']}
• Sentiment: {signal['details']['news_sentiment']:.2f}
• Regime: {signal['details']['market_regime']}
"""
        self.send_message(message)

    def send_trade_executed(self, trade_id: int, instrument: str, direction: str):
        """
        Send trade execution notification.

        Args:
            trade_id: Database trade ID
            instrument: Currency pair
            direction: LONG or SHORT
        """
        message = f"""
✅ <b>TRADE EXECUTED</b>

<b>Trade ID:</b> {trade_id}
<b>Instrument:</b> {instrument}
<b>Direction:</b> {direction}
"""
        self.send_message(message)

    def send_trade_closed(self, trade_id: int, instrument: str,
                         outcome: str, profit_pct: float):
        """
        Send trade closed notification.

        Args:
            trade_id: Database trade ID
            instrument: Currency pair
            outcome: win/loss
            profit_pct: Profit percentage
        """
        emoji = "🎉" if outcome == "win" else "😔"

        message = f"""
{emoji} <b>TRADE CLOSED</b>

<b>Trade ID:</b> {trade_id}
<b>Instrument:</b> {instrument}
<b>Outcome:</b> {outcome.upper()}
<b>Profit:</b> {profit_pct:+.2f}%
"""
        self.send_message(message)

    def send_error(self, error_msg: str):
        """
        Send error notification.

        Args:
            error_msg: Error message
        """
        message = f"""
❌ <b>ERROR</b>

{error_msg}
"""
        self.send_message(message)

    def send_daily_summary(self, summary: Dict):
        """
        Send daily performance summary.

        Args:
            summary: Performance summary dictionary
        """
        message = f"""
📊 <b>DAILY SUMMARY</b>

<b>Trades:</b> {summary['total_trades']}
<b>Wins:</b> {summary['wins']} | <b>Losses:</b> {summary['losses']}
<b>Win Rate:</b> {summary['win_rate']:.1%}
<b>Total Profit:</b> {summary['total_profit']:+.2f}%

<b>Best Trade:</b> {summary['best_trade']:+.2f}%
<b>Worst Trade:</b> {summary['worst_trade']:+.2f}%
"""
        self.send_message(message)

