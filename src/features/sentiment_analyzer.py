"""
Sentiment analysis using FinBERT (pre-trained).
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of financial news using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load FinBERT model and tokenizer."""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with 'positive', 'negative', 'neutral' probabilities and 'score'
        """
        if not text or not text.strip():
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'score': 0.0
            }

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            probs = predictions[0].tolist()

            # Calculate sentiment score (-1 to +1)
            # Score = positive - negative
            sentiment_score = probs[0] - probs[1]

            return {
                'positive': probs[0],
                'negative': probs[1],
                'neutral': probs[2],
                'score': sentiment_score
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34,
                'score': 0.0
            }

    def analyze_headlines(self, headlines: List[Dict]) -> Dict[str, float]:
        """
        Analyze sentiment of multiple headlines.

        Args:
            headlines: List of headline dictionaries with 'title' and optionally 'published'

        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if not headlines:
            return {
                'overall_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'avg_positive': 0.33,
                'avg_negative': 0.33,
                'avg_neutral': 0.34
            }

        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for headline in headlines:
            text = headline.get('title', '')
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment)

            # Categorize based on highest probability
            max_prob = max(sentiment['positive'], sentiment['negative'], sentiment['neutral'])
            if sentiment['positive'] == max_prob:
                positive_count += 1
            elif sentiment['negative'] == max_prob:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate averages
        avg_positive = np.mean([s['positive'] for s in sentiments])
        avg_negative = np.mean([s['negative'] for s in sentiments])
        avg_neutral = np.mean([s['neutral'] for s in sentiments])
        overall_score = np.mean([s['score'] for s in sentiments])

        return {
            'overall_score': float(overall_score),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'avg_positive': float(avg_positive),
            'avg_negative': float(avg_negative),
            'avg_neutral': float(avg_neutral)
        }

    def analyze_instrument_sentiment(self, headlines: List[Dict],
                                    instrument: str) -> float:
        """
        Analyze sentiment specific to an instrument and determine directional bias.

        For EUR_USD:
        - Positive EUR news or negative USD news = bullish (positive score)
        - Negative EUR news or positive USD news = bearish (negative score)

        Args:
            headlines: List of headline dictionaries
            instrument: Currency pair (e.g., "EUR_USD")

        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        if not headlines:
            return 0.0

        # Extract currencies
        currencies = instrument.split('_')
        if len(currencies) != 2:
            # Fallback to simple overall sentiment
            result = self.analyze_headlines(headlines)
            return result['overall_score']

        base_currency = currencies[0]  # EUR in EUR_USD
        quote_currency = currencies[1]  # USD in EUR_USD

        # Currency keywords for filtering
        currency_keywords = {
            'EUR': ['euro', 'eur', 'ecb', 'eurozone'],
            'USD': ['dollar', 'usd', 'fed', 'federal reserve'],
            'GBP': ['pound', 'gbp', 'sterling', 'boe'],
            'JPY': ['yen', 'jpy', 'boj'],
            'AUD': ['aussie', 'aud', 'rba'],
        }

        base_sentiments = []
        quote_sentiments = []

        for headline in headlines:
            text = headline.get('title', '').lower()
            sentiment = self.analyze_text(headline.get('title', ''))

            # Check if headline mentions base currency
            if base_currency in currency_keywords:
                if any(keyword in text for keyword in currency_keywords[base_currency]):
                    base_sentiments.append(sentiment['score'])

            # Check if headline mentions quote currency
            if quote_currency in currency_keywords:
                if any(keyword in text for keyword in currency_keywords[quote_currency]):
                    quote_sentiments.append(sentiment['score'])

        # Calculate directional score
        # Positive base sentiment = bullish for pair
        # Positive quote sentiment = bearish for pair
        base_score = np.mean(base_sentiments) if base_sentiments else 0.0
        quote_score = np.mean(quote_sentiments) if quote_sentiments else 0.0

        # Combine: positive base or negative quote = bullish
        directional_score = base_score - quote_score

        # Normalize to -1 to +1 range
        directional_score = np.clip(directional_score, -1.0, 1.0)

        return float(directional_score)


class KeywordSentimentAnalyzer:
    """
    Simplified sentiment analyzer using keyword matching.
    Faster alternative to FinBERT.
    """

    def __init__(self):
        """Initialize keyword lists."""
        self.positive_keywords = [
            'rally', 'surge', 'gain', 'rise', 'jump', 'soar', 'climb', 'advance',
            'strength', 'strong', 'bullish', 'optimism', 'confident', 'growth',
            'recovery', 'stimulus', 'support', 'breakthrough', 'agreement', 'deal',
            'positive', 'boost', 'improve', 'expansion'
        ]

        self.negative_keywords = [
            'fall', 'drop', 'decline', 'plunge', 'sink', 'tumble', 'slip', 'slide',
            'weak', 'weakness', 'bearish', 'concern', 'worry', 'fear', 'crisis',
            'tension', 'conflict', 'recession', 'slowdown', 'cut', 'negative',
            'disappoint', 'miss', 'contraction'
        ]

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text using keyword matching.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment score
        """
        if not text:
            return {'score': 0.0}

        text_lower = text.lower()

        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return {'score': 0.0}

        score = (positive_count - negative_count) / total

        return {'score': score}

    def analyze_headlines(self, headlines: List[Dict]) -> Dict[str, float]:
        """
        Analyze multiple headlines.

        Args:
            headlines: List of headline dictionaries

        Returns:
            Dictionary with overall score
        """
        if not headlines:
            return {'overall_score': 0.0}

        scores = []
        for headline in headlines:
            text = headline.get('title', '')
            result = self.analyze_text(text)
            scores.append(result['score'])

        return {'overall_score': float(np.mean(scores))}

