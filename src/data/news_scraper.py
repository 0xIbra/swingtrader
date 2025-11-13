"""
News scraper for fetching forex-related headlines from RSS feeds.
"""
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config


class NewsScraper:
    """Scrapes forex news from RSS feeds."""

    def __init__(self, sources: Optional[List[str]] = None):
        """
        Initialize news scraper.

        Args:
            sources: List of RSS feed URLs (defaults to config)
        """
        self.sources = sources or config.NEWS_SOURCES

    def fetch_headlines(self, hours_back: int = 24) -> List[Dict]:
        """
        Fetch recent headlines from all sources.

        Args:
            hours_back: How many hours of historical news to fetch

        Returns:
            List of headline dictionaries with 'title', 'published', 'source', 'link'
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        all_headlines = []

        for source_url in self.sources:
            try:
                feed = feedparser.parse(source_url)
                source_name = feed.feed.get('title', source_url)

                for entry in feed.entries:
                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])
                    else:
                        # If no timestamp, assume it's recent
                        published = datetime.utcnow()

                    # Only include recent headlines
                    if published >= cutoff_time:
                        headline = {
                            'title': entry.get('title', ''),
                            'published': published,
                            'source': source_name,
                            'link': entry.get('link', ''),
                            'summary': entry.get('summary', '')
                        }
                        all_headlines.append(headline)

            except Exception as e:
                print(f"Error fetching from {source_url}: {e}")
                continue

        # Sort by published time, most recent first
        all_headlines.sort(key=lambda x: x['published'], reverse=True)

        return all_headlines

    def fetch_instrument_specific_news(self, instrument: str,
                                       hours_back: int = 24) -> List[Dict]:
        """
        Fetch headlines related to a specific instrument.

        Args:
            instrument: Currency pair (e.g., "EUR_USD")
            hours_back: How many hours of historical news to fetch

        Returns:
            List of relevant headline dictionaries
        """
        all_headlines = self.fetch_headlines(hours_back)

        # Extract currency codes from instrument (e.g., EUR_USD -> EUR, USD)
        currencies = instrument.replace('_', ' ').split()

        # Define keywords for each major currency
        currency_keywords = {
            'EUR': ['euro', 'eur', 'ecb', 'eurozone', 'lagarde'],
            'USD': ['dollar', 'usd', 'fed', 'federal reserve', 'powell'],
            'GBP': ['pound', 'gbp', 'sterling', 'boe', 'bank of england'],
            'JPY': ['yen', 'jpy', 'boj', 'bank of japan'],
            'AUD': ['aussie', 'aud', 'rba', 'australia'],
            'NZD': ['kiwi', 'nzd', 'rbnz', 'new zealand'],
            'CAD': ['loonie', 'cad', 'boc', 'canada'],
            'CHF': ['franc', 'chf', 'snb', 'switzerland']
        }

        # Filter headlines that mention either currency
        relevant_headlines = []
        for headline in all_headlines:
            text = (headline['title'] + ' ' + headline.get('summary', '')).lower()

            for currency in currencies:
                if currency in currency_keywords:
                    keywords = currency_keywords[currency]
                    if any(keyword in text for keyword in keywords):
                        relevant_headlines.append(headline)
                        break

        return relevant_headlines

    def get_market_regime_indicators(self, hours_back: int = 24) -> Dict:
        """
        Analyze news to determine market regime (risk-on vs risk-off).

        Args:
            hours_back: How many hours of news to analyze

        Returns:
            Dictionary with regime indicators
        """
        headlines = self.fetch_headlines(hours_back)

        # Risk-off keywords
        risk_off_keywords = [
            'crisis', 'tension', 'war', 'conflict', 'recession', 'crash',
            'concern', 'worry', 'fear', 'uncertainty', 'volatility', 'decline',
            'fall', 'drop', 'plunge', 'tumble', 'sink'
        ]

        # Risk-on keywords
        risk_on_keywords = [
            'growth', 'recovery', 'rally', 'surge', 'rise', 'gain',
            'optimism', 'confidence', 'strong', 'stimulus', 'support',
            'breakthrough', 'agreement', 'deal', 'positive'
        ]

        risk_off_count = 0
        risk_on_count = 0

        for headline in headlines:
            text = (headline['title'] + ' ' + headline.get('summary', '')).lower()

            # Weight recent news more heavily
            hours_old = (datetime.utcnow() - headline['published']).total_seconds() / 3600
            weight = max(1.0 - (hours_old / hours_back), 0.1)

            # Count risk-off indicators
            for keyword in risk_off_keywords:
                if keyword in text:
                    risk_off_count += weight

            # Count risk-on indicators
            for keyword in risk_on_keywords:
                if keyword in text:
                    risk_on_count += weight

        # Calculate regime score (-1 = risk-off, 0 = neutral, +1 = risk-on)
        total = risk_on_count + risk_off_count
        if total > 0:
            regime_score = (risk_on_count - risk_off_count) / total
        else:
            regime_score = 0.0

        # Classify regime
        if regime_score > 0.3:
            regime = 'risk_on'
        elif regime_score < -0.3:
            regime = 'risk_off'
        else:
            regime = 'neutral'

        return {
            'regime': regime,
            'regime_score': regime_score,
            'risk_on_count': risk_on_count,
            'risk_off_count': risk_off_count,
            'headline_count': len(headlines)
        }

    def check_high_impact_events(self, hours_ahead: int = 24) -> bool:
        """
        Check if there are high-impact economic events coming up.

        This is a simplified version. In production, you'd want to use
        an economic calendar API like ForexFactory or Trading Economics.

        Args:
            hours_ahead: How many hours ahead to check

        Returns:
            True if high-impact events are expected, False otherwise
        """
        headlines = self.fetch_headlines(hours_back=hours_ahead)

        # High-impact event keywords
        high_impact_keywords = [
            'fomc', 'fed meeting', 'nfp', 'non-farm payroll', 'interest rate decision',
            'ecb meeting', 'gdp', 'cpi', 'inflation data', 'employment report',
            'central bank', 'policy decision', 'rate hike', 'rate cut'
        ]

        for headline in headlines:
            text = (headline['title'] + ' ' + headline.get('summary', '')).lower()
            if any(keyword in text for keyword in high_impact_keywords):
                return True

        return False

