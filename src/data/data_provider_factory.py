"""
Factory for creating data provider clients.
Easily switch between OANDA, Alpha Vantage, Yahoo Finance, etc.
"""
import config
import logging

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """Factory for creating data provider clients."""

    @staticmethod
    def create_client(provider: str = None):
        """
        Create a data provider client.

        Args:
            provider: Provider name ('oanda', 'alpha_vantage', 'yahoo', 'auto')
                     If 'auto' or None, tries providers in order until one works

        Returns:
            Data provider client instance
        """
        provider = provider or config.DATA_PROVIDER or 'auto'
        provider = provider.lower()

        if provider == 'auto':
            return DataProviderFactory._create_auto_client()
        elif provider == 'oanda':
            return DataProviderFactory._create_oanda_client()
        elif provider == 'alpha_vantage' or provider == 'alphavantage':
            return DataProviderFactory._create_alpha_vantage_client()
        elif provider == 'yahoo' or provider == 'yfinance':
            return DataProviderFactory._create_yahoo_client()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _create_auto_client():
        """Try providers in order until one works."""
        providers = ['oanda', 'yahoo', 'alpha_vantage']

        for provider_name in providers:
            try:
                logger.info(f"Trying {provider_name}...")

                if provider_name == 'oanda':
                    client = DataProviderFactory._create_oanda_client()
                elif provider_name == 'yahoo':
                    client = DataProviderFactory._create_yahoo_client()
                elif provider_name == 'alpha_vantage':
                    client = DataProviderFactory._create_alpha_vantage_client()

                # Test the connection with a small request
                df = client.get_candles("EUR_USD", "D", count=2)

                if not df.empty:
                    logger.info(f"✅ Using {provider_name} as data provider")
                    return client
                else:
                    logger.warning(f"{provider_name} returned no data")

            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                continue

        raise RuntimeError("No data provider available. Check your API credentials.")

    @staticmethod
    def _create_oanda_client():
        """Create OANDA client."""
        from .oanda_client import OANDAClient
        return OANDAClient()

    @staticmethod
    def _create_alpha_vantage_client():
        """Create Alpha Vantage client."""
        from .alpha_vantage_client import AlphaVantageClient
        return AlphaVantageClient()

    @staticmethod
    def _create_yahoo_client():
        """Create Yahoo Finance client."""
        from .yahoo_finance_client import YahooFinanceClient
        return YahooFinanceClient()


def get_data_client(provider: str = None):
    """
    Convenience function to get a data client.

    Args:
        provider: Provider name or 'auto'

    Returns:
        Data provider client
    """
    return DataProviderFactory.create_client(provider)

