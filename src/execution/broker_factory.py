"""
Factory for creating broker clients (for order execution).
Supports real brokers (OANDA) and simulated broker (paper trading).
"""
import config
import logging

logger = logging.getLogger(__name__)


class BrokerFactory:
    """Factory for creating broker clients."""

    @staticmethod
    def create_broker(broker: str = None):
        """
        Create a broker client for order execution.

        Args:
            broker: Broker name ('oanda', 'simulated')

        Returns:
            Broker client instance
        """
        broker = broker or config.BROKER or 'simulated'
        broker = broker.lower()

        if broker == 'simulated' or broker == 'paper':
            return BrokerFactory._create_simulated_broker()
        elif broker == 'oanda':
            return BrokerFactory._create_oanda_broker()
        else:
            raise ValueError(f"Unknown broker: {broker}")

    @staticmethod
    def _create_simulated_broker():
        """Create simulated broker for paper trading."""
        from ..data.simulated_broker import SimulatedBroker

        # Check for custom initial balance
        import os
        initial_balance = float(os.getenv("SIMULATED_BROKER_BALANCE", 10000))

        logger.info(f"Using SimulatedBroker (paper trading) with ${initial_balance:,.2f}")
        return SimulatedBroker(initial_balance=initial_balance)

    @staticmethod
    def _create_oanda_broker():
        """Create OANDA broker client."""
        from ..data.oanda_client import OANDAClient
        logger.info("Using OANDA broker")
        return OANDAClient()


def get_broker(broker: str = None):
    """
    Convenience function to get a broker client.

    Args:
        broker: Broker name or None for config default

    Returns:
        Broker client
    """
    return BrokerFactory.create_broker(broker)

