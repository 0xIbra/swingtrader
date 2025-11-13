"""
Test script to verify data providers are working.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.data_provider_factory import get_data_client
from src.monitoring.logger_config import setup_logging

logger = setup_logging()


def test_provider(provider_name):
    """Test a specific data provider."""
    print("\n" + "="*70)
    print(f"Testing {provider_name.upper()}")
    print("="*70)

    try:
        client = get_data_client(provider_name)
        print(f"✅ Client created: {type(client).__name__}")

        # Test fetching data
        print("\nFetching EUR_USD daily data (last 5 bars)...")
        df = client.get_candles("EUR_USD", "D", count=5)

        if df.empty:
            print("❌ No data returned")
            return False

        print(f"✅ Fetched {len(df)} bars")
        print("\nSample data:")
        print(df.tail())

        # Test current price
        print("\nFetching current price...")
        price = client.get_current_price("EUR_USD")
        print(f"✅ Current price: {price['mid']:.5f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test all available providers."""
    print("="*70)
    print("DATA PROVIDER TEST")
    print("="*70)

    providers = ['yahoo', 'alpha_vantage', 'oanda']
    results = {}

    for provider in providers:
        results[provider] = test_provider(provider)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for provider, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{provider.ljust(20)}: {status}")

    working = [p for p, s in results.items() if s]

    if working:
        print(f"\n✅ Working providers: {', '.join(working)}")
        print(f"\nRecommendation: Set DATA_PROVIDER={working[0]} in .env")
    else:
        print("\n❌ No providers working")
        print("\nFor immediate use without API keys:")
        print("  1. Make sure yfinance is installed: pip install yfinance")
        print("  2. Set DATA_PROVIDER=yahoo in .env")
        print("  3. No API key needed!")


if __name__ == "__main__":
    main()

