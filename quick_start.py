"""
Quick start script to verify installation and setup.
"""
import sys
from pathlib import Path

print("=" * 70)
print("SWING TRADING BOT - QUICK START")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✅ OK")

# Check dependencies
print("\n2. Checking dependencies...")
required_packages = [
    'pandas',
    'numpy',
    'xgboost',
    'sklearn',
    'transformers',
    'torch',
    'oandapyV20',
    'feedparser',
    'matplotlib',
    'seaborn'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing.append(package)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Check .env file
print("\n3. Checking configuration...")
env_file = Path(__file__).parent / ".env"
if not env_file.exists():
    print("   ⚠️  .env file not found")
    print("   Copy .env.example to .env and fill in your credentials")
else:
    print("   ✅ .env file exists")

    # Load and validate
    from dotenv import load_dotenv
    import os
    load_dotenv()

    oanda_key = os.getenv('OANDA_API_KEY')
    oanda_account = os.getenv('OANDA_ACCOUNT_ID')

    if not oanda_key or oanda_key == 'your_api_key_here':
        print("   ⚠️  OANDA_API_KEY not configured")
    else:
        print("   ✅ OANDA_API_KEY configured")

    if not oanda_account or oanda_account == 'your_account_id_here':
        print("   ⚠️  OANDA_ACCOUNT_ID not configured")
    else:
        print("   ✅ OANDA_ACCOUNT_ID configured")

# Check directories
print("\n4. Checking directories...")
sys.path.insert(0, str(Path(__file__).parent / "src"))
import config

for dir_name, dir_path in [('data', config.DATA_DIR),
                            ('models', config.MODELS_DIR),
                            ('logs', config.LOGS_DIR)]:
    if dir_path.exists():
        print(f"   ✅ {dir_name}/ directory exists")
    else:
        print(f"   ✅ {dir_name}/ directory created")

# Check data provider connection
print("\n5. Testing data provider connection...")
try:
    from src.data.data_provider_factory import get_data_client
    client = get_data_client()

    print(f"   ℹ️  Using: {type(client).__name__}")

    # Try to fetch a small amount of data
    df = client.get_candles("EUR_USD", "D", count=5)

    if not df.empty:
        print(f"   ✅ Successfully connected")
        print(f"   ✅ Fetched {len(df)} candles")
        print(f"\n   💡 To specify a provider, set DATA_PROVIDER in .env")
        print(f"      Options: yahoo (free!), alpha_vantage, oanda, auto")
    else:
        print("   ❌ No data returned")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   💡 Quick fix: Set DATA_PROVIDER=yahoo in .env (no API key needed)")

# Check if models are trained
print("\n6. Checking for trained models...")
bounce_model = config.MODELS_DIR / "bounce_predictor.joblib"
direction_model = config.MODELS_DIR / "direction_predictor.joblib"

if bounce_model.exists() and direction_model.exists():
    print("   ✅ Models found")
else:
    print("   ⚠️  Models not found")
    print("   Run: python train_models.py")

# Summary
print("\n" + "=" * 70)
print("SETUP SUMMARY")
print("=" * 70)

if missing:
    print("❌ Setup incomplete - install missing packages")
elif not env_file.exists():
    print("⚠️  Setup incomplete - create .env file")
    print("\n💡 For immediate start (no API keys):")
    print('   echo "DATA_PROVIDER=yahoo" > .env')
else:
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("  1. Test providers: python test_data_providers.py")
    print("  2. Train models: python train_models.py --instrument EUR_USD")
    print("  3. Run backtest: python backtest.py --instrument EUR_USD --days 365")
    print("  4. Paper trade: python paper_trade.py --once")
    print("\n📖 Guides:")
    print("   • No OANDA? → QUICK_START_NO_OANDA.md")
    print("   • Paper trading → PAPER_TRADING_GUIDE.md")
    print("   • Data providers → docs/DATA_PROVIDERS.md")
    print("   • Full docs → README.md")

print("=" * 70)

