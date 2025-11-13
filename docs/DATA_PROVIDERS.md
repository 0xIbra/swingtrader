# Data Provider Guide

The swing trading bot supports multiple data providers. Here's how to use each one:

## 🔄 Auto Mode (Recommended)

Set `DATA_PROVIDER=auto` in `.env` and the system will automatically try providers in this order:
1. OANDA (if configured)
2. Yahoo Finance (no API key needed)
3. Alpha Vantage (if configured)

```bash
DATA_PROVIDER=auto
```

## 📊 Available Providers

### 1. Yahoo Finance (Easiest - No API Key!)

**Pros:**
- ✅ Completely free, no API key needed
- ✅ Already included via `yfinance`
- ✅ Good data quality
- ✅ Perfect for backtesting

**Cons:**
- ❌ Data only (no live trading)
- ❌ Rate limited (but generous)

**Setup:**
```bash
# In .env
DATA_PROVIDER=yahoo

# That's it! No API key needed
```

**Usage:**
```python
python train_models.py --instrument EUR_USD
python backtest.py --instrument EUR_USD --days 365
```

---

### 2. Alpha Vantage

**Pros:**
- ✅ Free tier available
- ✅ Good forex data
- ✅ Real-time quotes

**Cons:**
- ❌ Limited to 25 requests/day (free tier)
- ❌ 5 requests/minute limit
- ❌ Data only (no trading)

**Setup:**
1. Get free API key: https://www.alphavantage.co/support/#api-key
2. Add to `.env`:
```bash
DATA_PROVIDER=alpha_vantage
ALPHA_VANTAGE_API_KEY=YOUR_KEY_HERE
```

**Free Tier Limits:**
- 25 API calls per day
- 5 API calls per minute
- Good for: Development, testing, manual analysis

**Paid Tiers:** Available for higher limits

---

### 3. OANDA (Full Trading)

**Pros:**
- ✅ Live trading support
- ✅ Practice account available
- ✅ Professional-grade data
- ✅ Order execution

**Cons:**
- ❌ Requires account setup
- ❌ Currently down (as of your message)

**Setup:**
1. Create account: https://www.oanda.com/
2. Get API credentials from account settings
3. Add to `.env`:
```bash
DATA_PROVIDER=oanda
OANDA_API_KEY=your_key
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice  # or 'live'
```

---

## 🎯 Recommended Workflow

### For Development & Backtesting:
```bash
# Use Yahoo Finance (free, no setup)
DATA_PROVIDER=yahoo

python train_models.py --instrument EUR_USD --days 1825
python backtest.py --instrument EUR_USD --days 365
```

### For Paper Trading:
```bash
# Use OANDA practice account
DATA_PROVIDER=oanda
OANDA_ENVIRONMENT=practice

python main.py --once  # Dry run mode
```

### For Live Trading:
```bash
# Use OANDA live account
DATA_PROVIDER=oanda
OANDA_ENVIRONMENT=live

python main.py --live  # ⚠️ Real money!
```

---

## 🔧 Provider Comparison

| Feature | Yahoo Finance | Alpha Vantage | OANDA |
|---------|--------------|---------------|--------|
| **Cost** | Free | Free tier | Free practice |
| **API Key** | None | Required | Required |
| **Data Quality** | Good | Good | Excellent |
| **Historical Data** | Years | Years | Years |
| **Real-time** | Yes | Yes | Yes |
| **Trading** | No | No | Yes |
| **Rate Limits** | Generous | 25/day free | Professional |
| **Setup Time** | 0 min | 5 min | 15 min |

---

## 💡 Quick Start (No API Keys)

Want to start immediately without any API keys?

```bash
# 1. .env file (minimal)
DATA_PROVIDER=yahoo

# 2. Train models (using Yahoo Finance)
python train_models.py --instrument EUR_USD --days 1825

# 3. Backtest
python backtest.py --instrument EUR_USD --days 365

# 4. See results
ls logs/
```

That's it! You'll have trained models and backtest results without signing up for anything.

---

## 🆘 Troubleshooting

### "No data provider available"

**If using auto mode:**
- Yahoo Finance should work automatically
- Check internet connection
- Try: `DATA_PROVIDER=yahoo` explicitly

**If using OANDA:**
- Verify credentials in `.env`
- Check OANDA service status
- Try practice environment first

**If using Alpha Vantage:**
- Verify API key is correct
- Check daily limit (25 requests/day free)
- Wait a minute between requests

### "Rate limit exceeded"

**Yahoo Finance:**
- Wait a few minutes
- Reduce number of requests

**Alpha Vantage:**
- Free tier: 25 requests/day, 5/minute
- Upgrade to paid tier for more
- Use Yahoo Finance for development

### "Connection error"

- Check internet connection
- Try different provider: `DATA_PROVIDER=yahoo`
- Check provider service status

---

## 🔌 Adding New Providers

Want to add another data provider? Create a new client in `src/data/` that implements:

```python
class NewProviderClient:
    def get_candles(self, instrument, granularity, count):
        """Return DataFrame with: open, high, low, close, volume"""
        pass

    def get_historical_data(self, instrument, granularity, days_back):
        """Return historical DataFrame"""
        pass

    def get_current_price(self, instrument):
        """Return dict with: bid, ask, mid, time"""
        pass
```

Then register it in `data_provider_factory.py`.

---

## 📞 Support

- Yahoo Finance: No support needed (just works!)
- Alpha Vantage: https://www.alphavantage.co/support/
- OANDA: https://www.oanda.com/support/

