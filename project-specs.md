# **📌 EPIC: Intraday Trading AI — Full Data + Feature + Model Pipeline**

This includes:

* FX OHLC pipeline
* Feature engineering
* News calendar
* LLM sentiment
* Macro regime
* Dataset builder
* Labels (MFE/MAE + direction)
* Model training
* Backtest engine
* Live inference

Everything.

---

# **TASK 1 — Raw Market Data Pipeline**

**Title:** Implement OHLCV + Spread Data Fetcher
**Type:** Backend / Data
**Specs:**

* Integrate with data provider for **5m** OR fetch **1m/tick** → resample to **5m**.
* Fields required:

  * timestamp UTC
  * open, high, low, close
  * tick_volume
  * bid/ask for spread (if available)
* Output Parquet partitioned by `year/month/pair`.
* Handle gaps: forward-fill price but add `missing_flag = 1`.
* Enforce UTC alignment.

---

# **TASK 2 — Session Feature Generator**

**Title:** Compute global trading session encodings
**Type:** ML Features
**Specs:**

* For each timestamp:

  * Asia (00:00–06:00 UTC)
  * London (07:00–15:00 UTC)
  * NY (13:00–21:00 UTC)
  * Overlap (13:00–16:00 UTC)
* Generate binary flags:

  * `session_asia`, `session_london`, `session_ny`, `session_overlap`
* Add cyclical encodings:

  * `hour_sin`, `hour_cos`
  * `dow_sin`, `dow_cos`

---

# **TASK 3 — Economic Calendar Pipeline**

**Title:** Retrieve high/medium/low impact events + currencies
**Type:** Data Integration
**Specs:**

* Fetch from:

  * FinancialModelingPrep, Forex Factory API, EODHD, or custom scraping.
* Required fields:

  * event_time (UTC)
  * currency (USD, EUR, GBP, JPY…)
  * impact_level (0–2)
  * event_type (categorical)
* For each 5m bar compute:

  * `minutes_to_next_event`
  * `minutes_since_last_event`
  * `is_event_now`
  * one-hot encode event currency
  * severity score ∈ {0,1,2}
* Forward-fill features within ±30 minutes window.
* Save as separate column group.

---

# **TASK 4 — News Sentiment Pipeline**

**Title:** Build hourly LLM sentiment extraction for each currency
**Type:** NLP / Feature Engineering
**Specs:**

* Fetch live headlines from news API.
* Every hour, run through LLM prompt:

  * Extract sentiment for each currency: USD, EUR, GBP, JPY
  * Range normalized: −1 to +1
  * Extract global risk sentiment (risk-on/off)
* Features produced:

  * `sent_USD`, `sent_EUR`, `sent_GBP`, `sent_JPY`
  * `sent_risk`
* Forward-fill to all 5m bars within the hour.
* Store as float features.

---

# **TASK 5 — Macro Regime Pipeline**

**Title:** Fetch external macro indicators (VIX, SPX, yields, DXY)
**Type:** Data Integration
**Specs:**
From any market data vendor fetch:

* SP500 Futures price
* VIX index
* U.S. 10Y yield (and optionally 2Y)
* DXY (Dollar index)
* GOLD
* OIL

Compute per 5m bar:

* returns:

  * `spx_ret_5m`, `vix_ret_5m`, `dxy_ret_5m`
* absolute levels:

  * `vix_level`, `yield10_level`, `dxy_level`
* vol-adjusted z-scores:

  * `vix_z`, `dxy_z`, `yield10_z`

Align by timestamp, forward-fill missing data.

---

# **TASK 6 — Internal Price Features**

**Title:** Compute price-derived technical features
**Type:** ML Features
**Specs:**
For each bar compute:

* log returns (1, 3, 6, 12 bars)
* ATR(14)
* rolling volatility (20, 50 bars)
* EMA20, EMA50, EMA100
* RSI14
* Candle shape:

  * body_norm, upper_wick_norm, lower_wick_norm (normalized by ATR)
* Recent structure:

  * `dist_high_24`, `dist_low_24`
  * `bars_since_high`, `bars_since_low`

Store into unified feature table.

---

# **TASK 7 — Compute MFE/MAE Labels**

**Title:** Future excursion label generator
**Type:** ML Labeling
**Specs:**
For each timestamp `t`:

H = future horizon (12 bars)
entry = close[t]

Compute:

**LONG**

* `MFE_long = max(high[t+1..t+H]) - entry`
* `MAE_long = min(low[t+1..t+H]) - entry`

**SHORT**

* `MFE_short = entry - min(low[t+1..t+H])`
* `MAE_short = entry - max(high[t+1..t+H])`

Normalize by ATR(t):

```
mfe_l = MFE_long / ATR_t
mae_l = MAE_long / ATR_t
mfe_s = MFE_short / ATR_t
mae_s = MAE_short / ATR_t
```

Store all 4 values.

---

# **TASK 8 — Direction Label Generator**

**Title:** Compute LONG / SHORT / FLAT class
**Type:** ML Labeling
**Specs:**
Given cost factor in ATR units (e.g., 0.05–0.1):

```
reward_long  = max(mfe_l - cost, 0)
reward_short = max(mfe_s - cost, 0)
```

Label rule:

```
if max(reward_long, reward_short) < threshold:
    class = 0   # FLAT
elif reward_long > reward_short:
    class = 1   # LONG
else:
    class = 2   # SHORT
```

Store `class`.

---

# **TASK 9 — Sequence Window Builder**

**Title:** Construct TCN-ready sliding windows
**Type:** ML Infra
**Specs:**

* seq_len = 128 bars
* For each timestamp with labels:

  ```
  X = features[t-127 : t]   # [128, feature_dim]
  y_class = class[t]
  y_reg   = [mfe_l, mae_l, mfe_s, mae_s]
  ```
* Skip if window contains missing_flag bars (optional).
* Append into memory dataset.

---

# **TASK 10 — Dataset Split + Normalization**

**Title:** Prepare train/val/test + scalers
**Type:** Preprocessing
**Specs:**

* Split chronologically:

  * train (60%)
  * validation (20%)
  * test (20%)
* Fit z-score normalization on **train only**.
* Apply to all splits AND exogenous data.
* Store normalization parameters for inference.

---

# **TASK 11 — PyTorch Dataset + Dataloader**

**Title:** Implement Dataset class
**Type:** ML Engineering
**Specs:**
Dataset returns:

```
X_t        # [128, feature_dim]
y_class_t  # int
y_reg_t    # [4]
```

Use DataLoader with:

* batch_size=64
* shuffle=True for train only
* num_workers=4

---

# **TASK 12 — TCN Model Implementation**

**Title:** Build TCN encoder + dual heads
**Type:** ML Model
**Specs:**
TCN encoder:

* 5–7 Conv1D dilation layers
* kernel_size=3
* dilation = 1,2,4,8,16…
* residual blocks
* LayerNorm after each block
* output = last timestep hidden state or pooled

Heads:

* **Direction head** → Linear(hidden_dim → 3)
* **Excursion head** → Linear(hidden_dim → 4)

Loss:

```
L = CE(direction) + λ * MSE(mfe/mae regression)
```

---

# **TASK 13 — Training Loop**

**Title:** Implement training + validation engine
**Type:** ML Training
**Specs:**

* Optimizer: AdamW(lr=1e-4)
* Scheduler: cosine annealing
* Early stopping on:

  * validation direction F1
  * regression MAE
* Log metrics per epoch:

  * CE_loss
  * reg_loss
  * total_loss
  * F1(long), F1(short)
* Save best model checkpoint.

---

# **TASK 14 — Walk-Forward Evaluation**

**Title:** Sliding window backtesting evaluation
**Type:** ML Infra
**Specs:**

* Train on year 1
* Validate on next 3 months
* Test on next 3 months
* Slide forward 3 months and repeat
* Aggregate metrics:

  * direction accuracy
  * regression error
  * equity curve in simulation

---

# **TASK 15 — Backtest Engine**

**Title:** Build offline trading simulator
**Type:** Trading Logic
**Specs:**
For each bar in test segment:

* Load last 128-bar X window
* Predict:

  * `p_flat`, `p_long`, `p_short`
  * `mfe_pred`, `mae_pred`
* Convert predicted excursions to SL/TP:

  * `SL = abs(mae_pred) * ATR`
  * `TP = mfe_pred * ATR`
* Simulate trade with:

  * spread
  * commission
  * slippage
* Track:

  * PnL
  * winrate
  * max drawdown
  * expectancy
  * trades/day

---

# **TASK 16 — Live Inference System**

**Title:** Real-time feature engine + prediction loop
**Type:** Backend / ML Inference
**Specs:**

* Subscribe to broker 5m candles
* Maintain rolling 128-bar window
* Compute features in real time:

  * technicals
  * session
  * macro regime
  * sentiment
  * event proximity
* On each new bar:

  * run model
  * log `p_long, p_short, p_flat`
  * log excursion predictions
  * create virtual trade according to backtest logic

---

# **TASK 17 — Monitoring & Drift Detection**

**Title:** Live model monitoring
**Type:** Monitoring
**Specs:**
Real-time metrics:

* rolling accuracy
* realized vs predicted MFE/MAE
* long/short ratio
* confidence distribution
* discrepancy from backtest

Trigger alerts if:

* excursion error spikes
* long/short imbalance goes extreme
* session-specific deterioration
* volatility regime mismatch

---

# **TASK 18 — Deployment**

**Title:** Package model + feature pipelines into service
**Type:** DevOps
**Specs:**

* Export model weights
* Containerize app
* Deploy API:

  * `/predict`
  * `/features`
* Provide cron jobs for:

  * news sentiment updates
  * macro data fetch
  * economic calendar refresh
