"""
Stock Data Preprocessing Pipeline
===================================
Handles multi-ticker datasets where each ticker has its own
time series (2016–2026, quarterly). Works for Mine_Data.csv
and all other sector files with the same structure.

Key challenges solved:
  1. Each ticker is an independent time series — rolling features
     must be computed PER ticker, never across tickers.
  2. Fundamental data (Revenue, ROA etc.) is only available for
     recent quarters — we forward-fill WITHIN each ticker.
  3. Rolling windows create NaN at the start of each ticker —
     we drop them AFTER feature engineering.
  4. Tickers with very few rows are useless for rolling features
     — we filter them out with a minimum row threshold.
  5. Train/Val/Test split must be TIME-BASED — never random,
     to avoid data leakage from the future.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# STEP 0 — Load and sort
# ─────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load a sector CSV file and sort by Ticker then Date."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort so each ticker's rows are in chronological order.
    # This is critical — rolling() and pct_change() depend on order.
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    print(f"Loaded: {len(df)} rows | {df['Ticker'].nunique()} tickers")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ─────────────────────────────────────────
# STEP 1 — Filter tickers with too few rows
# ─────────────────────────────────────────
def filter_short_tickers(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    """
    Remove tickers that don't have enough history.

    Why: Rolling windows (e.g. 8-quarter MA) need at least 8 rows
    to produce a single valid value. Tickers with 1–5 rows are
    newly listed stocks — they'll produce all-NaN features and
    contribute nothing to training.

    min_rows=10 is a safe minimum (gives at least 2 valid rows
    after an 8-quarter rolling window).
    """
    rows_per_ticker = df.groupby('Ticker').size()
    valid_tickers = rows_per_ticker[rows_per_ticker >= min_rows].index

    removed = df['Ticker'].nunique() - len(valid_tickers)
    df = df[df['Ticker'].isin(valid_tickers)].reset_index(drop=True)

    print(f"\nStep 1 — Filtered {removed} tickers with < {min_rows} rows")
    print(f"  Remaining: {df['Ticker'].nunique()} tickers | {len(df)} rows")
    return df


# ─────────────────────────────────────────
# STEP 2 — Feature engineering (per ticker)
# ─────────────────────────────────────────
FUNDAMENTAL_COLS = [
    'Revenue', 'NetIncome', 'TotalAssets', 'TotalDebt',
    'Equity', 'Cash', 'Debt_Equity', 'ROA', 'ROE'
]

def _engineer_one_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a single ticker's time series.

    This function is called independently for EACH ticker.
    Never mix rows from different tickers here.
    """
    g = group.copy().reset_index(drop=True)

    # ── Price-based features ──────────────────────────────
    # Quarter-over-quarter % change in closing price
    g['price_change_pct'] = g['close'].pct_change() * 100

    # Intra-quarter range as % of low (measures volatility of the quarter)
    g['high_low_range'] = (g['high'] - g['low']) / g['low'] * 100

    # Open-to-close movement within the quarter
    g['open_close_diff'] = (g['close'] - g['open']) / g['open'] * 100

    # ── Momentum ─────────────────────────────────────────
    # 4-quarter (1-year) and 2-quarter (6-month) price momentum
    g['momentum_4q'] = g['close'].pct_change(periods=4) * 100
    g['momentum_2q'] = g['close'].pct_change(periods=2) * 100

    # ── Volume features ───────────────────────────────────
    g['volume_change'] = g['volume'].pct_change() * 100
    g['volume_ma4']    = g['volume'].rolling(window=4).mean()
    # Volume ratio: is this quarter's volume above/below the 1-year average?
    g['volume_ratio']  = g['volume'] / g['volume_ma4']

    # ── Volatility ────────────────────────────────────────
    # Rolling std of quarterly returns over past 4 quarters
    g['volatility_4q'] = g['close'].pct_change().rolling(window=4).std() * 100

    # ── RSI (Relative Strength Index, adapted for quarterly data) ─
    # Measures whether the stock is overbought (RSI>70) or oversold (RSI<30)
    delta    = g['close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=4).mean()
    avg_loss = loss.rolling(window=4).mean()
    rs       = avg_gain / (avg_loss + 1e-9)   # +epsilon to avoid div/0
    g['RSI'] = 100 - (100 / (1 + rs))

    # ── Moving averages ───────────────────────────────────
    g['MA4'] = g['close'].rolling(window=4).mean()   # 1-year MA
    g['MA8'] = g['close'].rolling(window=8).mean()   # 2-year MA
    # Price relative to its 1-year average (>1 = above average = bullish)
    g['price_to_MA4'] = g['close'] / g['MA4']

    # ── Fundamental data — forward fill within ticker ─────
    # Financials (Revenue, Assets etc.) are only filed annually or
    # semi-annually, so many quarters show NaN. We fill forward:
    # the most recently reported value carries over until the next filing.
    # We NEVER fill across tickers (that's why this is inside the per-ticker function).
    g[FUNDAMENTAL_COLS] = g[FUNDAMENTAL_COLS].ffill()

    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering independently to each ticker."""
    all_groups = []
    for ticker, group in df.groupby('Ticker'):
        all_groups.append(_engineer_one_ticker(group))

    result = pd.concat(all_groups, ignore_index=True)
    new_cols = [c for c in result.columns if c not in df.columns]
    print(f"\nStep 2 — Feature engineering complete")
    print(f"  New features added: {new_cols}")
    return result


# ─────────────────────────────────────────
# STEP 3 — Drop NaN rows from rolling windows
# ─────────────────────────────────────────
FEATURE_COLS = [
    # Price features
    'price_change_pct', 'high_low_range', 'open_close_diff',
    # Momentum
    'momentum_4q', 'momentum_2q',
    # Volume
    'volume_change', 'volume_ratio',
    # Volatility & indicators
    'volatility_4q', 'RSI',
    # Moving averages
    'MA4', 'MA8', 'price_to_MA4',
    # Raw fundamentals (already forward-filled)
    'PE', 'EPS', 'BookValue',
    'Debt_Equity', 'ROA', 'ROE',
]

def drop_warmup_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where rolling-window features are NaN.

    Why: Rolling(8) on a ticker starting in 2016-Q1 won't produce
    a valid value until 2017-Q4 (8 quarters later). These NaN rows
    at the START of each ticker's history are expected and must be
    removed before training. We drop based on feature columns only.
    """
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    dropped = before - len(df)
    print(f"\nStep 3 — Dropped {dropped} NaN rows (rolling window warmup)")
    print(f"  Remaining: {len(df)} rows")
    return df


# ─────────────────────────────────────────
# STEP 4 — Time-based train/val/test split
# ─────────────────────────────────────────
def time_split(df: pd.DataFrame,
               val_start:  str = '2023-01-01',
               test_start: str = '2024-01-01') -> tuple:
    """
    Split data strictly by time — NEVER randomly.

    Why not random split?
      A random split would put 2024 data in training and 2022 data
      in test. The model would "see the future" during training,
      leading to falsely high accuracy that collapses in production.

    Train: all data before val_start
    Val:   val_start to test_start (tune hyperparameters here)
    Test:  test_start onwards (final evaluation, touch only once)
    """
    train = df[df['Date'] <  val_start].copy()
    val   = df[(df['Date'] >= val_start) & (df['Date'] < test_start)].copy()
    test  = df[df['Date'] >= test_start].copy()

    print(f"\nStep 4 — Time-based split")
    print(f"  Train : {len(train):>5} rows | {train['Date'].min().date()} → {train['Date'].max().date()} | Labels: {train['Label'].value_counts().to_dict()}")
    print(f"  Val   : {len(val):>5} rows | {val['Date'].min().date()  } → {val['Date'].max().date()  } | Labels: {val['Label'].value_counts().to_dict()}")
    print(f"  Test  : {len(test):>5} rows | {test['Date'].min().date() } → {test['Date'].max().date() } | Labels: {test['Label'].value_counts().to_dict()}")
    return train, val, test


# ─────────────────────────────────────────
# STEP 5 — Scale features
# ─────────────────────────────────────────
def scale_features(train: pd.DataFrame,
                   val:   pd.DataFrame,
                   test:  pd.DataFrame) -> tuple:
    """
    Fit StandardScaler on TRAIN only, then transform all three splits.

    Why fit only on train?
      Fitting on val+test would leak their statistics (mean, std)
      into training — another form of data leakage.

    Returns X_train, X_val, X_test (numpy arrays) and y splits.
    """
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train[FEATURE_COLS])
    X_val   = scaler.transform(val[FEATURE_COLS])
    X_test  = scaler.transform(test[FEATURE_COLS])

    y_train = train['Label'].values
    y_val   = val['Label'].values
    y_test  = test['Label'].values

    print(f"\nStep 5 — Feature scaling complete")
    print(f"  Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


# ─────────────────────────────────────────
# MAIN — Run full pipeline
# ─────────────────────────────────────────
def run_pipeline(filepath: str):
    print("=" * 55)
    print(f" PREPROCESSING PIPELINE: {filepath.split('/')[-1]}")
    print("=" * 55)

    df = load_data(filepath)
    df = filter_short_tickers(df, min_rows=10)
    df = engineer_features(df)
    df = drop_warmup_nans(df)
    train, val, test = time_split(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = scale_features(train, val, test)

    print("\n" + "=" * 55)
    print(" PIPELINE COMPLETE — ready for model training")
    print("=" * 55)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
        'feature_cols': FEATURE_COLS,
        # Keep the full processed dataframes too (useful for analysis)
        'train_df': train,
        'val_df':   val,
        'test_df':  test,
    }


# ─────────────────────────────────────────
# Run it
# ─────────────────────────────────────────
if __name__ == '__main__':
    result = run_pipeline('/Users/shivam0701/Study/project_stock/dataset/Mine_Data.csv')

    # Quick sanity check
    import xgboost as xgb
    from sklearn.metrics import classification_report, roc_auc_score

    print("\n── Quick XGBoost sanity check ──")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(
        result['X_train'], result['y_train'],
        eval_set=[(result['X_val'], result['y_val'])],
        verbose=False
    )
    y_pred = model.predict(result['X_test'])
    y_prob = model.predict_proba(result['X_test'])[:, 1]

    print(classification_report(result['y_test'], y_pred, target_names=['Sell/Hold', 'Buy']))
    print(f"ROC-AUC: {roc_auc_score(result['y_test'], y_prob):.3f}")