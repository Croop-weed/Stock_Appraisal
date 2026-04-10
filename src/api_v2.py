"""
Stock Prediction FastAPI Backend (Reorganized)
==============================================
Updated to work with organized project directory structure.

Run: python run.py
Or:  uvicorn src.api:app --reload
"""

import os
import sys
import pickle
import warnings
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═════════════════════════════════════════════════════════════════════

# api_v2.py is in src/, so go up one level to find project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

# Create dirs if missing
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SRC_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Sector CSV files
SECTOR_FILES = {
    'Mine':       DATA_DIR / 'Mine_Data.csv',
    'Automobile': DATA_DIR / 'Automobile_Data.csv',
    'Bank':       DATA_DIR / 'Bank_Data.csv',
    'Pharma':     DATA_DIR / 'pharma_symbols_Data.csv',
    'FMCG':       DATA_DIR / 'FMCG_Data.csv',
    'Energy':     DATA_DIR / 'Energy_Data.csv',
    'Metal':      DATA_DIR / 'Metal_Data.csv',
    'Infra':      DATA_DIR / 'Infra_Data.csv',
    'HCEquip':    DATA_DIR / 'raw_Healthcare_Equipment_&_Supplies_Companies_Data.csv',
    'HCServ':     DATA_DIR / 'rawHealthcare_Services_Companies_Data.csv',
}

MODEL_PATH = MODEL_DIR / 'stock_model.pkl'
SCALER_PATH = MODEL_DIR / 'scaler.pkl'
METADATA_PATH = MODEL_DIR / 'metadata.pkl'

MIN_ROWS = 12
VAL_START = '2023-01-01'
TEST_START = '2024-01-01'

FEATURE_COLS_BASE = [
    'ret_1q', 'ret_2q', 'ret_4q', 'ret_8q',
    'price_to_ma4', 'price_to_ma8', 'ma4_to_ma8',
    'high_low_pct', 'open_close_pct',
    'vol_4q', 'vol_8q', 'RSI',
    'vol_change_1q', 'vol_ratio', 'vol_ratio_8q',
    'drawdown_8q',
    'P_to_E', 'P_to_Book', 'earnings_yield',
    'ROA_val', 'ROE_val', 'D_E',
    'label_lag1', 'label_lag2', 'wins_4q',
]

FUNDAMENTAL_COLS = [
    'Revenue', 'NetIncome', 'TotalAssets', 'TotalDebt',
    'Equity', 'Cash', 'Debt_Equity', 'ROA', 'ROE'
]

# ═════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    price_change_pct: float
    sector: str
    prediction: str
    confidence: float
    model_probability: float
    recommendation: str
    timestamp: str


class TickerInfoResponse(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    market_cap: str
    pe_ratio: float
    sector: str
    available: bool


class BatchPredictRequest(BaseModel):
    tickers: list[str]


class BatchPredictResponse(BaseModel):
    count: int
    timestamp: str
    predictions: list[PredictionResponse]


class ModelStatsResponse(BaseModel):
    model_trained: bool
    training_date: str
    total_features: int
    feature_columns: list[str]
    training_samples: int
    validation_auc: float
    test_auc: float
    top_10_features: dict


# ═════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════

def build_ticker_features(group: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Engineer features for a single ticker."""
    try:
        g = group.copy().reset_index(drop=True)
        
        # Flatten MultiIndex columns if they exist (from yfinance)
        if isinstance(g.columns, pd.MultiIndex):
            g.columns = [col[0] if col[1] == '' else col[0] for col in g.columns]
        
        c = g['close']
        
        # Momentum
        g['ret_1q'] = c.pct_change(1) * 100
        g['ret_2q'] = c.pct_change(2) * 100
        g['ret_4q'] = c.pct_change(4) * 100
        g['ret_8q'] = c.pct_change(8) * 100

        # Moving averages (relative)
        ma4 = c.rolling(4).mean()
        ma8 = c.rolling(8).mean()
        g['price_to_ma4'] = (c / ma4 - 1) * 100
        g['price_to_ma8'] = (c / ma8 - 1) * 100
        g['ma4_to_ma8']   = (ma4 / ma8 - 1) * 100

        # Intra-quarter
        g['high_low_pct']   = (g['high'] - g['low']) / g['low'] * 100
        g['open_close_pct'] = (g['close'] - g['open']) / g['open'] * 100

        # Volatility
        ret = c.pct_change()
        g['vol_4q'] = ret.rolling(4).std() * 100
        g['vol_8q'] = ret.rolling(8).std() * 100

        # RSI
        delta    = c.diff()
        avg_gain = delta.clip(lower=0).rolling(4).mean()
        avg_loss = (-delta.clip(upper=0)).rolling(4).mean()
        g['RSI'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-9)))

        # Volume
        vol = g['volume']
        g['vol_change_1q'] = vol.pct_change() * 100
        g['vol_ratio']     = vol / vol.rolling(4).mean()
        g['vol_ratio_8q']  = vol / vol.rolling(8).mean()

        # Drawdown
        g['drawdown_8q'] = (c / c.rolling(8).max() - 1) * 100

        # Fundamentals - only fill columns that exist
        for col in FUNDAMENTAL_COLS:
            if col in g.columns:
                g[col] = g[col].ffill()

        # Valuation - use .values to get Series instead of DataFrame
        if 'PE' in g.columns:
            g['P_to_E'] = g['PE'].values
            g['P_to_E'] = g['P_to_E'].clip(-500, 500)
        else:
            g['P_to_E'] = 0.0
            
        if 'BookValue' in g.columns:
            g['P_to_Book'] = (c / g['BookValue'].values.flatten()).clip(-200, 200)
        else:
            g['P_to_Book'] = 0.0
            
        g['earnings_yield'] = (1 / (g['P_to_E'].replace(0, np.nan)) * 100).clip(-500, 500)
        
        if 'ROA' in g.columns:
            g['ROA_val'] = g['ROA'].fillna(0)
        else:
            g['ROA_val'] = 0.0
            
        if 'ROE' in g.columns:
            g['ROE_val'] = g['ROE'].fillna(0)
        else:
            g['ROE_val'] = 0.0
            
        if 'Debt_Equity' in g.columns:
            g['D_E'] = g['Debt_Equity'].values.flatten()
            g['D_E'] = g['D_E'].clip(-50, 50).fillna(0)
        else:
            g['D_E'] = 0.0

        # Lagged labels
        if 'Label' in g.columns:
            g['label_lag1'] = g['Label'].shift(1)
            g['label_lag2'] = g['Label'].shift(2)
            g['wins_4q'] = g['Label'].shift(1).rolling(4).sum()
        else:
            g['label_lag1'] = 0
            g['label_lag2'] = 0
            g['wins_4q'] = 0

        g['Sector'] = sector
        return g
    except Exception as e:
        print(f"[DEBUG] Feature engineering failed! Error: {e}")
        print(f"[DEBUG] Input group shape: {group.shape}")
        print(f"[DEBUG] Input columns: {list(group.columns)}")
        print(f"[DEBUG] Copy shape: {g.shape if 'g' in locals() else 'N/A'}")
        print(f"[DEBUG] 'close' in group: {'close' in group.columns}")
        import traceback
        traceback.print_exc()
        raise


# ═════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════

def load_all_sectors() -> pd.DataFrame:
    """Load and process all sector files."""
    all_dfs = []
    for sector, fpath in SECTOR_FILES.items():
        if not fpath.exists():
            print(f"  ⚠️  {sector} file not found: {fpath}")
            continue
        try:
            df = pd.read_csv(fpath)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

            groups = []
            for ticker, group in df.groupby('Ticker'):
                if len(group) < MIN_ROWS:
                    continue
                groups.append(build_ticker_features(group, sector))

            if groups:
                sec_df = pd.concat(groups, ignore_index=True)
                all_dfs.append(sec_df)
                print(f"  ✓ {sector:<12}: {len(sec_df):>5} rows | {sec_df['Ticker'].nunique():>3} tickers")
        except Exception as e:
            print(f"  ✗ Error loading {sector}: {e}")

    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def train_model():
    """Train and save the GradientBoosting model."""
    print("[TRAIN] Loading and preprocessing data...")
    combined = load_all_sectors()

    if combined.empty:
        print("ERROR: No data loaded!")
        return False

    # Sector encoding
    combined = pd.get_dummies(combined, columns=['Sector'], prefix='sec')
    sec_cols = [c for c in combined.columns if c.startswith('sec_')]
    all_feats = FEATURE_COLS_BASE + sec_cols

    # Clean NaN
    combined[all_feats] = combined[all_feats].replace([np.inf, -np.inf], np.nan)
    combined[sec_cols] = combined[sec_cols].fillna(0)

    required_feat = [f for f in FEATURE_COLS_BASE if f not in
                     ['P_to_E', 'P_to_Book', 'earnings_yield', 'ROA_val', 'ROE_val', 'D_E']]
    clean = combined.dropna(subset=required_feat).reset_index(drop=True)

    for col in ['P_to_E', 'P_to_Book', 'earnings_yield', 'ROA_val', 'ROE_val', 'D_E']:
        if col in clean.columns:
            clean[col] = clean[col].fillna(clean[col].median())

    # Time split
    train = clean[clean['Date'] < VAL_START].copy()
    val = clean[(clean['Date'] >= VAL_START) & (clean['Date'] < TEST_START)].copy()
    test = clean[clean['Date'] >= TEST_START].copy()

    print(f"[TRAIN] Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[all_feats])
    X_val = scaler.transform(val[all_feats])
    X_test = scaler.transform(test[all_feats])
    y_train = train['Label'].values
    y_val = val['Label'].values
    y_test = test['Label'].values

    # Train
    print("[TRAIN] Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"[TRAIN] ✓ Val AUC: {val_auc:.3f} | Test AUC: {test_auc:.3f}")

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        'training_date': datetime.now().isoformat(),
        'feature_cols': all_feats,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'feature_importance': dict(zip(all_feats, model.feature_importances_)),
        'training_samples': len(train),
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"[TRAIN] ✓ Model saved to {MODEL_PATH}")
    return True


# ═════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Stock Prediction API",
    description="Real-time stock predictions using ML model",
    version="2.0.0"
)

g_model = None
g_scaler = None
g_metadata = None


def load_model():
    """Load trained model and scaler."""
    global g_model, g_scaler, g_metadata
    
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            g_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            g_scaler = pickle.load(f)
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'rb') as f:
                g_metadata = pickle.load(f)
        print("[API] ✓ Model loaded successfully")
        return True
    return False


@app.on_event("startup")
async def startup():
    """Initialize model on startup."""
    if not load_model():
        print("[API] ⚠️  Model not found. Training new model...")
        train_model()
        load_model()


# ═════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ═════════════════════════════════════════════════════════════════════

def get_sector_for_ticker(ticker: str) -> str:
    """Find which sector a ticker belongs to."""
    for sector, fpath in SECTOR_FILES.items():
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            if ticker in df['Ticker'].values:
                return sector
        except:
            pass
    return "Unknown"


def predict_for_ticker(ticker: str) -> dict:
    """Generate prediction for a single ticker with improved error handling."""
    if g_model is None or g_scaler is None or g_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Validate and clean ticker
        ticker = ticker.upper().strip()
        if not ticker or len(ticker) > 20:
            raise HTTPException(status_code=400, detail="Invalid ticker format (too long)")

        print(f"[PREDICT] Requesting: {ticker}")

        # Get current data from yfinance with error handling
        try:
            hist = yf.download(ticker, period="5y", progress=False, interval="3mo")
        except Exception as e:
            error_msg = str(e)
            if "delisted" in error_msg.lower() or "no data found" in error_msg.lower():
                raise HTTPException(status_code=404, detail=f"Ticker {ticker} appears to be delisted or invalid")
            raise HTTPException(status_code=404, detail=f"Failed to download data for {ticker}")
        
        # Validate data was retrieved
        if hist is None or hist.empty or len(hist) < 3:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient data for {ticker}. Ticker may be delisted, invalid, or newly listed (needs 3+ years)"
            )

        try:
            current_info = yf.Ticker(ticker)
            info = current_info.info if hasattr(current_info, 'info') else {}
        except:
            info = {}
        
        # Prepare data - handle different column names
        try:
            df = hist.reset_index()
            
            # Standardize column names
            col_mapping = {
                'Open': 'open', 'HIGH': 'high', 'High': 'high',
                'Low': 'low', 'LOW': 'low',
                'Close': 'close', 'CLOSE': 'close',
                'Volume': 'volume', 'VOLUME': 'volume',
                'Adj Close': 'closeadj', 'Adjusted Close': 'closeadj'
            }
            
            for old_col, new_col in col_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required = ['Date', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing columns in data: {missing}")
            
            df['Ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[['Date', 'Ticker', 'open', 'high', 'low', 'close', 'volume']]
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Data format error: {str(e)[:40]}")

        # Add fundamental data
        for col in FUNDAMENTAL_COLS:
            df[col] = 0
        df['PE'] = info.get('trailingPE', 0)
        df['EPS'] = info.get('trailingEps', 0)
        df['BookValue'] = info.get('bookValue', 0)
        df['Label'] = 0

        # Feature engineering
        try:
            features_df = build_ticker_features(df, get_sector_for_ticker(ticker))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)[:40]}")

        # Get last valid row
        features_df = features_df.dropna(subset=FEATURE_COLS_BASE)
        if features_df.empty or len(features_df) < 1:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient history for {ticker}. Need at least 8 quarters of valid data."
            )

        last_row = features_df.iloc[-1]

        # Create feature vector with robust error handling
        ticker_sector = get_sector_for_ticker(ticker)
        feature_vector = []
        
        for col in g_metadata['feature_cols']:
            try:
                if col.startswith('sec_'):
                    feature_vector.append(1.0 if col == f'sec_{ticker_sector}' else 0.0)
                elif col in last_row.index:
                    val = last_row[col]
                    if pd.isna(val) or np.isinf(val):
                        feature_vector.append(0.0)
                    else:
                        feature_vector.append(float(val))
                else:
                    feature_vector.append(0.0)
            except:
                feature_vector.append(0.0)

        X = np.array(feature_vector).reshape(1, -1)
        
        # Clean NaN/Inf in feature vector
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = g_scaler.transform(X)

        # Predict
        prediction_int = g_model.predict(X_scaled)[0]
        probability = g_model.predict_proba(X_scaled)[0, 1]

        current_price = float(last_row['close'])
        prev_close = float(df.iloc[-2]['close']) if len(df) > 1 else current_price
        price_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0

        prediction_label = "BUY" if prediction_int == 1 else "SELL"
        confidence = max(probability, 1 - probability)

        recommendation = (
            "Strong Buy - Stock likely to appreciate next quarter"
            if probability > 0.65
            else "Buy - Positive momentum expected"
            if prediction_int == 1
            else "Sell - Caution recommended"
            if probability < 0.35
            else "Hold - Neutral signals"
        )

        print(f"[PREDICT] ✓ {ticker}: {prediction_label} (conf={confidence:.1%})")

        return {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'previous_close': round(prev_close, 2),
            'price_change_pct': round(price_change_pct, 2),
            'sector': ticker_sector,
            'prediction': prediction_label,
            'confidence': round(confidence, 3),
            'model_probability': round(probability, 3),
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)[:80]}")


# ═════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    """API root - returns API information."""
    return {
        "name": "Stock Prediction API v2",
        "status": "running",
        "endpoints": {
            "/predict/{ticker}": "GET - Single ticker prediction",
            "/ticker-info/{ticker}": "GET - Ticker information",
            "/batch-predict": "POST - Batch predictions",
            "/model/stats": "GET - Model statistics",
            "/model/retrain": "POST - Retrain model",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        }
    }


@app.get("/predict/{ticker}", response_model=PredictionResponse, tags=["Predictions"])
async def predict(ticker: str):
    """Get prediction for a single ticker."""
    result = predict_for_ticker(ticker.upper())
    return PredictionResponse(**result)


@app.get("/ticker-info/{ticker}", response_model=TickerInfoResponse, tags=["Info"])
async def ticker_info(ticker: str):
    """Get basic information for a ticker."""
    try:
        data = yf.Ticker(ticker)
        info = data.info
        
        return TickerInfoResponse(
            ticker=ticker.upper(),
            company_name=info.get('longName', 'Unknown'),
            current_price=info.get('currentPrice', 0),
            market_cap=info.get('marketCap', 'N/A'),
            pe_ratio=info.get('trailingPE', 0),
            sector=get_sector_for_ticker(ticker),
            available=True
        )
    except:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")


@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["Predictions"])
async def batch_predict(request: BatchPredictRequest):
    """Get predictions for multiple tickers."""
    results = []
    for ticker in request.tickers:
        try:
            result = predict_for_ticker(ticker.upper())
            results.append(PredictionResponse(**result))
        except Exception as e:
            print(f"[BATCH] Skipping {ticker}: {str(e)[:50]}")

    return BatchPredictResponse(
        count=len(results),
        timestamp=datetime.now().isoformat(),
        predictions=results
    )


@app.get("/model/stats", response_model=ModelStatsResponse, tags=["Model"])
async def model_stats():
    """Get model statistics and performance metrics."""
    if g_metadata is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    fi = g_metadata.get('feature_importance', {})
    top_10 = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10])

    return ModelStatsResponse(
        model_trained=g_model is not None,
        training_date=g_metadata.get('training_date', 'Unknown'),
        total_features=len(g_metadata.get('feature_cols', [])),
        feature_columns=g_metadata.get('feature_cols', []),
        training_samples=g_metadata.get('training_samples', 0),
        validation_auc=g_metadata.get('val_auc', 0),
        test_auc=g_metadata.get('test_auc', 0),
        top_10_features=top_10
    )


@app.post("/model/retrain", tags=["Model"])
async def retrain_model():
    """Retrain the model with latest data."""
    try:
        success = train_model()
        if success:
            load_model()
            return {"status": "success", "message": "Model retrained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/health", tags=["Info"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": g_model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


# ═════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
