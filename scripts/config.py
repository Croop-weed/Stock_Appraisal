"""
API Configuration
=================
Modify these settings to customize your API behavior.
"""

# ═════════════════════════════════════════════════════════════════════
# SERVER SETTINGS
# ═════════════════════════════════════════════════════════════════════

# Host and port for the API server
API_HOST = "0.0.0.0"  # 0.0.0.0 = accessible from any machine
API_PORT = 8000       # Change to 8001, 8002, etc. if 8000 is busy
API_RELOAD = True     # Auto-reload on file changes (dev mode)

# ═════════════════════════════════════════════════════════════════════
# MODEL SETTINGS
# ═════════════════════════════════════════════════════════════════════

# Minimum quarters of data required per ticker
MIN_QUARTERS = 12

# Time-based train/validation/test split
TRAIN_END = "2023-01-01"   # Train data: before this date
VAL_END = "2024-01-01"     # Validation: between dates
TEST_START = "2024-01-01"  # Test data: after this date

# GradientBoosting hyperparameters
MODEL_CONFIG = {
    'n_estimators': 300,      # Number of trees
    'max_depth': 4,           # Max tree depth
    'learning_rate': 0.05,    # How fast model learns
    'subsample': 0.8,         # Fraction of samples per tree
    'random_state': 42,       # For reproducibility
}

# ═════════════════════════════════════════════════════════════════════
# PREDICTION THRESHOLDS
# ═════════════════════════════════════════════════════════════════════

# Confidence threshold for "Strong Buy" recommendation
STRONG_BUY_THRESHOLD = 0.65

# Confidence threshold for "Strong Sell" recommendation
STRONG_SELL_THRESHOLD = 0.35

# ═════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════

# Base features used by the model
FEATURES = {
    'momentum': {
        'ret_1q': 'Last quarter return (%)',
        'ret_2q': '2-quarter return (%)',
        'ret_4q': '1-year return (%)',
        'ret_8q': '2-year return (%)',
    },
    'trend': {
        'price_to_ma4': 'Price vs 1-year MA (%)',
        'price_to_ma8': 'Price vs 2-year MA (%)',
        'ma4_to_ma8': 'Golden/death cross',
    },
    'volatility': {
        'vol_4q': '4-quarter volatility',
        'vol_8q': '8-quarter volatility',
        'RSI': 'Relative Strength Index',
    },
    'volume': {
        'vol_change_1q': 'Volume change QoQ',
        'vol_ratio': 'Volume vs 4q avg',
        'vol_ratio_8q': 'Volume vs 8q avg',
    },
    'valuation': {
        'P_to_E': 'Price-to-Earnings',
        'P_to_Book': 'Price-to-Book',
        'earnings_yield': 'Earnings yield',
    },
    'fundamental': {
        'ROA_val': 'Return on Assets',
        'ROE_val': 'Return on Equity',
        'D_E': 'Debt-to-Equity ratio',
    }
}

# ═════════════════════════════════════════════════════════════════════
# DATA FILES
# ═════════════════════════════════════════════════════════════════════

# CSV file locations (relative to project root)
SECTOR_DATA_FILES = {
    'Mine': 'Mine_Data.csv',
    'Automobile': 'Automobile_Data.csv',
    'Bank': 'Bank_Data.csv',
    'Pharma': 'pharma_symbols_Data.csv',
    'FMCG': 'FMCG_Data.csv',
    'Energy': 'Energy_Data.csv',
    'Metal': 'Metal_Data.csv',
    'Infra': 'Infra_Data.csv',
    'HCEquip': 'raw_Healthcare_Equipment_&_Supplies_Companies_Data.csv',
    'HCServ': 'rawHealthcare_Services_Companies_Data.csv',
}

# Model files locations (auto-created)
MODELS_DIR = 'models'
MODEL_FILE = f'{MODELS_DIR}/stock_model.pkl'
SCALER_FILE = f'{MODELS_DIR}/scaler.pkl'
METADATA_FILE = f'{MODELS_DIR}/metadata.pkl'

# ═════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════

# Customize recommendation messages
RECOMMENDATIONS = {
    'strong_buy': {
        'prediction': 'BUY',
        'message': 'Strong Buy - Stock likely to appreciate next quarter',
        'confidence': 0.65,  # >= this
    },
    'buy': {
        'prediction': 'BUY',
        'message': 'Buy - Positive momentum expected',
        'confidence': 0.55,  # >= this
    },
    'hold': {
        'prediction': 'BUY',
        'message': 'Hold - Neutral signals',
        'confidence': 0.50,  # >= this
    },
    'strong_sell': {
        'prediction': 'SELL',
        'message': 'Strong Sell - Caution recommended',
        'confidence': 0.35,  # <= this
    },
    'sell': {
        'prediction': 'SELL',
        'message': 'Sell - Downside risk',
        'confidence': 0.50,  # > this
    },
}

# ═════════════════════════════════════════════════════════════════════
# LOGGING & DEBUG
# ═════════════════════════════════════════════════════════════════════

# Enable debug logging
DEBUG = True

# Save prediction cache (speeds up repeated requests)
USE_CACHE = True
CACHE_EXPIRY_HOURS = 1

# ═════════════════════════════════════════════════════════════════════
# API DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════

API_TITLE = "Stock Prediction API"
API_DESCRIPTION = """
Real-time stock buy/sell predictions using ML model trained on 
quarterly K-line data across 10 sectors (16,000+ stocks).

### Features:
- Single ticker predictions
- Batch predictions
- Model performance metrics
- Automatic model training

### Endpoints:
- `/predict/{ticker}` - Get prediction for one stock
- `/batch-predict` - Predict for multiple stocks
- `/model/stats` - View model performance
- `/docs` - Interactive API documentation
"""
API_VERSION = "1.0.0"

# ═════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═════════════════════════════════════════════════════════════════════

EXAMPLE_TICKERS = [
    "RELIANCE.NS",    # Energy
    "TCS.NS",         # IT
    "INFY.NS",        # IT
    "HDFC.NS",        # Banking
    "WIPRO.NS",       # IT
    "AXISBANK.NS",    # Banking
    "NTPC.NS",        # Energy
    "MARUTI.NS",      # Automobile
    "SUNPHARMA.NS",   # Pharma
]

# ═════════════════════════════════════════════════════════════════════
# ADVANCED: OVERRIDE THESE ONLY IF YOU KNOW WHAT YOU'RE DOING
# ═════════════════════════════════════════════════════════════════════

# Market data source (yfinance by default)
MARKET_DATA_SOURCE = "yfinance"

# Historical data lookback (years)
HISTORICAL_YEARS = 5

# Quarterly data interval
DATA_INTERVAL = "3mo"  # 1mo, 3mo, etc.

# Feature scaling method
SCALER_TYPE = "StandardScaler"  # MinMaxScaler, RobustScaler, etc.

# Model type
MODEL_TYPE = "GradientBoostingClassifier"

# ═════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════

# Suppress warnings
SUPPRESS_WARNINGS = True

# Number of workers for parallel processing
WORKERS = 4

print("✓ Configuration loaded successfully")
