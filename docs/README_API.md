# Stock Prediction API Backend

A FastAPI-based REST API for real-time stock predictions using a trained GradientBoosting machine learning model.

## Features

✅ Real-time stock price and prediction via API
✅ Batch predictions for multiple tickers
✅ Model statistics and feature importance
✅ Automatic model training on first run
✅ Interactive Swagger documentation
✅ Built-in health checks

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python api.py
```

Or with uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will start at `http://localhost:8000`

## OpenAPI Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Get Prediction for a Single Ticker

```bash
GET /predict/{ticker}
```

**Example:**
```bash
curl "http://localhost:8000/predict/RELIANCE.NS"
```

**Response:**
```json
{
  "ticker": "RELIANCE.NS",
  "current_price": 2854.50,
  "previous_close": 2847.00,
  "price_change_pct": 0.26,
  "sector": "Energy",
  "prediction": "BUY",
  "confidence": 0.628,
  "model_probability": 0.628,
  "recommendation": "Buy - Positive momentum expected",
  "timestamp": "2024-04-05T10:30:45.123456"
}
```

**Parameters:**
- `ticker` (string, required): Stock ticker symbol (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`)

**Returns:**
- `current_price`: Latest closing price
- `prediction`: "BUY" or "SELL" signal
- `confidence`: Model confidence (0-1)
- `model_probability`: Raw probability of BUY
- `recommendation`: Detailed recommendation text

---

### Get Ticker Information

```bash
GET /ticker-info/{ticker}
```

**Example:**
```bash
curl "http://localhost:8000/ticker-info/TCS.NS"
```

**Response:**
```json
{
  "ticker": "TCS.NS",
  "company_name": "Tata Consultancy Services Limited",
  "current_price": 3625.00,
  "market_cap": "3.5T",
  "pe_ratio": 28.5,
  "sector": "Bank",
  "available": true
}
```

---

### Batch Predictions

```bash
POST /batch-predict
Content-Type: application/json

{
  "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}'
```

**Response:**
```json
{
  "count": 3,
  "timestamp": "2024-04-05T10:35:12.654321",
  "predictions": [
    {
      "ticker": "RELIANCE.NS",
      "current_price": 2854.50,
      "prediction": "BUY",
      "confidence": 0.628,
      "model_probability": 0.628,
      "recommendation": "Buy - Positive momentum expected",
      "timestamp": "2024-04-05T10:35:12.654321"
    },
    ...
  ]
}
```

---

### Model Statistics

```bash
GET /model/stats
```

**Example:**
```bash
curl "http://localhost:8000/model/stats"
```

**Response:**
```json
{
  "model_trained": true,
  "training_date": "2024-04-05T08:45:30.123456",
  "total_features": 35,
  "feature_columns": [
    "ret_1q", "ret_2q", "ret_4q", ..., "sec_Energy", "sec_Bank"
  ],
  "training_samples": 15500,
  "validation_auc": 0.556,
  "test_auc": 0.584,
  "top_10_features": {
    "ret_8q": 0.0845,
    "vol_4q": 0.0756,
    "price_to_ma8": 0.0692,
    ...
  }
}
```

---

### Retrain Model

```bash
POST /model/retrain
```

**Example:**
```bash
curl -X POST "http://localhost:8000/model/retrain"
```

**Response:**
```json
{
  "status": "success",
  "message": "Model retrained successfully"
}
```

This loads all CSV data files, retrains the model, and saves it to disk.

---

### Health Check

```bash
GET /health
```

**Example:**
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-04-05T10:30:45.123456"
}
```

---

## Model Features

The model uses 35 engineered features across 4 categories:

### 1. Momentum Features
- `ret_1q`, `ret_2q`, `ret_4q`, `ret_8q`: Returns over different timeframes
- `label_lag1`, `label_lag2`: Previous quarters' signals

### 2. Trend Features
- `price_to_ma4`, `price_to_ma8`: Price vs moving average
- `ma4_to_ma8`: Golden/death cross indicator
- `drawdown_8q`: Distance from 2-year high

### 3. Volume & Volatility
- `vol_ratio`, `vol_ratio_8q`: Volume anomalies
- `vol_4q`, `vol_8q`: Historical volatility
- `RSI`: Relative strength indicator

### 4. Valuation
- `P_to_E`: Price-to-earnings ratio
- `P_to_Book`: Price-to-book value
- `earnings_yield`: Earnings yield
- `ROA_val`, `ROE_val`, `D_E`: Profitability metrics

### 5. Sector Encoding
- One-hot encoded sector (Mine, Automobile, Bank, Energy, etc.)

## Model Performance

The model trains on ~15,500 quarterly records across 10 sectors:

- **Validation AUC**: ~0.556
- **Test AUC**: ~0.584

These metrics are typical for quarterly stock direction prediction. A 0.58 AUC indicates the model performs better than random (0.5) and is useful for directional trading signals.

## Prediction Interpretation

### Confidence Levels

- **Confidence > 0.65**: Strong signal (High conviction)
- **Confidence 0.55-0.65**: Moderate signal
- **Confidence 0.50-0.55**: Weak signal (Use with caution)

### Recommendation Categories

1. **Strong Buy**: Probability > 65% — Stock likely to appreciate
2. **Buy**: Signal =  BUY with moderate confidence
3. **Sell**: Caution recommended — Downside risk
4. **Hold**: Mixed signals — Neutral outlook

## Data Flow

1. **Input**: Ticker symbol (e.g., `RELIANCE.NS`)
2. **Data Collection**: Download 5 years of quarterly data from yfinance
3. **Feature Engineering**: Compute 35 features including momentum, volatility, fundamentals
4. **Scaling**: Standardize using pre-trained scaler
5. **Prediction**: Pass through GradientBoosting model
6. **Output**: {prediction, confidence, recommendation}

## Technical Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **ML Model**: GradientBoostingClassifier (scikit-learn)
- **Data**: pandas, numpy
- **Market Data**: yfinance
- **API Documentation**: Swagger/OpenAPI

## Directory Structure

```
Stock predictor/
├── api.py                          # Main FastAPI application
├── d1.py                           # Preprocessing pipeline (reference)
├── d2.py                           # Model training pipeline (reference)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Mine_Data.csv                   # Sector data
├── Automobile_Data.csv
├── Bank_Data.csv
├── ... (other sector CSVs)
└── models/                         # Auto-created on first run
    ├── stock_model.pkl             # Trained model
    ├── scaler.pkl                  # Feature scaler
    └── metadata.pkl                # Model metadata
```

## First Run

On first execution, the API will:
1. Check for existing trained model in `models/` directory
2. If not found, load all CSV files and automatically train a new model
3. Save the model, scaler, and metadata to disk
4. Use cached model on subsequent runs

Training takes ~2-5 minutes depending on data size. You'll see progress in the console.

## Example: Python Client

```python
import requests

# Single prediction
response = requests.get("http://localhost:8000/predict/RELIANCE.NS")
data = response.json()

print(f"{data['ticker']}: {data['prediction']}")
print(f"  Price: ₹{data['current_price']}")
print(f"  Confidence: {data['confidence']*100:.1f}%")
print(f"  Recommendation: {data['recommendation']}")


# Batch predictions
response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}
)
predictions = response.json()

for pred in predictions['predictions']:
    print(f"{pred['ticker']}: {pred['prediction']}")


# Model stats
response = requests.get("http://localhost:8000/model/stats")
stats = response.json()

print(f"Model trained on: {stats['training_samples']} samples")
print(f"Test AUC: {stats['test_auc']:.3f}")
print(f"Top features: {list(stats['top_10_features'].keys())[:5]}")
```

## Troubleshooting

### "Model not loaded" error
- Ensure data CSV files are in the project directory
- Check `models/` directory permissions
- Run `/model/retrain` endpoint to force retraining

### "Ticker not found" error
- Verify ticker symbol is correct (e.g., `RELIANCE.NS` for NSE listing)
- Ensure yfinance has internet connectivity
- Try a popular ticker like `RELIANCE.NS` first

### "Insufficient data for ticker" error
- Ticker has too few quarters of history
- Model requires at least 12 quarters of data
- Typical for newly listed stocks

### Port already in use
```bash
# Run on different port
python api.py --port 8001
# Or kill existing process
lsof -i :8000
```

## Performance Tips

1. **Batch predictions** are faster than individual calls — use `/batch-predict` for multiple tickers
2. **Model retraining** runs monthly or when new sector data is available
3. **Caching**: Model is loaded into memory on startup — no reload needed
4. **Concurrency**: Uvicorn handles concurrent requests automatically

## Future Enhancements

- [ ] Real-time streaming predictions via WebSocket
- [ ] Ensemble models (LSTM + GBM)
- [ ] Macro factors (Nifty return, FII flows)
- [ ] Earnings surprise features
- [ ] Portfolio optimization endpoint
- [ ] Alert system for trading signals
- [ ] Backtesting framework

## License

Internal use only.

## Support

For issues or errors, check:
1. Console logs for training progress
2. Model metadata at `/model/stats` endpoint
3. Ensure all CSV files are present and readable
