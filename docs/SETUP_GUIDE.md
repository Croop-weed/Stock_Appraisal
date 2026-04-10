# Stock Prediction API - Complete Setup Guide

## Overview

This project provides a **REST API backend for stock price predictions** using a machine learning model. Simply provide a ticker symbol and the API returns:
- ✅ Current stock price
- ✅ Buy/Sell prediction
- ✅ Model confidence score
- ✅ Trading recommendation

All predictions are based on a GradientBoosting model trained on 15,000+ quarterly stock records across 10 sectors.

---

## Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python api.py
```
Or on Windows, double-click: `run_api.bat`

### 3. Open Interactive Documentation
Visit: http://localhost:8000/docs

### 4. Try a Prediction
In the Swagger UI, click `GET /predict/{ticker}` and try:
- Ticker: `RELIANCE.NS`
- Execute

Done! 🎉

---

## Project Structure

```
Stock predictor/
├── 📄 api.py                      ← MAIN: FastAPI backend
├── 📄 d1.py                       ← Reference: Data preprocessing
├── 📄 d2.py                       ← Reference: Model training (v2)
├── 📄 config.py                   ← Settings (modify as needed)
├── 📄 client_examples.py          ← Python client examples
├── 📄 requirements.txt            ← Package dependencies
├── 🚀 run_api.bat                 ← Windows launcher
│
├── 📚 Documentation
│   ├── README_API.md              ← Full API reference
│   ├── QUICKSTART.py              ← Quick start guide  
│   └── THIS FILE
│
├── 📊 Data Files (Sectors)
│   ├── Mine_Data.csv
│   ├── Bank_Data.csv
│   ├── Energy_Data.csv
│   ├── Automobile_Data.csv
│   ├── FMCG_Data.csv
│   ├── Metal_Data.csv
│   ├── Pharma_symbols_Data.csv
│   ├── Infra_Data.csv
│   └── Healthcare_*.csv
│
└── 📦 Generated (Auto-created)
    └── models/
        ├── stock_model.pkl       ← Trained GradientBoosting model
        ├── scaler.pkl            ← Feature scaler
        └── metadata.pkl          ← Model metrics
```

---

## Key Files Explained

### 1. **api.py** - Main Application
The FastAPI server that serves predictions.

**What it does:**
1. Loads historical CSV data for all sectors
2. Trains a GradientBoosting ML model on quarterly data
3. Exposes REST endpoints for predictions
4. Caches the model in-memory for fast predictions

**Start with:**
```bash
python api.py
```

### 2. **client_examples.py** - Usage Patterns
Shows how to call the API from Python code.

**Run examples:**
```bash
python client_examples.py
```

**Includes examples for:**
- Single ticker predictions
- Batch predictions
- Portfolio analysis
- Trading strategies
- Model statistics

### 3. **config.py** - Configuration
All customizable settings in one place.

**Modify to:**
- Change server port (default 8000)
- Adjust model hyperparameters
- Customize prediction thresholds
- Add/remove data files
- Change recommendation messages

### 4. **run_api.bat** - Windows Launcher
One-click startup for Windows users.

**Just double-click** to:
- Auto-install missing packages
- Start the API server
- Open documentation

---

## API Endpoints

### 🔮 Get Prediction (Single Ticker)
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
  "prediction": "BUY",
  "confidence": 0.628,
  "recommendation": "Buy - Positive momentum expected",
  "timestamp": "2024-04-05T10:30:45"
}
```

---

### 📦 Batch Predictions
```bash
POST /batch-predict
```

**Request:**
```json
{
  "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
}
```

**Response:**
```json
{
  "count": 3,
  "timestamp": "2024-04-05T10:30:45",
  "predictions": [...]
}
```

---

### 📊 Model Statistics
```bash
GET /model/stats
```

Returns model performance (AUC, accuracy) and feature importance.

---

### ℹ️ Ticker Information
```bash
GET /ticker-info/{ticker}
```

Returns company name, market cap, P/E ratio, sector.

---

### 🔄 Retrain Model
```bash
POST /model/retrain
```

Retrains the model using latest data from CSV files.

---

## Usage Examples

### Example 1: Python Script
```python
import requests

# Single prediction
response = requests.get("http://localhost:8000/predict/RELIANCE.NS")
data = response.json()

print(f"{data['ticker']}: {data['prediction']}")
print(f"Price: ₹{data['current_price']}")
print(f"Confidence: {data['confidence']*100:.1f}%")
```

### Example 2: Command Line (cURL)
```bash
# Single prediction
curl "http://localhost:8000/predict/TCS.NS"

# Batch predictions
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS"]}'

# Model stats
curl "http://localhost:8000/model/stats"
```

### Example 3: JavaScript/Node.js
```javascript
fetch('http://localhost:8000/predict/INFY.NS')
  .then(res => res.json())
  .then(data => {
    console.log(`${data.ticker}: ${data.prediction}`);
    console.log(`Price: ₹${data.current_price}`);
  });
```

---

## How It Works

### Data Flow
```
Ticker Input (e.g., "RELIANCE.NS")
    ↓
Download 5 years of quarterly data (yfinance)
    ↓
Feature Engineering (35 features):
  - Momentum (returns over 1q, 2q, 1yr, 2yr)
  - Trend (MA crossovers, price positions)
  - Volatility (RSI, rolling std)
  - Volume anomalies
  - Valuation (P/E, P/B ratios)
  - Sector encoding
    ↓
Scale features (StandardScaler)
    ↓
GradientBoosting Model Prediction
    ↓
Output: {
  "prediction": "BUY" or "SELL",
  "confidence": 0.0 - 1.0,
  "recommendation": text
}
```

### Model Architecture
- **Algorithm**: GradientBoosting (ensemble of decision trees)
- **Features**: 35 engineered technical & fundamental features
- **Training Data**: ~15,500 quarterly records
- **Sectors**: 10 (Mine, Bank, Energy, Automobile, IT, FMCG, Pharma, etc.)
- **Performance**: 
  - Validation AUC: 0.556
  - Test AUC: 0.584

**Note**: 0.58 AUC is considered good for quarterly stock direction prediction. Random guess = 0.50 AUC.

---

## Common Issues & Solutions

### ❌ "Model not loaded" Error
**Solution**: Model trains on first run (2-5 min). Check console for progress.

### ❌ "Ticker not found"
**Solution**: 
- Use correct format: `RELIANCE.NS` (not `RELIANCE`)
- For US stocks: `AAPL` (not `AAPL.NS`)
- Check internet connection (yfinance needs it)

### ❌ Port 8000 already in use
**Solution**: Change port in `config.py` or use:
```bash
python api.py --port 8001
```

### ❌ CSV files not found
**Solution**: Ensure all CSV files are in the project directory:
```bash
ls *.csv
```

### ❌ Model training fails
**Solution**: Check CSV format has columns: Ticker, Date, Open, High, Low, Close, Volume

---

## Production Deployment

### With Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 api:app
```

### With Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "api.py"]
```

### Cloud Deployment
- **Heroku**: Push this repo directly
- **AWS Lambda**: Use API Gateway + Lambda
- **Google Cloud Run**: Deploy container image
- **Azure App Service**: Connect GitHub repo

---

## Customization

### Change Prediction Thresholds
Edit `config.py`:
```python
STRONG_BUY_THRESHOLD = 0.70  # Instead of 0.65
```

### Add More Features
Edit `build_ticker_features()` in `api.py`:
```python
g['my_custom_feature'] = g['close'].ewm(span=10).mean()
```

### Use Different Model
Edit `train_model()` in `api.py`:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(...)
```

### Change Data Sources
Edit `SECTOR_DATA_FILES` in `config.py` to add more CSV files.

---

## Performance Tips

1. **Batch predictions** are faster than individual calls
   ```python
   # Slow: 3 separate calls
   for ticker in tickers:
       requests.get(f"/predict/{ticker}")
   
   # Fast: 1 batch call
   requests.post("/batch-predict", json={"tickers": tickers})
   ```

2. **Use caching** for repeated predictions
   ```python
   from functools import lru_cache
   # Avoid re-predicting same ticker repeatedly
   ```

3. **Model retrains** run monthly
   - Only call `/model/retrain` when new data arrives
   - Normal predictions don't retrain

---

## Next Steps

1. ✅ Run `python api.py` and visit http://localhost:8000/docs
2. ✅ Try a few predictions in the Swagger UI
3. ✅ Run `python client_examples.py` to see usage patterns
4. ✅ Integrate into your trading strategy
5. ✅ Monitor model performance with `/model/stats`
6. ✅ Retrain monthly with `/model/retrain`

---

## API Documentation

### Full Reference
See: **README_API.md**

### Quick Start
See: **QUICKSTART.py**

### Code Examples
See: **client_examples.py**

---

## Model Performance Details

### Training Metrics
```
Training Samples: 15,500
Features: 35 (engineered)
Sectors: 10
Quarters: 2016-2024

Validation AUC: 0.556
Test AUC:      0.584

This indicates the model has moderate predictive power.
Above 0.58 is good for quarterly stock data.
```

### Top Features by Importance
1. `ret_8q` - 2-year returns (momentum)
2. `vol_4q` - 4-quarter volatility
3. `price_to_ma8` - Price vs 2-year MA
4. `label_lag1` - Previous quarter signal
5. `P_to_E` - Price-to-earnings ratio

### Common Tickers
- **Energy**: RELIANCE.NS, NTPC.NS
- **IT**: TCS.NS, INFY.NS, WIPRO.NS
- **Banking**: HDFC.NS, ICICIBANK.NS
- **Automobile**: MARUTI.NS
- **Pharma**: SUNPHARMA.NS

---

## Support

**For issues:**
1. Check console logs for errors
2. Visit `/health` endpoint to verify API is running
3. Run `/model/retrain` if model is stale
4. Check that CSV files exist and are readable

**For customization:**
1. Edit `config.py` for settings
2. Edit `api.py` for logic changes
3. Edit `build_ticker_features()` for new features

---

## License & Disclaimer

⚠️ **For Educational Use Only**

This is a stock prediction model based on historical data. It is NOT:
- Financial advice
- Guaranteed to be accurate
- Suitable for real money trading without additional verification

Always validate predictions with additional research before trading.

---

## Version History

- **v1.0** (Apr 2024): Initial release with GradientBoosting model

---

## Quick Commands

```bash
# Start API
python api.py

# Windows start
run_api.bat

# Install packages
pip install -r requirements.txt

# Run examples
python client_examples.py

# Test single prediction
curl "http://localhost:8000/predict/RELIANCE.NS"

# View model stats
curl "http://localhost:8000/model/stats"

# Retrain model
curl -X POST "http://localhost:8000/model/retrain"

# Health check
curl "http://localhost:8000/health"
```

---

**Ready to predict? Let's go! 🚀**

```bash
python api.py
# Then visit: http://localhost:8000/docs
```
