## ⚠️ Disclaimer

**This project is for educational purposes only.** 
- The stock predictions are based on historical data and ML models
- Past performance does NOT guarantee future results
- DO NOT use this model for actual trading without thorough backtesting
- The creator is NOT responsible for financial losses
- Consult a qualified financial advisor before making investment decisions
# Stock Predictor

A powerful **REST API for real-time stock price predictions** using machine learning. Get instant buy/sell signals with confidence scores for Indian stocks across 10 different sectors.

## 🎯 Features

- 🤖 **AI-Powered Predictions** - GradientBoosting model trained on 15,000+ quarterly stock records
- 📊 **Real-time Data** - Live stock prices via yfinance integration
- 🚀 **Fast API** - Built with FastAPI for high performance
- 📈 **Multi-Sector Support** - Covers Banking, Energy, Pharma, FMCG, Automobile, Infrastructure, Mining, Metals, and Healthcare sectors
- 📱 **Interactive Documentation** - Swagger UI for easy testing
- 🔄 **Batch Predictions** - Predict for multiple tickers simultaneously
- 📊 **Model Analytics** - Access model statistics and feature importance

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python run.py
```

The API will start at `http://localhost:8000`

### 3. Test the API

Open your browser and visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### 4. Make Your First Prediction

Using the Swagger UI at `/docs`, or via curl:

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
  "recommendation": "Buy - Positive momentum expected",
  "timestamp": "2024-04-05T10:30:45.123456"
}
```

## 📋 API Endpoints

### Get Prediction for a Single Ticker
```
GET /predict/{ticker}
```
- Returns current price, prediction (BUY/SELL), and confidence score
- **Example**: `/predict/TCS.NS`, `/predict/INFY.NS`

### Get Ticker Information
```
GET /ticker-info/{ticker}
```
- Returns detailed company information and market data
- **Example**: `/ticker-info/RELIANCE.NS`

### Get Model Statistics
```
GET /model/stats
```
- Returns model performance metrics and feature importance
- Useful for understanding model accuracy and behavior

### Batch Predictions
```
POST /batch-predict
```
- Predict for multiple tickers in a single request
- **Body**: `{"tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]}`

For detailed API documentation, see [README_API.md](docs/README_API.md)

## 🏗️ Project Structure

```
Stock predictor/
├── README.md                           ← You are here
├── run.py                              ← Main entry point (starts API server)
├── requirements.txt                    ← Python dependencies
│
├── src/
│   └── api_v2.py                      ← FastAPI application core
│
├── scripts/
│   ├── api.py                         ← Previous API version
│   ├── client_examples.py             ← Usage examples in Python
│   ├── d1.py                          ← Data preprocessing reference
│   ├── d2.py                          ← Model training reference
│   ├── config.py                      ← Configuration settings
│   └── verify_api.py                  ← API verification script
│
├── data/
│   ├── Mine_Data.csv
│   ├── Bank_Data.csv
│   ├── Energy_Data.csv
│   ├── Automobile_Data.csv
│   ├── FMCG_Data.csv
│   ├── Metal_Data.csv
│   ├── Infra_Data.csv
│   ├── pharma_symbols_Data.csv
│   └── raw_Healthcare_*.csv
│
├── models/                            ← Generated directory
│   ├── stock_model.pkl               ← Trained model
│   ├── scaler.pkl                    ← Feature scaler
│   └── metadata.pkl                  ← Model metadata
│
└── docs/
    ├── README_API.md                 ← Full API reference
    ├── SETUP_GUIDE.md                ← Detailed setup instructions
    ├── QUICK_REFERENCE.md            ← Quick API reference
    ├── MIGRATION_GUIDE.md            ← Project reorganization guide
    ├── WHAT_WAS_CREATED.md           ← Architecture documentation
    └── COMPLETION_SUMMARY.md         ← Project completion notes
```

## 📚 Key Files

### `api_v2.py` - Main Application Engine
The FastAPI server that powers all predictions:
- Loads historical stock data from CSVs
- Trains/loads the GradientBoosting model
- Serves predictions via REST API
- Caches data and models for performance

**Location**: `src/api_v2.py`

### `client_examples.py` - Usage Patterns
Python examples showing how to use the API programmatically:

```bash
python scripts/client_examples.py
```

### Data Files
Historical quarterly data for Indian stocks organized by sector:
- 10 sectors with 1,000+ companies
- Price movement labels (0 = SELL, 1 = BUY)
- Features: open, high, low, close prices, volume

## 🔧 Configuration

### Running on a Custom Port

```bash
python run.py --port 8001
```

### Running with Auto-Reload (Development)

```bash
python run.py --reload
```

### Running on a Specific Host

```bash
python run.py --host 0.0.0.0 --port 8000
```

## 📊 Supported Sectors & Tickers

The model supports stocks from these sectors:

| Sector | Sample Tickers |
|--------|-----------------|
| Banking | RELIANCE.NS, HDFC.NS, ICICIBANK.NS |
| IT & Software | TCS.NS, INFY.NS, WIPRO.NS |
| Pharma | CIPLA.NS, DIVI.NS, SUNPHARMA.NS |
| Automobiles | MARUTI.NS, BAJAJFINSV.NS |
| FMCG | ITC.NS, NESTLEIND.NS, MARICO.NS |
| Energy | ONGC.NS, NTPC.NS, POWERGRID.NS |
| Metals & Mining | TATASTEEL.NS, HINDALCO.NS, JSW.NS |
| Infrastructure | ADANIPORTS.NS, BHARTIARTL.NS |
| Healthcare | APOLLOHOSP.NS, LUPIN.NS |

## 💻 Usage Examples

### Python

```python
import requests

# Single ticker prediction
response = requests.get("http://localhost:8000/predict/RELIANCE.NS")
prediction = response.json()
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")

# Batch predictions
response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"tickers": ["TCS.NS", "INFY.NS", "WIPRO.NS"]}
)
predictions = response.json()
for pred in predictions:
    print(f"{pred['ticker']}: {pred['prediction']}")
```

### cURL

```bash
# Single prediction
curl "http://localhost:8000/predict/TCS.NS"

# Get model stats
curl "http://localhost:8000/model/stats"
```

## 🛠️ Installation Requirements

- **Python**: 3.8+
- **FastAPI**: 0.104.1+
- **Scikit-learn**: 1.3.2+
- **Pandas**: 2.1.3+
- **yfinance**: 0.2.32+

See [requirements.txt](requirements.txt) for full dependency list.

## 📖 Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)** - Detailed installation and configuration
- **[API Reference](docs/README_API.md)** - Complete API endpoint documentation
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Common API patterns
- **[Architecture](docs/WHAT_WAS_CREATED.md)** - System design explanation
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Project reorganization notes

## 🚨 Important Notes

1. **Stock Tickers**: Indian stock tickers must include `.NS` (NSE) suffix (e.g., `RELIANCE.NS`)
2. **Market Hours**: Best predictions when using market closing prices
3. **Model Confidence**: Confidence scores are model probabilities (0-1). Higher = more certain
4. **Batch Limits**: For optimal performance, limit batch requests to 50 tickers at a time
5. **Data Updates**: Historical data is loaded from CSV files on server startup

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Port already in use
```bash
python run.py --port 8001
```

### API not responding
1. Check if server is running (you should see: `✅ Server is running on http://0.0.0.0:8000`)
2. Visit http://localhost:8000/docs in your browser
3. Check the console for error messages

### Invalid ticker symbol
- Ensure ticker includes sector suffix (e.g., `.NS` for NSE)
- Check ticker is in the supported sectors list
- Example valid tickers: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`

## 📈 Model Performance

The GradientBoosting model achieves:
- **Training Data**: 15,000+ quarterly records
- **Sectors Covered**: 10 major Indian sectors
- **Feature Set**: Open, High, Low, Close, Volume analytics
- **Output**: Binary classification (BUY/SELL)

See [Model Statistics](http://localhost:8000/model/stats) endpoint for detailed metrics.

## 🤝 Contributing

To extend this project:

1. **Add New Sectors**: Add CSV file to `data/` and update `SECTOR_FILES` in `api_v2.py`
2. **Improve Model**: Modify training logic in `scripts/d2.py` and retrain
3. **Add Endpoints**: Extend `src/api_v2.py` with new FastAPI routes
4. **Update Documentation**: Modify relevant files in `docs/`

## 📝 License

This project is provided as-is for educational and research purposes.

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn ML Models](https://scikit-learn.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Indian Stock Market](https://www.nseindia.com/)

---

**Built with ❤️ for stock market enthusiasts**

For issues or questions, check the [docs/](docs/) folder for detailed documentation.
