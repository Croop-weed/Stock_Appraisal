# 📋 Quick Reference Card

## 🚀 Quick Start (30 Seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API
python run.py

# 3. Open in browser
http://localhost:8000/docs

# 4. Try a prediction
GET /predict/RELIANCE.NS
```

---

## ✅ What Was Fixed

| Issue | Error | Now | Status |
|-------|-------|-----|--------|
| Delisted tickers (APOLLOHOSP) | 500 Error | 404 with clear message | ✅ Fixed |
| Invalid tickers | 500 Error | 400 with validation | ✅ Fixed |
| Insufficient data | 500 Error | 400 with reason | ✅ Fixed |
| Data format issues | 500 Error | 400 with details | ✅ Fixed |
| NaN/Inf in features | 500 Error | Handles gracefully | ✅ Fixed |

---

## 📁 New Project Structure

```
Stock predictor/
├── run.py ........................ Main launcher (NEW)
├── README.md ..................... Project overview (NEW)
├── MIGRATION_GUIDE.md ............ What changed (NEW)
├── organize.py ................... Reorganizer (NEW)
├── src/api_v2.py ................ Improved API (NEW)
├── data/ ......................... CSV files (NEW folder)
├── models/ ....................... ML models (NEW folder)
├── docs/ ......................... Documentation (NEW folder)
└── scripts/ ...................... Utilities (NEW folder)
```

---

## 🎯 Common Commands

| Task | Command |
|------|---------|
| **Start API** | `python run.py` |
| **Custom port** | `python run.py --port 8001` |
| **Dev mode** | `python run.py --reload` |
| **Reorganize files** | `python organize.py` |
| **Test API** | `python scripts/verify_api.py` |
| **Run examples** | `python scripts/client_examples.py` |
| **Windows start** | `run_api.bat` |

---

## 🔮 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict/{ticker}` | GET | Single prediction |
| `/batch-predict` | POST | Multiple predictions |
| `/ticker-info/{ticker}` | GET | Company info |
| `/model/stats` | GET | Model metrics |
| `/model/retrain` | POST | Retrain |
| `/health` | GET | Health check |
| `/docs` | GET | API docs |

---

## 💡 Example: Python Client

```python
import requests

# Prediction
r = requests.get("http://localhost:8000/predict/RELIANCE.NS")
data = r.json()

print(f"{data['ticker']}: {data['prediction']}")
print(f"Price: ₹{data['current_price']}")
print(f"Confidence: {data['confidence']*100:.1f}%")
```

---

## 📊 Test Tickers

```
India (NSE):
• RELIANCE.NS
• TCS.NS
• INFY.NS

US:
• AAPL
• MSFT
• GOOGL
```

---

## ⚠️ Error Reference

| Error | Cause | Solution |
|-------|-------|----------|
| 404 Ticker delisted | Stock no longer trades | Try different ticker |
| 400 Invalid format | Wrong ticker name | Use correct format (e.g., RELIANCE.NS) |
| 400 Insufficient data | < 8 quarters history | Use established company |
| 500 Model not loaded | First run | Wait 2-5 min for training |
| Port in use | 8000 occupied | `python run.py --port 8001` |

---

## 📈 Expected Response (Example)

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

## 🔧 Troubleshooting

```bash
# Check if API is running
curl "http://localhost:8000/health"

# View model stats
curl "http://localhost:8000/model/stats"

# Retrain model
curl -X POST "http://localhost:8000/model/retrain"

# Test with cURL
curl "http://localhost:8000/predict/TCS.NS"
```

---

## 📚 Full Documentation

- **README**: Main project overview
- **SETUP_GUIDE**: Detailed setup instructions
- **MIGRATION_GUIDE**: What changed & how to upgrade
- **API Docs**: http://localhost:8000/docs (interactive)

---

## 🎁 What You Get

✅ Production-ready REST API
✅ Real-time stock predictions
✅ Error handling & validation
✅ Organized project structure
✅ Comprehensive documentation
✅ Example code & scripts
✅ Testing utilities

---

## 🚦 Status

| Component | Status |
|-----------|--------|
| API Server | ✅ Running |
| Model Training | ✅ Automatic |
| Error Handling | ✅ Fixed |
| Documentation | ✅ Complete |
| Organization | ✅ Done |

---

## ⏭️ Next Steps

1. **Start**: `python run.py`
2. **Visit**: http://localhost:8000/docs
3. **Test**: Try `RELIANCE.NS`
4. **Explore**: Check other endpoints
5. **Integrate**: Use in your app

---

## 🎉 Ready to Go!

Everything is set up and organized. Just run:

```bash
python run.py
```

Then visit the interactive API documentation in your browser! 🚀
