"""
Stock Prediction API - Python Client Examples
==============================================
Shows how to use the API from Python code.

Run the API server first:
  python api.py

Then run this script:
  python client_examples.py
"""

import requests
import json
from typing import List, Dict


# API Base URL
API_URL = "http://localhost:8000"


class StockPredictionClient:
    """Client for Stock Prediction API."""

    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url

    def predict(self, ticker: str) -> Dict:
        """
        Get prediction for a single ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'RELIANCE.NS')
            
        Returns:
            Dictionary with prediction, price, and recommendation
        """
        response = requests.get(f"{self.base_url}/predict/{ticker}")
        response.raise_for_status()
        return response.json()

    def get_ticker_info(self, ticker: str) -> Dict:
        """Get company info for a ticker."""
        response = requests.get(f"{self.base_url}/ticker-info/{ticker}")
        response.raise_for_status()
        return response.json()

    def batch_predict(self, tickers: List[str]) -> Dict:
        """Get predictions for multiple tickers."""
        response = requests.post(
            f"{self.base_url}/batch-predict",
            json={"tickers": tickers}
        )
        response.raise_for_status()
        return response.json()

    def model_stats(self) -> Dict:
        """Get model performance statistics."""
        response = requests.get(f"{self.base_url}/model/stats")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict:
        """Check if API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def retrain(self) -> Dict:
        """Trigger model retraining."""
        response = requests.post(f"{self.base_url}/model/retrain")
        response.raise_for_status()
        return response.json()


def example_single_prediction():
    """Example: Get prediction for single ticker."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Ticker Prediction")
    print("="*70)
    
    client = StockPredictionClient()
    
    try:
        ticker = "RELIANCE.NS"
        print(f"\nFetching prediction for {ticker}...")
        
        result = client.predict(ticker)
        
        print(f"\n{'Ticker':<15} {result['ticker']}")
        print(f"{'Current Price':<15} ₹{result['current_price']}")
        print(f"{'Previous Close':<15} ₹{result['previous_close']}")
        print(f"{'Change %':<15} {result['price_change_pct']}%")
        print(f"{'Sector':<15} {result['sector']}")
        print(f"{'Prediction':<15} {result['prediction']}")
        print(f"{'Confidence':<15} {result['confidence']*100:.1f}%")
        print(f"{'Probability':<15} {result['model_probability']*100:.1f}%")
        print(f"{'Recommendation':<15} {result['recommendation']}")
        print(f"{'Timestamp':<15} {result['timestamp']}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_ticker_info():
    """Example: Get ticker company info."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Ticker Information")
    print("="*70)
    
    client = StockPredictionClient()
    
    try:
        ticker = "TCS.NS"
        print(f"\nFetching info for {ticker}...")
        
        info = client.get_ticker_info(ticker)
        
        print(f"\n{'Company':<20} {info['company_name']}")
        print(f"{'Ticker':<20} {info['ticker']}")
        print(f"{'Current Price':<20} {info['current_price']}")
        print(f"{'Market Cap':<20} {info['market_cap']}")
        print(f"{'P/E Ratio':<20} {info['pe_ratio']}")
        print(f"{'Sector':<20} {info['sector']}")
        print(f"{'Available':<20} {info['available']}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_prediction():
    """Example: Get predictions for multiple tickers."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Predictions")
    print("="*70)
    
    client = StockPredictionClient()
    
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFC.NS"]
    print(f"\nFetching predictions for: {', '.join(tickers)}...")
    
    try:
        result = client.batch_predict(tickers)
        
        print(f"\nTotal predictions: {result['count']}")
        print(f"Timestamp: {result['timestamp']}\n")
        
        # Print table
        print(f"{'Ticker':<15} {'Prediction':<12} {'Confidence':<12} {'Recommendation':<30}")
        print("-" * 70)
        
        for pred in result['predictions']:
            print(
                f"{pred['ticker']:<15} "
                f"{pred['prediction']:<12} "
                f"{pred['confidence']*100:>10.1f}% "
                f"{pred['recommendation']:<30}"
            )
            
    except Exception as e:
        print(f"Error: {e}")


def example_portfolio_analysis():
    """Example: Analyze a portfolio of stocks."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Portfolio Analysis")
    print("="*70)
    
    client = StockPredictionClient()
    
    # Your portfolio
    portfolio = {
        "RELIANCE.NS": 5,    # 5 shares
        "TCS.NS": 10,        # 10 shares
        "INFY.NS": 20,       # 20 shares
    }
    
    print(f"\nAnalyzing portfolio:\n")
    
    try:
        predictions = client.batch_predict(list(portfolio.keys()))
        
        total_investment = 0
        buy_signals = 0
        
        print(f"{'Ticker':<15} {'Qty':<8} {'Price':<12} {'Signal':<8} {'Value':<12}")
        print("-" * 60)
        
        for pred in predictions['predictions']:
            ticker = pred['ticker']
            qty = portfolio[ticker]
            price = pred['current_price']
            signal = pred['prediction']
            value = qty * price
            
            total_investment += value
            if signal == "BUY":
                buy_signals += 1
            
            print(
                f"{ticker:<15} "
                f"{qty:<8} "
                f"₹{price:<11.2f} "
                f"{signal:<8} "
                f"₹{value:<11.2f}"
            )
        
        print("-" * 60)
        print(f"{'TOTAL':<15} {'-':<8} {'-':<12} {'-':<8} ₹{total_investment:<11.2f}")
        print(f"\nBuy signals: {buy_signals}/{len(portfolio)}")
        print(f"Portfolio allocation: {total_investment:,.2f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_model_stats():
    """Example: View model statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Model Performance Statistics")
    print("="*70)
    
    client = StockPredictionClient()
    
    try:
        stats = client.model_stats()
        
        print(f"\n{'Model Status':<20} {'Yes' if stats['model_trained'] else 'No'}")
        print(f"{'Training Date':<20} {stats['training_date']}")
        print(f"{'Training Samples':<20} {stats['training_samples']:,}")
        print(f"{'Total Features':<20} {stats['total_features']}")
        print(f"{'Validation AUC':<20} {stats['validation_auc']:.3f}")
        print(f"{'Test AUC':<20} {stats['test_auc']:.3f}")
        
        print(f"\nTop 10 Most Predictive Features:")
        print("-" * 50)
        for i, (feature, importance) in enumerate(stats['top_10_features'].items(), 1):
            width = int(importance * 50)
            bar = "█" * width
            print(f"{i:2}. {feature:<25} {importance:>7.4f} {bar}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_trading_strategy():
    """Example: Simple trading strategy based on predictions."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Trading Strategy Example")
    print("="*70)
    
    client = StockPredictionClient()
    
    # Sector to analyze
    sectors = {
        "Bank": ["HDFC.NS", "ICICIBANK.NS", "AXISBANK.NS"],
        "Energy": ["RELIANCE.NS", "NTPC.NS"],
        "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    }
    
    print("\nScanning sectors for strong buy signals...\n")
    
    try:
        strong_buys = []
        
        for sector, tickers in sectors.items():
            result = client.batch_predict(tickers)
            
            for pred in result['predictions']:
                if pred['confidence'] > 0.65 and pred['prediction'] == "BUY":
                    strong_buys.append(pred)
        
        if strong_buys:
            print(f"Found {len(strong_buys)} strong buy signals:\n")
            for stock in strong_buys:
                print(f"  ✓ {stock['ticker']}: {stock['prediction']} "
                      f"(Confidence: {stock['confidence']*100:.1f}%)")
                print(f"    Current: ₹{stock['current_price']} → {stock['recommendation']}\n")
        else:
            print("No strong buy signals found in the scanned sectors.")
            
    except Exception as e:
        print(f"Error: {e}")


def check_api_health():
    """Check if API is running."""
    print("\n" + "="*70)
    print("Checking API Health...")
    print("="*70)
    
    client = StockPredictionClient()
    
    try:
        health = client.health_check()
        print(f"\n✓ API Status: {health['status']}")
        print(f"✓ Model Loaded: {health['model_loaded']}")
        print(f"✓ Timestamp: {health['timestamp']}")
        return True
        
    except Exception as e:
        print(f"\n✗ API Error: {e}")
        print("\nMake sure the API server is running:")
        print("  python api.py")
        return False


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STOCK PREDICTION API - CLIENT EXAMPLES")
    print("="*70)
    
    # Check if API is running
    if not check_api_health():
        exit(1)
    
    # Run examples
    example_single_prediction()
    example_ticker_info()
    example_batch_prediction()
    example_portfolio_analysis()
    example_model_stats()
    example_trading_strategy()
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)
    print("\nFor full API documentation, visit:")
    print("  http://localhost:8000/docs")
