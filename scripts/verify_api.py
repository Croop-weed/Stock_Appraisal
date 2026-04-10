"""
Stock Prediction API - Verification & Testing Script
====================================================
Tests all API endpoints to verify the system is working correctly.

Run this AFTER starting the API server:
  python verify_api.py
"""

import sys
import time
import requests
from typing import Dict, List


class APITester:
    """Test and verify the Stock Prediction API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.passed = 0
        self.failed = 0

    def log(self, test_name: str, status: str, message: str = ""):
        """Log test result."""
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {test_name:<40} {status:<10} {message}")
        
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1

    def test_health(self) -> bool:
        """Test if API is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = "✓" if data.get("model_loaded") else "?"
                self.log("Health Check", "PASS", f"API running, Model loaded: {status}")
                return True
            else:
                self.log("Health Check", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Health Check", "FAIL", f"Cannot connect: {e}")
            return False

    def test_model_stats(self) -> bool:
        """Test model statistics endpoint."""
        try:
            response = requests.get(f"{self.base_url}/model/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                features = data.get("total_features", 0)
                val_auc = data.get("validation_auc", 0)
                test_auc = data.get("test_auc", 0)
                samples = data.get("training_samples", 0)
                
                msg = f"Features={features}, Val AUC={val_auc:.3f}, Test AUC={test_auc:.3f}, Samples={samples}"
                self.log("Model Statistics", "PASS", msg)
                return True
            else:
                self.log("Model Statistics", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Model Statistics", "FAIL", str(e))
            return False

    def test_single_prediction(self, ticker: str) -> bool:
        """Test single ticker prediction."""
        try:
            response = requests.get(f"{self.base_url}/predict/{ticker}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                pred = data.get("prediction", "?")
                conf = data.get("confidence", 0)
                price = data.get("current_price", 0)
                
                msg = f"Price=₹{price:.2f}, Prediction={pred}, Confidence={conf*100:.1f}%"
                self.log(f"Predict {ticker}", "PASS", msg)
                return True
            else:
                self.log(f"Predict {ticker}", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log(f"Predict {ticker}", "FAIL", str(e)[:50])
            return False

    def test_ticker_info(self, ticker: str) -> bool:
        """Test ticker information endpoint."""
        try:
            response = requests.get(f"{self.base_url}/ticker-info/{ticker}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                company = data.get("company_name", "Unknown")[:30]
                price = data.get("current_price", 0)
                
                msg = f"Company={company}..., Price=₹{price}"
                self.log(f"Info {ticker}", "PASS", msg)
                return True
            else:
                self.log(f"Info {ticker}", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log(f"Info {ticker}", "FAIL", str(e)[:50])
            return False

    def test_batch_predict(self, tickers: List[str]) -> bool:
        """Test batch predictions."""
        try:
            response = requests.post(
                f"{self.base_url}/batch-predict",
                json={"tickers": tickers},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                count = data.get("count", 0)
                
                msg = f"Predicted {count}/{len(tickers)} tickers"
                self.log("Batch Predict", "PASS" if count > 0 else "FAIL", msg)
                return count > 0
            else:
                self.log("Batch Predict", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Batch Predict", "FAIL", str(e)[:50])
            return False

    def test_documentation(self) -> bool:
        """Test if Swagger docs are available."""
        try:
            response = requests.get(f"{self.base_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log("Swagger Docs", "PASS", "Available at /docs")
                return True
            else:
                self.log("Swagger Docs", "FAIL", f"Status {response.status_code}")
                return False
        except Exception as e:
            self.log("Swagger Docs", "FAIL", str(e))
            return False

    def test_invalid_ticker(self) -> bool:
        """Test handling of invalid ticker."""
        try:
            response = requests.get(f"{self.base_url}/predict/INVALID_TICKER_XYZ", timeout=10)
            if response.status_code >= 400:  # Should fail
                self.log("Error Handling (Invalid Ticker)", "PASS", "Correctly rejected")
                return True
            else:
                self.log("Error Handling (Invalid Ticker)", "FAIL", "Should have rejected")
                return False
        except:
            self.log("Error Handling (Invalid Ticker)", "PASS", "Correctly rejected")
            return True

    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*80)
        print("STOCK PREDICTION API - VERIFICATION TEST SUITE")
        print("="*80 + "\n")

        # Basic connectivity
        print("📋 CONNECTIVITY TESTS")
        print("-" * 80)
        if not self.test_health():
            print("\n⚠️  API is not running!")
            print("Start it with: python api.py")
            return False

        print()

        # Model tests
        print("📊 MODEL TESTS")
        print("-" * 80)
        self.test_model_stats()
        print()

        # Single predictions
        print("🔮 SINGLE TICKER TESTS")
        print("-" * 80)
        test_tickers = [
            "RELIANCE.NS",
            "TCS.NS",
            "INFY.NS",
        ]
        for ticker in test_tickers:
            self.test_single_prediction(ticker)
            time.sleep(0.5)  # Rate limiting
        print()

        # Ticker info
        print("ℹ️  TICKER INFO TESTS")
        print("-" * 80)
        self.test_ticker_info("HDFC.NS")
        print()

        # Batch predictions
        print("📦 BATCH PREDICTION TESTS")
        print("-" * 80)
        self.test_batch_predict(["RELIANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS"])
        print()

        # Documentation
        print("📚 DOCUMENTATION TESTS")
        print("-" * 80)
        self.test_documentation()
        print()

        # Error handling
        print("⚠️  ERROR HANDLING TESTS")
        print("-" * 80)
        self.test_invalid_ticker()
        print()

        # Summary
        print("="*80)
        print("TEST SUMMARY")
        print("="*80)
        total = self.passed + self.failed
        pct = (self.passed / total * 100) if total > 0 else 0
        
        print(f"✓ Passed: {self.passed}/{total} ({pct:.1f}%)")
        print(f"✗ Failed: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\n🎉 All tests passed! API is working correctly.")
            return True
        else:
            print("\n⚠️  Some tests failed. Check the output above.")
            return False


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("WAIT: Is the API server running?")
    print("="*80)
    print("\nIf not, start it in another terminal:")
    print("  python api.py")
    print("\nThis script will test: http://localhost:8000")
    print("\nStarting tests in 3 seconds...")
    
    time.sleep(3)
    
    tester = APITester()
    success = tester.run_all_tests()
    
    # Print usage hints
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("""
1. View interactive API documentation:
   → http://localhost:8000/docs

2. Try example client code:
   → python client_examples.py

3. Integrate into your application:
   → See README_API.md for full documentation

4. Build your trading strategy using the API!
""")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
