"""
Stock Prediction API - Main Launcher
====================================

This script starts the API server with the organized project structure.

Usage:
    python run.py              (default: port 8000)
    python run.py --port 8001  (custom port)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    parser = argparse.ArgumentParser(description="Start Stock Prediction API")
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload on file changes')
    args = parser.parse_args()

    print("=" * 70)
    print("Stock Prediction API v2.0")
    print("=" * 70)
    print(f"\n📍 Starting server on {args.host}:{args.port}")
    print(f"🌐 API Docs: http://localhost:{args.port}/docs")
    print(f"📊 Open this URL in your browser to test predictions!\n")

    import uvicorn
    from api_v2 import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
