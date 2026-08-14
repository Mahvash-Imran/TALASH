"""
run_backend.py  –  Uvicorn Server Launcher for TALASH Backend API
================================================================

Usage
-----
  python run_backend.py
  python run_backend.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

# Allow importing from root directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env BEFORE importing api.main (which reads os.environ at import time)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="TALASH Backend Uvicorn Server Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind server")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload on code change")
    args = parser.parse_args()

    print(f"Starting TALASH Backend API on http://{args.host}:{args.port}")
    print(f"Interactive API Documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
