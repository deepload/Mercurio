#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to extract stock symbols from a CSV and run best_assets_screener.py on the top 50.
Usage: python scripts/screen_top_50_from_csv.py
"""
import pandas as pd
import subprocess
from pathlib import Path

# Path to your CSV file
CSV_PATH = Path('data/all_stocks_20250520.csv')

# Read the CSV and extract the symbols
# Read the CSV and extract the symbols, filtering out NaN and non-string values
symbols = pd.read_csv(CSV_PATH)['symbol']
# Filter out crypto-like symbols (ending with -USD, -USDT, -EUR, etc.)
crypto_suffixes = ("-USD", "-USDT", "-EUR", "-BTC", "-ETH")
symbols = [str(s).strip() for s in symbols if pd.notnull(s) and str(s).strip() and str(s).lower() != 'nan']
stock_symbols = [s for s in symbols if not any(s.endswith(suffix) for suffix in crypto_suffixes)]

# Use only the first 50 stock symbols
symbols_arg = ','.join(stock_symbols[:50])

# Build the command to run the screener for top 50 stocks
cmd = [
    'python',
    'scripts/best_assets_screener.py',
    '--top_stocks', '50',
    '--top_crypto', '0',  # Skip crypto
    '--stocks', symbols_arg
]

print(f"Running: {' '.join(cmd[:4])} --stocks <{len(symbols)} symbols>")

# Run the screener
subprocess.run(cmd)
