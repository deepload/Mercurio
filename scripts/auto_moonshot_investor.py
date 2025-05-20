#!/usr/bin/env python
"""
MercurioAI - Auto Moonshot Investor

Discovers ultra-low price cryptos (<$0.01), invests $1 in those with high potential, and monitors for optimal exits.

Usage:
    python scripts/auto_moonshot_investor.py --simulate --max-assets 20
    python scripts/auto_moonshot_investor.py --live --max-assets 10

Flags:
    --simulate    Run in simulation (paper) mode (default)
    --live        Run in live trading mode
    --max-assets  Maximum number of assets to invest in
    --min-price   Minimum price filter (default: 0)
    --max-price   Maximum price filter (default: 0.01)
    --help        Show help

Note: Use --simulate (not --paper) for paper trading mode.

"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Ensure project root is in sys.path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root for sibling script imports

# Mercurio imports
from scripts.fetch_all_alpaca_cryptos import get_all_alpaca_cryptos
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from project root (Mercurio standard)
root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / '.env')
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from scripts.best_assets_screener import AssetEvaluator

POSITIONS_FILE = Path(__file__).parent / 'positions.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auto_moonshot_investor")

def parse_args():
    parser = argparse.ArgumentParser(description="MercurioAI - Auto Moonshot Investor")
    parser.add_argument('--simulate', action='store_true', help='Simulate trades (default)')
    parser.add_argument('--live', action='store_true', help='Enable live trading mode')
    parser.add_argument('--max-assets', type=int, default=20, help='Max number of assets to invest in')
    parser.add_argument('--min-price', type=float, default=0.0, help='Minimum price filter (default: 0)')
    parser.add_argument('--max-price', type=float, default=0.01, help='Maximum price filter (default: 0.01)')
    parser.add_argument('--quote-currency', type=str, default='', help='Only include pairs with this quote currency (e.g., USD, USDT, USDC). Leave blank for all.')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous monitoring mode (loop)')
    parser.add_argument('--interval', type=int, default=60, help='Interval between scans in seconds (default: 60)')
    return parser.parse_args()

def load_positions() -> Dict[str, Any]:
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_positions(positions: Dict[str, Any]):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(positions, f, indent=2)

def log_trade(action: str, symbol: str, price: float, reason: str, pnl: float = None):
    logger.info(f"{action.upper()} | {symbol} | Price: {price:.8f} | Reason: {reason}" + (f" | P/L: {pnl:.2f}" if pnl is not None else ""))

# Main logic will be implemented next...

import time

import asyncio

async def run_once(args, market_data, trading_service, evaluator):
    is_paper = not args.live
    max_assets = args.max_assets
    min_price = args.min_price
    max_price = args.max_price
    quote_currency = args.quote_currency.upper() if args.quote_currency else ''

    logger.info(f"Mode: {'Paper' if is_paper else 'Live'} | Max assets: {max_assets} | Price range: [{min_price}, {max_price}] | Quote currency: {quote_currency or 'ALL'}")

    # Fetch all cryptos
    crypto_assets = get_all_alpaca_cryptos()
    logger.info(f"Total assets fetched from Alpaca: {len(crypto_assets)}")

    # Filter for tradable
    tradable_cryptos = [c for c in crypto_assets if c['tradable']]
    logger.info(f"Tradable cryptos: {len(tradable_cryptos)}")

    # Optional: filter by quote currency (e.g., USD, USDT, USDC)
    if quote_currency:
        tradable_cryptos = [c for c in tradable_cryptos if c['symbol'].endswith('/' + quote_currency) or c['symbol'].endswith('-' + quote_currency)]
        logger.info(f"Tradable cryptos after quote currency filter ({quote_currency}): {len(tradable_cryptos)}")

    # Fetch latest prices and filter by price
    candidates = []
    for asset in tradable_cryptos:
        symbol = asset['symbol_dash'] if 'symbol_dash' in asset else asset['symbol']
        try:
            price = await market_data.get_latest_price(symbol)
            if price is not None and min_price <= price <= max_price:
                candidates.append({'symbol': symbol, 'price': price, 'name': asset['name']})
        except Exception as e:
            logger.warning(f"Could not fetch price for {symbol}: {e}")

    logger.info(f"Found {len(candidates)} ultra-low price candidates after price filter.")

    # TODO: Score/filter using evaluator, invest, monitor, and log

async def main():
    args = parse_args()
    market_data = MarketDataService()
    trading_service = TradingService(is_paper=not args.live)
    evaluator = AssetEvaluator()

    if args.continuous:
        logger.info("Continuous monitoring mode enabled.")
        while True:
            await run_once(args, market_data, trading_service, evaluator)
            logger.info(f"Sleeping for {args.interval} seconds...")
            await asyncio.sleep(args.interval)
    else:
        await run_once(args, market_data, trading_service, evaluator)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
