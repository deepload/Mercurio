#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Top Stocks Options Day Trader

This script runs the daily options trader with the top stocks from the latest report
for a 4-hour day trading session.
"""

import os
import sys
import asyncio
import argparse
import csv
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import the run_options_trader function from the daily options trader script
sys.path.append(os.path.join(project_root, 'options'))
from options.run_daily_options_trader import run_options_trader

# Patch the OptionsStrategyAdapter class to fix the broker attribute issue
from app.strategies.options.strategy_adapter import OptionsStrategyAdapter

# Save the original create_strategy method
original_create_strategy = OptionsStrategyAdapter.create_strategy

# Create a patched version that adds the broker attribute
def patched_create_strategy(*args, **kwargs):
    # Call the original method to create the strategy
    strategy = original_create_strategy(*args, **kwargs)
    
    # Get the trading_service from the arguments (it's the 4th argument)
    if len(args) >= 4 and args[3] is not None:
        trading_service = args[3]
        # If the strategy doesn't have a broker attribute but trading_service has a broker
        if not hasattr(strategy, 'broker') and hasattr(trading_service, 'broker'):
            # Set the broker attribute on the strategy
            strategy.broker = trading_service.broker
    
    return strategy

# Apply the patch
OptionsStrategyAdapter.create_strategy = patched_create_strategy

def get_top_stocks(csv_file, limit=25):
    """Get top stock symbols from the CSV report"""
    symbols = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(symbols) >= limit:
                    break
                # Skip symbols with dots (preferred shares) as they might not have options
                if '.' not in row['Symbol']:
                    symbols.append(row['Symbol'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    return symbols

def main():
    """Main function to run the top stocks options trader"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run options day trader with top stocks')
    parser.add_argument('--strategy', type=str, default='CASH_SECURED_PUT',
                        choices=['COVERED_CALL', 'CASH_SECURED_PUT', 'LONG_CALL', 'LONG_PUT', 
                                'IRON_CONDOR', 'BUTTERFLY', 'MIXED'],
                        help='Options strategy to use (CASH_SECURED_PUT recommended for stability)')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Total capital to allocate for options trading')
    parser.add_argument('--allocation-per-trade', type=float, default=0.05,
                        help='Maximum allocation per trade as percentage of capital (0.05 = 5%)')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Maximum number of positions to hold simultaneously')
    parser.add_argument('--days-to-expiry', type=int, default=14,
                        help='Target days to expiration for options')
    parser.add_argument('--delta-target', type=float, default=0.4,
                        help='Target delta for option selections')
    parser.add_argument('--profit-target', type=float, default=0.3,
                        help='Profit target as percentage of option premium (0.3 = 30%)')
    parser.add_argument('--stop-loss', type=float, default=0.7,
                        help='Stop loss as percentage of option premium (0.7 = 70%)')
    parser.add_argument('--paper-trading', action='store_true', default=True,
                        help='Use paper trading mode instead of live trading')
    parser.add_argument('--limit', type=int, default=25,
                        help='Maximum number of symbols to trade')
    parser.add_argument('--csv-file', type=str, 
                        default=os.path.join(project_root, '..', 'reports', 'best_assets', 
                                           datetime.now().strftime('%Y-%m-%d'), 'top_stocks.csv'),
                        help='Path to the CSV file with top stocks')
    
    args = parser.parse_args()
    
    # Get top stocks from the CSV file
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        # Try to find the most recent file in the reports directory
        reports_dir = os.path.join(project_root, '..', 'reports', 'best_assets')
        if os.path.exists(reports_dir):
            date_dirs = sorted([d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))], reverse=True)
            for date_dir in date_dirs:
                potential_file = os.path.join(reports_dir, date_dir, 'top_stocks.csv')
                if os.path.exists(potential_file):
                    csv_file = potential_file
                    print(f"Using most recent file: {csv_file}")
                    break
    
    symbols = get_top_stocks(csv_file, args.limit)
    if not symbols:
        print("No symbols found in the CSV file")
        return
    
    print(f"Trading options for {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Create args object for the run_options_trader function
    class OptionsArgs:
        pass
    
    options_args = OptionsArgs()
    options_args.strategy = args.strategy
    options_args.capital = args.capital
    options_args.allocation_per_trade = args.allocation_per_trade
    options_args.max_positions = args.max_positions
    options_args.days_to_expiry = args.days_to_expiry
    options_args.delta_target = args.delta_target
    options_args.profit_target = args.profit_target
    options_args.stop_loss = args.stop_loss
    options_args.duration = 0.17  # 4 hours (0.17 days)
    options_args.paper_trading = args.paper_trading
    
    # Add symbols to the args
    options_args.symbols = symbols
    
    # Add paper trading flag if specified
    if args.paper_trading:
        options_args.paper_trading = True
    
    # Run the options trader (we've already patched the OptionsStrategyAdapter)
    asyncio.run(run_options_trader(options_args))

if __name__ == '__main__':
    main()
