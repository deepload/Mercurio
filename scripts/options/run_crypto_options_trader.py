#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Options Trader Script

This script runs options trading strategies for cryptocurrency derivatives
using the Alpaca API for crypto market data and options trading.

Usage:
    python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000
"""

import os
import sys
import asyncio
import logging
import sys
import threading
import concurrent.futures
from dotenv import load_dotenv
from datetime import datetime, timedelta
from argparse import ArgumentParser
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Charger les variables d'environnement
load_dotenv()

# Charger la liste de symboles personnalisés depuis le CSV si --use-custom-symbols est passé
import csv
CUSTOM_SYMBOLS_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "all_crypto_20250511.csv")
default_crypto_list = "BTC/USD,ETH/USD,SOL/USD,DOT/USD,AVAX/USD,ADA/USD,XRP/USD,LUNA/USD,DOGE/USD,MATIC/USD,LINK/USD,LTC/USD,UNI/USD,ALGO/USD,ATOM/USD,FIL/USD,AAVE/USD,MKR/USD,COMP/USD,SNX/USD,BAT/USD,YFI/USD,CRV/USD,GRT/USD,UMA/USD,ZRX/USD"
custom_crypto_list_str = os.getenv("PERSONALIZED_CRYPTO_LIST", default_crypto_list)
PERSONALIZED_CRYPTO_OPTIONS_LIST = [s.strip() for s in custom_crypto_list_str.split(',')]

def load_custom_symbols_from_csv():
    symbols = []
    try:
        with open(CUSTOM_SYMBOLS_CSV, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = row.get('symbol')
                if symbol:
                    # Convert to Alpaca format if needed (BTC-USD -> BTC/USD)
                    symbol = symbol.replace('-', '/').upper()
                    symbols.append(symbol)
    except Exception as e:
        logger.warning(f"Impossible de charger le CSV de symboles personnalisés: {e}")
    return symbols

from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService
from app.services.trading import TradingService
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.utils.logging import setup_logging


# Configure logging
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Console handler
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description='Run crypto options trading strategy')
    
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['LONG_CALL', 'LONG_PUT', 'IRON_CONDOR', 'BUTTERFLY', 'MIXED'],
                        help='Options trading strategy to employ')
    
    parser.add_argument('--symbols', type=str, nargs='+', required=False,
                        help='Crypto symbols to trade options on (e.g., BTC ETH)')
                        
    parser.add_argument('--capital', type=float, default=50000.0,
                        help='Total capital to allocate for crypto options trading')
                        
    parser.add_argument('--allocation-per-trade', type=float, default=0.03,
                        help='Maximum allocation per trade as percentage of capital (0.03 = 3%)')
                        
    parser.add_argument('--max-positions', type=int, default=5,
                        help='Maximum number of positions to hold simultaneously')
                        
    parser.add_argument('--days-to-expiry', type=int, default=14,
                        help='Target days to expiration for options')
                        
    parser.add_argument('--delta-target', type=float, default=0.4,
                        help='Target delta for option selections')
                        
    parser.add_argument('--profit-target', type=float, default=0.7,
                        help='Profit target as percentage of option premium (0.7 = 70%)')
                        
    parser.add_argument('--stop-loss', type=float, default=0.5,
                        help='Stop loss as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--volatility-threshold', type=float, default=0.01,
                        help='Minimum implied volatility to enter a trade')
                        
    parser.add_argument('--paper-trading', action='store_true',
                        help='Use paper trading mode instead of live trading')
                        
    parser.add_argument('--use-threads', action='store_true',
                        help='Process symbols using multiple threads for faster execution')
                        
    parser.add_argument('--use-custom-symbols', '--use_custom_symbols', action='store_true',
                        help='Use symbols as provided without adding USD suffix')
                        
    parser.add_argument('--duration', type=str, default='1d',
                        help='Trading duration: format as 1d (1 day), 5h (5 hours), 30m (30 minutes)')
                        
    return parser.parse_args()


def get_strategy_class(strategy_name: str):
    """Map strategy name to strategy class."""
    strategy_map = {
        'LONG_CALL': LongCallStrategy,
        'LONG_PUT': LongPutStrategy,
        'IRON_CONDOR': IronCondorStrategy,
        'BUTTERFLY': ButterflySpreadStrategy
    }
    
    return strategy_map.get(strategy_name)


def format_crypto_symbol(symbol: str) -> str:
    """Format crypto symbol for Alpaca API.
    
    Converts symbols like 'BTC' to 'BTC/USD' format which is required by Alpaca for crypto
    """
    # Remove any USD suffix if present
    symbol = symbol.upper().replace('USD', '')
    return f"{symbol}/USD"


async def run_crypto_options_trader(args):
    """Run crypto options trader with the provided arguments."""
    logger.info(f"Starting crypto options trader with strategy: {args.strategy}")
    
    # Add custom symbols support
    if args.use_custom_symbols:
        crypto_symbols = load_custom_symbols_from_csv()
        logger.info(f"Utilisation de la liste personnalisée de {len(crypto_symbols)} cryptomonnaies depuis CSV: {CUSTOM_SYMBOLS_CSV}")
        if not crypto_symbols:
            logger.error(f"Aucun symbole personnalisé trouvé dans le CSV {CUSTOM_SYMBOLS_CSV}. Abandon.")
            return
    else:
        if not args.symbols:
            logger.error("Erreur: Vous devez spécifier des symboles avec --symbols ou utiliser --use-custom-symbols")
            return
        crypto_symbols = [format_crypto_symbol(symbol) for symbol in args.symbols]
    logger.info(f"Trading on crypto symbols: {crypto_symbols}")
    
    # Initialize services in the correct order
    market_data_service = MarketDataService()
    
    # Forcer l'utilisation des données réelles pour les crypto en utilisant le même niveau d'abonnement
    # que dans run_strategy_crypto_trader.py
    market_data_service.subscription_level = 3
    
    # Initialiser le service de trading
    trading_service = TradingService(is_paper=args.paper_trading)  # Ceci gère déjà la création du client Alpaca
    options_service = OptionsService(trading_service, market_data_service)
    
    # Get account information
    account = await trading_service.get_account_info()
    account_value = float(account.get('equity', args.capital))
    logger.info(f"Account value: ${account_value:.2f}")
    
    # Create strategy instances
    strategy_class = get_strategy_class(args.strategy)
    
    if not strategy_class and args.strategy != 'MIXED':
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    strategies = []
    
    if args.strategy == 'MIXED':
        # Create a mix of different strategies for diversification
        strategy_allocation = {
            'LONG_CALL': 0.4,
            'LONG_PUT': 0.3,
            'IRON_CONDOR': 0.15,
            'BUTTERFLY': 0.15
        }
        
        for strategy_name, allocation in strategy_allocation.items():
            strategy_class = get_strategy_class(strategy_name)
            for symbol in crypto_symbols:
                # Adjust allocation based on strategy weight
                position_size = args.allocation_per_trade * allocation * 3
                strategy = strategy_class(
                    underlying_symbol=symbol,
                    account_size=account_value,
                    max_position_size=position_size,
                    min_implied_volatility=0.3,  # Higher values for crypto
                    max_implied_volatility=2.0,  # Crypto can have high volatility
                    max_days_to_expiry=args.days_to_expiry,
                    min_days_to_expiry=max(1, args.days_to_expiry // 2),
                    target_delta=args.delta_target,
                    delta_range=0.15,
                    profit_target_pct=args.profit_target,
                    stop_loss_pct=args.stop_loss,
                    roll_when_dte=5,  # Roll positions with 5 days to expiry
                    use_technical_filters=True
                )
                # Connect strategy to our services
                strategy.trading_service = trading_service
                strategy.options_service = options_service
                # Initialiser le broker pour permettre l'exécution des ordres
                strategy.broker = trading_service
                strategies.append(strategy)
    else:
        # Use single requested strategy
        for symbol in crypto_symbols:
            strategy = strategy_class(
                underlying_symbol=symbol,
                account_size=account_value,
                max_position_size=args.allocation_per_trade,
                min_implied_volatility=0.3,  # Higher values for crypto
                max_implied_volatility=2.0,  # Crypto can have high volatility
                max_days_to_expiry=args.days_to_expiry,
                min_days_to_expiry=max(1, args.days_to_expiry // 2),
                target_delta=args.delta_target,
                delta_range=0.15,
                profit_target_pct=args.profit_target,
                stop_loss_pct=args.stop_loss,
                roll_when_dte=5,  # Roll positions with 5 days to expiry
                use_technical_filters=True
            )
            # Connect strategy to our services
            strategy.trading_service = trading_service
            strategy.options_service = options_service
            # Initialiser le broker pour permettre l'exécution des ordres
            strategy.broker = trading_service
            strategies.append(strategy)
    
    # Parse duration string to timedelta
    duration_str = args.duration.lower()
    if duration_str.endswith('d'):
        # Days format (e.g., '1d')
        days = float(duration_str[:-1])
        duration = timedelta(days=days)
    elif duration_str.endswith('h'):
        # Hours format (e.g., '5h')
        hours = float(duration_str[:-1])
        duration = timedelta(hours=hours)
    elif duration_str.endswith('m'):
        # Minutes format (e.g., '30m')
        minutes = float(duration_str[:-1])
        duration = timedelta(minutes=minutes)
    else:
        # Default to days if no unit specified
        try:
            days = float(duration_str)
            duration = timedelta(days=days)
        except ValueError:
            logger.error(f"Durée non reconnue: {duration_str}. Utilisation de la durée par défaut de 1 jour.")
            duration = timedelta(days=1)
    
    # Run trading loop
    end_time = datetime.now() + duration
    position_count = 0
    
    logger.info(f"Trading will run until: {end_time}")
    
    # Function to process a single strategy/symbol
    async def process_strategy(strategy, position_lock):
        nonlocal position_count, end_time
        symbol = strategy.underlying_symbol
        
        while datetime.now() < end_time:
            # Skip if we've reached max positions
            with position_lock:
                if position_count >= args.max_positions:
                    await asyncio.sleep(60)  # Wait and check again later
                    continue
            
            # Get market data for crypto
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days of historical data
            
            try:
                # Format correct pour l'API Alpaca - gardons le format BTC/USD car le MarketDataService s'attend à ce format
                # et fera la conversion appropriée en interne
                market_data = await market_data_service.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1h"  # Use hourly data for crypto due to higher volatility
                )
                
                # Add volatility metric for crypto (not included in raw data)
                if not market_data.empty and len(market_data) > 20:
                    market_data['returns'] = market_data['close'].pct_change()
                    market_data['volatility'] = market_data['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized from hourly data
                
                # Only enter trades if volatility meets minimum threshold
                # Lower threshold for demo mode to allow more trades with sample data
                volatility_threshold = args.volatility_threshold * 0.5 if 'sample' in str(market_data_service) else args.volatility_threshold
                
                if (market_data.empty or 
                    'volatility' not in market_data.columns or 
                    market_data['volatility'].iloc[-1] < volatility_threshold):
                    logger.info(f"Skipping {symbol} due to insufficient volatility")
                    await asyncio.sleep(300)  # Check again in 5 minutes
                    continue
                
                # Check for entry conditions
                if await strategy.should_enter(market_data):
                    logger.info(f"Entry signal detected for {symbol} using {strategy.__class__.__name__}")
                    
                    # Execute entry
                    entry_result = await strategy.execute_entry()
                    
                    if entry_result.get('success', False):
                        logger.info(f"Entry executed: {entry_result}")
                        with position_lock:
                            position_count += 1
                    else:
                        logger.warning(f"Entry failed: {entry_result.get('error', 'Unknown error')}")
                
                # Check for exit conditions if we have a position
                # Use the appropriate method signature based on the strategy implementation
                if hasattr(strategy, 'open_positions') and strategy.open_positions:
                    if await strategy.should_exit("dummy_position_id", market_data):
                        logger.info(f"Exit signal detected for {symbol}")
                        
                        # Execute exit
                        exit_result = await strategy.execute_exit("dummy_position_id")
                        
                        if exit_result.get('success', False):
                            logger.info(f"Exit executed: {exit_result}")
                            with position_lock:
                                position_count -= 1
                    else:
                        logger.warning(f"Exit failed: {exit_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Sleep before next iteration
            await asyncio.sleep(300)  # Check every 5 minutes for crypto
    
    # Run trading loop using threads or sequential processing
    position_lock = threading.Lock()
    log_lock = threading.Lock()
    # Use a container for shared variables to avoid nonlocal declarations
    shared_state = {'position_count': 0}
    
    # Wrap the logger to ensure thread-safe logging
    def safe_log(level, message):
        with log_lock:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
            elif level == 'debug':
                logger.debug(message)
    
    # Modify process_strategy to use the thread-safe logger
    async def process_strategy_safe(strategy, position_lock, shared_state):
        symbol = strategy.underlying_symbol
        safe_log('info', f"Starting processing for {symbol} with {strategy.__class__.__name__}")
        
        while datetime.now() < end_time:
            try:
                # Skip if we've reached max positions
                with position_lock:
                    if shared_state['position_count'] >= args.max_positions:
                        safe_log('info', f"Max positions reached, skipping {symbol}")
                        await asyncio.sleep(300)  # Check again in 5 minutes
                        continue
                
                # Get market data for crypto
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # 30 days of historical data
                
                # For crypto, we might need specific handling in the market data service
                market_data = await market_data_service.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1h"  # Use hourly data for crypto due to higher volatility
                )
                
                # Add volatility metric for crypto (not included in raw data)
                if not market_data.empty and len(market_data) > 20:
                    market_data['returns'] = market_data['close'].pct_change()
                    market_data['volatility'] = market_data['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized from hourly data
                
                # Only enter trades if volatility meets minimum threshold
                if (market_data.empty or 
                    'volatility' not in market_data.columns or 
                    market_data['volatility'].iloc[-1] < args.volatility_threshold):
                    safe_log('info', f"Skipping {symbol} due to insufficient volatility")
                    await asyncio.sleep(300)  # Check again in 5 minutes
                    continue
                
                # Check for entry conditions
                try:
                    should_enter = await strategy.should_enter(market_data)
                    if should_enter:
                        safe_log('info', f"Entry signal detected for {symbol} using {strategy.__class__.__name__}")
                        
                        # Execute entry
                        entry_result = await strategy.execute_entry()
                        
                        if entry_result.get('success', False):
                            safe_log('info', f"Entry executed for {symbol}: {entry_result}")
                            with position_lock:
                                shared_state['position_count'] += 1
                        else:
                            safe_log('warning', f"Entry failed for {symbol}: {entry_result.get('error', 'Unknown error')}")
                except Exception as strat_e:
                    safe_log('error', f"Error in strategy execution for {symbol}: {str(strat_e)}")
                
                # Check for exit conditions if we have a position
                if hasattr(strategy, 'open_positions') and strategy.open_positions:
                    try:
                        if await strategy.should_exit("dummy_position_id", market_data):
                            safe_log('info', f"Exit signal detected for {symbol}")
                            
                            # Execute exit
                            exit_result = await strategy.execute_exit("dummy_position_id")
                            
                            if exit_result.get('success', False):
                                safe_log('info', f"Exit executed for {symbol}: {exit_result}")
                                with position_lock:
                                    shared_state['position_count'] -= 1
                            else:
                                safe_log('warning', f"Exit failed for {symbol}: {exit_result.get('error', 'Unknown error')}")
                    except Exception as exit_e:
                        safe_log('error', f"Error in exit execution for {symbol}: {str(exit_e)}")
            
            except Exception as e:
                safe_log('error', f"Error processing {symbol}: {str(e)}")
            
            # Sleep before next iteration
            await asyncio.sleep(300)  # Check every 5 minutes for crypto
    
    try:
        # Use multithreading if requested
        if args.use_threads:
            safe_log('info', f"Using threaded processing for {len(strategies)} symbols")
            # Create and run tasks for each strategy with improved error handling
            tasks = [asyncio.create_task(process_strategy_safe(strategy, position_lock, shared_state)) for strategy in strategies]
            # Wait for all tasks to complete or until the end time
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing (original behavior)
            logger.info(f"Using sequential processing for {len(strategies)} symbols")
            while datetime.now() < end_time and position_count < args.max_positions:
                for strategy in strategies:
                    symbol = strategy.underlying_symbol
                    
                    # Skip if we've reached max positions
                    if position_count >= args.max_positions:
                        break
                    
                    # Get market data for crypto
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)  # 30 days of historical data
                    
                    try:
                        # For crypto, we might need specific handling in the market data service
                        market_data = await market_data_service.get_historical_data(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            timeframe="1h"  # Use hourly data for crypto due to higher volatility
                        )
                        
                        # Add volatility metric for crypto (not included in raw data)
                        if not market_data.empty and len(market_data) > 20:
                            market_data['returns'] = market_data['close'].pct_change()
                            market_data['volatility'] = market_data['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized from hourly data
                        
                        # Only enter trades if volatility meets minimum threshold
                        if (market_data.empty or 
                            'volatility' not in market_data.columns or 
                            market_data['volatility'].iloc[-1] < args.volatility_threshold):
                            logger.info(f"Skipping {symbol} due to insufficient volatility")
                            continue
                        
                        # Check for entry conditions
                        if await strategy.should_enter(market_data):
                            logger.info(f"Entry signal detected for {symbol} using {strategy.__class__.__name__}")
                            
                            # Execute entry
                            entry_result = await strategy.execute_entry()
                            
                            if entry_result.get('success', False):
                                logger.info(f"Entry executed: {entry_result}")
                                position_count += 1
                            else:
                                logger.warning(f"Entry failed: {entry_result.get('error', 'Unknown error')}")
                        
                        # Check for exit conditions if we have a position
                        # Use the appropriate method signature based on the strategy implementation
                        if hasattr(strategy, 'open_positions') and strategy.open_positions:
                            if await strategy.should_exit("dummy_position_id", market_data):
                                logger.info(f"Exit signal detected for {symbol}")
                                
                                # Execute exit
                                exit_result = await strategy.execute_exit("dummy_position_id")
                                
                                if exit_result.get('success', False):
                                    logger.info(f"Exit executed: {exit_result}")
                                    position_count -= 1
                                else:
                                    logger.warning(f"Exit failed: {exit_result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                
                # Sleep before next iteration
                await asyncio.sleep(300)  # Check every 5 minutes for crypto
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    
    finally:
        # Print open positions summary at the end
        logger.info("Fetching any remaining open positions...")
        try:
            open_positions = await trading_service.get_positions()
            if open_positions:
                logger.info(f"Positions ouvertes restantes: {open_positions}")
            else:
                logger.info("Aucune position ouverte restante.")
        except Exception as e:
            logger.warning(f"Impossible de récupérer les positions ouvertes: {e}")
        
        # Print trading summary
        logger.info("Trading completed")


if __name__ == '__main__':
    import numpy as np  # For volatility calculation
    
    args = parse_arguments()
    asyncio.run(run_crypto_options_trader(args))
