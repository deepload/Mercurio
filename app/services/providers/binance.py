"""
Binance Market Data Provider

This module provides an implementation of the MarketDataProvider interface
for Binance cryptocurrency exchange. It uses the public REST API to fetch
historical and real-time market data for cryptocurrencies.
"""
import os
import logging
import pandas as pd
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

# Import base provider
from app.services.providers.base import MarketDataProvider

# Setup logger
logger = logging.getLogger(__name__)

class BinanceProvider(MarketDataProvider):
    """
    Market data provider implementation for Binance cryptocurrency exchange.
    
    Provides access to historical and real-time market data through
    the Binance public REST API. This provider doesn't require API keys
    for basic market data access, making it accessible for all users.
    """
    
    def __init__(self):
        """Initialize the Binance provider."""
        self.base_url = "https://api.binance.com"
        # Cache for storing data to reduce API calls
        self._cache = {}
        self._cache_expiry = {}
        self._default_cache_seconds = 60  # Default cache expiry in seconds
        
        # Common timeframe mappings between standard format and Binance format
        self.interval_map = {
            "1m": "1m",
            "5m": "5m",
            "5Min": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1w": "1w"
        }
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return "Binance"
    
    @property
    def requires_api_key(self) -> bool:
        """Check if API key is required"""
        # Basic market data doesn't require API keys on Binance
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if this provider is available to use"""
        # Simple check - we'll try to ping the Binance API
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/v3/ping", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Binance API not available: {e}")
            return False
    
    def _convert_symbol_to_binance_format(self, symbol: str) -> str:
        """
        Convert a symbol from standard format (BTC/USD) to Binance format (BTCUSDT).
        
        Args:
            symbol: Symbol in standard format (e.g., 'BTC/USD')
            
        Returns:
            Symbol in Binance format (e.g., 'BTCUSDT')
        """
        if '/' not in symbol and '-' not in symbol:
            # Already in the right format
            return symbol
            
        # Separate base and quote
        if '/' in symbol:
            base, quote = symbol.split('/')
        elif '-' in symbol:
            base, quote = symbol.split('-')
        else:
            return symbol
            
        # Rules for conversion
        if quote == 'USD':
            return f"{base}USDT"
        elif quote == 'USDC':
            return f"{base}USDC"
        elif quote == 'USDT':
            return f"{base}USDT"
        elif quote == 'BTC':
            return f"{base}BTC"
        else:
            # General case
            return f"{base}{quote}"
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'BTC/USD')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Convert the symbol to Binance format
            binance_symbol = self._convert_symbol_to_binance_format(symbol)
            
            # Convert dates to timestamps (milliseconds)
            if isinstance(start_date, datetime):
                start_ts = int(start_date.timestamp() * 1000)
            else:
                start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
                
            if isinstance(end_date, datetime):
                end_ts = int(end_date.timestamp() * 1000)
            else:
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            # Convert the interval to Binance format
            binance_interval = self.interval_map.get(timeframe, "1h")
            
            # Construct the URL for the Binance API
            endpoint = f"/api/v3/klines"
            url = f"{self.base_url}{endpoint}"
            
            params = {
                "symbol": binance_symbol,
                "interval": binance_interval,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": 1000  # Maximum allowed by Binance
            }
            
            logger.info(f"Retrieving Binance data for {symbol} ({binance_symbol}) from {start_date} to {end_date} with interval {binance_interval}")
            
            # Fetch the data
            data = await self._fetch_historical_data(url, params)
            
            if not data:
                return pd.DataFrame()
                
            # Convert the data to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamps to dates
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Use open_time as index
            df.set_index('open_time', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving Binance data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_historical_data(self, url: str, params: Dict[str, Any]) -> List[List[Any]]:
        """
        Helper method to fetch historical data.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            List of kline data or None on error
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not data:
                            logger.warning(f"No data available for {params['symbol']} on Binance")
                            return None
                        
                        return data
                    else:
                        error_msg = await response.text()
                        logger.error(f"Binance API Error ({response.status}): {error_msg}")
                        return None
        except Exception as e:
            logger.error(f"Error during Binance API request: {e}")
            return None
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'BTC/USD')
            
        Returns:
            Latest price as float
        """
        try:
            # Convert the symbol to Binance format
            binance_symbol = self._convert_symbol_to_binance_format(symbol)
            
            # Construct the URL for the Binance API
            endpoint = f"/api/v3/ticker/price"
            url = f"{self.base_url}{endpoint}"
            
            params = {"symbol": binance_symbol}
            
            logger.info(f"Getting current price for {symbol} ({binance_symbol}) via Binance")
            
            # Check cache first
            cache_key = f"price_{binance_symbol}"
            cached_price = self._get_from_cache(cache_key)
            if cached_price is not None:
                logger.info(f"Using cached price for {symbol}: ${cached_price:.4f}")
                return cached_price
            
            # Fetch the data
            price_data = await self._fetch_latest_price(url, params, symbol, binance_symbol)
            
            # Cache the result
            if price_data is not None:
                self._add_to_cache(cache_key, price_data)
                
            return price_data if price_data is not None else 0.0
                
        except Exception as e:
            logger.error(f"Error getting current price for {symbol} via Binance: {e}")
            return 0.0
    
    async def _fetch_latest_price(self, url: str, params: Dict[str, Any], symbol: str, binance_symbol: str) -> Optional[float]:
        """
        Helper method to fetch the latest price.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            symbol: Original symbol
            binance_symbol: Converted Binance symbol
            
        Returns:
            Price as float or None on error
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'price' in data:
                            price = float(data['price'])
                            logger.info(f"Current price for {symbol}: ${price:.4f}")
                            return price
                        else:
                            logger.warning(f"Unexpected response format from Binance for {binance_symbol}")
                            return None
                    else:
                        error_msg = await response.text()
                        logger.error(f"Binance API Error ({response.status}): {error_msg}")
                        return None
        except Exception as e:
            logger.error(f"Error during Binance price request: {e}")
            return None
    
    async def get_market_symbols(self, market_type: str = "crypto") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market (only 'crypto' is supported for Binance)
            
        Returns:
            List of available symbols
        """
        if market_type.lower() != "crypto":
            logger.warning(f"Binance only supports crypto markets, not {market_type}")
            return []
            
        try:
            # Check cache first
            cache_key = "market_symbols_crypto"
            cached_symbols = self._get_from_cache(cache_key)
            if cached_symbols is not None:
                return cached_symbols
                
            # Fetch the exchange info
            url = f"{self.base_url}/api/v3/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'symbols' in data:
                            # Extract symbols and filter for USDT pairs (most common)
                            symbols = [
                                f"{item['baseAsset']}/{item['quoteAsset']}"
                                for item in data['symbols']
                                if item['status'] == 'TRADING'
                            ]
                            
                            # Cache the result for 1 hour (this rarely changes)
                            self._add_to_cache(cache_key, symbols, 3600)
                            
                            logger.info(f"Retrieved {len(symbols)} available symbols from Binance")
                            return symbols
                        else:
                            logger.warning("Unexpected response format from Binance exchangeInfo")
                            return []
                    else:
                        error_msg = await response.text()
                        logger.error(f"Binance API Error ({response.status}): {error_msg}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching market symbols from Binance: {e}")
            
            # Return a default list of common crypto symbols as fallback
            return [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
                "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT"
            ]
    
    def _get_from_cache(self, key: str) -> Any:
        """
        Get a value from the cache if it exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        current_time = time.time()
        if key in self._cache and key in self._cache_expiry:
            if current_time < self._cache_expiry[key]:
                return self._cache[key]
        return None
    
    def _add_to_cache(self, key: str, value: Any, expiry_seconds: Optional[int] = None) -> None:
        """
        Add a value to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry_seconds: Optional custom expiry time in seconds
        """
        self._cache[key] = value
        expiry = expiry_seconds if expiry_seconds is not None else self._default_cache_seconds
        self._cache_expiry[key] = time.time() + expiry
