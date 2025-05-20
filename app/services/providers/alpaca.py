"""
Alpaca Market Data Provider

Provides market data through Alpaca's API.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import alpaca_trade_api as tradeapi
import requests
import re

from app.services.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

class AlpacaProvider(MarketDataProvider):
    """
    Provider for Alpaca market data.
    
    Uses Alpaca's API directly to retrieve historical and real-time market data.
    This provider adapts to different Alpaca subscription levels:
    - Level 1 (Basic/Starter): Limited market data access
    - Level 2 (Pro): Extended data access and faster rates
    - Level 3 (AlgoTrader Plus): Premium data with options and full market depth
    """
    
    def __init__(self):
        """Initialize the Alpaca provider with API credentials and determine subscription level."""
        # Determine Alpaca mode (paper or live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration based on mode
        if alpaca_mode == "live":
            self.alpaca_key = os.getenv("ALPACA_LIVE_KEY")
            self.alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("AlpacaProvider: Configured for LIVE trading mode")
        else:  # paper mode by default
            self.alpaca_key = os.getenv("ALPACA_PAPER_KEY")
            self.alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("AlpacaProvider: Configured for PAPER trading mode")
        
        # Data URLs for stocks/options and crypto
        self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        self.crypto_data_url = os.getenv("ALPACA_CRYPTO_DATA_URL", "https://data.alpaca.markets/v1beta3/crypto")
        if not self.crypto_data_url:
            logger.warning("AlpacaProvider: ALPACA_CRYPTO_DATA_URL not set in environment. Crypto data may fail.")
        
        # Subscription level (default to 1 if not specified)
        self.subscription_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
        logger.info(f"AlpacaProvider: Using Alpaca subscription level {self.subscription_level}")
        
        # Initialize Alpaca client
        self.client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                # Remove /v2 from URL if present
                if self.base_url.endswith("/v2"):
                    self.base_url = self.base_url.rstrip("/v2")
                
                # Initialize the client without data_url parameter to avoid errors
                self.client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=self.base_url
                )
                
                # Store data_url separately for direct API calls
                self.data_url = self.data_url
                logger.info(f"AlpacaProvider: Initialized Alpaca client with base_url: {self.base_url}")
                logger.info(f"AlpacaProvider: Will use data_url: {self.data_url} for direct API calls")
            except Exception as e:
                logger.error(f"AlpacaProvider: Failed to initialize Alpaca client: {e}")
                self.client = None
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Alpaca"
    
    @property
    def requires_api_key(self) -> bool:
        """Return whether this provider requires API keys."""
        return True
    
    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self.client is not None
        
    @property
    def has_options_data(self) -> bool:
        """Check if options data is available (subscription level 3)."""
        return self.subscription_level >= 3
        
    @property
    def has_extended_data(self) -> bool:
        """Check if extended market data is available (subscription level 2+)."""
        return self.subscription_level >= 2
    
    async def get_historical_data(self, symbol: str, start_date: Union[datetime, str], end_date: Union[datetime, str], timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data (can be datetime or string in ISO format)
            end_date: End date for data (can be datetime or string in ISO format)
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with historical data
        """
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch historical data")
            return pd.DataFrame()
        
        try:
            # Convert string dates to datetime objects if needed
            start_dt = start_date
            end_dt = end_date
            
            # Use our specialized date formatter that handles 1d, 5d, etc. correctly
            # This handles both datetime objects and string dates including relative ones like '1d'
            start_str = self._format_date_param(start_date)
            end_str = self._format_date_param(end_date)
            
            logger.debug(f"AlpacaProvider: Formatted dates: start={start_str}, end={end_str}")
            
            logger.info(f"AlpacaProvider: Using timestamps: {start_str} to {end_str}")
            
            # Map timeframe to Alpaca format
            alpaca_timeframe = timeframe
            if timeframe == "1d":
                alpaca_timeframe = "1Day"
            elif timeframe == "1h":
                alpaca_timeframe = "1Hour"
            
            # Check if it's a crypto symbol (contains '/' or '-')
            if '/' in symbol or '-' in symbol:
                # Format symbol for Alpaca crypto API (BTC/USD)
                crypto_symbol = symbol.replace('-', '/')
                logger.info(f"AlpacaProvider: Detected crypto symbol {symbol}, using crypto data API with formatted symbol {crypto_symbol}")
                return await self._get_crypto_data(crypto_symbol, start_str, end_str, alpaca_timeframe)
            
            # Default path for stocks
            logger.info(f"AlpacaProvider: Fetching historical data for {symbol} from {start_str} to {end_str} with timeframe {alpaca_timeframe}")
            
            # Ensure API call is compatible with installed version
            try:
                # Try the newer API first
                # Always make sure our formatted dates are used in the API call
                formatted_start = self._format_date_param(start_str)
                formatted_end = self._format_date_param(end_str)
                
                logger.debug(f"AlpacaProvider: Making API call with dates: start={formatted_start}, end={formatted_end}")
                
                bars = self.client.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=formatted_start,
                    end=formatted_end,
                    limit=10000
                ).df
            except (TypeError, AttributeError):
                # Fall back to older API if needed
                logger.info(f"AlpacaProvider: Falling back to older Alpaca API for {symbol}")
                # Format dates for older API version as well
                formatted_start = self._format_date_param(start_str)
                formatted_end = self._format_date_param(end_str)
                
                bars = self.client.get_barset(
                    symbols=symbol,
                    timeframe=alpaca_timeframe,
                    start=formatted_start,
                    end=formatted_end,
                    limit=10000
                ).df[symbol]
            
            # Process the data if we got any
            if not bars.empty:
                # Make sure the index is a datetime
                if not isinstance(bars.index, pd.DatetimeIndex):
                    bars.index = pd.to_datetime(bars.index)
                    
                logger.info(f"AlpacaProvider: Successfully retrieved {len(bars)} bars for {symbol}")
                return bars
            else:
                logger.warning(f"AlpacaProvider: No data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching historical data: {str(e)}")
            # Try to be helpful with specific error messages
            if "not found" in str(e).lower():
                logger.warning(f"AlpacaProvider: Symbol {symbol} not found in Alpaca - make sure you have the right subscription level")
            return pd.DataFrame()
    
    async def _get_crypto_data(self, symbol: str, start_str: str, end_str: str, timeframe: str) -> pd.DataFrame:
        """
        Get crypto data directly from Alpaca API.
        
        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD')
            start_str: Start date string in YYYY-MM-DD format
            end_str: End date string in YYYY-MM-DD format
            timeframe: Timeframe for data (e.g., '1Day', '1Hour')
            
        Returns:
            DataFrame with crypto data
        """
        try:
            # Use the crypto data URL from .env
            base_url = f"{getattr(self, 'crypto_data_url', os.getenv('ALPACA_CRYPTO_DATA_URL', 'https://data.alpaca.markets'))}/v1beta3/crypto/bars"
            
            # Map timeframe to v1beta3 format
            v1beta3_timeframe = timeframe
            if timeframe == "1Day":
                v1beta3_timeframe = "1D"
            elif timeframe == "1Hour":
                v1beta3_timeframe = "1H"
            
            # Request parameters
            params = {
                "symbols": symbol,  # symbol should already be formatted as BTC/USD
                "timeframe": v1beta3_timeframe,
                "start": start_str,
                "end": end_str,
                "limit": 1000,
                "_cache_buster": datetime.now().timestamp()  # Force refresh by preventing caching
            }
            
            # Authentication headers
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Execute request
            logger.info(f"AlpacaProvider: Making direct API call to Alpaca crypto endpoint for {symbol} at {base_url}")
            response = requests.get(base_url, params=params, headers=headers)
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                
                # Verify we have data for this symbol
                if data and "bars" in data and symbol in data["bars"] and len(data["bars"][symbol]) > 0:
                    # Convert data to DataFrame
                    bars = data["bars"][symbol]
                    df = pd.DataFrame(bars)
                    
                    # Rename and format columns to match expected format
                    df.rename(columns={
                        "t": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume"
                    }, inplace=True)
                    
                    # Convert timestamp column to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    
                    logger.info(f"AlpacaProvider: Successfully retrieved {len(df)} bars for {symbol} from Alpaca v1beta3 API")
                    return df
                else:
                    logger.warning(f"AlpacaProvider: No data returned for {symbol} from Alpaca v1beta3 API")
                    return pd.DataFrame()
            else:
                error_msg = f"API error: {response.status_code} {response.text[:100]}"
                logger.warning(f"AlpacaProvider: {error_msg}")
                # Specifically check for authorization errors
                if response.status_code == 403:
                    logger.warning("AlpacaProvider: Received 403 Forbidden. Your Alpaca plan likely does not include crypto data access.")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"AlpacaProvider: Error in direct API call to Alpaca: {str(e)[:200]}")
            return pd.DataFrame()

    # Cache for latest prices to avoid redundant API calls
    _price_cache = {}
    _price_cache_time = {}
    _price_cache_expiry = 5  # seconds - reduced from 60 to enable more frequent price updates
    
    def _get_from_cache(self, key):
        """
        Get a value from the cache if it exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        if key in self._price_cache and key in self._price_cache_time:
            timestamp = self._price_cache_time[key]
            if (datetime.now() - timestamp).total_seconds() < self._price_cache_expiry:
                return self._price_cache[key]
        return None
        
    def _add_to_cache(self, key, value, expiry_seconds=None):
        """
        Add a value to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expiry_seconds: Optional custom expiry time in seconds
        """
        self._price_cache[key] = value
        self._price_cache_time[key] = datetime.now()
        
        # Set custom expiry if provided
        if expiry_seconds is not None:
            # Store the custom expiry with the key
            self._price_cache_expiry = expiry_seconds
    
    def _format_date_param(self, date_param):
        """
        Ensures date parameters are in the correct format for Alpaca API calls.
        Converts string dates like '1d', '5d' into proper ISO format dates.
        
        Args:
            date_param: The date parameter which could be a datetime, a string in ISO format, or a relative timeframe
            
        Returns:
            A properly formatted date string or the original parameter if it's already valid
        """
        if isinstance(date_param, datetime):
            # If it's already a datetime, format it correctly
            return date_param.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        if isinstance(date_param, str):
            # Check if it's a relative timeframe like '1d', '5d', etc.
            relative_pattern = re.compile(r'^(\d+)([dhwmy])$')
            match = relative_pattern.match(date_param.lower())
            
            if match:
                # It's a relative time reference
                amount = int(match.group(1))
                unit = match.group(2)
                
                now = datetime.now()
                
                if unit == 'd':  # days
                    date = now - timedelta(days=amount)
                elif unit == 'h':  # hours
                    date = now - timedelta(hours=amount)
                elif unit == 'w':  # weeks
                    date = now - timedelta(weeks=amount)
                elif unit == 'm':  # months (approximate)
                    date = now - timedelta(days=amount*30)
                elif unit == 'y':  # years (approximate)
                    date = now - timedelta(days=amount*365)
                else:
                    # No valid unit, return as-is
                    return date_param
                    
                return date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Check if it's already in ISO format
            if 'T' in date_param and ('Z' in date_param or '+' in date_param):
                # Likely already in ISO format
                return date_param
                
            # Try to parse as a simple date
            try:
                date = pd.to_datetime(date_param)
                return date.strftime('%Y-%m-%dT%H:%M:%SZ')
            except:
                # If all parsing fails, return as-is
                return date_param
        
        # For any other type, return as-is
        return date_param
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            The latest price as a float
        """
        if not self.client:
            logger.error(f"AlpacaProvider: Client not initialized, cannot get latest price")
            return 0.0
            
        # Check if it's a crypto symbol
        if '/' in symbol:
            return await self.get_latest_crypto_price_realtime(symbol)
        
        # Try to get from cache first if not expired
        cache_key = f"price_{symbol}"
        cached_price = self._get_from_cache(cache_key)
        if cached_price is not None:
            logger.debug(f"AlpacaProvider: Using cached price for {symbol}: {cached_price}")
            return cached_price
        
        # Not in cache or expired, fetch new price
        try:
            # For non-crypto symbols
            logger.debug(f"AlpacaProvider: Getting latest price for {symbol}")
            end = datetime.now()
            start = end - timedelta(hours=24)  # Look back 24 hours max
            
            # Try with different timeframes if needed
            timeframes = ["1Min", "5Min", "1Day"]
            
            for timeframe in timeframes:
                try:
                    bars = await self.get_historical_data(symbol, start, end, timeframe)
                    if not bars.empty:
                        # Get the latest bar's closing price
                        latest_price = float(bars['close'].iloc[-1])
                        logger.info(f"{symbol} prix actuel (dernière barre): ${latest_price:.4f}")
                        
                        # Cache the price
                        self._add_to_cache(cache_key, latest_price, expiry_seconds=60)
                        return latest_price
                except Exception as e:
                    logger.warning(f"AlpacaProvider: Failed to get {timeframe} data for {symbol}: {e}")
                    continue
            
            # If we got here, we couldn't get data from any timeframe
            logger.error(f"AlpacaProvider: Could not get latest price for {symbol} after trying all timeframes")
            return 0.0
        except Exception as e:
            logger.error(f"AlpacaProvider: Error getting price for {symbol}: {str(e)}")
            return 0.0
        
    async def get_latest_crypto_price_realtime(self, symbol: str) -> float:
        """
        Get the latest real-time price for a crypto symbol using direct API call.
        This bypasses the historical bar API to get truly real-time prices.
        
        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD')
            
        Returns:
            The latest real-time price as a float
        """
        # Check cache first with very short expiry
        cache_key = f"rt_price_{symbol}"
        cached_price = self._get_from_cache(cache_key)
        if cached_price is not None:
            logger.debug(f"AlpacaProvider: Using cached real-time price for {symbol}: {cached_price}")
            return cached_price
            
        try:
            # Ensure proper symbol format (BTC/USD)
            if '/' not in symbol and '-' in symbol:
                symbol = symbol.replace('-', '/')
            
            # Try multiple endpoints and formats to maximize chances of success
            methods_to_try = [
                # Method 1: v1beta3 quotes API (latest Alpaca crypto endpoint)
                {
                    'url': f"{self.data_url}/v1beta3/crypto/us/latest/quotes",
                    'params': {"symbols": symbol},
                    'parser': lambda data: (data['quotes'][symbol]['ap'] + data['quotes'][symbol]['bp']) / 2 if symbol in data.get('quotes', {}) else None,
                    'description': 'v1beta3 quotes endpoint'
                },
                # Method 2: v1beta3 trades API
                {
                    'url': f"{self.data_url}/v1beta3/crypto/us/latest/trades",
                    'params': {"symbols": symbol},
                    'parser': lambda data: data['trades'][symbol][0]['p'] if symbol in data.get('trades', {}) and data['trades'][symbol] else None,
                    'description': 'v1beta3 trades endpoint'
                },
                # Method 3: v1beta2 quotes API (older format)
                {
                    'url': f"{self.data_url}/v1beta2/crypto/quotes",
                    'params': {"symbols": symbol},
                    'parser': lambda data: data['quotes'][symbol][0]['ap'] if symbol in data.get('quotes', {}) and data['quotes'][symbol] else None,
                    'description': 'v1beta2 quotes endpoint'
                },
                # Method 4: v2 latest trade API (for legacy compatibility)
                {
                    'url': f"{self.data_url}/v2/stocks/{symbol.replace('/', '')}/trades/latest",
                    'params': {},
                    'parser': lambda data: data['trade']['p'] if 'trade' in data else None,
                    'description': 'v2 trades endpoint'
                },
            ]
            
            # Authentication headers
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Try each method in order
            for method in methods_to_try:
                try:
                    logger.info(f"AlpacaProvider: Making direct API call to {method['description']} for {symbol}")
                    response = requests.get(method['url'], params=method['params'], headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        price = method['parser'](data)
                        
                        if price and price > 0:
                            logger.info(f"{symbol} price from {method['description']}: ${price:.4f}")
                            self._add_to_cache(cache_key, price, expiry_seconds=5)
                            return price
                    else:
                        logger.debug(f"AlpacaProvider: {method['description']} returned {response.status_code}: {response.text[:100]}")
                except Exception as endpoint_error:
                    logger.debug(f"AlpacaProvider: Error with {method['description']}: {str(endpoint_error)[:100]}")
                    continue
            
            # If we get here, all direct API calls failed, try the dedicated crypto data endpoint
            logger.info(f"AlpacaProvider: Trying dedicated crypto data endpoint for {symbol}")
            current_price = await self._get_crypto_price_from_bars(symbol)
            if current_price > 0:
                logger.info(f"{symbol} price from historical bars: ${current_price:.4f}")
                self._add_to_cache(cache_key, current_price, expiry_seconds=30)
                return current_price
            
            # If everything fails, log a clear error message    
            logger.error(f"AlpacaProvider: Failed to get crypto price for {symbol} after trying all methods")
            return 0.0
        except Exception as e:
            logger.error(f"AlpacaProvider: Error getting real-time price: {str(e)}")
            # Fall back to the historical method
            return await self._get_crypto_price_from_bars(symbol)
            
    async def _get_crypto_price_from_bars(self, symbol: str) -> float:
        """
        Fallback method to get crypto price from historical bars when real-time fails.
        This method tries multiple approaches and APIs to maximize success with Alpaca.
        
        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD')
            
        Returns:
            The latest bar price as a float
        """
        try:
            # Ensure proper symbol format
            if '/' not in symbol and '-' in symbol:
                symbol = symbol.replace('-', '/')
            
            # Make direct API call to Alpaca's v1beta3 bars endpoint (most reliable for crypto)
            end = datetime.now()
            start = end - timedelta(minutes=30)  # Look back 30 minutes to ensure we get data
            end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
            start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            #Mytest
            # Try direct API call to v1beta3 bars endpoint first
            urls_to_try = [
                # Primary endpoint - v1beta3/crypto/us/bars
                {
                    'url': f"{self.data_url}/v1beta3/crypto/us/bars",
                    'params': {
                        "symbols": symbol,
                        "timeframe": "1Min",
                        "start": start_str,
                        "end": end_str,
                        "limit": 1
                    },
                    'parser': lambda data: float(data['bars'][symbol][0]['c']) if symbol in data.get('bars', {}) and data['bars'][symbol] else None,
                    'description': 'v1beta3 crypto bars endpoint'
                },
                # Secondary endpoint - v2/stocks/{symbol}/bars
                {
                    'url': f"{self.data_url}/v2/stocks/{symbol.replace('/', '')}/bars",
                    'params': {
                        "timeframe": "1Min",
                        "start": start_str,
                        "end": end_str,
                        "limit": 1
                    },
                    'parser': lambda data: float(data['bars'][0]['c']) if 'bars' in data and data['bars'] else None,
                    'description': 'v2 stock bars endpoint (legacy)'
                },
                # Tertiary endpoint - v1beta2/crypto/bars
                {
                    'url': f"{self.data_url}/v1beta2/crypto/bars",
                    'params': {
                        "symbols": symbol,
                        "timeframe": "1Min",
                        "start": start_str,
                        "end": end_str,
                        "limit": 1
                    },
                    'parser': lambda data: float(data['bars'][symbol][0]['c']) if symbol in data.get('bars', {}) and data['bars'][symbol] else None,
                    'description': 'v1beta2 crypto bars endpoint'
                },
            ]
            
            # Headers for API calls
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Try each endpoint
            for endpoint in urls_to_try:
                try:
                    logger.info(f"AlpacaProvider: Trying {endpoint['description']} for {symbol}")
                    response = requests.get(endpoint['url'], params=endpoint['params'], headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        price = endpoint['parser'](data)
                        
                        if price and price > 0:
                            logger.info(f"{symbol} price from {endpoint['description']}: ${price:.4f}")
                            return price
                    else:
                        logger.debug(f"AlpacaProvider: {endpoint['description']} returned {response.status_code}: {response.text[:100]}")
                except Exception as endpoint_error:
                    logger.debug(f"AlpacaProvider: Error with {endpoint['description']}: {str(endpoint_error)[:100]}")
                    continue
            
            # If all API calls fail, try the standard historical data method through self.get_historical_data
            timeframes = ["1Min", "5Min", "1Day"]
            for timeframe in timeframes:
                try:
                    # Always use proper datetime objects here, not strings
                    # Ensure we're using datetime objects for start and end parameters
                    now = datetime.now()
                    past = now - timedelta(minutes=60)  # Look back 60 minutes to ensure we get enough data
                    
                    logger.info(f"AlpacaProvider: Trying to get {timeframe} bars for {symbol} from {past.isoformat()} to {now.isoformat()}")
                    bars = await self.get_historical_data(symbol, past, now, timeframe)
                    if not bars.empty:
                        latest_price = float(bars['close'].iloc[-1])
                        logger.info(f"{symbol} price from historical {timeframe} bars: ${latest_price:.4f}")
                        return latest_price
                except Exception as e:
                    logger.debug(f"AlpacaProvider: Failed to get {timeframe} bars for {symbol}: {str(e)[:100]}")
                    continue
            
            # Last resort: try to get snapshot price from v2 snapshot endpoint
            try:
                snapshot_url = f"{self.data_url}/v2/stocks/{symbol.replace('/', '')}/snapshot"
                response = requests.get(snapshot_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'latestTrade' in data and 'p' in data['latestTrade']:
                        price = float(data['latestTrade']['p'])
                        logger.info(f"{symbol} price from snapshot: ${price:.4f}")
                        return price
            except Exception as e:
                pass
            
            # If all attempts fail, log clear error and return 0
            logger.error(f"AlpacaProvider: Could not get any price data for {symbol} after trying all endpoints")
            return 0.0
        except Exception as e:
            logger.error(f"AlpacaProvider: Error in fallback price fetch: {str(e)}")
            return 0.0

    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', 'option', etc.)
            
        Returns:
            List of available symbols
        """
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch market symbols")
            return []
        
        try:
            if market_type.lower() == "option":
                # Options data requires subscription level 3
                if not self.has_options_data:
                    logger.warning("AlpacaProvider: Options data requires Alpaca subscription level 3")
                    return []
                    
                # This would need to access Alpaca's options API
                # Implementation depends on the exact Alpaca SDK version
                logger.info("AlpacaProvider: Fetching available options symbols")
                try:
                    # This is an example - the actual implementation will depend on Alpaca's API
                    # For most recent Alpaca API versions
                    options = self.client.get_option_chain("SPY")
                    return [option.symbol for option in options]
                except AttributeError:
                    logger.warning("AlpacaProvider: Options API not available in this version of Alpaca SDK")
                    return []
                    
            elif market_type.lower() == "crypto":
                # Crypto data may require subscription level 2+
                logger.info("AlpacaProvider: Fetching available crypto symbols")
                assets = self.client.list_assets(asset_class='crypto')
                return [asset.symbol for asset in assets if asset.tradable]
                
            else:  # stocks and other standard assets
                logger.info("AlpacaProvider: Fetching available stock symbols")
                assets = self.client.list_assets(asset_class='us_equity')
                return [asset.symbol for asset in assets if asset.tradable]
                
        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching market symbols for {market_type}: {str(e)}")
            if "rate limit" in str(e).lower():
                logger.warning("AlpacaProvider: Rate limit reached. Consider upgrading your subscription level.")
            return []
