"""
Alpaca Market Data Provider

Provides market data through Alpaca's API.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import alpaca_trade_api as tradeapi
import requests

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
        
        # Data URL is the same for both modes
        self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
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
    
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with historical data
        """
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch historical data")
            return pd.DataFrame()
        
        try:
            # Format dates for API with precise timestamps to ensure fresh data
            start_str = start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_str = end_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            logger.info(f"AlpacaProvider: Using precise timestamps: {start_str} to {end_str}")
            
            # Map timeframe to Alpaca format
            alpaca_timeframe = timeframe
            if timeframe == "1d":
                alpaca_timeframe = "1Day"
            elif timeframe == "1h":
                alpaca_timeframe = "1Hour"
            
            # Check if it's a crypto symbol (contains '/')
            if '/' in symbol:
                logger.info(f"AlpacaProvider: Detected crypto symbol {symbol}, using crypto data API")
                return await self._get_crypto_data(symbol, start_str, end_str, alpaca_timeframe)
            
            # Default path for stocks
            logger.info(f"AlpacaProvider: Fetching historical data for {symbol} from {start_str} to {end_str} with timeframe {alpaca_timeframe}")
            
            # Ensure API call is compatible with installed version
            try:
                # Try the newer API first
                bars = self.client.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_str,
                    end=end_str,
                    limit=10000
                ).df
            except (TypeError, AttributeError):
                # Fall back to older API if needed
                logger.info(f"AlpacaProvider: Falling back to older Alpaca API for {symbol}")
                bars = self.client.get_barset(
                    symbols=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_str,
                    end=end_str,
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
            # Use the crypto API endpoint format
            base_url = f"{self.data_url}/v1beta3/crypto/bars"
            
            # Map timeframe to v1beta3 format
            v1beta3_timeframe = timeframe
            if timeframe == "1Day":
                v1beta3_timeframe = "1D"
            elif timeframe == "1Hour":
                v1beta3_timeframe = "1H"
            
            # Request parameters
            params = {
                "symbols": symbol,
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
            logger.info(f"AlpacaProvider: Making direct API call to Alpaca crypto endpoint for {symbol}")
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
            # Direct API call to quotes endpoint
            timestamp = datetime.now().timestamp()
            base_url = f"{self.data_url}/v1beta3/crypto/quotes"
            
            # Request parameters - add timestamp to prevent caching
            params = {
                "symbols": symbol,
                "_nocache": timestamp
            }
            
            # Authentication headers
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Execute request
            logger.info(f"AlpacaProvider: Making direct quote API call for {symbol}")
            response = requests.get(base_url, params=params, headers=headers)
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                
                # Verify we have data for this symbol
                if data and "quotes" in data and symbol in data["quotes"] and len(data["quotes"][symbol]) > 0:
                    # Get the latest quote
                    quote = data["quotes"][symbol][0]
                    # Use ask price as the latest price
                    latest_price = float(quote.get("ap", 0))
                    
                    if latest_price > 0:
                        logger.info(f"{symbol} prix temps réel (cotation): ${latest_price:.4f}")
                        # Cache for very short time (1 second)
                        self._add_to_cache(cache_key, latest_price, expiry_seconds=1)
                        return latest_price
            
            # If direct quote API failed, fall back to trades API
            base_url = f"{self.data_url}/v1beta3/crypto/trades"
            response = requests.get(base_url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify we have data for this symbol
                if data and "trades" in data and symbol in data["trades"] and len(data["trades"][symbol]) > 0:
                    # Get the latest trade
                    trade = data["trades"][symbol][0]
                    # Use trade price
                    latest_price = float(trade.get("p", 0))
                    
                    if latest_price > 0:
                        logger.info(f"{symbol} prix temps réel (dernière transaction): ${latest_price:.4f}")
                        # Cache for very short time
                        self._add_to_cache(cache_key, latest_price, expiry_seconds=1)
                        return latest_price
            
            # If both realtime methods failed, fall back to the historical method
            logger.warning(f"AlpacaProvider: Real-time quote not available for {symbol}, falling back to bars")
            return await self._get_crypto_price_from_bars(symbol)
        except Exception as e:
            logger.warning(f"AlpacaProvider: Error getting real-time price: {str(e)}")
            # Fall back to the historical method
            return await self._get_crypto_price_from_bars(symbol)
            
    async def _get_crypto_price_from_bars(self, symbol: str) -> float:
        """
        Fallback method to get crypto price from historical bars when real-time fails.
        
        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD')
            
        Returns:
            The latest bar price as a float
        """
        try:
            # For crypto fallback, use the historical data approach
            end = datetime.now()
            start = end - timedelta(minutes=5)  # Look back just 5 minutes to get freshest data
            
            # Try with different timeframes if needed
            timeframes = ["1Min", "5Min", "1Day"]
            
            for timeframe in timeframes:
                try:
                    bars = await self.get_historical_data(symbol, start, end, timeframe)
                    if not bars.empty:
                        # Get the latest bar's closing price
                        latest_price = float(bars['close'].iloc[-1])
                        logger.info(f"{symbol} prix (barre historique): ${latest_price:.4f}")
                        return latest_price
                except Exception as e:
                    continue
            
            # If all attempts fail
            logger.error(f"AlpacaProvider: Could not get any price data for {symbol}")
            return 0.0
        except Exception as e:
            logger.error(f"AlpacaProvider: Error in fallback price fetch: {str(e)}")
            return 0.0
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
