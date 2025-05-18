"""
Trading Service

Provides functionality for executing trades and managing portfolios
using Alpaca as the broker.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

# For Alpaca API
import alpaca_trade_api as tradeapi

# Import environment loader to ensure .env variables are loaded
from app.utils import env_loader

from app.db.models import TradeAction
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class TradingService:
    """
    Service for executing trades and managing portfolios.
    
    Supports:
    - Paper trading with Alpaca
    - Live trading with Alpaca
    - Order tracking and position management
    """
    
    def __init__(self, is_paper: bool = True):
        """
        Initialize the trading service with Alpaca client.
        
        Args:
            is_paper: Whether to use Alpaca paper trading API
        """
        self.alpaca_key = os.getenv("ALPACA_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET")
        
        # Determine base URL based on paper trading mode
        if is_paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.alpaca_client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                self.alpaca_client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=self.base_url
                )
                logger.info(f"Alpaca client initialized successfully (paper: {is_paper})")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
        
        # Initialize market data service for price information
        self.market_data = MarketDataService()
        
        # Options trading support flag - default to True for options strategies
        self.enable_options = True
        
        # Add broker attribute for compatibility with options strategies
        self.broker = self
    
    async def check_market_status(self) -> Dict[str, Any]:
        """
        Check if the market is currently open.
        
        Returns:
            Dictionary with market status information
        """
        if not self.alpaca_client:
            return {"is_open": False, "error": "Alpaca client not initialized"}
        
        try:
            clock = self.alpaca_client.get_clock()
            market_status = {
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat(),
                "next_close": clock.next_close.isoformat(),
                "timestamp": clock.timestamp.isoformat()
            }
            return market_status
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return {"is_open": False, "error": str(e)}
    
    def get_account(self):
        """
        Get the raw account object for options strategies.
        This is a non-async version for compatibility with options strategies.
        
        Returns:
            Alpaca account object
        """
        if not self.alpaca_client:
            raise Exception("Alpaca client not initialized")
        
        try:
            return self.alpaca_client.get_account()
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            raise
            
    def _get_fallback_account_info(self) -> Dict[str, Any]:
        """
        Provides fallback demo account information when real account data cannot be accessed.
        Used for testing and development when API keys are not authorized for account operations.
        
        Returns:
            Dictionary with mock account information
        """
        logger.warning("Using DEMO account information - API key may not have proper trading permissions")
        
        # Create realistic demo account data
        return {
            "id": "demo-account-12345",
            "cash": 50000.0,  # $50,000 cash
            "equity": 65000.0,  # $65,000 total equity
            "buying_power": 100000.0,  # $100,000 buying power (2x leverage)
            "portfolio_value": 65000.0,
            "day_trade_count": 0,
            "pattern_day_trader": False,
            "status": "ACTIVE",
            "trading_blocked": False,
            "account_blocked": False,
            "is_demo": True  # Add flag to indicate this is demo data
        }
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.
        
        Returns:
            Dictionary with account information
        """
        if not self.alpaca_client:
            return self._get_fallback_account_info()
        
        try:
            account = self.alpaca_client.get_account()
            account_info = {
                "id": account.id,
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": account.day_trade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "status": account.status,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked
            }
            return account_info
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            logger.info("Using fallback demo account information for testing")
            return self._get_fallback_account_info()
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.alpaca_client:
            return [{"error": "Alpaca client not initialized"}]
        
        try:
            positions = self.alpaca_client.list_positions()
            positions_list = []
            
            for position in positions:
                positions_list.append({
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "market_value": float(position.market_value),
                    "avg_entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "side": position.side
                })
            
            return positions_list
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return [{"error": str(e)}]
    
    async def execute_trade(
        self,
        symbol: str,
        action: TradeAction,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        strategy_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute a trade order.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            action: TradeAction (BUY, SELL, HOLD)
            quantity: Quantity to trade
            order_type: Order type (market, limit, etc.)
            limit_price: Price for limit orders
            time_in_force: Time in force (day, gtc, etc.)
            strategy_name: Name of the strategy making the trade
            
        Returns:
            Dictionary with order information
        """
        if action == TradeAction.HOLD:
            return {"status": "skipped", "message": "HOLD action, no trade executed"}
        
        if not self.alpaca_client:
            return {"status": "error", "message": "Alpaca client not initialized"}
        
        try:
            # Convert TradeAction to Alpaca side
            side = "buy" if action == TradeAction.BUY else "sell"
            
            # Handle fractional quantities
            if quantity < 1 and not isinstance(quantity, int):
                # Alpaca supports fractional shares for market orders
                if order_type != "market":
                    order_type = "market"
                    logger.warning("Changing order type to market for fractional shares")
                
                # Use notional API for fractional shares
                try:
                    latest_price = await self.market_data.get_latest_price(symbol)
                    notional_amount = quantity * latest_price
                    
                    order = self.alpaca_client.submit_order(
                        symbol=symbol,
                        notional=notional_amount,
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force
                    )
                except Exception as notional_error:
                    logger.error(f"Error executing notional order: {notional_error}")
                    # Fall back to standard order API
                    order = self.alpaca_client.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price if order_type == "limit" else None
                    )
            else:
                # Standard order API
                order = self.alpaca_client.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price if order_type == "limit" else None
                )
            
            # Format order information
            order_info = {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "order_type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "strategy": strategy_name
            }
            
            logger.info(f"Order executed: {order_info}")
            return {"status": "success", "order": order_info}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_option_chain(self, symbol: str, option_type: str, expiration_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the option chain for a specific symbol, expiration date, and option type.
        
        Args:
            symbol: The underlying asset symbol
            expiration_date: The expiration date in YYYY-MM-DD format
            option_type: The option type ("call" or "put")
            
        Returns:
            List of option contracts
        """
        try:
            # If expiration_date is not provided, generate a default one
            if expiration_date is None:
                # Use a default expiration 30 days from now
                today = datetime.now().date()
                # Find the next Friday that is at least 30 days out
                expiry_date = today + timedelta(days=30)
                while expiry_date.weekday() != 4:  # Friday is weekday 4
                    expiry_date += timedelta(days=1)
                expiration_date = expiry_date.strftime("%Y-%m-%d")
                
            logger.info(f"Getting {option_type} option chain for {symbol} with expiration {expiration_date}")
            
            # For crypto options, we need to implement a simulated chain since
            # Alpaca does not currently support crypto options
            current_price = await self.market_data.get_latest_price(symbol)
            if not current_price or current_price <= 0:
                logger.error(f"Unable to get current price for {symbol}")
                return []
            
            # Generate a synthetic options chain with strikes around current price
            strike_multipliers = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
            options = []
            
            # Cryptocurrency prices can vary widely, so adjust strikes accordingly
            if current_price >= 1000:  # For high-value crypto like BTC
                strike_step = 500
            elif current_price >= 100:  # For medium-value crypto like ETH
                strike_step = 50
            else:  # For lower-value crypto
                strike_step = 5
            
            # Create synthetic option contracts
            for i, mult in enumerate(strike_multipliers):
                strike = round(current_price * mult / strike_step) * strike_step
                
                # Calculate synthetic Greeks and prices based on strike and current price
                if option_type.lower() == "call":
                    delta = max(0.01, min(0.99, 1 - (strike / current_price)))
                    option_value = max(0.01, current_price - strike)
                else:  # put
                    delta = max(-0.99, min(-0.01, -(strike / current_price)))
                    option_value = max(0.01, strike - current_price)
                
                # Add some spread to simulate bid/ask
                bid = max(0.01, option_value * 0.95)
                ask = option_value * 1.05
                
                # Create a simple synthetic IV
                iv = 0.3 + (abs(1 - (strike / current_price)) * 0.2)  # Higher IV for further OTM options
                
                # Create contract object
                contract = {
                    "symbol": f"{symbol.replace('/', '')}_{expiration_date}_{option_type[0].upper()}_{strike}",
                    "underlying": symbol,
                    "strike": strike,
                    "expiry_date": expiration_date,
                    "option_type": option_type.lower(),
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "last": round((bid + ask) / 2, 2),
                    "delta": delta,
                    "gamma": 0.01,
                    "theta": -0.01,
                    "vega": 0.05,
                    "implied_volatility": iv,
                    "volume": 100,
                    "open_interest": 500
                }
                
                options.append(contract)
            
            return options
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return []
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Dictionary with order status
        """
        if not self.alpaca_client:
            return {"status": "error", "message": "Alpaca client not initialized"}
        
        try:
            order = self.alpaca_client.get_order(order_id)
            
            order_status = {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "status": order.status,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "filled_qty": order.filled_qty,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": order.limit_price,
                "filled_avg_price": order.filled_avg_price,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
                "failed_at": order.failed_at.isoformat() if order.failed_at else None,
                "asset_class": order.asset_class,
                "asset_id": order.asset_id
            }
            
            return order_status
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def calculate_order_quantity(
        self,
        symbol: str,
        action: TradeAction,
        capital_pct: float = 0.1
    ) -> float:
        """
        Calculate the quantity to order based on available capital.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            action: TradeAction (BUY, SELL)
            capital_pct: Percentage of available capital to use (0.0 to 1.0)
            
        Returns:
            Order quantity
        """
        if action == TradeAction.HOLD:
            return 0.0
        
        if not self.alpaca_client:
            logger.error("Alpaca client not initialized")
            return 0.0
        
        try:
            # Get account information
            account = self.alpaca_client.get_account()
            available_capital = float(account.cash) if action == TradeAction.BUY else 0.0
            
            # If selling, check current position
            if action == TradeAction.SELL:
                try:
                    position = self.alpaca_client.get_position(symbol)
                    return float(position.qty)
                except Exception as e:
                    logger.warning(f"No position found for {symbol}: {e}")
                    return 0.0
            
            # For buying, calculate based on latest price and available capital
            latest_price = await self.market_data.get_latest_price(symbol)
            
            # Calculate quantity based on capital percentage
            capital_to_use = available_capital * capital_pct
            quantity = capital_to_use / latest_price
            
            # Round to 6 decimal places for fractional shares
            quantity = round(quantity, 6)
            
            logger.info(f"Calculated order quantity for {symbol}: {quantity}")
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating order quantity: {e}")
            return 0.0
