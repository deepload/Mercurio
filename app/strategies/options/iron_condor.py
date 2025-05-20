"""
Iron Condor strategy for options trading.

This strategy involves simultaneously selling an out-of-the-money put spread and 
an out-of-the-money call spread with the same expiration date, creating a range
where the strategy profits if the underlying asset stays within this range.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from app.core.models.option import OptionContract, OptionPosition, OptionType
from app.strategies.options.base_options_strategy import BaseOptionsStrategy

logger = logging.getLogger(__name__)

class IronCondorStrategy(BaseOptionsStrategy):
    @property
    def broker(self):
        return getattr(self, 'broker_adapter', None)

    """
    Implementation of the Iron Condor strategy.
    
    The Iron Condor is a market-neutral strategy that profits when the underlying
    asset remains within a specific price range, making it ideal for low-volatility
    environments. It consists of four legs:
    
    1. Sell an OTM put (short put)
    2. Buy a further OTM put (long put)
    3. Sell an OTM call (short call)
    4. Buy a further OTM call (long call)
    
    All options have the same expiration date, creating a position with limited risk
    and limited reward.
    """
    
    def __init__(
        self,
        underlying_symbol: str,
        account_size: float,
        max_position_size: float = 0.10,
        max_days_to_expiry: int = 45,
        min_days_to_expiry: int = 25,
        short_put_delta: float = -0.30,
        short_call_delta: float = 0.30,
        wing_width: int = 1,  # Width in strike prices
        profit_target_pct: float = 0.50,  # 50% of max profit
        stop_loss_pct: float = 1.50,     # 150% of max profit (i.e., 50% more than max loss)
        roll_when_dte: int = 10,
        use_technical_filters: bool = True,
        **kwargs
    ):
        """
        Initialize the Iron Condor strategy.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            account_size: Account size in USD
            max_position_size: Maximum position size as % of account
            max_days_to_expiry: Maximum number of days until expiration
            min_days_to_expiry: Minimum number of days until expiration
            short_put_delta: Target delta for short put
            short_call_delta: Target delta for short call
            wing_width: Width between short and long options in strike prices
            profit_target_pct: % of max profit to close position
            stop_loss_pct: % of max loss to trigger stop loss
            roll_when_dte: Number of days remaining before expiration to roll the position
            use_technical_filters: Use technical filters for entry
        """
        # Initialize base strategy with proper parameters
        name = f"Iron Condor on {underlying_symbol}"
        description = f"Iron Condor strategy for {underlying_symbol} with short put delta {short_put_delta} and short call delta {short_call_delta}"
        super().__init__(name=name, description=description)
        
        # Store strategy-specific parameters
        self.underlying_symbol = underlying_symbol
        self.account_size = account_size
        self.max_position_size = max_position_size
        
        self.max_days_to_expiry = max_days_to_expiry
        self.min_days_to_expiry = min_days_to_expiry
        self.short_put_delta = short_put_delta
        self.short_call_delta = short_call_delta
        self.wing_width = wing_width
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.roll_when_dte = roll_when_dte
        self.use_technical_filters = use_technical_filters
        
        # Position variables
        self.short_put: Optional[OptionContract] = None
        self.long_put: Optional[OptionContract] = None
        self.short_call: Optional[OptionContract] = None
        self.long_call: Optional[OptionContract] = None
        self.position_size: int = 0
        self.max_profit: float = 0
        self.max_loss: float = 0
        self.net_credit: float = 0
        
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if a new Iron Condor position should be opened.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if entry is recommended, False otherwise
        """
        if not await self._check_account_requirements():
            return False
            
        # Check if we already have an open position
        if self.short_put is not None or self.short_call is not None:
            return False
            
        # Check technical conditions if enabled
        if self.use_technical_filters and not self._check_technical_filters(market_data):
            logger.info(f"Technical filters not satisfied for {self.underlying_symbol}")
            return False
            
        # Get the last price of the underlying
        last_price = market_data['close'].iloc[-1] if not market_data.empty else None
        
        if not last_price:
            logger.warning(f"Unable to get last price for {self.underlying_symbol}")
            return False
            
        try:
            # Find suitable options for each leg
            success, legs_info = await self._find_iron_condor_legs(last_price)
            
            if not success:
                return False
                
            # Set the chosen options
            self.short_put = legs_info['short_put']
            self.long_put = legs_info['long_put']
            self.short_call = legs_info['short_call']
            self.long_call = legs_info['long_call']
            
            # Calculate position size based on risk management
            put_spread_width = (self.short_put.strike - self.long_put.strike) * 100
            call_spread_width = (self.long_call.strike - self.short_call.strike) * 100
            max_risk_per_contract = max(put_spread_width, call_spread_width) - self.net_credit * 100
            
            max_capital_at_risk = self.account_size * self.max_position_size
            self.position_size = max(1, int(max_capital_at_risk // max_risk_per_contract))
            
            # Limit position size
            self.position_size = min(self.position_size, 5)  # Arbitrary cap
            
            if self.position_size < 1:
                logger.info(f"Not enough capital to open an Iron Condor position on {self.underlying_symbol}")
                self._reset_position_variables()
                return False
                
            # Calculate max profit and loss for the position
            self.max_profit = self.net_credit * 100 * self.position_size
            self.max_loss = (max(put_spread_width, call_spread_width) - self.net_credit * 100) * self.position_size
            
            return True
            
        except Exception as e:
            logger.error(f"Error during Iron Condor entry evaluation: {str(e)}")
            self._reset_position_variables()
            return False
            
    async def should_exit(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if the Iron Condor position should be closed.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if exit is recommended, False otherwise
        """
        if not self._has_open_position():
            return False
            
        try:
            # Get current price info
            current_prices = await self._get_current_prices()
            if not current_prices:
                logger.warning("Unable to get current option prices")
                return False
                
            # Calculate current position value
            current_value = (
                -current_prices.get('short_put', 0) * 100 +
                current_prices.get('long_put', 0) * 100 +
                -current_prices.get('short_call', 0) * 100 +
                current_prices.get('long_call', 0) * 100
            ) * self.position_size
            
            # Calculate days to expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.short_put.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            # Exit criteria
            
            # 1. Profit target
            profit_target = self.max_profit * self.profit_target_pct
            current_profit = self.net_credit * 100 * self.position_size - current_value
            
            if current_profit >= profit_target:
                logger.info(f"Iron Condor exit: profit target reached ({current_profit:.2f} >= {profit_target:.2f})")
                return True
                
            # 2. Stop loss
            max_loss_threshold = self.max_loss * self.stop_loss_pct
            current_loss = current_value - self.net_credit * 100 * self.position_size
            
            if current_loss >= max_loss_threshold:
                logger.info(f"Iron Condor exit: stop loss triggered ({current_loss:.2f} >= {max_loss_threshold:.2f})")
                return True
                
            # 3. Close to expiration
            if days_to_expiry <= self.roll_when_dte:
                logger.info(f"Iron Condor exit: close to expiration ({days_to_expiry} days)")
                return True
                
            # 4. Underlying price approaching short strikes
            last_price = market_data['close'].iloc[-1] if not market_data.empty else None
            if last_price:
                distance_to_put = (last_price - self.short_put.strike) / last_price
                distance_to_call = (self.short_call.strike - last_price) / last_price
                
                if distance_to_put < 0.02 or distance_to_call < 0.02:
                    logger.info(f"Iron Condor exit: underlying price approaching short strikes")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error during Iron Condor exit evaluation: {str(e)}")
            return False
            
    async def execute_entry(self) -> Dict[str, Any]:
        """
        Executes the entry for the Iron Condor strategy.
        
        Returns:
            Dict[str, Any]: Result of the entry operation
        """
        if not self._has_open_position():
            return {"success": False, "error": "No Iron Condor position defined"}
            
        try:
            # Create multi-leg options strategy order
            legs = [
                {"symbol": self.short_put.symbol, "side": "sell", "quantity": self.position_size},
                {"symbol": self.long_put.symbol, "side": "buy", "quantity": self.position_size},
                {"symbol": self.short_call.symbol, "side": "sell", "quantity": self.position_size},
                {"symbol": self.long_call.symbol, "side": "buy", "quantity": self.position_size}
            ]
            
            order_result = await self.broker.place_option_strategy(
                strategy_type="iron_condor",
                underlying=self.underlying_symbol,
                legs=legs,
                qty=self.position_size
            )
            
            if order_result.get("success", False):
                logger.info(f"Iron Condor entry executed: {self.position_size} contracts on {self.underlying_symbol}")
                
                # Calculate position metrics
                entry_result = {
                    "success": True,
                    "order_id": order_result.get("order_id", "unknown"),
                    "strategy": "Iron Condor",
                    "action": "ENTRY",
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "short_put_strike": self.short_put.strike,
                    "long_put_strike": self.long_put.strike,
                    "short_call_strike": self.short_call.strike,
                    "long_call_strike": self.long_call.strike,
                    "expiry": self.short_put.expiry_date,
                    "net_credit": self.net_credit,
                    "max_profit": self.max_profit,
                    "max_loss": self.max_loss,
                    "trade_date": datetime.now().isoformat()
                }
                
                self._log_trade("ENTRY", entry_result)
                return entry_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Iron Condor entry failed: {error_msg}")
                self._reset_position_variables()
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Iron Condor entry execution: {str(e)}")
            self._reset_position_variables()
            return {"success": False, "error": str(e)}
            
    async def execute_exit(self) -> Dict[str, Any]:
        """
        Executes the exit from the Iron Condor position.
        
        Returns:
            Dict[str, Any]: Result of the exit operation
        """
        if not self._has_open_position():
            return {"success": False, "error": "No open Iron Condor position to close"}
            
        try:
            # Create multi-leg options strategy order (with opposite sides from entry)
            legs = [
                {"symbol": self.short_put.symbol, "side": "buy", "quantity": self.position_size},
                {"symbol": self.long_put.symbol, "side": "sell", "quantity": self.position_size},
                {"symbol": self.short_call.symbol, "side": "buy", "quantity": self.position_size},
                {"symbol": self.long_call.symbol, "side": "sell", "quantity": self.position_size}
            ]
            
            order_result = await self.broker.place_option_strategy(
                strategy_type="iron_condor",
                underlying=self.underlying_symbol,
                legs=legs,
                qty=self.position_size
            )
            
            if order_result.get("success", False):
                # Get current prices
                current_prices = await self._get_current_prices()
                
                # Calculate P&L
                exit_debit = (
                    current_prices.get('short_put', 0) -
                    current_prices.get('long_put', 0) +
                    current_prices.get('short_call', 0) -
                    current_prices.get('long_call', 0)
                )
                
                profit_loss = (self.net_credit - exit_debit) * 100 * self.position_size
                profit_loss_pct = profit_loss / self.max_loss * 100 if self.max_loss > 0 else 0
                
                logger.info(f"Iron Condor exit executed: {self.position_size} contracts on {self.underlying_symbol}")
                
                exit_result = {
                    "success": True,
                    "order_id": order_result.get("order_id", "unknown"),
                    "strategy": "Iron Condor",
                    "action": "EXIT",
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "entry_credit": self.net_credit,
                    "exit_debit": exit_debit,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": profit_loss_pct,
                    "trade_date": datetime.now().isoformat()
                }
                
                # Reset position variables
                self._reset_position_variables()
                
                self._log_trade("EXIT", exit_result)
                return exit_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Iron Condor exit failed: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Iron Condor exit execution: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_position_info(self) -> Dict[str, Any]:
        """
        Provides information about the current Iron Condor position.
        
        Returns:
            Dict: Details of the current position
        """
        if not self._has_open_position():
            return {"has_position": False}
            
        try:
            # Get current prices
            current_prices = await self._get_current_prices()
            
            if not current_prices:
                return {
                    "has_position": True,
                    "error": "Unable to get current option prices"
                }
                
            # Calculate current position value
            current_value = (
                -current_prices.get('short_put', 0) * 100 +
                current_prices.get('long_put', 0) * 100 +
                -current_prices.get('short_call', 0) * 100 +
                current_prices.get('long_call', 0) * 100
            ) * self.position_size
            
            # Calculate P&L
            current_profit = self.net_credit * 100 * self.position_size - current_value
            profit_pct = current_profit / self.max_profit * 100 if self.max_profit > 0 else 0
            
            # Days to expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.short_put.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            return {
                "has_position": True,
                "strategy": "Iron Condor",
                "underlying": self.underlying_symbol,
                "quantity": self.position_size,
                "short_put_strike": self.short_put.strike,
                "long_put_strike": self.long_put.strike,
                "short_call_strike": self.short_call.strike,
                "long_call_strike": self.long_call.strike,
                "expiry": self.short_put.expiry_date,
                "days_to_expiry": days_to_expiry,
                "net_credit": self.net_credit,
                "current_value": current_value / (self.position_size * 100),
                "max_profit": self.max_profit,
                "max_loss": self.max_loss,
                "current_profit": current_profit,
                "profit_pct": profit_pct
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Iron Condor position information: {str(e)}")
            return {
                "has_position": True,
                "error": str(e)
            }
            
    def _has_open_position(self) -> bool:
        """
        Checks if there is an open Iron Condor position.
        
        Returns:
            bool: True if there is an open position
        """
        return (self.short_put is not None and 
                self.long_put is not None and 
                self.short_call is not None and 
                self.long_call is not None and
                self.position_size > 0)
                
    def _reset_position_variables(self) -> None:
        """
        Resets all position-related variables.
        """
        self.short_put = None
        self.long_put = None
        self.short_call = None
        self.long_call = None
        self.position_size = 0
        self.max_profit = 0
        self.max_loss = 0
        self.net_credit = 0
        
    async def _find_iron_condor_legs(self, stock_price: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Finds appropriate options for all legs of the Iron Condor.
        
        Args:
            stock_price: Current price of the underlying asset
            
        Returns:
            Tuple[bool, Dict]: Success flag and dictionary with option legs
        """
        try:
            # Get put options
            put_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                option_type=OptionType.PUT
            )
            
            if not put_chain:
                logger.warning(f"No put options available for {self.underlying_symbol}")
                return False, {}
                
            # Get call options
            call_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                option_type=OptionType.CALL
            )
            
            if not call_chain:
                logger.warning(f"No call options available for {self.underlying_symbol}")
                return False, {}
                
            # Filter by expiration date
            today = datetime.now().date()
            valid_puts = []
            valid_calls = []
            
            expiry_dates = set()
            
            for option in put_chain:
                try:
                    expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
                    days_to_expiry = (expiry - today).days
                    
                    if self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry:
                        valid_puts.append(option)
                        expiry_dates.add(option.expiry_date)
                except:
                    continue
                    
            for option in call_chain:
                try:
                    expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
                    days_to_expiry = (expiry - today).days
                    
                    if self.min_days_to_expiry <= days_to_expiry <= self.max_days_to_expiry:
                        valid_calls.append(option)
                except:
                    continue
                    
            if not expiry_dates:
                logger.warning(f"No valid expiration dates found for {self.underlying_symbol}")
                return False, {}
                
            # Choose the furthest expiration date within our range
            selected_expiry = max(expiry_dates)
            
            # Filter options by the selected expiration date
            expiry_puts = [p for p in valid_puts if p.expiry_date == selected_expiry]
            expiry_calls = [c for c in valid_calls if c.expiry_date == selected_expiry]
            
            if not expiry_puts or not expiry_calls:
                logger.warning(f"Not enough options for the selected expiry date")
                return False, {}
                
            # Sort puts by strike price (ascending)
            sorted_puts = sorted(expiry_puts, key=lambda x: x.strike)
            
            # Sort calls by strike price (ascending)
            sorted_calls = sorted(expiry_calls, key=lambda x: x.strike)
            
            # Find nearest OTM put (with delta close to target)
            short_put = None
            for put in sorted_puts:
                if put.strike < stock_price and abs(put.delta - self.short_put_delta) < 0.10:
                    short_put = put
                    break
                    
            if not short_put:
                logger.warning(f"No suitable short put found for {self.underlying_symbol}")
                return False, {}
                
            # Find nearest OTM call (with delta close to target)
            short_call = None
            for call in sorted_calls:
                if call.strike > stock_price and abs(call.delta - self.short_call_delta) < 0.10:
                    short_call = call
                    break
                    
            if not short_call:
                logger.warning(f"No suitable short call found for {self.underlying_symbol}")
                return False, {}
                
            # Find long put (wing_width strikes below short put)
            long_put_index = sorted_puts.index(short_put) - self.wing_width
            if long_put_index < 0:
                logger.warning(f"No suitable long put found for {self.underlying_symbol}")
                return False, {}
                
            long_put = sorted_puts[long_put_index]
            
            # Find long call (wing_width strikes above short call)
            long_call_index = sorted_calls.index(short_call) + self.wing_width
            if long_call_index >= len(sorted_calls):
                logger.warning(f"No suitable long call found for {self.underlying_symbol}")
                return False, {}
                
            long_call = sorted_calls[long_call_index]
            
            # Calculate net credit
            self.net_credit = (short_put.bid + short_call.bid - long_put.ask - long_call.ask)
            
            if self.net_credit <= 0:
                logger.warning(f"Iron Condor would not receive a credit: {self.net_credit}")
                return False, {}
                
            return True, {
                "short_put": short_put,
                "long_put": long_put,
                "short_call": short_call,
                "long_call": long_call
            }
            
        except Exception as e:
            logger.error(f"Error finding Iron Condor legs: {str(e)}")
            return False, {}
            
    async def _get_current_prices(self) -> Dict[str, float]:
        """
        Gets current prices for all options in the Iron Condor.
        
        Returns:
            Dict: Current prices for each leg
        """
        if not self._has_open_position():
            return {}
            
        result = {}
        
        try:
            # Get put options
            put_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                expiration_date=self.short_put.expiry_date,
                option_type=OptionType.PUT
            )
            
            # Get call options
            call_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                expiration_date=self.short_call.expiry_date,
                option_type=OptionType.CALL
            )
            
            # Find short put
            for put in put_chain:
                if put.strike == self.short_put.strike:
                    result['short_put'] = (put.bid + put.ask) / 2
                    break
                    
            # Find long put
            for put in put_chain:
                if put.strike == self.long_put.strike:
                    result['long_put'] = (put.bid + put.ask) / 2
                    break
                    
            # Find short call
            for call in call_chain:
                if call.strike == self.short_call.strike:
                    result['short_call'] = (call.bid + call.ask) / 2
                    break
                    
            # Find long call
            for call in call_chain:
                if call.strike == self.long_call.strike:
                    result['long_call'] = (call.bid + call.ask) / 2
                    break
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting current prices: {str(e)}")
            return {}
            
    def _check_technical_filters(self, market_data: pd.DataFrame) -> bool:
        """
        Checks technical filters for position entry.
        
        Args:
            market_data: Market data
            
        Returns:
            bool: True if filters are satisfied, False otherwise
        """
        if market_data.empty or len(market_data) < 20:
            return False
            
        try:
            # Iron Condor is a neutral strategy, so we want:
            # 1. Low volatility environment
            # 2. Price in a range-bound condition
            
            # Calculate historical volatility
            market_data['returns'] = market_data['close'].pct_change()
            historical_vol = market_data['returns'].rolling(window=20).std() * np.sqrt(252) * 100
            current_vol = historical_vol.iloc[-1]
            
            # Check if volatility is appropriate (not too high)
            vol_ok = current_vol < 30  # Arbitrary threshold
            
            # Check if price is range-bound (not trending)
            # Calculate moving averages
            market_data['sma20'] = market_data['close'].rolling(window=20).mean()
            market_data['sma50'] = market_data['close'].rolling(window=50).mean()
            
            # Distance between moving averages should be small for range-bound markets
            last_close = market_data['close'].iloc[-1]
            last_sma20 = market_data['sma20'].iloc[-1]
            last_sma50 = market_data['sma50'].iloc[-1]
            
            ma_diff_pct = abs(last_sma20 - last_sma50) / last_close * 100
            range_bound = ma_diff_pct < 5  # Moving averages within 5% of each other
            
            # Check if the price has been trading in a range recently
            last_20_days = market_data.iloc[-20:]
            high_low_range = (last_20_days['high'].max() - last_20_days['low'].min()) / last_close * 100
            narrow_range = high_low_range < 15  # 15% range over the last 20 days
            
            # An Iron Condor works best in a range-bound, low volatility environment
            return vol_ok and range_bound and narrow_range
            
        except Exception as e:
            logger.error(f"Error checking technical filters: {str(e)}")
            return False
            
    async def _check_account_requirements(self) -> bool:
        """
        Checks if the account meets the requirements for this strategy.
        
        Returns:
            bool: True if conditions are met, False otherwise
        """
        try:
            # Check account balance - no await since get_account() is not async
            account = self.broker.get_account()
            if not account:
                logger.warning("Unable to retrieve account information")
                return False
                
            # Access account properties directly without using get()
            buying_power = float(account.buying_power) if hasattr(account, 'buying_power') else 0
            
            # Iron Condor requires more buying power due to potential assignment risk
            if buying_power < 5000:  # Higher requirement due to multiple options
                logger.warning(f"Insufficient buying power for Iron Condor strategy: {buying_power}")
                return False
                
            # Check if options trading is enabled
            if hasattr(self.broker, 'enable_options') and not self.broker.enable_options:
                logger.warning("Options trading is not enabled")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking account requirements: {str(e)}")
            return False
