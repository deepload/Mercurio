"""
Long Call strategy for options trading.

This strategy involves buying call options on an underlying asset, typically when 
expecting significant upward price movement with limited risk.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from app.core.models.option import OptionContract, OptionPosition, OptionType
from app.strategies.options.base_options_strategy import BaseOptionsStrategy

logger = logging.getLogger(__name__)

class LongCallStrategy(BaseOptionsStrategy):
    """
    Implementation of the Long Call strategy.
    
    This strategy is suitable for bullish market views with the potential for 
    significant upside. It offers leverage with defined risk (limited to the premium paid).
    """
    
    def __init__(
        self,
        underlying_symbol: str,
        account_size: float,
        max_position_size: float = 0.05,
        min_implied_volatility: float = 0.15,
        max_implied_volatility: float = 0.60,
        max_days_to_expiry: int = 60,
        min_days_to_expiry: int = 30,
        target_delta: float = 0.70,
        delta_range: float = 0.10,
        profit_target_pct: float = 1.00,  # 100% return
        stop_loss_pct: float = 0.50,      # 50% loss
        roll_when_dte: int = 15,
        use_technical_filters: bool = True,
        **kwargs
    ):
        """
        Initialize the Long Call strategy.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            account_size: Account size in USD
            max_position_size: Maximum position size as % of account
            min_implied_volatility: Minimum implied volatility for buying calls
            max_implied_volatility: Maximum implied volatility for buying calls
            max_days_to_expiry: Maximum number of days until expiration
            min_days_to_expiry: Minimum number of days until expiration
            target_delta: Target delta for the call (positive)
            delta_range: Acceptable range around the target delta
            profit_target_pct: % return to target for taking profits
            stop_loss_pct: % loss to trigger stop loss
            roll_when_dte: Number of days remaining before expiration to roll the position
            use_technical_filters: Use technical filters for entry
        """
        # Initialize base strategy with proper parameters
        name = f"Long Call on {underlying_symbol}"
        description = f"Long Call strategy for {underlying_symbol} with delta {target_delta}"
        super().__init__(name=name, description=description)
        
        # Store strategy-specific parameters
        self.underlying_symbol = underlying_symbol
        self.account_size = account_size
        self.max_position_size = max_position_size
        
        self.min_implied_volatility = min_implied_volatility
        self.max_implied_volatility = max_implied_volatility
        self.max_days_to_expiry = max_days_to_expiry
        self.min_days_to_expiry = min_days_to_expiry
        self.target_delta = target_delta
        self.delta_range = delta_range
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.roll_when_dte = roll_when_dte
        self.use_technical_filters = use_technical_filters
        
        self.current_call: Optional[OptionContract] = None
        self.position_size: int = 0
        self.entry_premium: float = 0
        self.max_drawdown: float = 0
        
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if a new Long Call position should be opened.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if entry is recommended, False otherwise
        """
        if not await self._check_account_requirements():
            return False
            
        # Check if we already have an open position
        if self.current_call is not None:
            return False
            
        # Check technical conditions of the underlying
        if self.use_technical_filters and not self._check_technical_filters(market_data):
            logger.info(f"Technical filters not satisfied for {self.underlying_symbol}")
            return False
            
        # Get the last price of the underlying
        last_price = market_data['close'].iloc[-1] if not market_data.empty else None
        
        if not last_price:
            logger.warning(f"Unable to get last price for {self.underlying_symbol}")
            return False
            
        # Look for a long call opportunity
        try:
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                option_type=OptionType.CALL
            )
            
            if not option_chain:
                logger.warning(f"No options available for {self.underlying_symbol}")
                return False
                
            # Filter options according to our criteria
            filtered_options = self._filter_options(option_chain, last_price)
            
            if not filtered_options:
                logger.info(f"No options match our criteria for {self.underlying_symbol}")
                return False
                
            # Select the best option
            self.current_call = self._select_best_option(filtered_options)
            
            if not self.current_call:
                return False
                
            # Calculate position size based on risk management
            max_capital_at_risk = self.account_size * self.max_position_size
            premium_per_contract = self.current_call.ask * 100  # One contract = 100 shares
            
            self.position_size = max(1, int(max_capital_at_risk // premium_per_contract))
            
            # Limit position size 
            self.position_size = min(self.position_size, 10)  # Arbitrary cap at 10 contracts
            
            if self.position_size < 1:
                logger.info(f"Not enough capital to open a Long Call position on {self.underlying_symbol}")
                self.current_call = None
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error during Long Call entry evaluation: {str(e)}")
            return False
            
    async def should_exit(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if the Long Call position should be closed.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if exit is recommended, False otherwise
        """
        if not self.current_call:
            return False
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                logger.warning(f"Unable to get current data for option {self.current_call.symbol}")
                return False
                
            # Calculate days to expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.current_call.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            # Current option price
            current_price = option_data.get('bid', 0)
            if not current_price and option_data.get('last', 0) > 0:
                # Use last price if bid is not available
                current_price = option_data.get('last', 0)
                
            # Exit criteria
            
            # 1. Profit target reached
            profit_target = self.entry_premium * (1 + self.profit_target_pct)
            if current_price >= profit_target:
                logger.info(f"Long Call exit: profit target reached for {self.current_call.symbol}")
                return True
                
            # 2. Stop loss
            stop_loss = self.entry_premium * (1 - self.stop_loss_pct)
            if current_price <= stop_loss:
                logger.info(f"Long Call exit: stop loss reached for {self.current_call.symbol}")
                return True
                
            # 3. Close to expiration
            if days_to_expiry <= self.roll_when_dte:
                logger.info(f"Long Call exit: close to expiration ({days_to_expiry} days)")
                return True
                
            # 4. Technical reversal in the underlying
            if self._detect_bearish_reversal(market_data):
                logger.info(f"Long Call exit: bearish reversal detected in {self.underlying_symbol}")
                return True
                
            # Track maximum drawdown
            if current_price < self.entry_premium:
                drawdown_pct = (self.entry_premium - current_price) / self.entry_premium
                self.max_drawdown = max(self.max_drawdown, drawdown_pct)
                
            return False
            
        except Exception as e:
            logger.error(f"Error during Long Call exit evaluation: {str(e)}")
            return False
            
    async def execute_entry(self) -> Dict[str, Any]:
        """
        Executes the entry for the Long Call strategy.
        
        Returns:
            Dict[str, Any]: Result of the entry operation
        """
        if not self.current_call:
            return {"success": False, "error": "No option selected for Long Call entry"}
            
        try:
            # Buy calls
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_call.symbol,
                qty=self.position_size,
                side="buy",
                order_type="limit",
                limit_price=self.current_call.ask,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                self.entry_premium = self.current_call.ask
                self.max_drawdown = 0
                
                # Log transaction details
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"Long Call entry executed: {self.position_size} contracts of {self.current_call.symbol} bought at {self.entry_premium}")
                
                # Calculate metrics
                total_cost = self.entry_premium * 100 * self.position_size
                
                entry_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Long Call",
                    "action": "BUY_CALL",
                    "symbol": self.current_call.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "premium": self.entry_premium,
                    "strike": self.current_call.strike,
                    "expiry": self.current_call.expiry_date,
                    "total_cost": total_cost,
                    "max_risk": total_cost,
                    "trade_date": datetime.now().isoformat()
                }
                
                self._log_trade("ENTRY", entry_result)
                return entry_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Long Call entry failed: {error_msg}")
                self.current_call = None
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Long Call entry execution: {str(e)}")
            self.current_call = None
            return {"success": False, "error": str(e)}
            
    async def execute_exit(self) -> Dict[str, Any]:
        """
        Executes the exit from the Long Call position.
        
        Returns:
            Dict[str, Any]: Result of the exit operation
        """
        if not self.current_call:
            return {"success": False, "error": "No open Long Call position to close"}
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {"success": False, "error": f"Unable to get data for {self.current_call.symbol}"}
                
            # Sell calls to close the position
            bid_price = option_data.get("bid", 0)
            if bid_price <= 0:
                logger.warning(f"Invalid bid price ({bid_price}) for {self.current_call.symbol}, using last price")
                bid_price = option_data.get("last", 0.05)  # Minimal price
                
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_call.symbol,
                qty=self.position_size,
                side="sell",  # Sell the calls
                order_type="limit",
                limit_price=bid_price,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                exit_premium = bid_price
                profit_loss = (exit_premium - self.entry_premium) * 100 * self.position_size
                
                # Log transaction details
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"Long Call exit executed: {self.position_size} contracts of {self.current_call.symbol} sold at {exit_premium}")
                
                exit_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Long Call",
                    "action": "SELL_CALL",
                    "symbol": self.current_call.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "entry_premium": self.entry_premium,
                    "exit_premium": exit_premium,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": (exit_premium - self.entry_premium) / self.entry_premium * 100,
                    "max_drawdown_pct": self.max_drawdown * 100,
                    "strike": self.current_call.strike,
                    "trade_date": datetime.now().isoformat()
                }
                
                # Reset position variables
                self.current_call = None
                self.position_size = 0
                self.entry_premium = 0
                self.max_drawdown = 0
                
                self._log_trade("EXIT", exit_result)
                return exit_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Long Call exit failed: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Long Call exit execution: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_position_info(self) -> Dict[str, Any]:
        """
        Provides information about the current Long Call position.
        
        Returns:
            Dict: Details of the current position
        """
        if not self.current_call:
            return {"has_position": False}
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {
                    "has_position": True,
                    "error": f"Unable to get current data for {self.current_call.symbol}"
                }
                
            # Current option price
            current_price = option_data.get('bid', 0)
            if not current_price and option_data.get('last', 0) > 0:
                # Use last price if bid is not available
                current_price = option_data.get('last', 0)
                
            # Calculate position metrics
            unrealized_pl = (current_price - self.entry_premium) * 100 * self.position_size
            unrealized_pl_pct = ((current_price - self.entry_premium) / self.entry_premium) * 100 if self.entry_premium > 0 else 0
            
            # Days to expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.current_call.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            return {
                "has_position": True,
                "strategy": "Long Call",
                "symbol": self.current_call.symbol,
                "underlying": self.underlying_symbol,
                "quantity": self.position_size,
                "strike": self.current_call.strike,
                "expiry": self.current_call.expiry_date,
                "days_to_expiry": days_to_expiry,
                "entry_premium": self.entry_premium,
                "current_premium": current_price,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
                "max_drawdown_pct": self.max_drawdown * 100,
                "total_investment": self.entry_premium * 100 * self.position_size
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Long Call position information: {str(e)}")
            return {
                "has_position": True,
                "error": str(e),
                "symbol": self.current_call.symbol if self.current_call else "unknown"
            }
            
    def _filter_options(self, options: List[OptionContract], stock_price: float) -> List[OptionContract]:
        """
        Filters options according to strategy criteria.
        
        Args:
            options: List of available option contracts
            stock_price: Current price of the underlying asset
            
        Returns:
            List[OptionContract]: Filtered options
        """
        filtered = []
        
        for option in options:
            # Verify it's a call
            if option.option_type != OptionType.CALL:
                continue
                
            # Calculate days to expiration
            today = datetime.now().date()
            try:
                expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
                days_to_expiry = (expiry - today).days
            except:
                # If date format is incorrect, skip to next option
                continue
                
            # Basic filters
            if (
                days_to_expiry < self.min_days_to_expiry or
                days_to_expiry > self.max_days_to_expiry or
                option.implied_volatility < self.min_implied_volatility or
                option.implied_volatility > self.max_implied_volatility or
                option.ask <= 0.05  # Minimal price to avoid illiquid options
            ):
                continue
                
            # Delta filter
            if not (self.target_delta - self.delta_range <= option.delta <= self.target_delta + self.delta_range):
                continue
                
            # Check if option has sufficient liquidity
            if option.volume < 10 or option.open_interest < 100:
                continue
                
            # Check bid-ask spread (not to exceed 15%)
            if option.ask > 0 and option.bid > 0:
                spread_pct = (option.ask - option.bid) / option.bid
                if spread_pct > 0.15:
                    continue
                    
            # Add option to filtered results
            filtered.append(option)
            
        return filtered
        
    def _select_best_option(self, filtered_options: List[OptionContract]) -> Optional[OptionContract]:
        """
        Selects the best option from the filtered options.
        
        Args:
            filtered_options: List of filtered options
            
        Returns:
            Optional[OptionContract]: Best option or None
        """
        if not filtered_options:
            return None
            
        # Sort by proximity to target delta
        sorted_by_delta = sorted(filtered_options, key=lambda x: abs(x.delta - self.target_delta))
        
        # Take the top 3 options by delta
        top_delta_options = sorted_by_delta[:3] if len(sorted_by_delta) >= 3 else sorted_by_delta
        
        # Among these options, take the one with the best price/value ratio
        best_option = None
        best_score = 0
        
        for option in top_delta_options:
            # Calculate days until expiration
            today = datetime.now().date()
            expiry = datetime.strptime(option.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = max(1, (expiry - today).days)  # Avoid division by zero
            
            # Metrics for scoring
            theta_cost = abs(option.theta) * days_to_expiry  # Total theta decay until expiration
            vega_value = option.vega * (option.implied_volatility / 0.10)  # Vega value relative to IV
            
            # Risk-adjusted score
            score = option.delta / (option.ask * (1 + theta_cost/option.ask))
            
            if score > best_score:
                best_score = score
                best_option = option
                
        return best_option
        
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
            # Calculate moving averages
            market_data['sma20'] = market_data['close'].rolling(window=20).mean()
            market_data['sma50'] = market_data['close'].rolling(window=50).mean()
            market_data['sma200'] = market_data['close'].rolling(window=200).mean()
            
            # Last values
            last_close = market_data['close'].iloc[-1]
            last_sma20 = market_data['sma20'].iloc[-1]
            last_sma50 = market_data['sma50'].iloc[-1]
            last_sma200 = market_data['sma200'].iloc[-1]
            
            # Calculate RSI
            delta = market_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            last_rsi = rsi.iloc[-1]
            
            # Check overall trend (buy calls in a bullish trend)
            uptrend = last_close > last_sma50 > last_sma200
            
            # Check RSI (not overbought)
            rsi_ok = last_rsi <= 70  # Avoid overbought markets
            
            # Check recent momentum
            recent_candles = market_data.iloc[-5:]
            bullish_candles = sum(1 for i in range(len(recent_candles)) if recent_candles['close'].iloc[i] > recent_candles['open'].iloc[i])
            momentum_bullish = bullish_candles >= 3  # At least 3 out of 5 recent candles are bullish
            
            # For Long Call, we want to buy calls when the market is bullish
            # and not overbought
            return uptrend and rsi_ok and momentum_bullish
            
        except Exception as e:
            logger.error(f"Error checking technical filters: {str(e)}")
            return False
            
    def _detect_bearish_reversal(self, market_data: pd.DataFrame) -> bool:
        """
        Detects bearish reversal patterns in the underlying asset.
        
        Args:
            market_data: Market data
            
        Returns:
            bool: True if bearish reversal detected, False otherwise
        """
        if market_data.empty or len(market_data) < 10:
            return False
            
        try:
            # Calculate RSI
            delta = market_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Check for overbought RSI that is turning down
            overbought_to_down = rsi.iloc[-2] > 70 and rsi.iloc[-1] < rsi.iloc[-2]
            
            # Check for price lower than recent moving average
            market_data['sma10'] = market_data['close'].rolling(window=10).mean()
            crossing_down = market_data['close'].iloc[-2] > market_data['sma10'].iloc[-2] and market_data['close'].iloc[-1] < market_data['sma10'].iloc[-1]
            
            # Check for bearish engulfing pattern
            last_candle = market_data.iloc[-1]
            prev_candle = market_data.iloc[-2]
            bearish_engulfing = (
                last_candle['open'] > prev_candle['close'] and
                last_candle['close'] < prev_candle['open'] and
                last_candle['close'] < last_candle['open'] and  # Current candle is red
                prev_candle['close'] > prev_candle['open']  # Previous candle is green
            )
            
            # Detect at least one of these bearish signals
            return overbought_to_down or crossing_down or bearish_engulfing
            
        except Exception as e:
            logger.error(f"Error detecting bearish reversal: {str(e)}")
            return False
            
    async def _get_current_option_data(self) -> Dict[str, Any]:
        """
        Retrieves current data for the current option.
        
        Returns:
            Dict: Current option data
        """
        if not self.current_call:
            return {}
            
        try:
            # Try to retrieve current price information
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                expiration_date=self.current_call.expiry_date,
                option_type=OptionType.CALL
            )
            
            if not option_chain:
                return {}
                
            # Find our specific option
            for option in option_chain:
                if (option.symbol == self.current_call.symbol or 
                    (option.strike == self.current_call.strike and 
                     option.expiry_date == self.current_call.expiry_date and
                     option.option_type == OptionType.CALL)):
                    return {
                        "bid": option.bid,
                        "ask": option.ask,
                        "last": option.last,
                        "delta": option.delta,
                        "implied_volatility": option.implied_volatility,
                        "volume": option.volume,
                        "open_interest": option.open_interest
                    }
                    
            return {}
            
        except Exception as e:
            logger.error(f"Error retrieving current option data: {str(e)}")
            return {}
            
    async def _check_account_requirements(self) -> bool:
        """
        Checks if the account meets the requirements for this strategy.
        
        Returns:
            bool: True if conditions are met, False otherwise
        """
        import inspect
        try:
            # Fix for broker attribute missing - use broker_adapter if available
            if not hasattr(self, 'broker'):
                if hasattr(self, 'broker_adapter'):
                    self.broker = self.broker_adapter
                    logger.info("Fixed broker attribute using broker_adapter")
                elif hasattr(self, 'trading_service') and hasattr(self.trading_service, 'broker'):
                    self.broker = self.trading_service.broker
                    logger.info("Fixed broker attribute using trading_service.broker")
                else:
                    logger.warning("No broker available for strategy")
                    return False

            # Support both sync and async get_account
            account = self.broker.get_account()
            if inspect.isawaitable(account):
                account = await account
            if not account:
                logger.warning("Unable to retrieve account information")
                return False

            # Support both Account object and dict
            buying_power = getattr(account, "buying_power", None)
            if buying_power is None and isinstance(account, dict):
                buying_power = account.get("buying_power")
            if buying_power is None:
                logger.warning("Unable to retrieve buying_power from account")
                return False

            buying_power = float(buying_power)

            if buying_power < 2000:  # Minimum required for Long Call
                logger.warning(f"Insufficient buying power for Long Call strategy: {buying_power}")
                return False

            # Check if options trading is enabled
            if hasattr(self.broker, 'enable_options') and not self.broker.enable_options:
                logger.warning("Options trading is not enabled")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking account requirements: {str(e)}")
            return False
