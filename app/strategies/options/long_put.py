"""
Long Put strategy for options trading.

This strategy involves buying put options on an underlying asset, typically when 
expecting significant downward price movement with limited risk.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from app.core.models.option import OptionContract, OptionPosition, OptionType
from app.strategies.options.base_options_strategy import BaseOptionsStrategy

logger = logging.getLogger(__name__)

class LongPutStrategy(BaseOptionsStrategy):
    @property
    def broker(self):
        return self.broker_adapter

    """
    Implementation of the Long Put strategy.
    
    This strategy is suitable for bearish market views with the potential for 
    significant downside. It offers leverage with defined risk (limited to the premium paid).
    Can also be used as a hedge against existing long positions.
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
        target_delta: float = -0.70,  # Negative because it's a put
        delta_range: float = 0.10,
        profit_target_pct: float = 1.00,  # 100% return
        stop_loss_pct: float = 0.50,      # 50% loss
        roll_when_dte: int = 15,
        use_technical_filters: bool = True,
        **kwargs
    ):
        """
        Initialize the Long Put strategy.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            account_size: Account size in USD
            max_position_size: Maximum position size as % of account
            min_implied_volatility: Minimum implied volatility for buying puts
            max_implied_volatility: Maximum implied volatility for buying puts
            max_days_to_expiry: Maximum number of days until expiration
            min_days_to_expiry: Minimum number of days until expiration
            target_delta: Target delta for the put (negative)
            delta_range: Acceptable range around the target delta
            profit_target_pct: % return to target for taking profits
            stop_loss_pct: % loss to trigger stop loss
            roll_when_dte: Number of days remaining before expiration to roll the position
            use_technical_filters: Use technical filters for entry
        """
        # Initialize base strategy with proper parameters
        name = f"Long Put on {underlying_symbol}"
        description = f"Long Put strategy for {underlying_symbol} with delta {target_delta}"
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
        
        self.current_put: Optional[OptionContract] = None
        self.position_size: int = 0
        self.entry_premium: float = 0
        self.max_drawdown: float = 0
        # Ensure broker_adapter is accessible as broker for compatibility
        # (Handled by @property above)
        
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if a new Long Put position should be opened.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if entry is recommended, False otherwise
        """
        if not await self._check_account_requirements():
            return False
            
        # Check if we already have an open position
        if self.current_put is not None:
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
            
        # Look for a long put opportunity
        try:
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                option_type=OptionType.PUT
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
            self.current_put = self._select_best_option(filtered_options)
            
            if not self.current_put:
                return False
                
            # Calculate position size based on risk management
            max_capital_at_risk = self.account_size * self.max_position_size
            premium_per_contract = self.current_put.ask * 100  # One contract = 100 shares
            
            self.position_size = max(1, int(max_capital_at_risk // premium_per_contract))
            
            # Limit position size 
            self.position_size = min(self.position_size, 10)  # Arbitrary cap at 10 contracts
            
            if self.position_size < 1:
                logger.info(f"Not enough capital to open a Long Put position on {self.underlying_symbol}")
                self.current_put = None
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error during Long Put entry evaluation: {str(e)}")
            return False
            
    async def should_exit(self, market_data: pd.DataFrame) -> bool:
        """
        Determines if the Long Put position should be closed.
        
        Args:
            market_data: Market data for the underlying asset
            
        Returns:
            bool: True if exit is recommended, False otherwise
        """
        if not self.current_put:
            return False
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                logger.warning(f"Unable to get current data for option {self.current_put.symbol}")
                return False
                
            # Calculate days to expiration
            today = datetime.now().date()
            expiry = datetime.strptime(self.current_put.expiry_date, "%Y-%m-%d").date()
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
                logger.info(f"Long Put exit: profit target reached for {self.current_put.symbol}")
                return True
                
            # 2. Stop loss
            stop_loss = self.entry_premium * (1 - self.stop_loss_pct)
            if current_price <= stop_loss:
                logger.info(f"Long Put exit: stop loss reached for {self.current_put.symbol}")
                return True
                
            # 3. Close to expiration
            if days_to_expiry <= self.roll_when_dte:
                logger.info(f"Long Put exit: close to expiration ({days_to_expiry} days)")
                return True
                
            # 4. Technical reversal in the underlying
            if self._detect_bullish_reversal(market_data):
                logger.info(f"Long Put exit: bullish reversal detected in {self.underlying_symbol}")
                return True
                
            # Track maximum drawdown
            if current_price < self.entry_premium:
                drawdown_pct = (self.entry_premium - current_price) / self.entry_premium
                self.max_drawdown = max(self.max_drawdown, drawdown_pct)
                
            return False
            
        except Exception as e:
            logger.error(f"Error during Long Put exit evaluation: {str(e)}")
            return False
            
    async def execute_entry(self) -> Dict[str, Any]:
        """
        Executes the entry for the Long Put strategy.
        
        Returns:
            Dict[str, Any]: Result of the entry operation
        """
        if not self.current_put:
            return {"success": False, "error": "No option selected for Long Put entry"}
            
        try:
            # Buy puts
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_put.symbol,
                qty=self.position_size,
                side="buy",
                order_type="limit",
                limit_price=self.current_put.ask,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                self.entry_premium = self.current_put.ask
                self.max_drawdown = 0
                
                # Log transaction details
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"Long Put entry executed: {self.position_size} contracts of {self.current_put.symbol} bought at {self.entry_premium}")
                
                # Calculate metrics
                total_cost = self.entry_premium * 100 * self.position_size
                
                entry_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Long Put",
                    "action": "BUY_PUT",
                    "symbol": self.current_put.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "premium": self.entry_premium,
                    "strike": self.current_put.strike,
                    "expiry": self.current_put.expiry_date,
                    "total_cost": total_cost,
                    "max_risk": total_cost,
                    "trade_date": datetime.now().isoformat()
                }
                
                self._log_trade("ENTRY", entry_result)
                return entry_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Long Put entry failed: {error_msg}")
                self.current_put = None
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Long Put entry execution: {str(e)}")
            self.current_put = None
            return {"success": False, "error": str(e)}
            
    async def execute_exit(self) -> Dict[str, Any]:
        """
        Executes the exit from the Long Put position.
        
        Returns:
            Dict[str, Any]: Result of the exit operation
        """
        if not self.current_put:
            return {"success": False, "error": "No open Long Put position to close"}
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {"success": False, "error": f"Unable to get data for {self.current_put.symbol}"}
                
            # Sell puts to close the position
            bid_price = option_data.get("bid", 0)
            if bid_price <= 0:
                logger.warning(f"Invalid bid price ({bid_price}) for {self.current_put.symbol}, using last price")
                bid_price = option_data.get("last", 0.05)  # Minimal price
                
            order_result = await self.broker.place_option_order(
                option_symbol=self.current_put.symbol,
                qty=self.position_size,
                side="sell",  # Sell the puts
                order_type="limit",
                limit_price=bid_price,
                time_in_force="day"
            )
            
            if order_result.get("success", False):
                exit_premium = bid_price
                profit_loss = (exit_premium - self.entry_premium) * 100 * self.position_size
                
                # Log transaction details
                order_id = order_result.get("order_id", "unknown")
                logger.info(f"Long Put exit executed: {self.position_size} contracts of {self.current_put.symbol} sold at {exit_premium}")
                
                exit_result = {
                    "success": True,
                    "order_id": order_id,
                    "strategy": "Long Put",
                    "action": "SELL_PUT",
                    "symbol": self.current_put.symbol,
                    "underlying": self.underlying_symbol,
                    "quantity": self.position_size,
                    "entry_premium": self.entry_premium,
                    "exit_premium": exit_premium,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": (exit_premium - self.entry_premium) / self.entry_premium * 100,
                    "max_drawdown_pct": self.max_drawdown * 100,
                    "strike": self.current_put.strike,
                    "trade_date": datetime.now().isoformat()
                }
                
                # Reset position variables
                self.current_put = None
                self.position_size = 0
                self.entry_premium = 0
                self.max_drawdown = 0
                
                self._log_trade("EXIT", exit_result)
                return exit_result
            else:
                error_msg = order_result.get("error", "Order failed without error message")
                logger.error(f"Long Put exit failed: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Exception during Long Put exit execution: {str(e)}")
            return {"success": False, "error": str(e)}
            
    async def get_position_info(self) -> Dict[str, Any]:
        """
        Provides information about the current Long Put position.
        
        Returns:
            Dict: Details of the current position
        """
        if not self.current_put:
            return {"has_position": False}
            
        try:
            # Get current option data
            option_data = await self._get_current_option_data()
            
            if not option_data:
                return {
                    "has_position": True,
                    "error": f"Unable to get current data for {self.current_put.symbol}"
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
            expiry = datetime.strptime(self.current_put.expiry_date, "%Y-%m-%d").date()
            days_to_expiry = (expiry - today).days
            
            return {
                "has_position": True,
                "strategy": "Long Put",
                "symbol": self.current_put.symbol,
                "underlying": self.underlying_symbol,
                "quantity": self.position_size,
                "strike": self.current_put.strike,
                "expiry": self.current_put.expiry_date,
                "days_to_expiry": days_to_expiry,
                "entry_premium": self.entry_premium,
                "current_premium": current_price,
                "unrealized_pl": unrealized_pl,
                "unrealized_pl_pct": unrealized_pl_pct,
                "max_drawdown_pct": self.max_drawdown * 100,
                "total_investment": self.entry_premium * 100 * self.position_size
            }
            
        except Exception as e:
            logger.error(f"Error retrieving Long Put position information: {str(e)}")
            return {
                "has_position": True,
                "error": str(e),
                "symbol": self.current_put.symbol if self.current_put else "unknown"
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
            # Verify it's a put
            if option.option_type != OptionType.PUT:
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
                
            # Delta filter (negative for puts)
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
            
            # Risk-adjusted score - for puts we want to maximize delta (negative) per dollar spent
            score = abs(option.delta) / (option.ask * (1 + theta_cost/option.ask))
            
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
            
            # Check overall trend (buy puts in a bearish trend)
            downtrend = last_close < last_sma50 < last_sma200
            
            # Check RSI (not oversold)
            rsi_ok = last_rsi >= 30  # Avoid oversold markets
            
            # Check recent momentum
            recent_candles = market_data.iloc[-5:]
            bearish_candles = sum(1 for i in range(len(recent_candles)) if recent_candles['close'].iloc[i] < recent_candles['open'].iloc[i])
            momentum_bearish = bearish_candles >= 3  # At least 3 out of 5 recent candles are bearish
            
            logger.info(f"{self.underlying_symbol}: downtrend={downtrend}, rsi_ok={rsi_ok} (RSI={last_rsi:.2f}), momentum_bearish={momentum_bearish} (bearish_candles={bearish_candles})")
            if not downtrend:
                logger.info(f"{self.underlying_symbol}: Downtrend condition not met (close={last_close}, sma50={last_sma50}, sma200={last_sma200})")
            if not rsi_ok:
                logger.info(f"{self.underlying_symbol}: RSI condition not met (RSI={last_rsi:.2f})")
            if not momentum_bearish:
                logger.info(f"{self.underlying_symbol}: Momentum condition not met (bearish_candles={bearish_candles})")
            return downtrend and rsi_ok and momentum_bearish
            
        except Exception as e:
            logger.error(f"Error checking technical filters: {str(e)}")
            return False
            
    def _detect_bullish_reversal(self, market_data: pd.DataFrame) -> bool:
        """
        Detects bullish reversal patterns in the underlying asset.
        
        Args:
            market_data: Market data
            
        Returns:
            bool: True if bullish reversal detected, False otherwise
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
            
            # Check for oversold RSI that is turning up
            oversold_to_up = rsi.iloc[-2] < 30 and rsi.iloc[-1] > rsi.iloc[-2]
            
            # Check for price higher than recent moving average
            market_data['sma10'] = market_data['close'].rolling(window=10).mean()
            crossing_up = market_data['close'].iloc[-2] < market_data['sma10'].iloc[-2] and market_data['close'].iloc[-1] > market_data['sma10'].iloc[-1]
            
            # Check for bullish engulfing pattern
            last_candle = market_data.iloc[-1]
            prev_candle = market_data.iloc[-2]
            bullish_engulfing = (
                last_candle['open'] < prev_candle['close'] and
                last_candle['close'] > prev_candle['open'] and
                last_candle['close'] > last_candle['open'] and  # Current candle is green
                prev_candle['close'] < prev_candle['open']  # Previous candle is red
            )
            
            # Detect at least one of these bullish signals
            return oversold_to_up or crossing_up or bullish_engulfing
            
        except Exception as e:
            logger.error(f"Error detecting bullish reversal: {str(e)}")
            return False
            
    async def _get_current_option_data(self) -> Dict[str, Any]:
        """
        Retrieves current data for the current option.
        
        Returns:
            Dict: Current option data
        """
        if not self.current_put:
            return {}
            
        try:
            # Try to retrieve current price information
            option_chain = await self.broker.get_option_chain(
                symbol=self.underlying_symbol,
                expiration_date=self.current_put.expiry_date,
                option_type=OptionType.PUT
            )
            
            if not option_chain:
                return {}
                
            # Find our specific option
            for option in option_chain:
                if (option.symbol == self.current_put.symbol or 
                    (option.strike == self.current_put.strike and 
                     option.expiry_date == self.current_put.expiry_date and
                     option.option_type == OptionType.PUT)):
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
        try:
            # Check account balance - no await since get_account() is not async
            account = self.broker.get_account()
            if not account:
                logger.warning("Unable to retrieve account information")
                return False
                
            # Access account properties directly without using get()
            buying_power = float(account.buying_power) if hasattr(account, 'buying_power') else 0
            
            if buying_power < 2000:  # Minimum required for Long Put
                logger.warning(f"Insufficient buying power for Long Put strategy: {buying_power}")
                return False
                
            # Check if options trading is enabled
            if hasattr(self.broker, 'enable_options') and not self.broker.enable_options:
                logger.warning("Options trading is not enabled")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking account requirements: {str(e)}")
            return False
