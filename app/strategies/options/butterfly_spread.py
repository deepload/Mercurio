"""
Butterfly Spread Options Strategy

This module implements the Butterfly Spread options strategy, which is a neutral strategy
that combines bull and bear spreads with a common middle strike price. It profits most
when the underlying price is at the middle strike price at expiration, with limited risk.

Classes:
    ButterflySpreadStrategy: Implementation of the butterfly spread strategy.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.core.models.option import OptionContract, OptionType


class ButterflySpreadStrategy(BaseOptionsStrategy):
    """
    Butterfly Spread Options Strategy.
    
    A butterfly spread is an options strategy combining a bull spread and a bear spread,
    with three strike prices (typically using all calls or all puts).
    It consists of:
    - Buy 1 option at a lower strike price (wing)
    - Sell 2 options at the middle strike price (body)
    - Buy 1 option at a higher strike price (wing)
    
    All options have the same expiration date. Maximum profit occurs when the underlying
    price is exactly at the middle strike price at expiration.
    """
    
    def __init__(self, 
                 underlying_symbol: str, 
                 max_position_size: float = 0.1,
                 option_type: str = "call",
                 delta_target: float = 0.30,
                 wing_width_pct: float = 0.05,
                 days_to_expiration: int = 30,
                 max_days_to_hold: int = 21,
                 profit_target_pct: float = 0.50,
                 stop_loss_pct: float = 0.50,
                 use_technical_filters: bool = True,
                 **kwargs):
        """
        Initialize the Butterfly Spread strategy.
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            max_position_size: Maximum position size as a fraction of portfolio value
            option_type: Type of options to use ("call" or "put")
            delta_target: Target delta for the middle strike option
            wing_width_pct: Width between strikes as percentage of underlying price
            days_to_expiration: Target days to expiration for new positions
            max_days_to_hold: Maximum days to hold the position
            profit_target_pct: Profit target as percentage of initial debit
            stop_loss_pct: Stop loss as percentage of initial debit
            use_technical_filters: Whether to use technical indicators as filters
        """
        self.underlying_symbol = underlying_symbol
        name = f"Butterfly Spread ({option_type.upper()})"
        description = "A neutral options strategy combining bull and bear spreads with a common middle strike price"
        super().__init__(name=name, description=description)
        
        self.option_type_str = option_type.lower()
        self.option_type = (OptionType.CALL if self.option_type_str == "call" 
                           else OptionType.PUT)
        self.delta_target = delta_target
        self.wing_width_pct = wing_width_pct
        self.days_to_expiration = days_to_expiration
        self.max_days_to_hold = max_days_to_hold
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.use_technical_filters = use_technical_filters
        
        self.logger = logging.getLogger(__name__)
        self.open_positions = {}
        self.initial_debit = 0
    
    async def should_enter(self, market_data: pd.DataFrame) -> bool:
        """
        Determine if a new butterfly spread position should be entered.
        
        Args:
            market_data: DataFrame containing market data for the underlying asset
            
        Returns:
            True if an entry signal is generated, False otherwise
        """
        if not await self._check_technical_filters(market_data):
            return False
            
        # Check if IV is in an acceptable range for a butterfly (not too low)
        if self.options_service:
            options_chain = await self.options_service.get_options_chain(
                self.underlying_symbol)
            
            if options_chain:
                atm_options = self._find_atm_options(options_chain)
                if atm_options:
                    avg_iv = np.mean([opt.implied_volatility for opt in atm_options 
                                     if opt.implied_volatility])
                    
                    # Butterfly spreads generally work better in moderate IV environments
                    # IV should not be too low (poor reward to risk) or too high (expensive wings)
                    if avg_iv < 0.15 or avg_iv > 0.60:
                        self.logger.info(f"IV conditions not ideal for butterfly spread: {avg_iv:.2f}")
                        return False
        
        # Check if we already have an open position
        if len(self.open_positions) > 0:
            self.logger.info("Butterfly spread position already open")
            return False
            
        return True
    
    async def execute_entry(self) -> Dict[str, Any]:
        """
        Execute entry into a new butterfly spread position.
        
        Returns:
            Dictionary with entry details
        """
        try:
            # Get current price and options chain
            current_price = await self._get_current_price()
            if not current_price:
                return {"success": False, "error": "Failed to get current price"}
            
            # Get available options
            if not self.options_service:
                return {"success": False, "error": "Options service not available"}
                
            options_chain = await self.options_service.get_options_chain(
                self.underlying_symbol)
            
            if not options_chain:
                return {"success": False, "error": "No options available"}
                
            # Find appropriate expiration date
            expiry = self._find_expiration_date(options_chain, self.days_to_expiration)
            if not expiry:
                return {"success": False, "error": "No suitable expiration date found"}
                
            # Filter options by expiration date and option type
            filtered_options = [opt for opt in options_chain 
                              if opt.expiration == expiry and 
                              opt.option_type == self.option_type]
            
            if not filtered_options:
                return {"success": False, "error": f"No {self.option_type_str} options found for {expiry}"}
                
            # Find strikes for butterfly based on delta or price
            wing_width = current_price * self.wing_width_pct
            
            # Find middle strike close to current price
            middle_strike = self._find_strike_by_delta(
                filtered_options, self.delta_target, current_price)
            
            if not middle_strike:
                return {"success": False, "error": "Could not find appropriate middle strike"}
                
            # Lower strike = middle strike - wing width
            lower_strike = self._find_closest_strike(
                filtered_options, middle_strike - wing_width)
            
            # Upper strike = middle strike + wing width
            upper_strike = self._find_closest_strike(
                filtered_options, middle_strike + wing_width)
            
            if not lower_strike or not upper_strike:
                return {"success": False, "error": "Could not find appropriate wing strikes"}
                
            # Find corresponding option contracts
            lower_option = next((opt for opt in filtered_options 
                              if abs(opt.strike - lower_strike) < 0.01), None)
            middle_option = next((opt for opt in filtered_options 
                               if abs(opt.strike - middle_strike) < 0.01), None)
            upper_option = next((opt for opt in filtered_options 
                              if abs(opt.strike - upper_strike) < 0.01), None)
            
            if not lower_option or not middle_option or not upper_option:
                return {"success": False, "error": "Could not find all required option contracts"}
                
            # Calculate maximum amount to invest based on account value
            account_value = await self._get_account_value()
            max_debit = account_value * self.max_position_size
            
            # Calculate the cost of the butterfly spread
            # Buy 1 lower strike, sell 2 middle strike, buy 1 upper strike
            lower_cost = (lower_option.ask + lower_option.bid) / 2
            middle_cost = (middle_option.ask + middle_option.bid) / 2
            upper_cost = (upper_option.ask + upper_option.bid) / 2
            
            # Total debit = lower + upper - 2*middle
            debit_per_spread = lower_cost + upper_cost - 2 * middle_cost
            
            # Calculate number of spreads to buy (100 shares per contract)
            num_spreads = int(max_debit / (debit_per_spread * 100))
            
            if num_spreads < 1:
                return {"success": False, "error": "Insufficient funds for butterfly spread"}
                
            # Cap at a reasonable number
            num_spreads = min(num_spreads, 10)
            
            # Execute orders
            if not self.broker_adapter:
                return {"success": False, "error": "Broker adapter not available"}
                
            # Buy lower strike
            lower_order = await self.broker_adapter.place_option_order(
                option_symbol=lower_option.symbol,
                qty=num_spreads,
                side="buy",
                order_type="limit",
                limit_price=lower_option.ask
            )
            
            # Sell middle strike (2x)
            middle_order = await self.broker_adapter.place_option_order(
                option_symbol=middle_option.symbol,
                qty=num_spreads * 2,
                side="sell",
                order_type="limit",
                limit_price=middle_option.bid
            )
            
            # Buy upper strike
            upper_order = await self.broker_adapter.place_option_order(
                option_symbol=upper_option.symbol,
                qty=num_spreads,
                side="buy",
                order_type="limit",
                limit_price=upper_option.ask
            )
            
            # Record the position
            self.initial_debit = debit_per_spread * 100 * num_spreads
            total_cost = self.initial_debit
            
            self.open_positions = {
                "entry_date": datetime.now(),
                "expiry_date": expiry,
                "num_spreads": num_spreads,
                "lower_strike": lower_strike,
                "middle_strike": middle_strike,
                "upper_strike": upper_strike,
                "initial_debit": self.initial_debit,
                "lower_option": lower_option.symbol,
                "middle_option": middle_option.symbol,
                "upper_option": upper_option.symbol,
                "current_price": current_price,
                "max_profit": (upper_strike - middle_strike) * 100 * num_spreads - total_cost,
                "max_loss": total_cost
            }
            
            entry_details = {
                "success": True,
                "strategy": "Butterfly Spread",
                "option_type": self.option_type_str,
                "lower_strike": lower_strike,
                "middle_strike": middle_strike,
                "upper_strike": upper_strike,
                "expiry": expiry,
                "num_spreads": num_spreads,
                "initial_debit": self.initial_debit,
                "max_profit": self.open_positions["max_profit"],
                "max_loss": self.open_positions["max_loss"]
            }
            
            self.logger.info(f"Executed butterfly spread entry: {entry_details}")
            return entry_details
            
        except Exception as e:
            self.logger.error(f"Error executing butterfly spread entry: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def should_exit(self, position_id: str, market_data: pd.DataFrame) -> bool:
        """
        Determine if an existing butterfly spread position should be exited.
        
        Args:
            market_data: DataFrame containing market data for the underlying asset
            
        Returns:
            True if an exit signal is generated, False otherwise
        """
        # Ignore position_id parameter for now since we only manage one position at a time
        # In a more sophisticated implementation, we would track multiple positions by ID
        if not self.open_positions:
            return False
            
        try:
            current_price = await self._get_current_price()
            if not current_price:
                return False
                
            # Calculate days to expiration
            entry_date = self.open_positions["entry_date"]
            expiry_date = datetime.strptime(
                self.open_positions["expiry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_date).days
            days_to_expiry = (expiry_date - datetime.now()).days
            
            # Get current value of the position
            lower_option_symbol = self.open_positions["lower_option"]
            middle_option_symbol = self.open_positions["middle_option"]
            upper_option_symbol = self.open_positions["upper_option"]
            
            # Get current option prices
            if not self.options_service:
                return False
                
            lower_option = await self.options_service.get_option_details(lower_option_symbol)
            middle_option = await self.options_service.get_option_details(middle_option_symbol)
            upper_option = await self.options_service.get_option_details(upper_option_symbol)
            
            if not lower_option or not middle_option or not upper_option:
                return False
                
            # Calculate current value
            num_spreads = self.open_positions["num_spreads"]
            lower_value = (lower_option.bid + lower_option.ask) / 2
            middle_value = (middle_option.bid + middle_option.ask) / 2
            upper_value = (upper_option.bid + upper_option.ask) / 2
            
            current_value = (lower_value + upper_value - 2 * middle_value) * 100 * num_spreads
            initial_debit = self.open_positions["initial_debit"]
            
            profit_loss = current_value - initial_debit
            profit_loss_pct = profit_loss / initial_debit
            
            # Exit criteria
            # 1. Profit target reached
            if profit_loss_pct >= self.profit_target_pct:
                self.logger.info(f"Butterfly spread profit target reached: {profit_loss_pct:.2%}")
                return True
                
            # 2. Stop loss hit
            if profit_loss_pct <= -self.stop_loss_pct:
                self.logger.info(f"Butterfly spread stop loss hit: {profit_loss_pct:.2%}")
                return True
                
            # 3. Maximum hold time reached
            if days_held >= self.max_days_to_hold:
                self.logger.info(f"Butterfly spread max hold time reached: {days_held} days")
                return True
                
            # 4. Near expiration (avoid gamma risk and exercise)
            if days_to_expiry <= 5:
                self.logger.info(f"Butterfly spread near expiration: {days_to_expiry} days remaining")
                return True
                
            # 5. Price moved significantly beyond upper or lower strikes
            middle_strike = self.open_positions["middle_strike"]
            lower_strike = self.open_positions["lower_strike"]
            upper_strike = self.open_positions["upper_strike"]
            
            # If price has moved far beyond wings and there's little chance of recovery
            if current_price < lower_strike * 0.95 or current_price > upper_strike * 1.05:
                # Only exit if we're also losing money
                if profit_loss_pct < 0:
                    self.logger.info(f"Butterfly spread price moved beyond wings: {current_price}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in butterfly spread exit decision: {str(e)}")
            return False
    
    async def execute_exit(self, position_id: str) -> Dict[str, Any]:
        """
        Execute exit from an existing butterfly spread position.
        
        Returns:
            Dictionary with exit details
        """
        # Ignore position_id parameter for now since we only manage one position at a time
        # In a more sophisticated implementation, we would track positions by ID
        if not self.open_positions:
            return {"success": False, "error": "No open positions to exit"}
            
        try:
            # Get current option details
            lower_option_symbol = self.open_positions["lower_option"]
            middle_option_symbol = self.open_positions["middle_option"]
            upper_option_symbol = self.open_positions["upper_option"]
            
            # Get current option prices
            if not self.options_service:
                return {"success": False, "error": "Options service not available"}
                
            lower_option = await self.options_service.get_option_details(lower_option_symbol)
            middle_option = await self.options_service.get_option_details(middle_option_symbol)
            upper_option = await self.options_service.get_option_details(upper_option_symbol)
            
            if not lower_option or not middle_option or not upper_option:
                return {"success": False, "error": "Could not get current option details"}
                
            num_spreads = self.open_positions["num_spreads"]
            
            # Place closing orders
            if not self.broker_adapter:
                return {"success": False, "error": "Broker adapter not available"}
                
            # Sell lower strike (close long position)
            lower_close = await self.broker_adapter.place_option_order(
                option_symbol=lower_option_symbol,
                qty=num_spreads,
                side="sell",
                order_type="limit",
                limit_price=lower_option.bid
            )
            
            # Buy middle strike (close short position)
            middle_close = await self.broker_adapter.place_option_order(
                option_symbol=middle_option_symbol,
                qty=num_spreads * 2,
                side="buy",
                order_type="limit",
                limit_price=middle_option.ask
            )
            
            # Sell upper strike (close long position)
            upper_close = await self.broker_adapter.place_option_order(
                option_symbol=upper_option_symbol,
                qty=num_spreads,
                side="sell",
                order_type="limit",
                limit_price=upper_option.bid
            )
            
            # Calculate exit value and P&L
            lower_exit_value = lower_option.bid * 100 * num_spreads
            middle_exit_value = -middle_option.ask * 100 * num_spreads * 2
            upper_exit_value = upper_option.bid * 100 * num_spreads
            
            total_exit_value = lower_exit_value + middle_exit_value + upper_exit_value
            initial_debit = self.open_positions["initial_debit"]
            
            profit_loss = total_exit_value - (-initial_debit)  # Initial debit is negative cash flow
            profit_loss_pct = profit_loss / initial_debit
            
            days_held = (datetime.now() - self.open_positions["entry_date"]).days
            
            exit_details = {
                "success": True,
                "exit_date": datetime.now(),
                "days_held": days_held,
                "exit_value": total_exit_value,
                "initial_value": -initial_debit,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "current_price": await self._get_current_price(),
                "entry_price": self.open_positions["current_price"]
            }
            
            # Clear the position
            self.open_positions = {}
            self.initial_debit = 0
            
            self.logger.info(f"Executed butterfly spread exit: {exit_details}")
            return exit_details
            
        except Exception as e:
            self.logger.error(f"Error executing butterfly spread exit: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _find_closest_strike(self, options: List[OptionContract], 
                          target_strike: float) -> float:
        """Find the option with strike price closest to the target."""
        if not options:
            return None
            
        # Get unique strike prices
        strikes = sorted(set(opt.strike for opt in options))
        
        if not strikes:
            return None
            
        # Find closest
        closest_strike = min(strikes, key=lambda x: abs(x - target_strike))
        return closest_strike
    
    def _find_strike_by_delta(self, options: List[OptionContract], 
                           target_delta: float, 
                           current_price: float) -> float:
        """Find strike price with delta closest to target delta."""
        if not options:
            return None
            
        # Filter options with valid delta
        valid_options = [opt for opt in options if opt.delta is not None]
        
        if not valid_options:
            # Fall back to ATM option
            return self._find_closest_strike(options, current_price)
            
        # For puts, delta is negative, so we take absolute value
        closest_option = min(
            valid_options, 
            key=lambda x: abs(abs(x.delta) - target_delta)
        )
        
        return closest_option.strike
    
    def _find_atm_options(self, options: List[OptionContract]) -> List[OptionContract]:
        """Find at-the-money options."""
        if not options:
            return []
            
        current_price = next((opt.underlying_price for opt in options 
                            if opt.underlying_price is not None), None)
        
        if not current_price:
            return []
            
        # Find the strike closest to current price
        closest_strike = self._find_closest_strike(options, current_price)
        
        if not closest_strike:
            return []
            
        # Return options with this strike
        return [opt for opt in options if abs(opt.strike - closest_strike) < 0.01]
    
    async def _check_technical_filters(self, market_data: pd.DataFrame) -> bool:
        """
        Apply technical indicators as filters for entry decisions.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            True if all technical filters pass, False otherwise
        """
        if not self.use_technical_filters:
            return True
            
        if market_data.empty:
            return False
            
        try:
            # Make sure we have enough data
            if len(market_data) < 50:
                return False
                
            # Get the latest data point
            latest = market_data.iloc[-1]
            
            # Check if we have the necessary columns
            if 'close' not in market_data.columns:
                return False
                
            # Calculate some basic technical indicators if not present
            if 'sma20' not in market_data.columns:
                market_data['sma20'] = market_data['close'].rolling(window=20).mean()
                
            if 'sma50' not in market_data.columns:
                market_data['sma50'] = market_data['close'].rolling(window=50).mean()
                
            latest = market_data.iloc[-1]
            
            # Neutral strategy works best when price is range-bound
            # Check if price is between SMAs (relatively stable)
            if 'sma20' in latest and 'sma50' in latest:
                min_sma = min(latest['sma20'], latest['sma50'])
                max_sma = max(latest['sma20'], latest['sma50'])
                
                # Check if price is between moving averages (range-bound)
                # or if SMAs are very close to each other (low volatility)
                if not (min_sma <= latest['close'] <= max_sma or 
                       abs(latest['sma20'] - latest['sma50']) / latest['sma50'] < 0.02):
                    return False
                    
            # Calculate historical volatility if not present
            if 'hist_vol' not in market_data.columns:
                # 20-day historical volatility
                market_data['returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
                market_data['hist_vol'] = market_data['returns'].rolling(window=20).std() * np.sqrt(252)
                
            # Check if historical volatility is in an acceptable range
            # Butterfly spreads work better in moderate volatility environments
            if 'hist_vol' in latest and latest['hist_vol'] is not None:
                if latest['hist_vol'] < 0.15 or latest['hist_vol'] > 0.50:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error in technical filters: {str(e)}")
            return False
            
    def _find_expiration_date(self, options: List[OptionContract], target_days: int) -> Optional[str]:
        """
        Find the most appropriate expiration date from available options.
        
        Args:
            options: List of option contracts
            target_days: Target number of days to expiration
            
        Returns:
            The selected expiration date in YYYY-MM-DD format or None if no suitable date found
        """
        try:
            # Extract unique expiration dates from the options chain
            expiration_dates = set()
            for opt in options:
                if hasattr(opt, 'expiration') and opt.expiration:
                    expiration_dates.add(opt.expiration)
                    
            if not expiration_dates:
                self.logger.warning("No expiration dates found in options chain")
                return None
                
            # Convert expiration dates to datetime objects for comparison
            expiry_dates = []
            today = datetime.now().date()
            
            for date_str in expiration_dates:
                try:
                    expiry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    days_to_expiry = (expiry_date - today).days
                    
                    # Only consider dates in the future
                    if days_to_expiry > 0:
                        expiry_dates.append((date_str, days_to_expiry))
                except ValueError as e:
                    self.logger.warning(f"Invalid expiration date format: {date_str}")
                    
            if not expiry_dates:
                self.logger.warning("No valid future expiration dates found")
                return None
                
            # Sort by closest to target days
            expiry_dates.sort(key=lambda x: abs(x[1] - target_days))
            
            # Return the closest match
            return expiry_dates[0][0]
            
        except Exception as e:
            self.logger.error(f"Error finding expiration date: {str(e)}")
            return None
            
    async def _get_current_price(self) -> Optional[float]:
        """
        Get the current price for the underlying asset.
        
        Returns:
            The current price of the underlying asset or None if not available
        """
        try:
            # Try to get the current price from the market data service
            if hasattr(self, 'market_data_service') and self.market_data_service is not None:
                # Get latest price using market data service
                latest_data = await self.market_data_service.get_latest_price(self.underlying_symbol)
                if latest_data and 'close' in latest_data:
                    return latest_data['close']
            
            # Alternative: try to get from the options chain if available
            if self.options_service:
                options_chain = await self.options_service.get_options_chain(self.underlying_symbol)
                if options_chain and len(options_chain) > 0:
                    # Get the underlying price from the first option in the chain
                    for option in options_chain:
                        if hasattr(option, 'underlying_price') and option.underlying_price is not None:
                            return option.underlying_price
            
            # Fallback for testing: use sample data
            self.logger.warning(f"Using fallback sample price for {self.underlying_symbol}")
            # Return a sample price based on the symbol
            sample_prices = {
                'BTC': 26500.0,
                'ETH': 1650.0,
                'SOL': 38.0,
                'BNB': 235.0,
                'XRP': 0.52,
                'ADA': 0.30,
                'AVAX': 10.50,
                'DOGE': 0.065
            }
            return sample_prices.get(self.underlying_symbol.split('-')[0], 100.0)
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {self.underlying_symbol}: {str(e)}")
            return None
