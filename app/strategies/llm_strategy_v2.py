"""
MercurioAI LLMStrategyV2 - Advanced LLM-Based Trading Strategy

This module implements an enhanced trading strategy that combines:
1. Technical analysis with multiple indicators
2. Social media sentiment from multiple sources
3. News and market event analysis
4. LLM-powered decision making that integrates all signals
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import re

from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.core.event_bus import EventBus, EventType
from app.services.market_data import MarketDataService
from app.utils.llm_utils import load_llm_model, call_llm
from app.strategies.sentiment.web_sentiment_agent import LLMWebSentimentAgent
from app.strategies.sentiment.enhanced_web_sentiment import EnhancedWebSentimentAgent
from app.strategies.sentiment.news_api_agent import NewsAPIAgent

logger = logging.getLogger(__name__)

class LLMStrategyV2(BaseStrategy):
    """
    Advanced LLM-based trading strategy that integrates market data with
    social sentiment analysis to make informed trading decisions.
    
    Features:
    - Multi-source sentiment analysis (social media, news, forums)
    - Technical indicator integration and analysis
    - Confidence-weighted decision making
    - Adaptive stop-loss and take-profit recommendations
    - Fallback mechanisms for missing data or API failures
    """

    def __init__(self, 
                 model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 sentiment_model_name: str = None,
                 use_local_model: bool = False,
                 local_model_path: str = None,
                 api_key: str = None,
                 use_web_sentiment: bool = True,
                 sentiment_weight: float = 0.5,
                 min_confidence: float = 0.65,
                 technical_indicators: List[str] = None,
                 news_lookback_hours: int = 24,
                 market_data_service: Any = None,
                 trading_service: Any = None,
                 **kwargs):
        """
        Initialize the enhanced LLM strategy
        
        Args:
            model_name: Name of the LLM to use for technical analysis
            sentiment_model_name: Name of the model for sentiment analysis (uses primary if None)
            use_local_model: Whether to use a local model or API
            local_model_path: Path to local model if applicable
            api_key: API key for remote model access
            use_web_sentiment: Whether to incorporate web sentiment analysis
            sentiment_weight: Weight given to sentiment vs technical (0-1)
            min_confidence: Minimum confidence threshold for trades
            technical_indicators: List of technical indicators to calculate
            news_lookback_hours: Hours of news to analyze
        """
        super().__init__(**kwargs)
        
        # Strategy identification
        self.name = "LLMStrategyV2"
        self.description = "Advanced LLM strategy combining technical analysis with multi-source sentiment data"
        
        # Model configuration
        self.model_name = model_name
        self.sentiment_model_name = sentiment_model_name or model_name
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        
        # Strategy parameters
        self.use_web_sentiment = use_web_sentiment
        self.sentiment_weight = max(0.0, min(1.0, sentiment_weight))
        self.technical_weight = 1.0 - self.sentiment_weight
        self.min_confidence = min_confidence
        self.news_lookback_hours = news_lookback_hours
        self.technical_indicators = technical_indicators or [
            'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 
            'rsi_14', 'macd', 'bollinger_bands', 'atr'
        ]
        
        # Initialize services
        # Use existing market_data if provided, otherwise create a new instance
        self.market_data = market_data_service or MarketDataService()
        self.event_bus = EventBus()
        
        # Initialize models
        self.model = None
        self.sentiment_model = None
        self._initialize_models()
        
        # Initialize sentiment agent if using web sentiment
        self.sentiment_agent = None
        if self.use_web_sentiment:
            self._initialize_sentiment_agent()
        
        # Cache for sentiment data to avoid redundant API calls
        self.cache = {}
        self.cache_ttl = timedelta(hours=2)
        self.cache_timestamps = {}
        
        logger.info(f"Initialized {self.name} with sentiment weight: {self.sentiment_weight:.2f}, "
                   f"technical weight: {self.technical_weight:.2f}")

    def _initialize_models(self):
        """Initialize LLM models for technical and sentiment analysis"""
        try:
            logger.info(f"Initializing main model: {self.model_name}")
            self.model = load_llm_model(
                model_name=self.model_name,
                use_local=self.use_local_model,
                local_path=self.local_model_path,
                api_key=self.api_key
            )
            
            # Initialize sentiment model (may be the same as main model)
            if self.sentiment_model_name != self.model_name:
                logger.info(f"Initializing separate sentiment model: {self.sentiment_model_name}")
                self.sentiment_model = load_llm_model(
                    model_name=self.sentiment_model_name,
                    use_local=self.use_local_model,
                    local_path=self.local_model_path,
                    api_key=self.api_key
                )
            else:
                self.sentiment_model = self.model
                
            logger.info("LLM models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM models: {str(e)}")
            logger.warning("Using fallback for LLM operations")
    
    def _initialize_sentiment_agent(self):
        """Initialize enhanced web sentiment analysis agent and NewsAPI agent"""
        try:
            # Initialize web sentiment agent
            self.sentiment_agent = EnhancedWebSentimentAgent(
                model_name=self.sentiment_model_name,
                use_local_model=self.use_local_model,
                local_model_path=self.local_model_path,
                api_key=self.api_key
            )
            logger.info(f"Initialized enhanced web sentiment agent with model {self.sentiment_model_name}")
            
            # Initialize NewsAPI agent for real-time news data
            self.news_agent = NewsAPIAgent(
                lookback_hours=self.news_lookback_hours,
                use_fallback=True
            )
            logger.info(f"Initialized NewsAPI agent with {self.news_lookback_hours} hour lookback")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment agents: {str(e)}")
            self.sentiment_agent = None
            self.news_agent = None
            logger.warning("Falling back to standard web sentiment agent")
            try:
                self.sentiment_agent = LLMWebSentimentAgent(
                    model_name=self.sentiment_model_name,
                    use_local_model=self.use_local_model,
                    local_model_path=self.local_model_path,
                    api_key=self.api_key
                )
                logger.info("Fallback web sentiment agent initialized successfully")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback sentiment agent: {str(e2)}")
                logger.warning("Web sentiment analysis will be unavailable")
    
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical market data for the given symbol and date range.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            DataFrame containing the historical data
        """
        try:
            logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date}")
            return await self.market_data.get_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            # Return empty DataFrame with expected columns as fallback
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for analysis by adding technical indicators.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data with technical indicators
        """
        if data.empty:
            logger.warning("Empty data received for preprocessing")
            return data
            
        df = data.copy()
        
        try:
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Add volatility metrics
            df['daily_return'] = df['close'].pct_change()
            df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
            
            # Add date-based features
            if isinstance(df.index, pd.DatetimeIndex):
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                
            return df.dropna()
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        df = data.copy()
        
        try:
            # Simple Moving Averages
            if 'sma_10' in self.technical_indicators:
                df['sma_10'] = df['close'].rolling(window=10).mean()
            if 'sma_20' in self.technical_indicators:
                df['sma_20'] = df['close'].rolling(window=20).mean()
            if 'sma_50' in self.technical_indicators:
                df['sma_50'] = df['close'].rolling(window=50).mean()
                
            # Exponential Moving Averages
            if 'ema_10' in self.technical_indicators:
                df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            if 'ema_20' in self.technical_indicators:
                df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
                
            # Relative Strength Index (RSI)
            if 'rsi_14' in self.technical_indicators:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
            # MACD
            if 'macd' in self.technical_indicators:
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
            # Bollinger Bands
            if 'bollinger_bands' in self.technical_indicators:
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                df['bb_std'] = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
                df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
                
            # Average True Range (ATR)
            if 'atr' in self.technical_indicators:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['atr'] = true_range.rolling(14).mean()
                
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
            
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate a trading signal based on the input data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Tuple of (TradeAction, confidence)
        """
        if data.empty or len(data) < 20:  # Need enough data for analysis
            logger.warning("Insufficient data for prediction")
            return TradeAction.HOLD, 0.0
            
        try:
            # Extract latest data for analysis
            latest_data = data.iloc[-60:].copy()  # Use last 60 points
            symbol = getattr(data, 'symbol', 'UNKNOWN')
            
            # Technical analysis signal
            tech_action, tech_confidence = await self._generate_technical_signal(latest_data)
            logger.info(f"Technical analysis signal: {tech_action.name} ({tech_confidence:.2f})")
            
            # Sentiment analysis signal (if enabled)
            sentiment_action = TradeAction.HOLD
            sentiment_confidence = 0.0
            
            if self.use_web_sentiment and self.sentiment_agent:
                sentiment_action, sentiment_confidence = await self._generate_sentiment_signal(symbol)
                logger.info(f"Sentiment analysis signal: {sentiment_action.name} ({sentiment_confidence:.2f})")
            
            # Combine signals with weighting
            final_action, final_confidence = self._combine_signals(
                tech_action, tech_confidence, 
                sentiment_action, sentiment_confidence
            )
            
            logger.info(f"Final signal for {symbol}: {final_action.name} ({final_confidence:.2f})")
            
            # Only return actionable signals if confidence is above threshold
            if final_confidence < self.min_confidence and final_action != TradeAction.HOLD:
                return TradeAction.HOLD, final_confidence
            
            return final_action, final_confidence
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return TradeAction.HOLD, 0.0
    
    async def _generate_technical_signal(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """Generate trading signal based on technical analysis"""
        try:
            # Get the latest data point
            latest = data.iloc[-1]
            
            # Simple rule-based approach as fallback
            if self.model is None:
                # MACD signal
                if 'macd' in latest and 'macd_signal' in latest:
                    if latest['macd'] > latest['macd_signal']:
                        return TradeAction.BUY, 0.6
                    elif latest['macd'] < latest['macd_signal']:
                        return TradeAction.SELL, 0.6
                
                # Moving average crossover
                if 'sma_10' in latest and 'sma_50' in latest:
                    if latest['sma_10'] > latest['sma_50']:
                        return TradeAction.BUY, 0.7
                    elif latest['sma_10'] < latest['sma_50']:
                        return TradeAction.SELL, 0.7
                
                # RSI signals
                if 'rsi_14' in latest:
                    if latest['rsi_14'] < 30:
                        return TradeAction.BUY, 0.65
                    elif latest['rsi_14'] > 70:
                        return TradeAction.SELL, 0.65
                
                return TradeAction.HOLD, 0.5
            
            # LLM-based approach
            prompt = self._create_technical_analysis_prompt(data)
            response = call_llm(self.model, prompt)
            
            action, confidence = self._extract_trading_signal(response)
            return action, confidence
            
        except Exception as e:
            logger.error(f"Error generating technical signal: {str(e)}")
            return TradeAction.HOLD, 0.0
    
    async def _generate_sentiment_signal(self, symbol: str) -> Tuple[TradeAction, float]:
        """Generate trading signal based on sentiment analysis from web and news sources"""
        if not self.use_web_sentiment or not self.sentiment_agent:
            return TradeAction.HOLD, 0.5
        
        try:
            # Get sentiment from web sources
            web_sentiment = await self.sentiment_agent.analyze_sentiment(symbol)
            web_score = web_sentiment.get("score", 0.5)
            
            # Get news articles from NewsAPI
            news_articles = []
            if self.news_agent:
                news_articles = await self.news_agent.get_news_for_symbol(symbol)
                logger.info(f"Found {len(news_articles)} news articles for {symbol}")
            
            # Analyze news sentiment if articles are available
            news_sentiment = {"sentiment": "neutral", "confidence": 0.5, "sources": 0}
            if news_articles and self.news_agent:
                news_sentiment = await self.news_agent.analyze_sentiment(news_articles, self.sentiment_agent)
                logger.info(f"News sentiment for {symbol}: {news_sentiment['sentiment']} with confidence {news_sentiment['confidence']}")
            
            # Combine web and news sentiment
            combined_score = web_score
            if news_sentiment["sources"] > 0:
                # Weight news more heavily if we have articles (60% news, 40% web)
                news_score = 1.0 if news_sentiment["sentiment"] == "bullish" else (
                              0.0 if news_sentiment["sentiment"] == "bearish" else 0.5)
                combined_score = (news_score * 0.6 * news_sentiment["confidence"]) + (web_score * 0.4)
                logger.info(f"Combined sentiment score for {symbol}: {combined_score:.2f}")
            return action, confidence
            
        except Exception as e:
            logger.error(f"Error generating sentiment signal: {str(e)}")
            return TradeAction.HOLD, 0.0
            
    def _combine_signals(self, 
                        tech_action: TradeAction, 
                        tech_confidence: float,
                        sentiment_action: TradeAction, 
                        sentiment_confidence: float) -> Tuple[TradeAction, float]:
        """
        Combine technical and sentiment signals to produce a final signal.
        
        Args:
            tech_action: Action from technical analysis
            tech_confidence: Confidence from technical analysis
            sentiment_action: Action from sentiment analysis
            sentiment_confidence: Confidence from sentiment analysis
            
        Returns:
            Final action and confidence
        """
        # If sentiment is disabled or unavailable, use only technical
        if not self.use_web_sentiment or sentiment_confidence == 0:
            return tech_action, tech_confidence
            
        # If actions agree, increase confidence
        if tech_action == sentiment_action:
            combined_confidence = (
                tech_confidence * self.technical_weight + 
                sentiment_confidence * self.sentiment_weight
            ) * 1.2  # Bonus for agreement
            
            # Cap at 1.0
            combined_confidence = min(combined_confidence, 1.0)
            return tech_action, combined_confidence
            
        # Actions disagree, use weighted confidence to determine winner
        tech_score = tech_confidence * self.technical_weight
        sentiment_score = sentiment_confidence * self.sentiment_weight
        
        if tech_score > sentiment_score:
            # Technical wins but reduce confidence due to disagreement
            return tech_action, tech_score * 0.8
        elif sentiment_score > tech_score:
            # Sentiment wins but reduce confidence due to disagreement
            return sentiment_action, sentiment_score * 0.8
        else:
            # Tie, default to HOLD with reduced confidence
            return TradeAction.HOLD, max(tech_confidence, sentiment_confidence) * 0.6
            
    def _create_technical_analysis_prompt(self, data: pd.DataFrame) -> str:
        """Create a prompt for technical analysis"""
        # Convert last N rows to dictionary for JSON serialization
        data_dict = data.tail(10).reset_index().to_dict(orient='records')
        
        # Simplify timestamp format if present
        if 'timestamp' in data_dict[0] or isinstance(data_dict[0].get('index'), (datetime, pd.Timestamp)):
            for item in data_dict:
                if isinstance(item.get('timestamp'), (datetime, pd.Timestamp)):
                    item['timestamp'] = item['timestamp'].strftime('%Y-%m-%d')
                elif isinstance(item.get('index'), (datetime, pd.Timestamp)):
                    item['date'] = item['index'].strftime('%Y-%m-%d')
                    del item['index']
                    
        # Format the indicators for the prompt
        indicators = []
        latest = data.iloc[-1]
        
        if 'sma_10' in latest and 'sma_50' in latest:
            indicators.append(f"SMA Crossover: 10-day SMA is {'above' if latest['sma_10'] > latest['sma_50'] else 'below'} 50-day SMA")
            
        if 'rsi_14' in latest:
            rsi_status = "oversold" if latest['rsi_14'] < 30 else "overbought" if latest['rsi_14'] > 70 else "neutral"
            indicators.append(f"RSI (14): {latest['rsi_14']:.2f} - {rsi_status}")
            
        if 'macd' in latest and 'macd_signal' in latest:
            macd_signal = "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
            indicators.append(f"MACD: {macd_signal} ({latest['macd']:.4f} vs signal {latest['macd_signal']:.4f})")
            
        if 'volatility_10d' in latest:
            indicators.append(f"10-day Volatility: {latest['volatility_10d']:.4f}")
            
        if 'bb_upper' in latest and 'bb_lower' in latest:
            price = latest['close']
            bb_position = ""  # position description
            if price > latest['bb_upper']:
                bb_position = "above upper band (potential overbought)"
            elif price < latest['bb_lower']:
                bb_position = "below lower band (potential oversold)"
            else:
                bb_position = "between bands"
            indicators.append(f"Bollinger Bands: Price is {bb_position}")
            
        # Create the prompt
        prompt = f"""
You are an expert financial analyst analyzing market data to generate a trading signal.

Latest market data (recent price history):
{json.dumps(data_dict, indent=2)}

Key Technical Indicators:
{chr(10).join(f"- {indicator}" for indicator in indicators)}

Current close price: {latest['close']:.4f}

Based on this technical analysis, determine whether to BUY, SELL, or HOLD.
Output your response in JSON format as follows:
{{
  "action": "BUY | SELL | HOLD",
  "confidence": 0.0 - 1.0 (your confidence level),
  "justification": "Brief technical analysis explanation"
}}

Focus on objective technical patterns without personal opinions. Consider trend direction,
momentum, support/resistance levels, and the overall market context.
"""
        return prompt
    
    def _extract_trading_signal(self, llm_response: str) -> Tuple[TradeAction, float]:
        """Extract trading signal from LLM response"""
        try:
            # Extract JSON from the response
            json_pattern = r'(\{.*\})'
            match = re.search(json_pattern, llm_response, re.DOTALL)
            
            if not match:
                logger.warning(f"No JSON found in LLM response: {llm_response[:100]}...")
                return TradeAction.HOLD, 0.5
                
            json_str = match.group(1)
            data = json.loads(json_str)
            
            # Extract action
            action_str = data.get('action', 'HOLD').upper()
            action_map = {
                'BUY': TradeAction.BUY,
                'SELL': TradeAction.SELL,
                'HOLD': TradeAction.HOLD
            }
            action = action_map.get(action_str, TradeAction.HOLD)
            
            # Extract confidence
            confidence = float(data.get('confidence', 0.5))
            
            # Log justification if present
            if 'justification' in data:
                logger.info(f"LLM justification: {data['justification']}")
                
            return action, confidence
            
        except Exception as e:
            logger.error(f"Error extracting trading signal: {str(e)}")
            return TradeAction.HOLD, 0.5
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        LLMStrategyV2 doesn't require traditional training, but this
        method can be used for parameter tuning or model fine-tuning.
        
        Args:
            data: Training data
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("LLMStrategyV2 doesn't require traditional training")
        return {"status": "success", "message": "LLM strategy does not require training"}
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Historical market data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        if data.empty:
            logger.warning("Empty data for backtesting")
            return {"error": "No data available for backtesting"}
            
        # Ensure data is preprocessed
        processed_data = await self.preprocess_data(data)
        
        # Initialize backtest state
        capital = initial_capital
        position = 0  # Number of shares/units
        trades = []
        equity_curve = []
        
        # Get symbol from data if available
        symbol = getattr(processed_data, 'symbol', 'UNKNOWN')
        
        # Step through each data point (skip first ones that might have NaN due to indicators)
        start_idx = max(50, len(processed_data) - min(len(processed_data), 252))  # Start after initial NaNs and limit to ~1 year
        
        for i in range(start_idx, len(processed_data)):
            # Use data up to current point only (avoid lookahead bias)
            current_data = processed_data.iloc[:i+1]
            current_date = current_data.index[-1] if isinstance(current_data.index, pd.DatetimeIndex) else i
            current_price = current_data['close'].iloc[-1]
            
            # Generate prediction for current data point
            action, confidence = await self.predict(current_data)
            
            # Update position based on action
            previous_position = position
            
            if action == TradeAction.BUY and position <= 0:
                # Close any short position
                if position < 0:
                    trade_value = position * current_price * -1  # Close short
                    capital += trade_value
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'quantity': position * -1,
                        'value': trade_value,
                        'capital': capital
                    })
                    position = 0
                
                # Open long position - invest 95% of capital
                if confidence >= self.min_confidence:
                    quantity = (capital * 0.95) / current_price
                    trade_value = quantity * current_price
                    capital -= trade_value
                    position += quantity
                    trades.append({
                        'timestamp': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': quantity,
                        'value': trade_value,
                        'capital': capital
                    })
                
            elif action == TradeAction.SELL and position >= 0:
                # Close any long position
                if position > 0:
                    trade_value = position * current_price
                    capital += trade_value
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'quantity': position,
                        'value': trade_value,
                        'capital': capital
                    })
                    position = 0
                
                # Open short position - 95% of capital
                if confidence >= self.min_confidence:
                    quantity = (capital * 0.95) / current_price
                    trade_value = quantity * current_price
                    capital += trade_value  # Short proceeds
                    position -= quantity
                    trades.append({
                        'timestamp': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': quantity,
                        'value': trade_value,
                        'capital': capital
                    })
            
            # Calculate current equity (capital + position value)
            equity = capital + (position * current_price)
            
            # Record equity and position state
            equity_curve.append({
                'timestamp': current_date,
                'price': current_price,
                'action': action.name if action else 'UNKNOWN',
                'confidence': confidence,
                'position': position,
                'capital': capital,
                'equity': equity
            })
        
        # Close final position for accurate return calculation
        final_price = processed_data['close'].iloc[-1]
        final_equity = capital
        if position != 0:
            final_equity += position * final_price
            
        # Calculate performance metrics
        total_return = ((final_equity / initial_capital) - 1) * 100  # Percentage
        equity_df = pd.DataFrame(equity_curve)
        
        # Add more sophisticated metrics if enough data
        metrics = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'symbol': symbol,
            'strategy': self.name
        }
        
        # Add more metrics if we have enough data
        if len(equity_df) > 1:
            # Calculate daily returns
            if isinstance(equity_df['timestamp'].iloc[0], (datetime, pd.Timestamp)):
                equity_df = equity_df.set_index('timestamp')
            
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            
            # Calculate additional metrics
            if len(equity_df) > 30:
                metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(equity_df['daily_return'])
                metrics['max_drawdown_pct'] = self._calculate_max_drawdown(equity_df['equity']) * 100
                metrics['win_rate'] = self._calculate_win_rate(trades) * 100
        
        # Prepare final results
        results = {
            'metrics': metrics,
            'equity_curve': equity_df.reset_index().to_dict(orient='records') if not equity_df.empty else [],
            'trades': trades
        }
        
        return results
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if returns.empty or returns.isna().all():
            return 0.0
            
        # Assume daily returns and annualize
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if equity.empty:
            return 0.0
            
        # Calculate running maximum
        running_max = equity.cummax()
        # Calculate drawdown in percentage terms
        drawdown = (equity - running_max) / running_max
        # Find the maximum drawdown
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown) if not pd.isna(max_drawdown) else 0.0
    
    def _calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate win rate from list of trades"""
        if not trades:
            return 0.0
            
        # Count profitable trades
        profitable_trades = sum(1 for t in trades if 'value' in t and 'action' in t and 
                               ((t['action'] == 'CLOSE_LONG' and t['value'] > 0) or 
                                (t['action'] == 'CLOSE_SHORT' and t['value'] > 0)))
        
        # Count total closed trades
        closed_trades = sum(1 for t in trades if 'action' in t and 
                           t['action'] in ['CLOSE_LONG', 'CLOSE_SHORT'])
        
        return profitable_trades / closed_trades if closed_trades > 0 else 0.0


async def test_strategy():
    """Test the strategy with a sample"""
    logging.basicConfig(level=logging.INFO)
    
    # Create the strategy with NewsAPI integration
    strategy = LLMStrategyV2(
        news_lookback_hours=24,  # Look for news from the last 24 hours
        sentiment_weight=0.6     # Give more weight to sentiment vs technical analysis
    )
    
    # Load some data
    symbol = "BTC/USD"  # Use standard format for crypto pairs
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    data = await strategy.load_data(symbol, start_date, end_date)
    
    if not data.empty:
        # Preprocess the data
        processed_data = await strategy.preprocess_data(data)
        
        # Generate a prediction
        action, confidence = await strategy.predict(processed_data)
        
        print(f"\nResults for {symbol}:")
        print(f"Action: {action.name}")
        print(f"Confidence: {confidence:.2f}")
        
        # Show news sentiment if available
        if hasattr(strategy, 'news_agent') and strategy.news_agent:
            news_articles = await strategy.news_agent.get_news_for_symbol(symbol)
            if news_articles:
                print(f"\nFound {len(news_articles)} recent news articles about {symbol.split('/')[0]}")
                print("Top 3 headlines:")
                for i, article in enumerate(news_articles[:3]):
                    print(f"  {i+1}. {article.get('title', 'No title')} - {article.get('source', {}).get('name', 'Unknown')}")
    else:
        print("No data available for testing")


if __name__ == "__main__":
    asyncio.run(test_strategy())
