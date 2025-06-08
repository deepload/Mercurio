"""
MercurioAI NewsAPI Agent - Real-time news data for trading strategies

This module provides a specialized agent for fetching and analyzing news data
from NewsAPI.org to enhance trading decisions with real-time market sentiment.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class NewsAPIAgent:
    """
    Agent for fetching and analyzing news from NewsAPI.org
    to enhance trading decisions with real-time news sentiment.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 lookback_hours: int = 24,
                 use_fallback: bool = True):
        """
        Initialize the NewsAPI agent
        
        Args:
            api_key: NewsAPI.org API key
            lookback_hours: Hours of news to analyze
            use_fallback: Whether to use fallback data if API fails
        """
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        self.lookback_hours = lookback_hours
        self.use_fallback = use_fallback
        self.base_url = "https://newsapi.org/v2"
        
        # Cache to avoid repeated API calls for the same symbol
        self.news_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # Cache news for 1 hour
        
        logger.info(f"NewsAPI Agent initialized with lookback of {lookback_hours} hours")
        
    def get_news_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get relevant news for a trading symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'ETH/USDT')
            
        Returns:
            List of news articles
        """
        # Check cache first
        now = datetime.now().timestamp()
        if symbol in self.news_cache and symbol in self.cache_expiry:
            if now < self.cache_expiry[symbol]:
                logger.debug(f"Using cached news for {symbol}")
                return self.news_cache[symbol]
        
        # Handle crypto vs stock symbols
        search_terms = self._get_search_terms(symbol)
        
        # Calculate time window
        from_date = (datetime.now() - timedelta(hours=self.lookback_hours)).strftime('%Y-%m-%dT%H:%M:%S')
        
        try:
            # Make API request
            logger.info(f"Fetching news for {symbol} with search terms: {search_terms}")
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    'q': search_terms,
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': self.api_key
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Update cache
                self.news_cache[symbol] = articles
                self.cache_expiry[symbol] = now + self.cache_duration
                
                logger.info(f"Found {len(articles)} news articles for {symbol}")
                return articles
            else:
                logger.warning(f"NewsAPI request failed: {response.status_code}, {response.text}")
                
                if self.use_fallback:
                    return self._get_fallback_news(symbol)
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {str(e)}")
            if self.use_fallback:
                return self._get_fallback_news(symbol)
            return []
    
    def _get_search_terms(self, symbol: str) -> str:
        """Generate appropriate search terms based on the symbol"""
        # Clean up the symbol
        clean_symbol = symbol.replace('/', ' ')
        base_currency = clean_symbol.split()[0]
        
        # Handle crypto symbols
        if 'USD' in symbol or 'USDT' in symbol:
            return f"({base_currency} OR {base_currency} crypto OR {base_currency} cryptocurrency OR {base_currency} price)"
        
        # Handle stock symbols - could expand with company name lookup
        return f"({symbol} OR {symbol} stock OR {symbol} company)"
    
    def _get_fallback_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Return fallback news data when API fails"""
        # Check if we have cached data even if expired
        if symbol in self.news_cache:
            logger.info(f"Using expired cached news for {symbol} as fallback")
            return self.news_cache[symbol]
            
        logger.info(f"Using synthetic news data for {symbol}")
        # Generate synthetic news data
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol
        
        # Create a minimal synthetic article
        return [{
            "source": {"name": "MercurioAI Fallback"},
            "author": "System",
            "title": f"Market data for {base_currency}",
            "description": f"No recent news found for {base_currency}. Trading will rely on technical analysis.",
            "url": "",
            "urlToImage": "",
            "publishedAt": datetime.now().isoformat(),
            "content": f"No recent news found for {base_currency}. Trading will rely on technical analysis."
        }]
        
    async def analyze_sentiment(self, articles: List[Dict[str, Any]], llm_agent) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles using LLM
        
        Args:
            articles: List of news articles
            llm_agent: LLM agent for sentiment analysis
            
        Returns:
            Sentiment analysis results
        """
        if not articles:
            return {"sentiment": "neutral", "confidence": 0.5, "sources": 0}
        
        # Prepare content for LLM analysis
        content = self._prepare_news_for_llm(articles)
        
        # Use LLM to analyze sentiment
        prompt = self._create_sentiment_analysis_prompt(content)
        
        try:
            # Call LLM for analysis
            response = await llm_agent.generate_response(prompt)
            
            # Extract sentiment from LLM response
            sentiment_data = self._extract_sentiment_from_response(response)
            sentiment_data["sources"] = len(articles)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with LLM: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.5, "sources": len(articles)}
    
    def _prepare_news_for_llm(self, articles: List[Dict[str, Any]]) -> str:
        """Prepare news articles for LLM analysis"""
        # Limit to most recent articles if there are too many
        recent_articles = articles[:10]
        
        # Format articles for LLM consumption
        formatted_articles = []
        for i, article in enumerate(recent_articles):
            formatted_articles.append(
                f"Article {i+1}:\n"
                f"Title: {article.get('title', 'No title')}\n"
                f"Source: {article.get('source', {}).get('name', 'Unknown')}\n"
                f"Published: {article.get('publishedAt', 'Unknown')}\n"
                f"Description: {article.get('description', 'No description')}\n"
            )
        
        return "\n\n".join(formatted_articles)
    
    def _create_sentiment_analysis_prompt(self, content: str) -> str:
        """Create prompt for LLM sentiment analysis"""
        return f"""
        Analyze the following news articles for market sentiment:
        
        {content}
        
        Based on these articles, determine the overall market sentiment for the related asset.
        Provide your analysis in the following JSON format:
        {{
            "sentiment": "[bullish/bearish/neutral]",
            "confidence": [0.0-1.0 confidence score],
            "reasoning": "[brief explanation of your analysis]"
        }}
        
        Only respond with the JSON object, no additional text.
        """
    
    def _extract_sentiment_from_response(self, response: str) -> Dict[str, Any]:
        """Extract sentiment data from LLM response"""
        try:
            # Try to parse JSON from response
            import json
            import re
            
            # Extract JSON object if embedded in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Map sentiment to trade action
                sentiment_map = {
                    "bullish": "BUY",
                    "bearish": "SELL",
                    "neutral": "HOLD"
                }
                
                return {
                    "sentiment": data.get("sentiment", "neutral"),
                    "action": sentiment_map.get(data.get("sentiment", "neutral"), "HOLD"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", "")
                }
            
        except Exception as e:
            logger.error(f"Error extracting sentiment from LLM response: {str(e)}")
        
        # Default response if parsing fails
        return {"sentiment": "neutral", "action": "HOLD", "confidence": 0.5, "reasoning": ""}
