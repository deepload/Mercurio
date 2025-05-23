import os
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Demo configuration
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
PAPER_TRADING = True  # Always use paper trading for demo
INITIAL_CAPITAL = 100000.0
DAYS_HISTORY = 180

# Mock data generation
def generate_mock_data(symbol, days=180):
    """Generate mock stock price data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    
    # Start with a random price between 50 and 500
    start_price = random.uniform(50, 500)
    
    # Generate random price movements with a slight upward bias
    prices = [start_price]
    for i in range(1, len(date_range)):
        # Random daily movement between -3% and +4%
        daily_return = random.uniform(-0.03, 0.04)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": date_range,
        "open": prices,
        "high": [p * random.uniform(1.0, 1.05) for p in prices],
        "low": [p * random.uniform(0.95, 1.0) for p in prices],
        "close": [p * random.uniform(0.98, 1.02) for p in prices],
        "volume": [random.randint(100000, 10000000) for _ in range(len(date_range))]
    })
    
    return df

class MockTradeAction:
    """Enum for trade actions (BUY, SELL, HOLD)"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy for testing"""
    
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        
    async def preprocess_data(self, data):
        """Add moving averages to the data"""
        df = data.copy()
        df["short_ma"] = df["close"].rolling(window=self.short_window).mean()
        df["long_ma"] = df["close"].rolling(window=self.long_window).mean()
        return df
    
    async def predict(self, data):
        """Generate trading signal based on moving average crossover"""
        if len(data) < self.long_window:
            return MockTradeAction.HOLD, 0.5
            
        # Get the latest values
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check for crossover
        if latest["short_ma"] > latest["long_ma"] and previous["short_ma"] <= previous["long_ma"]:
            return MockTradeAction.BUY, 0.8
        elif latest["short_ma"] < latest["long_ma"] and previous["short_ma"] >= previous["long_ma"]:
            return MockTradeAction.SELL, 0.8
        else:
            return MockTradeAction.HOLD, 0.5
    
    async def backtest(self, data, initial_capital=100000.0):
        """Run a simple backtest on the strategy"""
        df = data.dropna().copy()
        
        # Initialize variables
        position = 0  # 0 = no position, 1 = long
        cash = initial_capital
        equity = [initial_capital]
        trades = []
        
        for i in range(1, len(df)):
            current_day = df.iloc[i]
            previous_day = df.iloc[i-1]
            
            # Check for buy signal
            if previous_day["short_ma"] <= previous_day["long_ma"] and current_day["short_ma"] > current_day["long_ma"]:
                if position == 0:
                    # Buy at close price
                    price = current_day["close"]
                    shares = cash / price
                    cash = 0
                    position = 1
                    trades.append({
                        "date": current_day["date"],
                        "action": "BUY",
                        "price": price,
                        "shares": shares
                    })
            
            # Check for sell signal
            elif previous_day["short_ma"] >= previous_day["long_ma"] and current_day["short_ma"] < current_day["long_ma"]:
                if position == 1:
                    # Sell at close price
                    price = current_day["close"]
                    cash = price * shares
                    position = 0
                    trades.append({
                        "date": current_day["date"],
                        "action": "SELL",
                        "price": price,
                        "shares": shares
                    })
            
            # Calculate equity
            current_equity = cash
            if position == 1:
                current_equity = shares * current_day["close"]
            equity.append(current_equity)
        
        # Calculate results
        equity_curve = pd.Series(equity, index=df.index)
        final_capital = equity[-1]
        total_return = (final_capital / initial_capital) - 1
        
        return {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "equity_curve": equity_curve,
            "trades": trades
        }

async def run_simplified_demo():
    """Run a simplified demo with mock data"""
    logger.info("=" * 50)
    logger.info("MERCURIO AI SIMPLIFIED DEMO - Starting")
    logger.info("=" * 50)
    
    # Step 1: Generate mock data
    logger.info("\n\nStep 1: Generating mock market data...")
    data_by_symbol = {}
    for symbol in SYMBOLS:
        logger.info(f"Generating data for {symbol}...")
        data = generate_mock_data(symbol, days=DAYS_HISTORY)
        data_by_symbol[symbol] = data
        logger.info(f"Generated {len(data)} data points for {symbol}")
    
    # Use the first symbol for demonstration
    symbol = SYMBOLS[0]
    data = data_by_symbol[symbol]
    
    # Step 2: Run backtest with moving average strategy
    logger.info("\n\nStep 2: Running backtest with Moving Average strategy...")
    try:
        # Initialize strategy
        ma_strategy = SimpleMovingAverageStrategy(short_window=20, long_window=50)
        
        # Preprocess data
        processed_data = await ma_strategy.preprocess_data(data)
        
        # Run backtest
        logger.info(f"Running backtest on {symbol} with SimpleMovingAverageStrategy...")
        backtest_results = await ma_strategy.backtest(
            processed_data, 
            initial_capital=INITIAL_CAPITAL
        )
        
        # Show results
        logger.info(f"Backtest Results:")
        logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        logger.info(f"Final Capital: ${backtest_results["final_capital"]:.2f}")
        logger.info(f"Total Return: {backtest_results["total_return"]*100:.2f}%")
        logger.info(f"Number of Trades: {len(backtest_results["trades"])}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results["equity_curve"])
        plt.title(f"Moving Average Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        equity_curve_path = f"./data/{symbol}_MA_equity_curve.png"
        plt.savefig(equity_curve_path)
        plt.close()
        logger.info(f"Equity curve saved to {equity_curve_path}")
        
        # Generate a trade recommendation
        logger.info("\n\nStep 3: Generating trade recommendation...")
        action, confidence = await ma_strategy.predict(processed_data)
        logger.info(f"Trade Recommendation for {symbol}:")
        logger.info(f"Action: {action}")
        logger.info(f"Confidence: {confidence:.2f}")
        
        # Show summary
        logger.info("\n\nSummary:")
        logger.info("=" * 50)
        logger.info("This simplified demo demonstrates the core functionality of Mercurio AI:")
        logger.info("1. Data processing and analysis")
        logger.info("2. Strategy backtesting")
        logger.info("3. Trade signal generation")
        logger.info("\nIn the full version, Mercurio AI includes:")
        logger.info("- Real market data integration")
        logger.info("- Multiple sophisticated trading strategies")
        logger.info("- Advanced ML models for price prediction")
        logger.info("- Paper and live trading capabilities")
        logger.info("- Portfolio management and optimization")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in simplified demo: {e}")

if __name__ == "__main__":
    # Run the simplified demo
    asyncio.run(run_simplified_demo())

