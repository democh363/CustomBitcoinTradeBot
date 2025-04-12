#!/usr/bin/env python
import os
import time
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bitcoin_trader")

class BitcoinTraderBot:
    def __init__(self, exchange_id='binance', symbol='BTC/USDT', timeframe='1h', initial_balance=1000):
        """
        Initialize the Bitcoin trading bot
        
        Parameters:
        - exchange_id: The exchange to use for data (default: 'binance')
        - symbol: The trading pair (default: 'BTC/USDT')
        - timeframe: The timeframe for analysis (default: '1h')
        - initial_balance: Initial USDT balance for simulation (default: 1000)
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Simulation mode with real market data
        logger.info("Running in simulation mode with real market data from CCXT")
        self.demo_mode = True
        
        # Initialize exchange connection for data
        try:
            # Connect to exchange without API keys (public data only)
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # Use spot market
                }
            })
            logger.info(f"Connected to {self.exchange_id} for market data")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise
        
        # Initial portfolio for simulation
        self.demo_balance = {
            'USDT': initial_balance,
            'BTC': 0.0
        }
        
        # Bot state and configuration
        self.price_data = None
        self.last_operation = None
        self.portfolio_value = initial_balance
        self.trade_history = []
        self.initial_portfolio_value = initial_balance
        
        # Trading parameters (can be adjusted)
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.ema_short = 9
        self.ema_long = 21
        self.stop_loss_percentage = 0.05  # 5%
        self.take_profit_percentage = 0.1  # 10%
        
    def fetch_balance(self):
        """Fetch account balance (simulated)"""
        logger.info("Simulation: Fetching balance")
        return {'USDT': {'free': self.demo_balance['USDT']}, 'BTC': {'free': self.demo_balance['BTC']}}
            
    def fetch_market_data(self, limit=100):
        """Fetch market data from CCXT"""
        try:
            logger.info(f"Fetching real market data from {self.exchange_id}")
            
            try:
                # Get OHLCV data from exchange
                if not self.exchange.has['fetchOHLCV']:
                    raise Exception(f"{self.exchange_id} does not support fetchOHLCV")
                
                # Ensure exchange is loaded
                if not self.exchange.markets:
                    self.exchange.load_markets()
                    
                # Fetch OHLCV data
                since = None  # Default to most recent data
                timeframe = self.timeframe
                
                # Some exchanges might require different timeframe format
                if self.exchange_id == 'binance':
                    if timeframe == '1h':
                        timeframe = '1h'  # Default is already 1h
                
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, since, limit)
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Make sure we have enough data
                if len(df) < limit / 2:
                    raise Exception(f"Not enough data returned from {self.exchange_id}, got {len(df)} candles")
                
                logger.info(f"Market data fetched successfully from {self.exchange_id}: {len(df)} candles")
                self.price_data = df
                return df
                
            except Exception as e:
                logger.error(f"Error fetching from {self.exchange_id}: {e}")
                # Try a different exchange if the first one fails
                fallback_exchange = 'kucoin'
                if self.exchange_id != fallback_exchange:
                    logger.info(f"Trying fallback exchange: {fallback_exchange}")
                    try:
                        exchange_class = getattr(ccxt, fallback_exchange)
                        fallback = exchange_class({'enableRateLimit': True})
                        if not fallback.has['fetchOHLCV']:
                            raise Exception(f"{fallback_exchange} does not support fetchOHLCV")
                            
                        fallback.load_markets()
                        ohlcv = fallback.fetch_ohlcv(self.symbol, self.timeframe, None, limit)
                        
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        logger.info(f"Market data fetched successfully from {fallback_exchange}: {len(df)} candles")
                        self.price_data = df
                        return df
                    except Exception as fallback_error:
                        logger.error(f"Error with fallback exchange: {fallback_error}")
                
                # If all exchanges fail, try fetching from CoinGecko
                logger.info("Trying CoinGecko API")
                try:
                    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
                    
                    # Convert timeframe to days for CoinGecko
                    days = 30
                    if self.timeframe == '1h':
                        days = 5  # 5 days for hourly data
                    
                    params = {
                        'vs_currency': 'usd',
                        'days': days,
                        'interval': 'daily' if self.timeframe != '1h' else 'hourly'
                    }
                    
                    response = requests.get(url, params=params)
                    data = response.json()
                    
                    # Process price data
                    prices = data['prices']  # [[timestamp, price], ...]
                    volumes = data['total_volumes']  # [[timestamp, volume], ...]
                    
                    # Create DataFrame
                    price_df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                    volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    
                    # Convert timestamps from milliseconds
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
                    volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
                    
                    # Merge price and volume data
                    df = pd.merge(price_df, volume_df, on='timestamp')
                    
                    # Add missing OHLC columns using close price with small variations
                    df['open'] = df['close'] * np.random.uniform(0.99, 1.01, size=len(df))
                    df['high'] = df['close'] * np.random.uniform(1.01, 1.03, size=len(df))
                    df['low'] = df['close'] * np.random.uniform(0.97, 0.99, size=len(df))
                    
                    # Set timestamp as index
                    df.set_index('timestamp', inplace=True)
                    
                    # Limit to the most recent 'limit' data points
                    if len(df) > limit:
                        df = df.iloc[-limit:]
                    
                    logger.info(f"Market data fetched successfully from CoinGecko: {len(df)} candles")
                    self.price_data = df
                    return df
                except Exception as coingecko_error:
                    logger.error(f"Error fetching from CoinGecko: {coingecko_error}")
                
                # All attempts failed, generate synthetic data as last resort
                logger.warning("All data sources failed. Using synthetic data as last resort.")
                self.generate_synthetic_data(limit)
                return self.price_data
                
        except Exception as e:
            logger.error(f"Error fetching market data from all sources: {e}")
            logger.warning("Falling back to synthetic data generation")
            self.generate_synthetic_data(limit)
            return self.price_data
    
    def generate_synthetic_data(self, limit=100):
        """Generate synthetic price data when real data is unavailable"""
        logger.info("Generating synthetic market data")
        
        # Create date range and seed with current time
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=self.timeframe)
        
        # Try to get current price, or use a reasonable estimate
        try:
            current_price = self.get_current_btc_price()
        except:
            current_price = None
            
        latest_price = current_price or 65000  # Reasonable BTC price if current price is unavailable
        
        # Generate random seed based on current time
        seed = int(time.time()) % 10000
        np.random.seed(seed)
        
        # Generate price movement with trend and volatility
        trend = np.random.choice([-1, 1])  # Random trend direction
        volatility_factor = 0.05  # 5% volatility
        
        # Generate price changes
        price_changes = np.zeros(limit)
        for i in range(1, limit):
            # Random walk model with trend
            random_change = np.random.normal(0, latest_price * volatility_factor)
            trend_change = trend * np.random.uniform(0, latest_price * 0.01)  # Up to 1% trend effect
            price_changes[i] = price_changes[i-1] + random_change + trend_change
        
        # Calculate prices
        close_prices = latest_price + price_changes
        
        # Ensure prices are positive and reasonable
        close_prices = np.maximum(close_prices, latest_price * 0.5)
        
        # Create OHLCV DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * np.random.uniform(0.99, 1.01, size=limit),
            'high': close_prices * np.random.uniform(1.01, 1.03, size=limit),
            'low': close_prices * np.random.uniform(0.97, 0.99, size=limit),
            'close': close_prices,
            'volume': np.random.uniform(100, 1000, size=limit) * 10
        })
        df.set_index('timestamp', inplace=True)
        
        self.price_data = df
        logger.info(f"Synthetic market data generated: {len(df)} candles")
        return df
    
    def get_current_btc_price(self):
        """Get current BTC price"""
        try:
            # First try using the exchange
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except:
            # Fallback to CoinGecko
            try:
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': 'bitcoin',
                    'vs_currencies': 'usd'
                }
                response = requests.get(url, params=params)
                data = response.json()
                return data['bitcoin']['usd']
            except:
                return None
            
    def calculate_indicators(self):
        """Calculate technical indicators for trading decisions"""
        if self.price_data is None or len(self.price_data) < self.rsi_period:
            logger.warning("Not enough data to calculate indicators")
            return False
            
        df = self.price_data.copy()
        
        # Calculate RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate EMAs (Exponential Moving Averages)
        df['ema_short'] = df['close'].ewm(span=self.ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long, adjust=False).mean()
        
        # Calculate MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_short'] - df['ema_long']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        self.price_data = df
        logger.info("Technical indicators calculated successfully")
        return True
        
    def get_trading_signals(self):
        """Get buy/sell signals based on technical indicators"""
        if self.price_data is None or 'rsi' not in self.price_data.columns:
            logger.warning("Indicators not calculated, cannot generate signals")
            return None
            
        df = self.price_data.iloc[-1]  # Get the latest data point
        current_price = df['close']
        rsi = df['rsi']
        ema_short = df['ema_short']
        ema_long = df['ema_long']
        macd = df['macd']
        macd_signal = df['macd_signal']
        
        # Define signals
        buy_signals = []
        sell_signals = []
        
        # RSI signals
        if rsi < self.rsi_oversold:
            buy_signals.append(f"RSI oversold ({rsi:.2f})")
        if rsi > self.rsi_overbought:
            sell_signals.append(f"RSI overbought ({rsi:.2f})")
            
        # EMA crossover signals
        if ema_short > ema_long:
            buy_signals.append("EMA bullish crossover")
        if ema_short < ema_long:
            sell_signals.append("EMA bearish crossover")
            
        # MACD signals
        if macd > macd_signal:
            buy_signals.append("MACD bullish")
        if macd < macd_signal:
            sell_signals.append("MACD bearish")
            
        logger.info(f"Current price: ${current_price:.2f}, RSI: {rsi:.2f}")
        logger.info(f"Buy signals: {buy_signals}")
        logger.info(f"Sell signals: {sell_signals}")
        
        return {
            'current_price': current_price,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
        
    def execute_buy(self, amount_usdt=None):
        """Execute a buy order (simulation)"""
        signals = self.get_trading_signals()
        if not signals or not signals['buy_signals']:
            logger.info("No buy signals detected")
            return False
            
        # Get current balance
        usdt_balance = self.demo_balance['USDT']
        
        if amount_usdt is None:
            amount_usdt = usdt_balance * 0.9  # Use 90% of available USDT
        if amount_usdt > usdt_balance:
            amount_usdt = usdt_balance
            
        price = signals['current_price']
        btc_amount = amount_usdt / price
        
        # Update balances
        self.demo_balance['USDT'] -= amount_usdt
        self.demo_balance['BTC'] += btc_amount
        
        logger.info(f"Simulation: BUY {btc_amount:.6f} BTC at ${price:.2f}")
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'buy',
            'price': price,
            'amount': btc_amount,
            'value': amount_usdt,
            'signals': signals['buy_signals']
        }
        self.trade_history.append(trade)
        self.last_operation = 'buy'
        return True
    
    def execute_sell(self, btc_amount=None):
        """Execute a sell order (simulation)"""
        signals = self.get_trading_signals()
        if not signals or not signals['sell_signals']:
            logger.info("No sell signals detected")
            return False
            
        # Get current balance
        btc_balance = self.demo_balance['BTC']
        
        if btc_amount is None:
            btc_amount = btc_balance  # Sell all available BTC
        if btc_amount > btc_balance:
            btc_amount = btc_balance
            
        price = signals['current_price']
        usdt_value = btc_amount * price
        
        # Update balances
        self.demo_balance['USDT'] += usdt_value
        self.demo_balance['BTC'] -= btc_amount
        
        logger.info(f"Simulation: SELL {btc_amount:.6f} BTC at ${price:.2f}")
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'sell',
            'price': price,
            'amount': btc_amount,
            'value': usdt_value,
            'signals': signals['sell_signals']
        }
        self.trade_history.append(trade)
        self.last_operation = 'sell'
        return True
    
    def check_stop_loss_take_profit(self):
        """Check if stop loss or take profit conditions are met"""
        if not self.trade_history or self.last_operation != 'buy':
            return False
            
        last_buy = None
        for trade in reversed(self.trade_history):
            if trade['type'] == 'buy':
                last_buy = trade
                break
                
        if not last_buy:
            return False
            
        signals = self.get_trading_signals()
        current_price = signals['current_price']
        buy_price = last_buy['price']
        
        # Calculate price change percentage
        price_change = (current_price - buy_price) / buy_price
        
        # Check stop loss
        if price_change < -self.stop_loss_percentage:
            logger.info(f"Stop loss triggered: {price_change:.2%} loss")
            self.execute_sell()
            return True
            
        # Check take profit
        if price_change > self.take_profit_percentage:
            logger.info(f"Take profit triggered: {price_change:.2%} gain")
            self.execute_sell()
            return True
            
        return False
    
    def calculate_portfolio_value(self):
        """Calculate current portfolio value"""
        signals = self.get_trading_signals()
        if signals:
            current_price = signals['current_price']
            usdt_balance = self.demo_balance['USDT']
            btc_balance = self.demo_balance['BTC']
            
            btc_value = btc_balance * current_price
            total_value = usdt_balance + btc_value
            
            # Calculate profit/loss percentage
            profit_loss = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value) * 100
            
            logger.info(f"Portfolio: {btc_balance:.6f} BTC (${btc_value:.2f}) + ${usdt_balance:.2f} USDT = ${total_value:.2f}")
            logger.info(f"Profit/Loss: {profit_loss:.2f}%")
            
            self.portfolio_value = total_value
            return total_value
                
        return 0
    
    def plot_performance(self):
        """Plot trading performance and indicators"""
        if not self.price_data is None and len(self.price_data) > 0:
            plt.figure(figsize=(12, 10))
            
            # Price chart
            ax1 = plt.subplot(4, 1, 1)
            self.price_data['close'].plot(ax=ax1, color='blue', label='Close Price')
            self.price_data['ema_short'].plot(ax=ax1, color='red', label=f'EMA {self.ema_short}')
            self.price_data['ema_long'].plot(ax=ax1, color='green', label=f'EMA {self.ema_long}')
            
            # Plot buy/sell points
            buy_times = []
            buy_prices = []
            sell_times = []
            sell_prices = []
            
            for trade in self.trade_history:
                if trade['type'] == 'buy':
                    buy_times.append(trade['timestamp'])
                    buy_prices.append(trade['price'])
                else:
                    sell_times.append(trade['timestamp'])
                    sell_prices.append(trade['price'])
            
            if buy_times:
                plt.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy')
            if sell_times:
                plt.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell')
                
            plt.title('Bitcoin Price with EMAs')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid()
            
            # RSI chart
            ax2 = plt.subplot(4, 1, 2)
            self.price_data['rsi'].plot(ax=ax2, color='purple', label='RSI')
            plt.axhline(y=self.rsi_overbought, color='red', linestyle='--', label=f'Overbought ({self.rsi_overbought})')
            plt.axhline(y=self.rsi_oversold, color='green', linestyle='--', label=f'Oversold ({self.rsi_oversold})')
            plt.title('Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.legend()
            plt.grid()
            
            # MACD chart
            ax3 = plt.subplot(4, 1, 3)
            self.price_data['macd'].plot(ax=ax3, color='blue', label='MACD')
            self.price_data['macd_signal'].plot(ax=ax3, color='red', label='Signal Line')
            plt.title('Moving Average Convergence Divergence (MACD)')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.legend()
            plt.grid()
            
            # Portfolio Value Chart
            ax4 = plt.subplot(4, 1, 4)
            
            # Create portfolio value history
            portfolio_times = []
            portfolio_values = []
            
            # Calculate portfolio value at each trade
            running_btc = 0
            running_usdt = self.initial_portfolio_value
            
            # Add initial point
            portfolio_times.append(self.price_data.index[0])
            portfolio_values.append(self.initial_portfolio_value)
            
            for trade in self.trade_history:
                if trade['type'] == 'buy':
                    running_btc += trade['amount']
                    running_usdt -= trade['value']
                else:  # sell
                    running_btc -= trade['amount']
                    running_usdt += trade['value']
                
                # Add trade point
                portfolio_times.append(trade['timestamp'])
                portfolio_values.append(running_usdt + running_btc * trade['price'])
            
            # Add final point at current value
            if self.portfolio_value > 0:
                portfolio_times.append(datetime.now())
                portfolio_values.append(self.portfolio_value)
            
            # Plot portfolio value
            if portfolio_times and portfolio_values:
                plt.plot(portfolio_times, portfolio_values, color='orange', label='Portfolio Value')
                
                # Add horizontal line for initial value
                plt.axhline(y=self.initial_portfolio_value, color='black', linestyle='--', 
                           label=f'Initial Value (${self.initial_portfolio_value:.2f})')
                
                plt.title('Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Value (USD)')
                plt.legend()
                plt.grid()
            
            plt.tight_layout()
            plt.savefig('trading_performance.png')
            logger.info("Performance chart saved to trading_performance.png")
            plt.close()
    
    def run_single_iteration(self):
        """Run a single iteration of the trading bot"""
        logger.info("Starting trading iteration")
        
        # Update market data
        self.fetch_market_data()
        
        # Calculate indicators
        if self.calculate_indicators():
            # Check stop loss / take profit
            if not self.check_stop_loss_take_profit():
                # Get signals
                signals = self.get_trading_signals()
                
                if signals:
                    # Daha agresif ticaret stratejisi - sadece bir sinyal olması yeterli
                    if signals['buy_signals'] and self.last_operation != 'buy':
                        logger.info("Buy signal detected, executing buy order")
                        self.execute_buy()
                    elif signals['sell_signals'] and self.last_operation != 'sell':
                        logger.info("Sell signal detected, executing sell order")
                        self.execute_sell()
                    else:
                        logger.info("No clear trading signal, holding position")
        
        # Calculate portfolio value
        self.calculate_portfolio_value()
        
        # Update performance chart
        self.plot_performance()
        
        logger.info("Trading iteration completed")


if __name__ == "__main__":
    # Create and run the trading bot
    bot = BitcoinTraderBot(
        exchange_id='binance',  # Use binance for data
        symbol='BTC/USDT',
        timeframe='1h',
        initial_balance=1000  # 1000 USDT initial balance
    )
    
    # Run trading bot for 20 iterations with 1-minute intervals
    print("Starting 20-minute trading simulation...")
    print("Initial Portfolio Value: $1000.00")
    print("-" * 50)
    
    # Run first iteration immediately
    bot.run_single_iteration()
    
    # Run 19 more iterations with 1-minute intervals
    for i in range(19):
        print(f"\nWaiting 1 minute for next iteration ({i+2}/20)...")
        time.sleep(60)  # Wait for 1 minute
        bot.run_single_iteration()
    
    # Final portfolio results
    final_value = bot.portfolio_value
    profit_loss_amount = final_value - 1000
    profit_loss_percent = (profit_loss_amount / 1000) * 100
    
    print("\n" + "=" * 50)
    print("TRADING SIMULATION RESULTS (20 MINUTES)")
    print("=" * 50)
    print(f"Initial Portfolio Value: $1000.00")
    print(f"Final Portfolio Value:   ${final_value:.2f}")
    print(f"Profit/Loss:             ${profit_loss_amount:.2f} ({profit_loss_percent:.2f}%)")
    print("=" * 50)
    print(f"Trading activity: {len(bot.trade_history)} trades")
    
    # Print trade details
    if bot.trade_history:
        print("\nTRADE HISTORY:")
        print("-" * 50)
        for i, trade in enumerate(bot.trade_history):
            print(f"Trade #{i+1} - {trade['type'].upper()} at ${trade['price']:.2f}")
            print(f"  Amount: {trade['amount']:.6f} BTC")
            print(f"  Value: ${trade['value']:.2f}")
            print(f"  Signals: {', '.join(trade['signals'])}")
            print(f"  Time: {trade['timestamp']}")
            print("-" * 30)
    
    print("\nCheck trading_performance.png for detailed charts") 