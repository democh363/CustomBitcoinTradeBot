#!/usr/bin/env python
# ====================================================================
# WIF TRADER BOT - Gelişmiş Bitcoin ve WIF Coin Alım-Satım Botu
# ====================================================================
# Bu bot, gerçek piyasa verilerini kullanarak kripto para alım-satımı simüle eder.
# Çoklu teknik göstergeler kullanarak (RSI, EMA, MACD) ve gelişmiş sinyal ağırlıklandırma 
# sistemi ile işlem kararları verir. Stop-loss, take-profit ve trailing stop
# özellikleri ile riski sınırlar ve kârı optimize eder.
# ====================================================================

import os
import time
import ccxt  # Kripto borsalarından veri çekmek için
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import logging
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import ssl

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
logger = logging.getLogger("wif_trader")

class WIFTraderBot:
    """
    WIF Trader Bot sınıfı, gerçek piyasa verilerini kullanarak WIF/USDT 
    alım-satım stratejisini uygular. Çoklu teknik göstergeler, trend analizi
    ve volatilite adaptasyonu ile akıllı işlem kararları verir.
    
    Önemli Özellikler:
    - Gelişmiş sinyal ağırlıklandırma sistemi
    - Dinamik pozisyon boyutlandırma
    - Volatiliteye uyum sağlayan adaptif stratejiler
    - Trailing stop ve akıllı kâr alma stratejileri
    - Destek/direnç analizi ile güçlü alım/satım noktaları belirleme
    - Düzenli e-posta bildirimleri ile durum takibi
    """
    def __init__(self, exchange_id='binance', symbol='WIF/USDT', timeframe='5m', initial_balance=1000):
        """
        WIF trading bot'un başlatılması ve temel ayarların yapılması
        
        Parameters:
        - exchange_id: Veri çekilecek borsa (default: 'binance')
        - symbol: İşlem çifti (default: 'WIF/USDT')
        - timeframe: Analiz için zaman dilimi (default: '5m')
        - initial_balance: Başlangıç USDT bakiyesi (default: 1000)
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Simulation mode with real market data
        logger.info(f"Running in simulation mode with real market data for {symbol} from CCXT")
        self.demo_mode = True
        
        # Initialize exchange connection for data
        try:
            # Connect to exchange without API keys (public data only)
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'timeout': 30000,  # 30 seconds timeout
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
            'WIF': 0.0
        }
            
        # Bot state and configuration
        self.price_data = None
        self.last_operation = None
        self.portfolio_value = initial_balance
        self.trade_history = []
        self.initial_portfolio_value = initial_balance
        
        # Trading parameters (optimized)
        self.rsi_period = 14
        self.rsi_overbought = 70    # Standard value
        self.rsi_oversold = 30      # Standard value
        self.ema_short = 9
        self.ema_long = 21
        self.stop_loss_percentage = 0.03  # Daha sıkı stop-loss (5% -> 3%)
        self.take_profit_percentage = 0.06  # Daha hızlı kar alma (10% -> 6%)
        self.trailing_stop_enabled = True  # Trailing stop etkinleştir
        self.trailing_stop_activation = 0.02  # %2 kar olunca trailing stop başlasın
        self.trailing_stop_distance = 0.015  # %1.5 mesafeden takip et
        self.position_sizing = True  # Pozisyon boyutlandırma aktif
        self.max_position_size = 0.8  # Maksimum pozisyon büyüklüğü (total balance'ın %80'i)
        
        # Email notification settings
        self.email_notifications = True
        self.notification_email = "yustun355@gmail.com"
        self.email_interval_hours = 1
        self.last_email_time = None
        
    def fetch_balance(self):
        """Fetch account balance (simulated)"""
        logger.info("Simulation: Fetching balance")
        return {'USDT': {'free': self.demo_balance['USDT']}, 'WIF': {'free': self.demo_balance['WIF']}}
            
    def fetch_market_data(self, limit=100):
        """
        CCXT kütüphanesini kullanarak borsadan piyasa verilerini çeker
        
        İşleyiş:
        1. Öncelikle Binance veya seçilen borsadan veri çekmeyi dener
        2. Başarısız olursa yedek olarak başka bir borsayı dener
        3. O da başarısız olursa CoinGecko API'den veri çeker
        4. Tüm kaynaklar başarısız olursa, tutarlı test için sentetik veri üretir
        
        Bu yaklaşım, botun her durumda çalışmaya devam etmesini sağlar
        """
        try:
            logger.info(f"Fetching real market data from {self.exchange_id}")
            
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
                if timeframe == '5m':
                    timeframe = '5m'  # Default is already 5m
            
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
            
            try:
                if self.exchange_id != fallback_exchange:
                    logger.info(f"Trying fallback exchange: {fallback_exchange}")
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
                
                # If fallback exchange is not available or we're already using it, try CoinGecko
                logger.info("Trying CoinGecko API")
                url = "https://api.coingecko.com/api/v3/coins/dogwifhat/market_chart"
                
                # Convert timeframe to days for CoinGecko
                days = 30
                if self.timeframe == '5m':
                    days = 1  # 1 day for 5-minute data
                
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily' if self.timeframe != '5m' else 'hourly'  # Get hourly data for 5m approximation
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
            except Exception as data_error:
                logger.error(f"Error fetching data from alternative sources: {data_error}")
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
            current_price = self.get_current_wif_price()
        except:
            current_price = None
            
        latest_price = current_price or 0.47  # Reasonable WIF price if current price is unavailable
        
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
    
    def get_current_wif_price(self):
        """Get current WIF price"""
        try:
            # First try using the exchange
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except:
            # Fallback to CoinGecko
            try:
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    'ids': 'dogwifhat',
                    'vs_currencies': 'usd'
                }
                response = requests.get(url, params=params)
                data = response.json()
                return data['dogwifhat']['usd']
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
        wif_amount = amount_usdt / price
        
        # Update balances
        self.demo_balance['USDT'] -= amount_usdt
        self.demo_balance['WIF'] += wif_amount
        
        logger.info(f"Simulation: BUY {wif_amount:.2f} WIF at ${price:.4f}")
                
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'buy',
            'price': price,
            'amount': wif_amount,
            'value': amount_usdt,
            'signals': signals['buy_signals']
        }
        self.trade_history.append(trade)
        self.last_operation = 'buy'
        return True
                
    def execute_sell(self, wif_amount=None):
        """Execute a sell order (simulation)"""
        signals = self.get_trading_signals()
        if not signals or not signals['sell_signals']:
            logger.info("No sell signals detected")
            return False
            
        # Get current balance
        wif_balance = self.demo_balance['WIF']
            
        if wif_amount is None:
            wif_amount = wif_balance  # Sell all available WIF
        if wif_amount > wif_balance:
            wif_amount = wif_balance
                
            price = signals['current_price']
        usdt_value = wif_amount * price
            
        # Update balances
        self.demo_balance['USDT'] += usdt_value
        self.demo_balance['WIF'] -= wif_amount
        
        logger.info(f"Simulation: SELL {wif_amount:.2f} WIF at ${price:.4f}")
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'type': 'sell',
            'price': price,
            'amount': wif_amount,
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
        
        # Check stop loss - standard stop loss
        if price_change < -self.stop_loss_percentage:
            logger.info(f"Stop loss triggered: {price_change:.2%} loss")
            self.execute_sell()
            return True
            
        # Trailing stop loss
        if self.trailing_stop_enabled and price_change > self.trailing_stop_activation:
            # Calculate highest price since buy
            highest_price = buy_price
            for trade_index in range(len(self.trade_history) - 1, -1, -1):
                trade = self.trade_history[trade_index]
                if trade['type'] == 'buy' and trade == last_buy:
                    break
                if 'trailing_high' in trade:  # Check if we've recorded a trailing high previously
                    highest_price = trade['trailing_high']
                    break
            
            # Update highest price seen if current price is higher
            if current_price > highest_price:
                highest_price = current_price
                # Record this new high in trade history
                self.trade_history.append({
                    'timestamp': datetime.now(),
                    'type': 'trailing_update', 
                    'trailing_high': highest_price
                })
                logger.info(f"Updated trailing stop high: ${highest_price:.4f}")
            
            # Check if we're below our trailing stop
            trailing_stop_price = highest_price * (1 - self.trailing_stop_distance)
            if current_price < trailing_stop_price:
                logger.info(f"Trailing stop triggered: Current ${current_price:.4f} below stop ${trailing_stop_price:.4f}")
                self.execute_sell()
                return True
                
        # Check take profit - reduced from original to take profits quicker
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
        wif_balance = self.demo_balance['WIF']
                    
        wif_value = wif_balance * current_price
        total_value = usdt_balance + wif_value
            
        # Calculate profit/loss percentage
        profit_loss = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value) * 100
                    
        logger.info(f"Portfolio: {wif_balance:.2f} WIF (${wif_value:.2f}) + ${usdt_balance:.2f} USDT = ${total_value:.2f}")
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
                
            plt.title('WIF Price with EMAs')
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
            running_wif = 0
            running_usdt = self.initial_portfolio_value
            
            # Add initial point
            portfolio_times.append(self.price_data.index[0])
            portfolio_values.append(self.initial_portfolio_value)
            
            for trade in self.trade_history:
                if trade['type'] == 'buy':
                    running_wif += trade['amount']
                    running_usdt -= trade['value']
                else:  # sell
                    running_wif -= trade['amount']
                    running_usdt += trade['value']
                
                # Add trade point
                portfolio_times.append(trade['timestamp'])
                portfolio_values.append(running_usdt + running_wif * trade['price'])
            
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
        """
        Ticaret botunun tek bir döngüsünü çalıştırır
        
        İşleyiş:
        1. Piyasa verilerini çeker ve indikatörleri hesaplar
        2. Volatiliteye göre strateji ayarlamaları yapar
        3. Stop-loss ve take-profit durumlarını kontrol eder
        4. Tüm sinyalleri toplayıp ağırlıklı karar verir
        5. Piyasa trendini dikkate alarak kararı filtreler
        6. Gerekli alım/satım işlemlerini gerçekleştirir
        7. Portföy değerini hesaplayıp performans grafiğini günceller
        8. E-posta bildirim zamanı geldiyse rapor gönderir
        
        Her iterasyon bağımsız çalışır, böylece sürekli piyasa değişimlerine
        adapte olur ve yeni fırsatları/riskleri değerlendirir
        """
        logger.info("Starting trading iteration")
        
        try:
            # Update market data
            self.fetch_market_data()
            
            # Calculate indicators
            if self.calculate_indicators():
                # Volatilite ölçümü ve strateji ayarı
                self.adjust_for_market_volatility()
                
                # Check stop loss / take profit
                if not self.check_stop_loss_take_profit():
                    # Get enhanced signals with all indicators
                    enhanced_signals = self.get_enhanced_trading_signals()
                    
                    if enhanced_signals:
                        # Evaluate signals with weights to make a decision
                        decision = self.evaluate_signals_with_weights(enhanced_signals)
                        
                        logger.info(f"Trading decision based on weighted signals: {decision.upper()}")
                        
                        # Market trend bazlı filtre
                        market_trend = enhanced_signals.get('market_trend', 'unknown')
                        
                        # Trendle uyumlu işlem yap
                        if decision == 'buy':
                            # Düşüş trendinde sadece güçlü sinyallerde al
                            if market_trend == 'downtrend':
                                # RSI veya StochRSI aşırı satım bölgesinde mi kontrol et
                                strong_oversold = False
                                for signal in enhanced_signals['buy_signals']:
                                    if 'RSI oversold' in signal or 'StochRSI oversold' in signal:
                                        strong_oversold = True
                                        break
                                
                                if not strong_oversold:
                                    logger.info(f"Skipping buy in downtrend without strong oversold signal")
                                    decision = 'hold'
                        elif decision == 'sell':
                            # Yükseliş trendinde sadece güçlü sinyallerde sat
                            if market_trend == 'uptrend':
                                # RSI veya StochRSI aşırı alım bölgesinde mi kontrol et
                                strong_overbought = False
                                for signal in enhanced_signals['sell_signals']:
                                    if 'RSI overbought' in signal or 'StochRSI overbought' in signal:
                                        strong_overbought = True
                                        break
                                        
                                if not strong_overbought:
                                    logger.info(f"Skipping sell in uptrend without strong overbought signal")
                                    decision = 'hold'
                        
                        # Execute trade based on decision
                        if decision == 'buy' and self.last_operation != 'buy':
                            logger.info("Multiple indicators suggest BUY, executing buy order")
                            self.execute_buy_weighted()  # Use weighted buy
                        elif decision == 'sell' and self.last_operation != 'sell':
                            logger.info("Multiple indicators suggest SELL, executing sell order")
                            self.execute_sell_weighted()  # Use weighted sell
                        else:
                            logger.info(f"Decision is to {decision.upper()}, holding position")
                    else:
                        logger.warning("No enhanced signals available")
            
            # Calculate portfolio value
            self.calculate_portfolio_value()
            
            # Update performance chart
            self.plot_performance()
            
            # Check if it's time to send an email update
            self.check_email_notification()
            
            logger.info("Trading iteration completed")
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            # Generate synthetic data if we had an error (likely with data fetch)
            if self.price_data is None:
                logger.info("Generating synthetic data for testing purposes")
                self.generate_synthetic_data()
                self.calculate_indicators()
                self.calculate_portfolio_value()
                self.plot_performance()
            logger.info("Trading iteration completed with fallback to synthetic data")
    
    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        df = self.price_data.copy()
        df['middle_band'] = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        df['upper_band'] = df['middle_band'] + (rolling_std * num_std)
        df['lower_band'] = df['middle_band'] - (rolling_std * num_std)
        
        # Bollinger Band genişliği - volatilite göstergesi
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        return df

    def analyze_volume(self):
        """Analyze volume patterns"""
        df = self.price_data.copy()
        # Ortalama hacim hesapla
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        # Hacim artışı/azalışı
        df['volume_change'] = df['volume'] / df['volume_sma']
        
        # Fiyat-hacim ilişkisi (hacimle doğrulanan fiyat hareketleri)
        df['price_up_volume_up'] = ((df['close'] > df['close'].shift(1)) & 
                                   (df['volume'] > df['volume'].shift(1))).astype(int)
        
        return df

    def get_market_sentiment(self):
        """Crypto market sentiment analysis"""
        try:
            # Fear & Greed Index API
            url = "https://api.alternative.me/fng/"
            response = requests.get(url)
            data = response.json()
            fear_greed_value = int(data['data'][0]['value'])
            fear_greed_classification = data['data'][0]['value_classification']
            
            logger.info(f"Fear & Greed Index: {fear_greed_value} ({fear_greed_classification})")
            
            # Extreme fear: alım fırsatı olabilir
            # Extreme greed: satım fırsatı olabilir
            return {
                'fear_greed_value': fear_greed_value,
                'fear_greed_classification': fear_greed_classification
            }
        except:
            logger.warning("Sentiment data could not be fetched")
            return None

    def calculate_fibonacci_levels(self, trend="up", lookback_period=100):
        """Calculate Fibonacci retracement levels"""
        df = self.price_data.copy().iloc[-lookback_period:]
        
        if trend == "up":
            # Yükselen trend için, minimum ve maksimum fiyat
            price_min = df['low'].min()
            price_max = df['high'].max()
        else:
            # Düşen trend için, maksimum ve minimum
            price_max = df['high'].max()
            price_min = df['low'].min()
        
        # Fibonacci seviyeleri hesapla
        diff = price_max - price_min
        levels = {
            '0.0': price_min,
            '0.236': price_min + 0.236 * diff,
            '0.382': price_min + 0.382 * diff,
            '0.5': price_min + 0.5 * diff,
            '0.618': price_min + 0.618 * diff,
            '0.786': price_min + 0.786 * diff,
            '1.0': price_max
        }
        
        return levels

    def find_support_resistance(self, window=10):
        """Find support and resistance levels"""
        df = self.price_data.copy()
        support_levels = []
        resistance_levels = []
        
        # Pivot noktaları bul
        for i in range(window, len(df) - window):
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                support_levels.append(df['low'].iloc[i])
                
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                resistance_levels.append(df['high'].iloc[i])
        
        # Yakın seviyeleri birleştir
        support_levels = self._consolidate_levels(support_levels)
        resistance_levels = self._consolidate_levels(resistance_levels)
        
        return support_levels, resistance_levels

    def multi_timeframe_analysis(self):
        """Analyze multiple timeframes"""
        timeframes = ['1h', '4h', '1d'] 
        signals = {}
        
        for tf in timeframes:
            original_tf = self.timeframe
            self.timeframe = tf
            
            # Bu zaman dilimi için verileri çek
            self.fetch_market_data()
            self.calculate_indicators()
            
            # Sinyalleri kaydet
            signals[tf] = self.get_trading_signals()
            
            # Orijinal zaman dilimine geri dön
            self.timeframe = original_tf
        
        # Farklı zaman dilimlerinden gelen sinyalleri birleştir
        consensus = self._calculate_signal_consensus(signals)
        return consensus

    def calculate_mfi(self, period=14):
        """Calculate Money Flow Index"""
        df = self.price_data.copy()
        
        # Tipik fiyat
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Para akışı
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # Pozitif ve negatif para akışı
        df['positive_flow'] = np.where(df['typical_price'] > df['typical_price'].shift(1), df['money_flow'], 0)
        df['negative_flow'] = np.where(df['typical_price'] < df['typical_price'].shift(1), df['money_flow'], 0)
        
        # Belirli periyotta pozitif ve negatif para akışı toplamı
        positive_mf = df['positive_flow'].rolling(window=period).sum()
        negative_mf = df['negative_flow'].rolling(window=period).sum()
        
        # Money Ratio ve MFI hesapla
        df['money_ratio'] = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + df['money_ratio']))
        
        return df

    def calculate_stochastic_rsi(self, period=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic RSI"""
        df = self.price_data.copy()
        
        # Önce RSI hesapla
        if 'rsi' not in df.columns:
            self.calculate_indicators()
            df = self.price_data.copy()
        
        # StochRSI hesapla
        df['stoch_rsi'] = (df['rsi'] - df['rsi'].rolling(period).min()) / \
                          (df['rsi'].rolling(period).max() - df['rsi'].rolling(period).min())
        
        # K ve D çizgileri
        df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=smooth_k).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=smooth_d).mean()
        
        return df

    def calculate_adx(self, period=14):
        """Calculate Average Directional Index"""
        df = self.price_data.copy()
        
        # True Range hesapla
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Directional Movement hesapla
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Directional Indicators
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
        
        # Directional Index
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # Average Directional Index
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df

    def get_macro_factors(self):
        """Get macroeconomic factors that could impact crypto"""
        try:
            # Örnek: Fed faiz oranı, enflasyon, DXY (dolar endeksi) gibi verileri API'lerden çekme
            # Bu örnek için basitleştirilmiş
            dxy_url = "https://some-api.com/dxy"  # Gerçek bir API URL'si kullanın
            dxy_response = requests.get(dxy_url)
            dxy_data = dxy_response.json()
            
            # Dolar endeksi genellikle kripto ile ters korelasyon gösterir
            if 'value' in dxy_data:
                dxy_value = dxy_data['value']
                dxy_change = dxy_data['change']
                
                logger.info(f"Dollar Index (DXY): {dxy_value} (Change: {dxy_change}%)")
                
                # DXY düşüyorsa, kripto alım sinyalini güçlendir
                return {
                    'dxy_value': dxy_value,
                    'dxy_change': dxy_change,
                    'dxy_signal': 'buy' if dxy_change < 0 else 'sell'
                }
        except:
            logger.warning("Macroeconomic data could not be fetched")
        
        return None

    def get_enhanced_trading_signals(self):
        """Get enhanced buy/sell signals with multiple indicators"""
        # Temel sinyalleri al
        base_signals = self.get_trading_signals()
        if not base_signals:
            return None
        
        # Gelişmiş indikatörler hesapla
        bollinger_data = self.calculate_bollinger_bands()
        volume_data = self.analyze_volume()
        sentiment_data = self.get_market_sentiment()
        adx_data = self.calculate_adx()
        mfi_data = self.calculate_mfi()
        
        # Ek analizler
        stoch_rsi_data = self.calculate_stochastic_rsi()
        
        # Market trend analizi
        market_trend = self.identify_market_trend()
        
        current_price = base_signals['current_price']
        
        # Ek sinyaller
        additional_buy = []
        additional_sell = []
        
        # Bollinger Bant sinyalleri
        if current_price < bollinger_data['lower_band'].iloc[-1]:
            additional_buy.append("Price below lower Bollinger Band")
        if current_price > bollinger_data['upper_band'].iloc[-1]:
            additional_sell.append("Price above upper Bollinger Band")
        
        # Bollinger Bant sıkışması (volatilite azaldığında ortaya çıkabilecek büyük hareket öncesi)
        bollinger_width = bollinger_data['bb_width'].iloc[-1]
        bollinger_width_prev = bollinger_data['bb_width'].iloc[-2]
        
        if bollinger_width < 0.03:  # Çok dar bantlar
            if bollinger_width < bollinger_width_prev:
                additional_buy.append(f"Bollinger Squeeze: Volatility contraction ({bollinger_width:.3f})")
        
        # Hacim sinyalleri
        if volume_data['volume_change'].iloc[-1] > 1.5 and current_price > volume_data['close'].iloc[-2]:
            additional_buy.append("Strong volume supporting price increase")
        
        # Hacimle fiyat ilişkisi
        if volume_data['volume_change'].iloc[-1] > 1.8 and volume_data['price_up_volume_up'].iloc[-1] == 1:
            additional_buy.append("Price-volume confirmation (strong buying)")
        
        # ADX sinyalleri (trend gücü)
        adx_value = adx_data['adx'].iloc[-1]
        if adx_value > 25:
            if adx_data['plus_di'].iloc[-1] > adx_data['minus_di'].iloc[-1]:
                additional_buy.append(f"Strong uptrend (ADX: {adx_value:.2f})")
            else:
                additional_sell.append(f"Strong downtrend (ADX: {adx_value:.2f})")
        elif adx_value < 20:
            # Zayıf trend, range-bound market - trend izleme yerine momentum stratejileri
            if current_price < bollinger_data['lower_band'].iloc[-1]:
                additional_buy.append(f"Range-bound market - lower band bounce (ADX: {adx_value:.2f})")
            if current_price > bollinger_data['upper_band'].iloc[-1]:
                additional_sell.append(f"Range-bound market - upper band resistance (ADX: {adx_value:.2f})")
        
        # MFI sinyalleri
        if mfi_data['mfi'].iloc[-1] < 20:
            additional_buy.append(f"MFI oversold ({mfi_data['mfi'].iloc[-1]:.2f})")
        if mfi_data['mfi'].iloc[-1] > 80:
            additional_sell.append(f"MFI overbought ({mfi_data['mfi'].iloc[-1]:.2f})")
        
        # Sentiment sinyalleri
        if sentiment_data and 'fear_greed_value' in sentiment_data:
            if sentiment_data['fear_greed_value'] < 25:  # Extreme fear
                additional_buy.append(f"Market sentiment: Extreme Fear ({sentiment_data['fear_greed_value']})")
            elif sentiment_data['fear_greed_value'] > 75:  # Extreme greed
                additional_sell.append(f"Market sentiment: Extreme Greed ({sentiment_data['fear_greed_value']})")
        
        # StochRSI sinyalleri - daha hassas al/sat noktaları için
        if stoch_rsi_data['stoch_rsi_k'].iloc[-1] < 0.2:
            additional_buy.append(f"StochRSI oversold ({stoch_rsi_data['stoch_rsi_k'].iloc[-1]:.2f})")
        if stoch_rsi_data['stoch_rsi_k'].iloc[-1] > 0.8:
            additional_sell.append(f"StochRSI overbought ({stoch_rsi_data['stoch_rsi_k'].iloc[-1]:.2f})")
        
        # StochRSI çaprazlamaları
        if (stoch_rsi_data['stoch_rsi_k'].iloc[-2] < stoch_rsi_data['stoch_rsi_d'].iloc[-2] and 
            stoch_rsi_data['stoch_rsi_k'].iloc[-1] > stoch_rsi_data['stoch_rsi_d'].iloc[-1]):
            additional_buy.append("StochRSI bullish crossover")
        if (stoch_rsi_data['stoch_rsi_k'].iloc[-2] > stoch_rsi_data['stoch_rsi_d'].iloc[-2] and 
            stoch_rsi_data['stoch_rsi_k'].iloc[-1] < stoch_rsi_data['stoch_rsi_d'].iloc[-1]):
            additional_sell.append("StochRSI bearish crossover")
            
        # Market trend değerlendirmesi
        if market_trend == 'uptrend':
            additional_buy.append("Overall market in uptrend")
        elif market_trend == 'downtrend':
            additional_sell.append("Overall market in downtrend")
        
        # Geri çekilmeler / pullbacklar
        if market_trend == 'uptrend' and current_price < bollinger_data['middle_band'].iloc[-1]:
            additional_buy.append("Pullback in uptrend (buying opportunity)")
        
        # Fiyat formasyonları ve destek/direnç seviyeleri
        supports, resistances = self.find_support_resistance()
        
        # En yakın destek/direnç seviyesini bul
        closest_support = None
        closest_resistance = None
        
        if supports:
            # Fiyatın altındaki en yakın destek
            supports_below = [s for s in supports if s < current_price]
            if supports_below:
                closest_support = max(supports_below)
        
        if resistances:
            # Fiyatın üstündeki en yakın direnç
            resistances_above = [r for r in resistances if r > current_price]
            if resistances_above:
                closest_resistance = min(resistances_above)
        
        # Destek/direnç sinyalleri
        if closest_support and (current_price - closest_support) / current_price < 0.02:
            additional_buy.append(f"Price near support level (${closest_support:.4f})")
        
        if closest_resistance and (closest_resistance - current_price) / current_price < 0.02:
            additional_sell.append(f"Price near resistance level (${closest_resistance:.4f})")
            
        # Divergence analizi
        price_action_last_two = bollinger_data['close'].iloc[-2:].values
        rsi_action_last_two = bollinger_data['rsi'].iloc[-2:].values
        
        # Bearish divergence: Fiyat yükselirken RSI düşüyor
        if price_action_last_two[1] > price_action_last_two[0] and rsi_action_last_two[1] < rsi_action_last_two[0]:
            additional_sell.append("Bearish divergence detected (price up, RSI down)")
            
        # Bullish divergence: Fiyat düşerken RSI yükseliyor
        if price_action_last_two[1] < price_action_last_two[0] and rsi_action_last_two[1] > rsi_action_last_two[0]:
            additional_buy.append("Bullish divergence detected (price down, RSI up)")
        
        # Tüm sinyalleri birleştir
        enhanced_signals = {
            'current_price': current_price,
            'buy_signals': base_signals['buy_signals'] + additional_buy,
            'sell_signals': base_signals['sell_signals'] + additional_sell,
            'market_trend': market_trend
        }
        
        return enhanced_signals
        
    def identify_market_trend(self):
        """
        Genel piyasa trendini belirleyen fonksiyon
        
        İşleyiş:
        1. EMA'ları kontrol eder (kısa > uzun = yukarı trend)
        2. Son mum hareketlerini analiz eder (ardışık yükselişler/düşüşler)
        3. Fiyatın ortalamaya göre konumunu değerlendirir 
        4. ADX göstergesi ile trend gücünü ölçer
        5. Tüm faktörleri birleştirerek 'uptrend', 'downtrend' veya 'sideways' kararı verir
        
        Trend belirleme, trende karşı işlem yapma riskini azaltır
        """
        if self.price_data is None:
            return 'unknown'
            
        df = self.price_data.copy()
        
        # EMA'ları kontrol et
        short_above_long = df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1]
        
        # Son X mum trend yönü
        last_candles = 5
        last_n_closes = df['close'].iloc[-last_candles:].values
        closes_increasing = all(last_n_closes[i] <= last_n_closes[i+1] for i in range(len(last_n_closes)-1))
        closes_decreasing = all(last_n_closes[i] >= last_n_closes[i+1] for i in range(len(last_n_closes)-1))
        
        # Fiyat ve ortalama ilişkisi
        price_above_ema = df['close'].iloc[-1] > df['ema_long'].iloc[-1]
        
        # ADX ile trend gücü
        adx_data = self.calculate_adx()
        adx_value = adx_data['adx'].iloc[-1]
        plus_di = adx_data['plus_di'].iloc[-1]
        minus_di = adx_data['minus_di'].iloc[-1]
        
        strong_trend = adx_value > 25
        bullish_di = plus_di > minus_di
        
        # Fiyat ve ortalama momentumu
        momentum_bullish = (df['close'].iloc[-1] - df['close'].iloc[-10]) > 0
        
        # Çoklu faktörlere dayalı trend belirleme
        if (short_above_long and price_above_ema and 
            (closes_increasing or (strong_trend and bullish_di)) and momentum_bullish):
            return 'uptrend'
        elif (not short_above_long and not price_above_ema and 
              (closes_decreasing or (strong_trend and not bullish_di)) and not momentum_bullish):
            return 'downtrend'
        else:
            return 'sideways'
    
    def adjust_for_market_volatility(self):
        """
        Piyasa volatilitesine göre stratejiyi dinamik olarak ayarlar
        
        İşleyiş:
        1. Bollinger bant genişliği ile volatiliteyi ölçer
        2. ATR göstergesi ile fiyat hareketliliğini hesaplar
        3. Yüksek volatilitede:
           - Daha sıkı stop-loss (%2.5)
           - Daha küçük pozisyon boyutu (max %60)
        4. Düşük volatilitede:
           - Daha geniş stop-loss (%4)
           - Daha büyük pozisyon boyutu (max %80)
        
        Volatiliteye adaptasyon, farklı piyasa koşullarında riskin 
        daha iyi yönetilmesini sağlar
        """
        if self.price_data is None:
            return
        
        # Bollinger bant genişliği ölçümü
        bollinger_data = self.calculate_bollinger_bands()
        current_volatility = bollinger_data['bb_width'].iloc[-1]
        
        # ATR (Average True Range) hesaplama
        df = self.price_data.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        atr_volatility = df['atr'].iloc[-1] / df['close'].iloc[-1]  # Normalize to price
        
        # Volatilite değerlendirmesi
        high_volatility = current_volatility > 0.05 or atr_volatility > 0.03
        low_volatility = current_volatility < 0.02 and atr_volatility < 0.015
        
        # Strateji ayarlamaları
        if high_volatility:
            logger.info(f"High volatility environment detected (BB: {current_volatility:.3f}, ATR: {atr_volatility:.3f})")
            # Daha sıkı stop-loss ve daha küçük pozisyon boyutu
            self.stop_loss_percentage = 0.025  # %2.5
            self.trailing_stop_distance = 0.01  # %1
            self.max_position_size = 0.6  # Max %60 kullanım
        elif low_volatility:
            logger.info(f"Low volatility environment detected (BB: {current_volatility:.3f}, ATR: {atr_volatility:.3f})")
            # Volatilite düşükken daha geniş stop-loss (breakout beklentisi)
            self.stop_loss_percentage = 0.04  # %4
            self.trailing_stop_distance = 0.02  # %2
            self.max_position_size = 0.8  # Max %80 kullanım
        else:
            # Normal volatilite - varsayılan değerlere dön
            logger.info(f"Normal volatility environment (BB: {current_volatility:.3f}, ATR: {atr_volatility:.3f})")
            self.stop_loss_percentage = 0.03  # %3
            self.trailing_stop_distance = 0.015  # %1.5
            self.max_position_size = 0.7  # Max %70 kullanım
            
        # Volatilite bazlı ek sinyal
        return {
            "volatility_level": "high" if high_volatility else "low" if low_volatility else "normal",
            "bb_width": current_volatility,
            "atr_ratio": atr_volatility
        }
        
    def evaluate_signals_with_weights(self, signals):
        """
        Farklı teknik göstergelere ağırlık vererek alım/satım kararı verir
        
        İşleyiş:
        1. Her göstergeye farklı ağırlıklar atar (RSI: 2.5, Bollinger: 1.8 gibi)
        2. Özel durumlarda ekstra ağırlık verir (örn. RSI aşırı satım durumunda 1.5x)
        3. Buy/sell skorlarını hesaplar ve eşik değerlerle karşılaştırır
        4. Daha seçici işlemler için yüksek eşik değerleri kullanır
        5. Pozisyonda olma durumuna göre kâr alma eğilimini ayarlar
        
        Bu ağırlıklandırma sistemi, daha akıllı ve daha az hatalı kararlar verir
        """
        if not signals:
            return None
        
        buy_score = 0
        sell_score = 0
        
        # Sinyal ağırlıkları - optimize edilmiş değerler
        weights = {
            'RSI': 2.5,         # RSI'a daha fazla ağırlık (1.5 -> 2.5)
            'EMA': 1.2,         # Biraz artırıldı (1.0 -> 1.2)
            'MACD': 1.0,        # Azaltıldı (1.2 -> 1.0)
            'Bollinger': 1.8,   # Artırıldı (1.3 -> 1.8)
            'ADX': 1.7,         # Trend gücüne daha fazla ağırlık (1.1 -> 1.7)
            'MFI': 1.5,         # Artırıldı (1.2 -> 1.5)
            'Sentiment': 0.6,   # Azaltıldı (0.8 -> 0.6) 
            'Volume': 1.3       # Hacime daha fazla ağırlık (0.9 -> 1.3)
        }
        
        # Sinyalleri değerlendir ve puan hesapla
        for signal in signals['buy_signals']:
            if 'RSI oversold' in signal:  # RSI oversold'a özel artırılmış ağırlık
                buy_score += weights['RSI'] * 1.5
            elif 'RSI' in signal:
                buy_score += weights['RSI']
            elif 'EMA' in signal:
                buy_score += weights['EMA']
            elif 'MACD' in signal:
                buy_score += weights['MACD']
            elif 'Bollinger' in signal:
                buy_score += weights['Bollinger']
            elif 'Strong uptrend' in signal:  # Güçlü yükselen trende daha fazla ağırlık
                buy_score += weights['ADX'] * 1.3
            elif 'ADX' in signal:
                buy_score += weights['ADX']
            elif 'MFI oversold' in signal:  # MFI aşırı satıma özel artırılmış ağırlık
                buy_score += weights['MFI'] * 1.4
            elif 'MFI' in signal:
                buy_score += weights['MFI']
            elif 'sentiment' in signal.lower():
                buy_score += weights['Sentiment']
            elif 'volume' in signal.lower():
                buy_score += weights['Volume']
            else:
                buy_score += 0.5  # Diğer sinyaller
        
        for signal in signals['sell_signals']:
            if 'RSI overbought' in signal:  # RSI overbought'a özel artırılmış ağırlık
                sell_score += weights['RSI'] * 1.5
            elif 'RSI' in signal:
                sell_score += weights['RSI']
            elif 'EMA' in signal:
                sell_score += weights['EMA']
            elif 'MACD' in signal:
                sell_score += weights['MACD']
            elif 'Bollinger' in signal:
                sell_score += weights['Bollinger']
            elif 'Strong downtrend' in signal:  # Güçlü düşen trende daha fazla ağırlık
                sell_score += weights['ADX'] * 1.3
            elif 'ADX' in signal:
                sell_score += weights['ADX'] 
            elif 'MFI overbought' in signal:  # MFI aşırı alıma özel artırılmış ağırlık
                sell_score += weights['MFI'] * 1.4
            elif 'MFI' in signal:
                sell_score += weights['MFI']
            elif 'sentiment' in signal.lower():
                sell_score += weights['Sentiment']
            elif 'volume' in signal.lower():
                sell_score += weights['Volume']
            else:
                sell_score += 0.5  # Diğer sinyaller
        
        # Kararı belirle
        buy_threshold = 2.5   # Artırıldı (2.0 -> 2.5) - Daha seçici alımlar için
        sell_threshold = 2.8  # Artırıldı (2.0 -> 2.8) - Daha seçici satışlar için
        
        # Zarar durdurma ve kar alma eğilimini ayarla
        if self.last_operation == 'buy':
            # Eğer zaten alım yapılmışsa, satımlar için daha düşük bir eşik değeri kullan
            # Bu, pozisyonda kar varsa daha çabuk satma eğilimi sağlar
            sell_threshold = 2.2
            
        if buy_score > sell_score and buy_score >= buy_threshold:
            return 'buy'
        elif sell_score > buy_score and sell_score >= sell_threshold:
            return 'sell'
        else:
            return 'hold'

    def _consolidate_levels(self, levels, threshold_percent=1.0):
        """Helper method to consolidate close price levels"""
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        consolidated = []
        
        current_level = sorted_levels[0]
        
        for level in sorted_levels[1:]:
            # If this level is close to the current one, skip it
            if (level - current_level) / current_level * 100 < threshold_percent:
                continue
            else:
                consolidated.append(current_level)
                current_level = level
                
        # Add the last level
        consolidated.append(current_level)
        
        return consolidated
    
    def _calculate_signal_consensus(self, timeframe_signals):
        """Calculate consensus across multiple timeframes"""
        buy_count = 0
        sell_count = 0
        
        for tf, signals in timeframe_signals.items():
            if not signals:
                continue
                
            if len(signals['buy_signals']) > len(signals['sell_signals']):
                buy_count += 1
            elif len(signals['sell_signals']) > len(signals['buy_signals']):
                sell_count += 1
                
        # Create consensus signals
        consensus = {
            'buy_signals': [],
            'sell_signals': [],
            'current_price': timeframe_signals[self.timeframe]['current_price'] if self.timeframe in timeframe_signals else 0
        }
        
        if buy_count > sell_count:
            consensus['buy_signals'].append(f"Multi-timeframe consensus ({buy_count}/{len(timeframe_signals)} timeframes)")
        elif sell_count > buy_count:
            consensus['sell_signals'].append(f"Multi-timeframe consensus ({sell_count}/{len(timeframe_signals)} timeframes)")
            
        return consensus
        
    def execute_buy_weighted(self, amount_usdt=None):
        """
        Gelişmiş alım stratejisi uygulayan fonksiyon
        
        İşleyiş:
        1. Ağırlıklı sinyal değerlendirmesi sonucu alım kararı verilmişse
        2. Sinyal gücüne göre dinamik pozisyon boyutlandırma yapar
        3. Güçlü sinyaller daha büyük pozisyon alımına neden olur
        4. Yüksek volatilite durumunda pozisyon boyutunu otomatik azaltır
        5. Maksimum pozisyon boyutunu sınırlar (ana bakiyenin %80'i gibi)
        
        Dinamik pozisyon boyutlandırma, güçlü fırsatlarda daha çok kazanç
        sağlarken riski de dengeler
        """
        enhanced_signals = self.get_enhanced_trading_signals()
        decision = self.evaluate_signals_with_weights(enhanced_signals)
        
        if decision != 'buy':
            logger.info(f"Weighted decision is {decision}, not buying")
            return False
        
        # Get current balance
        usdt_balance = self.demo_balance['USDT']
        
        if amount_usdt is None:
            # Dynamic position sizing based on signal strength
            buy_score = 0
            for signal in enhanced_signals['buy_signals']:
                if 'RSI oversold' in signal:
                    buy_score += 1
                if 'Strong uptrend' in signal:
                    buy_score += 1
                if 'EMA bullish' in signal:
                    buy_score += 0.5
                if 'MACD bullish' in signal:
                    buy_score += 0.5
                if 'Price below lower Bollinger' in signal:
                    buy_score += 1
                
            # Normalize score between 0.4 and self.max_position_size
            normalized_score = min(max(0.4 + (buy_score / 10), 0.4), self.max_position_size)
            
            # Use smaller position sizes in very volatile markets
            if self.position_sizing and 'bb_width' in self.price_data.columns:
                volatility = self.price_data['bb_width'].iloc[-1]
                if volatility > 0.05:  # High volatility threshold
                    normalized_score *= 0.8  # Reduce position size by 20%
                    logger.info(f"High volatility detected ({volatility:.2f}), reducing position size")
                    
            amount_usdt = usdt_balance * normalized_score
            logger.info(f"Position sizing: Using {normalized_score:.2%} of available balance")
        
        if amount_usdt > usdt_balance:
            amount_usdt = usdt_balance
            
        price = enhanced_signals['current_price']
        wif_amount = amount_usdt / price
        
        # Update balances
        self.demo_balance['USDT'] -= amount_usdt
        self.demo_balance['WIF'] += wif_amount
        
        logger.info(f"Simulation: WEIGHTED BUY {wif_amount:.2f} WIF at ${price:.4f}")
        
        # Record trade with all signals
        trade = {
            'timestamp': datetime.now(),
            'type': 'buy',
            'price': price,
            'amount': wif_amount,
            'value': amount_usdt,
            'signals': enhanced_signals['buy_signals'],
            'decision': 'weighted_signals'
        }
        self.trade_history.append(trade)
        self.last_operation = 'buy'
        return True
        
    def execute_sell_weighted(self, wif_amount=None):
        """Execute a sell order with enhanced signals"""
        enhanced_signals = self.get_enhanced_trading_signals()
        decision = self.evaluate_signals_with_weights(enhanced_signals)
        
        if decision != 'sell':
            logger.info(f"Weighted decision is {decision}, not selling")
            return False
        
        # Get current balance
        wif_balance = self.demo_balance['WIF']
        
        if wif_amount is None:
            wif_amount = wif_balance  # Sell all available WIF
        if wif_amount > wif_balance:
            wif_amount = wif_balance
            
        price = enhanced_signals['current_price']
        usdt_value = wif_amount * price
        
        # Update balances
        self.demo_balance['USDT'] += usdt_value
        self.demo_balance['WIF'] -= wif_amount
        
        logger.info(f"Simulation: WEIGHTED SELL {wif_amount:.2f} WIF at ${price:.4f}")
        
        # Record trade with all signals
        trade = {
            'timestamp': datetime.now(),
            'type': 'sell',
            'price': price,
            'amount': wif_amount,
            'value': usdt_value,
            'signals': enhanced_signals['sell_signals'],
            'decision': 'weighted_signals'
        }
        self.trade_history.append(trade)
        self.last_operation = 'sell'
        return True

    def send_portfolio_email(self):
        """
        Portföy performans raporunu e-posta olarak gönderir
        
        İşleyiş:
        1. HTML formatında güzel bir e-posta şablonu oluşturur
        2. Mevcut bakiye, toplam değer, kar/zarar bilgilerini ekler
        3. Son işlemleri detaylı bir tablo halinde gösterir
        4. Performans grafiğini ek olarak ekler
        5. Gmail SMTP üzerinden belirlenen adrese gönderir
        
        Düzenli e-posta bildirimleri, uzaktan takibi kolaylaştırır
        """
        try:
            # Create email content
            msg = MIMEMultipart()
            msg['Subject'] = f'WIF Trader Bot - Portfolio Update {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            msg['From'] = 'wiftraderbot@notification.com'  # Use a valid sender if needed
            msg['To'] = self.notification_email
            
            # Get portfolio information
            current_price = 0
            if self.price_data is not None and len(self.price_data) > 0:
                current_price = self.price_data['close'].iloc[-1]
                
            usdt_balance = self.demo_balance['USDT']
            wif_balance = self.demo_balance['WIF']
            wif_value = wif_balance * current_price
            total_value = usdt_balance + wif_value
            profit_loss = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value) * 100
            
            # Create email body
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 10px; text-align: center; }}
                    .container {{ padding: 20px; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .profit {{ color: green; }}
                    .loss {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>WIF Trader Bot - Portfolio Report</h2>
                </div>
                <div class="container">
                    <h3>Portfolio Summary</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>WIF Balance</td>
                            <td>{wif_balance:.2f} WIF (${wif_value:.2f})</td>
                        </tr>
                        <tr>
                            <td>USDT Balance</td>
                            <td>${usdt_balance:.2f}</td>
                        </tr>
                        <tr>
                            <td>Total Portfolio Value</td>
                            <td>${total_value:.2f}</td>
                        </tr>
                        <tr>
                            <td>Current WIF Price</td>
                            <td>${current_price:.4f}</td>
                        </tr>
                        <tr>
                            <td>Profit/Loss</td>
                            <td class="{'profit' if profit_loss >= 0 else 'loss'}">{profit_loss:.2f}%</td>
                        </tr>
                        <tr>
                            <td>Initial Investment</td>
                            <td>${self.initial_portfolio_value:.2f}</td>
                        </tr>
                    </table>
                    
                    <h3>Recent Trade Activity</h3>
                    <table>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>Price</th>
                            <th>Amount</th>
                            <th>Value</th>
                        </tr>
            """
            
            # Add last 5 trades (or less if fewer exist)
            recent_trades = []
            for trade in reversed(self.trade_history):
                if trade['type'] in ['buy', 'sell']:  # Only include actual trades
                    recent_trades.append(trade)
                    if len(recent_trades) >= 5:
                        break
                        
            for trade in reversed(recent_trades):  # Show most recent first
                trade_type = trade['type'].upper()
                trade_time = trade['timestamp'].strftime("%H:%M:%S")
                trade_price = f"${trade['price']:.4f}"
                trade_amount = f"{trade['amount']:.2f}"
                trade_value = f"${trade['value']:.2f}"
                
                html += f"""
                        <tr>
                            <td>{trade_time}</td>
                            <td>{trade_type}</td>
                            <td>{trade_price}</td>
                            <td>{trade_amount}</td>
                            <td>{trade_value}</td>
                        </tr>
                """
            
            html += """
                    </table>
                    <p>This is an automated message from your WIF Trader Bot.</p>
                </div>
            </body>
            </html>
            """
            
            # Attach HTML content
            msg.attach(MIMEText(html, 'html'))
            
            # Attach performance chart if it exists
            if os.path.exists('trading_performance.png'):
                with open('trading_performance.png', 'rb') as img_file:
                    img_data = img_file.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-ID', '<performance_chart>')
                    image.add_header('Content-Disposition', 'inline', filename='trading_performance.png')
                    msg.attach(image)
            
            # Configure email server with TLS
            context = ssl.create_default_context()
            
            # Try to send email via Gmail SMTP
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls(context=context)
                
                # Note: For this to work, you would need to use an app password if using Gmail
                # You can set this up in the .env file or directly here
                email_password = os.getenv('EMAIL_PASSWORD', '')
                email_user = os.getenv('EMAIL_USER', 'your-email@gmail.com')
                
                if email_password:
                    server.login(email_user, email_password)
                    server.send_message(msg)
                    server.quit()
                    logger.info(f"Portfolio email sent to {self.notification_email}")
                else:
                    # Fallback to logging if no password is set
                    logger.warning("Email not sent: No email password provided in environment variables")
                    logger.info("To enable email, add EMAIL_USER and EMAIL_PASSWORD to your .env file")
            except Exception as email_error:
                logger.error(f"Failed to send email via Gmail: {email_error}")
                # Try alternate method with a different provider if desired
                
            self.last_email_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error sending portfolio email: {e}")
    
    def check_email_notification(self):
        """Check if it's time to send a portfolio update email"""
        if not self.email_notifications:
            return
            
        current_time = datetime.now()
        
        # Send email on first run
        if self.last_email_time is None:
            self.send_portfolio_email()
            return
            
        # Check if interval has passed since last email
        time_diff = current_time - self.last_email_time
        if time_diff.total_seconds() >= (self.email_interval_hours * 3600):
            self.send_portfolio_email()

if __name__ == "__main__":
    # Create and run the trading bot
    try:
        bot = WIFTraderBot(
            exchange_id='binance',  # Use binance for data
            symbol='WIF/USDT',
            timeframe='5m',         # Set to 5-minute timeframe
            initial_balance=1000    # 1000 USDT initial balance
        )
        
        # Run trading bot for 6 hours with 15-second checks
        print("=" * 60)
        print("WIF TRADER BOT - ENHANCED SIMULATION (6-HOUR ANALYSIS)")
        print("=" * 60)
        print("Features enabled:")
        print("- Technical indicators (RSI, EMA, MACD)")
        print("- Bollinger Bands for volatility analysis")
        print("- Volume analysis for trend confirmation")
        print("- Price pattern recognition with support/resistance") 
        print("- Market sentiment analysis")
        print("- Money Flow Index (MFI)")
        print("- ADX for trend strength")
        print("- Stochastic RSI for overbought/oversold")
        print("- Weighted decision making system")
        print("- Hourly portfolio email reports to yustun355@gmail.com")
        print("-" * 60)
        print("Settings:")
        print("- Timeframe: 5-minute candles")
        print("- Check interval: Every 15 seconds")
        print("- Analysis duration: 6 hours")
        print("-" * 60)
        print("Initial Portfolio Value: $1000.00")
        print("-" * 60)
        
        # Calculate number of iterations for 6 hours with 15-second intervals
        # 6 hours = 21600 seconds, 21600 / 15 = 1440 iterations
        total_iterations = 1440
        
        # Try to fetch data initially, but use synthetic data if it fails
        try:
            # Fetch initial data
            data = bot.fetch_market_data()
            if data is None or len(data) < 10:
                raise Exception("Insufficient market data")
            print(f"Successfully connected to exchange. Using real market data.")
        except Exception as e:
            print(f"Could not fetch market data: {e}")
            print("Using synthetic data for simulation instead.")
            bot.generate_synthetic_data()
            print("Synthetic data generated successfully.")
        
        # Run first iteration immediately
        bot.run_single_iteration()
        
        # Run remaining iterations with 15-second intervals
        for i in range(1, total_iterations):
            try:
                print(f"\nIteration {i+1}/{total_iterations} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"Time elapsed: {i*15//60} minutes {i*15%60} seconds")
                print(f"Estimated time remaining: {(total_iterations-i)*15//60} minutes {(total_iterations-i)*15%60} seconds")
                
                time.sleep(15)  # Wait for 15 seconds
                bot.run_single_iteration()
                
                # Save progress every hour (240 iterations)
                if i % 240 == 0:
                    print(f"\n{'='*30}")
                    print(f"PROGRESS UPDATE - {i//240} hour(s) completed")
                    print(f"Current portfolio value: ${bot.portfolio_value:.2f}")
                    profit_loss = ((bot.portfolio_value - 1000) / 1000) * 100
                    print(f"Current profit/loss: {profit_loss:.2f}%")
                    print(f"Trades executed: {len(bot.trade_history)}")
                    print(f"{'='*30}\n")
            except Exception as e:
                print(f"Error in iteration {i+1}: {e}")
                print("Continuing with next iteration...")
                continue
        
        # Final portfolio results
        final_value = bot.portfolio_value
        profit_loss_amount = final_value - 1000
        profit_loss_percent = (profit_loss_amount / 1000) * 100
        
        print("\n" + "=" * 60)
        print("ENHANCED TRADING SIMULATION RESULTS (6-HOUR ANALYSIS)")
        print("=" * 60)
        print(f"Initial Portfolio Value: $1000.00")
        print(f"Final Portfolio Value:   ${final_value:.2f}")
        print(f"Profit/Loss:             ${profit_loss_amount:.2f} ({profit_loss_percent:.2f}%)")
        print("=" * 60)
        print(f"Trading activity: {len(bot.trade_history)} trades")
        
        # Calculate trading statistics
        buy_trades = [t for t in bot.trade_history if t['type'] == 'buy']
        sell_trades = [t for t in bot.trade_history if t['type'] == 'sell']
        
        print(f"Total buy trades:  {len(buy_trades)}")
        print(f"Total sell trades: {len(sell_trades)}")
        
        # Calculate average trade statistics if there are trades
        if bot.trade_history:
            avg_holding_time = "N/A"
            
            # Calculate holding times for complete buy-sell cycles
            holding_times = []
            for i in range(len(bot.trade_history)-1):
                if bot.trade_history[i]['type'] == 'buy' and bot.trade_history[i+1]['type'] == 'sell':
                    buy_time = bot.trade_history[i]['timestamp']
                    sell_time = bot.trade_history[i+1]['timestamp']
                    holding_time = (sell_time - buy_time).total_seconds() / 60  # in minutes
                    holding_times.append(holding_time)
            
            if holding_times:
                avg_holding_time = sum(holding_times) / len(holding_times)
                print(f"Average holding time: {avg_holding_time:.2f} minutes")
        
        # Print trade details
        if bot.trade_history:
            print("\nTRADE HISTORY:")
            print("-" * 60)
            for i, trade in enumerate(bot.trade_history):
                print(f"Trade #{i+1} - {trade['type'].upper()} at ${trade['price']:.4f}")
                print(f"  Amount: {trade['amount']:.2f} WIF")
                print(f"  Value: ${trade['value']:.2f}")
                print(f"  Decision: {'Weighted Analysis' if 'decision' in trade and trade['decision'] == 'weighted_signals' else 'Basic Signal'}")
                print(f"  Signals:")
                for signal in trade['signals']:
                    print(f"    - {signal}")
                print(f"  Time: {trade['timestamp']}")
                print("-" * 30)
        
        print("\nCheck trading_performance.png for detailed charts")
    
    except Exception as e:
        print(f"Error running trading bot: {e}")
        print("Try running with synthetic data only by modifying the code.") 