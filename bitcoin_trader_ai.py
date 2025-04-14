#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ============================================================================
# BITCOIN TRADER AI - Reinforcement Learning Tabanlı Day Trading Botu
# ============================================================================
# Bu bot, geçmiş Bitcoin verilerini kullanarak Reinforcement Learning ile
# day trading stratejisi geliştirir ve uygular.
# ============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import argparse
import sys
import torch

# Günlük kaydını ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitcoin_trader_ai.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bitcoin_trader_ai")

# Çevre değişkenlerini yükle
load_dotenv()

class BitcoinTradingEnv(gym.Env):
    """
    Bitcoin day trading için Reinforcement Learning ortamı
    
    Bu sınıf, bir kripto para alım-satım simülasyonu oluşturur ve
    OpenAI Gym arayüzünü kullanarak RL algoritmaları ile çalışabilir.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=10000, commission=0.001, window_size=20):
        """
        Trading ortamını başlat
        
        Parametreler:
        - df: OHLCV verileri içeren DataFrame
        - initial_balance: Başlangıç bakiyesi (USDT)
        - commission: İşlem komisyonu (0.001 = %0.1)
        - window_size: Gözlem penceresinin boyutu
        """
        super(BitcoinTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        
        # Aksiyon uzayı: 0 (sat), 1 (tut), 2 (al)
        self.action_space = spaces.Discrete(3)
        
        # Gözlem uzayı: [fiyat değişimi, RSI, MACD, Bollinger Bands, bakiye, portföy değeri] x window_size
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(window_size, 10), dtype=np.float32
        )
        
        # State değişkenleri
        self.balance = None
        self.btc_held = None
        self.current_step = None
        self.current_price = None
        self.portfolio_value = None
        self.trades = None
        self.returns = None
        self.consecutive_same_action = 0
        self.prev_action = None
        self.prev_price_direction = 0  # -1: düşüş, 0: sabit, 1: yükseliş
        self.holding_duration = 0
        
        # Teknik göstergeler
        self._calculate_indicators()
        
        # NaN değerlerini kontrol et ve temizle
        self._clean_data()
        
        # Trend ve destek/direnç analizi için
        self._calculate_trend_support_resistance()
        
    def _clean_data(self):
        """
        NaN değerlerini temizle ve aşırı değerleri kırp
        """
        # NaN değerlerini doldur
        self.df = self.df.fillna(0)
        
        # Sonsuz değerleri sınırla
        for col in self.df.columns:
            if self.df[col].dtype == float or self.df[col].dtype == int:
                # Aşırı değerleri kırp
                if len(self.df[col]) > 0:
                    mean = self.df[col].mean()
                    std = self.df[col].std() if self.df[col].std() != 0 else 1
                    self.df[col] = self.df[col].clip(lower=mean-5*std, upper=mean+5*std)
        
        # Tekrar NaN kontrolü yap
        if self.df.isnull().values.any():
            self.df = self.df.fillna(0)
            logger.warning("Veri temizlendikten sonra hala NaN değerleri var. Sıfır ile dolduruldu.")
    
    def _calculate_trend_support_resistance(self):
        """
        Trend ve destek/direnç seviyelerini hesapla
        """
        # Uzun vadeli trend için EMA
        self.df['ema50'] = self.df['close'].ewm(span=50, adjust=False, min_periods=1).mean()
        self.df['ema200'] = self.df['close'].ewm(span=200, adjust=False, min_periods=1).mean()
        
        # Trend yönü (1: yukarı, -1: aşağı, 0: yatay)
        self.df['trend'] = 0
        self.df.loc[self.df['ema50'] > self.df['ema200'], 'trend'] = 1
        self.df.loc[self.df['ema50'] < self.df['ema200'], 'trend'] = -1
        
        # Destek ve direnç hesaplama
        window = 20
        self.df['support'] = self.df['low'].rolling(window=window, min_periods=1).min()
        self.df['resistance'] = self.df['high'].rolling(window=window, min_periods=1).max()
        
        # Trend güçlülüğü (ADX benzeri)
        self.df['price_change_pct'] = self.df['close'].pct_change(1).fillna(0)
        self.df['trend_strength'] = self.df['price_change_pct'].abs().rolling(window=14, min_periods=1).mean()
        
        # Normalize et
        self.df['trend'] = self.df['trend'] / 2  # -0.5 to 0.5
        self.df['trend_strength'] = np.clip(self.df['trend_strength'] * 100, 0, 1)  # 0 to 1
        
        # Fiyatın destek/dirençlere göre konumu (0-1 arasında)
        price_range = self.df['resistance'] - self.df['support']
        # Sıfıra bölme hatasını önle
        price_range = np.where(price_range == 0, self.df['close'] * 0.01, price_range)
        self.df['sr_position'] = (self.df['close'] - self.df['support']) / price_range
        self.df['sr_position'] = self.df['sr_position'].clip(0, 1)

    def _calculate_indicators(self):
        """
        Teknik göstergeleri hesapla:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Momentum
        - Volatilite
        """
        try:
            # Fiyat değişimini hesapla (NaN değerlerini ele al)
            self.df['price_change'] = self.df['close'].pct_change().fillna(0)
            
            # RSI hesapla (daha güvenli yöntem)
            delta = self.df['close'].diff().fillna(0)
            gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
            loss = (-delta).clip(lower=0).rolling(window=14, min_periods=1).mean()
            
            # Sıfıra bölme hatasını önle
            rs = np.where(loss == 0, 100, gain / (loss + 1e-9)).clip(0, 100)
            self.df['rsi'] = 100 - (100 / (1 + rs))
            
            # RSI'ı 0-1 aralığına normalize et
            self.df['rsi'] = self.df['rsi'] / 100
            
            # MACD hesapla (daha güvenli yöntem)
            exp1 = self.df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = self.df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
            self.df['macd'] = (exp1 - exp2) / (self.df['close'].mean() + 1e-9)  # Normalize MACD
            self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
            self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
            
            # Bollinger Bands hesapla (daha güvenli yöntem)
            self.df['bb_middle'] = self.df['close'].rolling(window=20, min_periods=1).mean()
            self.df['bb_std'] = self.df['close'].rolling(window=20, min_periods=1).std().fillna(0)
            
            # Sıfır std durumunu ele al
            self.df['bb_std'] = self.df['bb_std'].replace(0, self.df['close'].std() + 1e-9)
            
            self.df['bb_upper'] = self.df['bb_middle'] + 2 * self.df['bb_std']
            self.df['bb_lower'] = self.df['bb_middle'] - 2 * self.df['bb_std']
            
            # Bollinger pozisyonunu hesapla (0-1 aralığında)
            bb_range = self.df['bb_upper'] - self.df['bb_lower']
            bb_range = bb_range.replace(0, self.df['close'].std() * 4 + 1e-9)  # Sıfır bölme hatasını önle
            self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / bb_range
            
            # Fiyat momentumu (5 ve 10 periyod)
            self.df['momentum_5'] = self.df['close'].pct_change(5).fillna(0)
            self.df['momentum_10'] = self.df['close'].pct_change(10).fillna(0)
            
            # Volatilite (fiyat değişim standart sapması)
            self.df['volatility'] = self.df['price_change'].rolling(window=10, min_periods=1).std().fillna(0)
            
            # Hacim değişimi
            self.df['volume_change'] = self.df['volume'].pct_change().fillna(0)
            
            # OBV (On-Balance Volume)
            self.df['obv'] = np.where(
                self.df['close'] > self.df['close'].shift(1),
                self.df['volume'],
                np.where(
                    self.df['close'] < self.df['close'].shift(1),
                    -self.df['volume'],
                    0
                )
            ).cumsum()
            
            # OBV'yi normalize et
            obv_max = self.df['obv'].max()
            obv_min = self.df['obv'].min()
            
            # Sıfıra bölme hatasını önle
            if obv_max != obv_min:
                self.df['obv'] = (self.df['obv'] - obv_min) / (obv_max - obv_min + 1e-9)
            else:
                self.df['obv'] = 0
            
            # NaN değerlerini doldur
            self.df = self.df.fillna(0)
            
            # Aşırı değerleri kontrol et
            for col in ['price_change', 'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                       'momentum_5', 'momentum_10', 'volatility', 'volume_change', 'obv']:
                if self.df[col].abs().max() > 5:
                    self.df[col] = self.df[col] / self.df[col].abs().max() * 2  # -2 ile 2 arasına normalizasyon
            
        except Exception as e:
            logger.error(f"Gösterge hesaplama hatası: {e}")
            # Eksik sütunları varsayılan değerlerle doldur
            required_columns = [
                'price_change', 'rsi', 'macd', 'macd_signal', 'macd_histogram', 
                'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'bb_position',
                'momentum_5', 'momentum_10', 'volatility', 'volume_change', 'obv'
            ]
            for col in required_columns:
                if col not in self.df.columns:
                    self.df[col] = 0
    
    def reset(self):
        """
        Ortamı başlangıç durumuna getir ve ilk gözlemi döndür
        """
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.current_step = self.window_size
        self.trades = []
        self.returns = []
        self.portfolio_value = self.initial_balance
        self.consecutive_same_action = 0
        self.prev_action = None
        self.prev_price_direction = 0
        self.holding_duration = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        Mevcut gözlemi oluştur
        """
        try:
            # Son window_size adımlık veriyi al
            frame = self.df.iloc[self.current_step-self.window_size:self.current_step]
            
            # Fiyat yön değişimi (-1, 0, 1)
            current_price = frame['close'].values[-1]
            prev_price = frame['close'].values[-2] if self.current_step > 1 else current_price
            price_direction = 1 if current_price > prev_price else (-1 if current_price < prev_price else 0)
            self.prev_price_direction = price_direction
            
            # Gözlem matrisini oluştur (daha bilgilendirici)
            obs = np.array([
                frame['price_change'].values,  # Fiyat değişimi
                frame['rsi'].values,           # RSI (0-1 arası)
                frame['macd'].values,          # MACD
                frame['macd_histogram'].values, # MACD histogram
                frame['bb_position'].values,    # Bollinger Bant pozisyonu (0-1)
                frame['momentum_5'].values,     # 5 periyot momentumu
                frame['momentum_10'].values,    # 10 periyot momentumu
                frame['volatility'].values,     # Volatilite
                frame['trend'].values,          # Trend (-0.5 to 0.5)
                frame['sr_position'].values     # Destek/Direnç pozisyonu (0-1)
            ]).T
            
            # NaN veya inf değerlerini temizle
            obs[np.isnan(obs)] = 0
            obs[np.isinf(obs)] = 0
            
            # Aşırı değerleri kırp
            obs = np.clip(obs, -5, 5)
            
            return obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Gözlem oluşturma hatası: {e}")
            # Hata durumunda sıfır tensörü döndür
            return np.zeros((self.window_size, 10), dtype=np.float32)
    
    def _calculate_reward(self, action, portfolio_change):
        """
        Gelişmiş ödül fonksiyonu
        """
        reward = 0
        
        # 1. Portföy değişimi ödülü (temel ödül - daha güçlü)
        reward += np.tanh(portfolio_change * 30)  # Daha güçlü bir temel getiri sinyali
        
        # 2. Trend takip etme ödülü (daha güçlü)
        trend_score = 0
        current_trend = self.df.iloc[self.current_step]['trend']
        current_trend_strength = self.df.iloc[self.current_step]['trend_strength']
        
        if action == 2 and current_trend > 0:  # Yükselen trendde AL
            trend_score = 0.5  # Trend takibi için daha yüksek ödül
        elif action == 0 and current_trend < 0:  # Düşen trendde SAT
            trend_score = 0.5
        elif action == 1 and abs(current_trend) < 0.1:  # Yatay trendde TUT
            trend_score = 0.2
        else:  # Trende karşı hareket etmeye daha güçlü ceza
            trend_score = -0.3 * abs(current_trend) * current_trend_strength
            
        reward += trend_score * current_trend_strength
        
        # 3. RSI tabanlı ödül (iyileştirilmiş)
        rsi = self.df.iloc[self.current_step]['rsi']
        if action == 2 and rsi < 0.3:  # Aşırı satış bölgesinde AL
            reward += 0.4  # Daha güçlü RSI ödülü
        elif action == 0 and rsi > 0.7:  # Aşırı alım bölgesinde SAT
            reward += 0.4
        elif action == 2 and rsi > 0.7:  # Aşırı alım bölgesinde AL yapmaya ceza
            reward -= 0.3
        elif action == 0 and rsi < 0.3:  # Aşırı satış bölgesinde SAT yapmaya ceza
            reward -= 0.3
        
        # 4. Destek/Direnç tabanlı ödül (iyileştirilmiş)
        sr_position = self.df.iloc[self.current_step]['sr_position']
        if action == 2 and sr_position < 0.2:  # Destek seviyelerine yakınken AL
            reward += 0.4
        elif action == 0 and sr_position > 0.8:  # Direnç seviyelerine yakınken SAT
            reward += 0.4
        elif action == 2 and sr_position > 0.8:  # Direnç seviyelerinde AL yapmaya ceza
            reward -= 0.3
        elif action == 0 and sr_position < 0.2:  # Destek seviyelerinde SAT yapmaya ceza
            reward -= 0.3
        
        # 5. MACD sinyal takibi (yeni)
        macd = self.df.iloc[self.current_step]['macd']
        macd_signal = self.df.iloc[self.current_step]['macd_signal']
        macd_cross_up = macd > macd_signal and (self.df.iloc[self.current_step-1]['macd'] if self.current_step > 0 else 0) <= (self.df.iloc[self.current_step-1]['macd_signal'] if self.current_step > 0 else 0)
        macd_cross_down = macd < macd_signal and (self.df.iloc[self.current_step-1]['macd'] if self.current_step > 0 else 0) >= (self.df.iloc[self.current_step-1]['macd_signal'] if self.current_step > 0 else 0)
        
        if action == 2 and macd_cross_up:  # MACD yukarı kesişiminde AL
            reward += 0.4
        elif action == 0 and macd_cross_down:  # MACD aşağı kesişiminde SAT
            reward += 0.4
        
        # 6. Hacim ödülü (yeni)
        volume_change = self.df.iloc[self.current_step]['volume_change']
        if abs(volume_change) > 0.2:  # Önemli hacim değişiminde
            if action != 1:  # Al veya sat aksiyonlarında
                reward *= (1 + min(abs(volume_change), 1) * 0.5)  # Hacim değişimi büyükse, ödülü artır (max 1.5x)
        
        # 7. Aynı aksiyonu tekrarlama cezası (daha ılımlı)
        if self.prev_action is not None and action == self.prev_action:
            self.consecutive_same_action += 1
            if self.consecutive_same_action > 3:  # 3 adımdan fazla aynı aksiyon tekrarlanırsa
                repeat_penalty = min(0.05 * (self.consecutive_same_action - 3), 0.3)  # Maksimum 0.3 ceza
                reward -= repeat_penalty
        else:
            self.consecutive_same_action = 0
        
        # 8. BTC uzun süre tutma cezası/ödülü (geliştirilmiş trend bazlı strateji)
        if self.btc_held > 0:
            self.holding_duration += 1
            trend = self.df.iloc[self.current_step]['trend']
            if trend > 0:  # Yükselen trend
                if self.holding_duration > 5:  # Trend devam ediyorsa tutmaya devam et
                    hold_bonus = min(0.02 * (self.holding_duration - 5), 0.3)  # Maksimum 0.3 ödül
                    reward += hold_bonus
            elif trend < 0 and self.holding_duration > 3:  # Düşen trenddeyken tutuyorsa
                hold_penalty = min(0.03 * (self.holding_duration - 3), 0.4)  # Daha hızlı ceza (maksimum 0.4)
                reward -= hold_penalty
        else:
            self.holding_duration = 0
        
        # 9. Yüksek volatilitede yapılan işlemlere ek ödül/ceza (iyileştirilmiş)
        volatility = self.df.iloc[self.current_step]['volatility']
        if volatility > 0.03:  # Yüksek volatilite
            if action != 1:  # Al veya sat aksiyonlarında
                reward *= (1 + volatility * 7)  # Doğru tahmin ederse daha yüksek ödül (7x faktör)
        
        # 10. Fiyat momentumu dikkate alma (yeni)
        momentum_5 = self.df.iloc[self.current_step]['momentum_5']
        momentum_10 = self.df.iloc[self.current_step]['momentum_10']
        if action == 2 and momentum_5 > 0 and momentum_10 > 0:  # Çift momentum yukarı ve AL
            reward += 0.3
        elif action == 0 and momentum_5 < 0 and momentum_10 < 0:  # Çift momentum aşağı ve SAT
            reward += 0.3
        
        # 11. Portföy çeşitlendirme ve risk yönetimi (geliştirilmiş)
        portfolio_balance = self.btc_held * self.current_price / self.portfolio_value if self.portfolio_value > 0 else 0
        if current_trend > 0:  # Yükselen trend
            # Yükselen trenddeyken daha fazla kripto tut (0.5-0.7 arası ideal)
            if 0.5 <= portfolio_balance <= 0.7:
                reward += 0.2
            elif portfolio_balance < 0.3 and action != 2:  # Yükselen trendde az kripto tutuyorsa ve AL yapmıyorsa
                reward -= 0.2
        else:  # Düşen veya yatay trend
            # Düşen trenddeyken daha az kripto tut (0.2-0.4 arası ideal)
            if 0.2 <= portfolio_balance <= 0.4:
                reward += 0.2
            elif portfolio_balance > 0.6 and action != 0:  # Düşen trendde fazla kripto tutuyorsa ve SAT yapmıyorsa
                reward -= 0.2
        
        # 12. Destek direnç kırılması (yeni)
        close_price = self.df.iloc[self.current_step]['close']
        resistance = self.df.iloc[self.current_step]['resistance']
        support = self.df.iloc[self.current_step]['support']
        
        # Direnç kırılması
        if close_price > resistance * 1.01 and action == 2:  # %1 direnç kırılması ve AL
            reward += 0.5
        # Destek kırılması
        elif close_price < support * 0.99 and action == 0:  # %1 destek kırılması ve SAT
            reward += 0.5
        
        # Ödülü normalize et
        reward = np.clip(reward, -1, 1)
        
        return float(reward)
    
    def step(self, action):
        """
        Aksiyonu uygula ve bir sonraki duruma geç
        
        Parametreler:
        - action: 0 (sat), 1 (tut), 2 (al)
        
        Dönüş:
        - observation: Yeni gözlem
        - reward: Ödül
        - done: Eğer bölüm bittiyse True, değilse False
        - info: Ek bilgiler
        """
        try:
            self.current_step += 1
            
            # Eğer veri setinin sonuna geldiysek bölümü sonlandır
            if self.current_step >= len(self.df):
                return self._get_observation(), 0, True, {}
            
            # Mevcut fiyatı al
            self.current_price = self.df.iloc[self.current_step]['close']
            
            # Aksiyonu uygula
            prev_portfolio_value = self.portfolio_value
            
            successful_action = False  # Bir işlem yapıldı mı?
            
            if action == 0:  # SAT
                if self.btc_held > 0:
                    # Bitcoin'leri sat
                    sell_amount = self.btc_held * self.current_price
                    sell_amount_after_commission = sell_amount * (1 - self.commission)
                    
                    self.balance += sell_amount_after_commission
                    self.btc_held = 0
                    
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'sell',
                        'price': self.current_price,
                        'amount': sell_amount,
                        'commission': sell_amount * self.commission
                    })
                    
                    successful_action = True
                    
            elif action == 2:  # AL
                if self.balance > 0:
                    # Bakiyenin %20'si ile Bitcoin al
                    buy_balance = self.balance * 0.2
                    buy_amount_btc = buy_balance / self.current_price
                    buy_amount_after_commission = buy_amount_btc * (1 - self.commission)
                    
                    self.balance -= buy_balance
                    self.btc_held += buy_amount_after_commission
                    
                    self.trades.append({
                        'step': self.current_step,
                        'type': 'buy',
                        'price': self.current_price,
                        'amount': buy_balance,
                        'commission': buy_balance * self.commission
                    })
                    
                    successful_action = True
            
            # Portföy değerini hesapla
            self.portfolio_value = self.balance + (self.btc_held * self.current_price)
            
            # Portföy değişimini hesapla
            portfolio_change = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            
            # Gelişmiş ödül fonksiyonu
            reward = self._calculate_reward(action, portfolio_change)
            
            # Yapılan işlemin başarısını izle (bilgi olarak)
            action_success = False
            if successful_action:
                if action == 0 and portfolio_change > 0:  # Satış sonrası portföy değeri arttı
                    action_success = True
                elif action == 2 and portfolio_change > 0:  # Alış sonrası portföy değeri arttı
                    action_success = True
            
            self.prev_action = action
            self.returns.append(reward)
            
            # Gözlem, ödül ve durum bilgisini döndür
            done = self.current_step >= len(self.df) - 1
            obs = self._get_observation()
            
            info = {
                'portfolio_value': self.portfolio_value,
                'balance': self.balance,
                'btc_held': self.btc_held,
                'current_price': self.current_price,
                'portfolio_change': portfolio_change,
                'action_success': action_success,
                'successful_action': successful_action
            }
            
            return obs, float(reward), done, info
            
        except Exception as e:
            logger.error(f"Step fonksiyonu hatası: {e}")
            # Hata durumunda varsayılan dönüş
            return self._get_observation(), 0, True, {}
    
    def render(self, mode='human'):
        """
        Mevcut durumu göster
        """
        profit = self.portfolio_value - self.initial_balance
        profit_percent = (profit / self.initial_balance) * 100
        
        print(f"Adım: {self.current_step}")
        print(f"Tarih: {self.df.index[self.current_step]}")
        print(f"Bitcoin Fiyatı: ${self.current_price:.2f}")
        print(f"Bakiye: ${self.balance:.2f}")
        print(f"Bitcoin: {self.btc_held:.6f}")
        print(f"Portföy Değeri: ${self.portfolio_value:.2f}")
        print(f"Kar/Zarar: ${profit:.2f} (%{profit_percent:.2f})")
        
        # Teknik göstergeleri göster
        print(f"RSI: {self.df.iloc[self.current_step]['rsi']*100:.1f}")
        print(f"MACD: {self.df.iloc[self.current_step]['macd']:.6f}")
        print(f"Trend: {'+' if self.df.iloc[self.current_step]['trend'] > 0 else '-'}")
        print(f"Destek/Direnç Pozisyonu: {self.df.iloc[self.current_step]['sr_position']:.2f}")
        print("----------------------------")
        
    def close(self):
        """
        Ortamı kapat
        """
        pass

class AccuracyCallback(BaseCallback):
    """
    Eğitim sırasında doğruluk oranını izleyen özel callback
    
    Bu callback, belirli aralıklarla modelin test ortamında değerlendirmesini yapar
    ve doğruluk oranını (accuracy) hesaplar.
    """
    def __init__(self, eval_env, verbose=0, eval_freq=1000, accuracy_threshold=None):
        super(AccuracyCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.accuracies = []
        self.trend_accuracies = []  # Trend doğruluk oranı
        self.action_accuracies = []  # Aksiyon doğruluk oranı
        self.rewards = []
        self.timesteps = []
        self.profits = []
        
        # Doğruluk eşiği (bu değere ulaşılırsa eğitim durdurulabilir)
        if accuracy_threshold is None and 'ACCURACY_THRESHOLD' in os.environ:
            try:
                self.accuracy_threshold = float(os.environ['ACCURACY_THRESHOLD'])
            except (ValueError, TypeError):
                self.accuracy_threshold = None
        else:
            self.accuracy_threshold = accuracy_threshold
        
        if self.accuracy_threshold:
            logger.info(f"Doğruluk eşiği: %{self.accuracy_threshold:.1f} (bu değere ulaşılırsa eğitim durdurulacak)")
    
    def _on_step(self):
        """
        Her n adımda bir modeli değerlendir ve doğruluğu hesapla
        """
        if self.n_calls % self.eval_freq == 0:
            # Modeli değerlendir
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=1,
                return_episode_rewards=True,
                deterministic=True
            )
            
            # Accuracy hesaplamak için özel bir değerlendirme yap
            obs = self.eval_env.reset()
            done = False
            total_steps = 0
            successful_actions = 0
            correct_trend_predictions = 0
            correct_actions = 0
            total_actions = 0
            total_trend_predictions = 0
            
            # Farklı doğruluk metrikleri için değişkenler
            initial_portfolio = self.eval_env.envs[0].unwrapped.portfolio_value
            portfolio_values = [initial_portfolio]
            prev_portfolio_value = initial_portfolio
            
            while not done:
                # Mevcut fiyat trendini belirle
                df = self.eval_env.envs[0].unwrapped.df
                current_step = self.eval_env.envs[0].unwrapped.current_step
                
                # Modelin tahmini
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Aksiyonu uygula
                obs, reward, done, info = self.eval_env.step(action)
                
                # Portföy değerini takip et
                portfolio_value = info[0]['portfolio_value']
                portfolio_values.append(portfolio_value)
                
                # Portföy değişimini kontrol et
                portfolio_change = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
                
                # Doğruluk 1: İşlem başarısı
                if 'successful_action' in info[0] and info[0]['successful_action']:
                    total_actions += 1
                    if 'action_success' in info[0] and info[0]['action_success']:
                        successful_actions += 1
                
                # Doğruluk 2: Aksiyon doğruluğu
                if action[0] != 1:  # Sadece alım/satım aksiyonlarını değerlendir
                    # Doğru yön tahmini:
                    # - Al yönünde hareket ve fiyat artıyorsa doğru
                    # - Sat yönünde hareket ve fiyat düşüyorsa doğru
                    if current_step + 1 < len(df):
                        price_change = df.iloc[current_step + 1]['close'] - df.iloc[current_step]['close']
                        predict_up = action[0] == 2  # Al (yukarı tahmin)
                        actual_up = price_change > 0  # Gerçekte yukarı mı?
                        
                        total_trend_predictions += 1
                        if (predict_up and actual_up) or (not predict_up and not actual_up):
                            correct_trend_predictions += 1
                
                # Doğruluk 3: Aksiyon sonucu
                if 'action_success' in info[0]:
                    correct_actions += int(info[0]['action_success'])
                
                prev_portfolio_value = portfolio_value
                total_steps += 1
            
            # Üç farklı doğruluk metriği hesapla
            portfolio_accuracy = successful_actions / max(1, total_actions) * 100
            trend_accuracy = correct_trend_predictions / max(1, total_trend_predictions) * 100
            action_accuracy = correct_actions / max(1, total_steps) * 100
            
            # En son portföy değeri
            final_portfolio = portfolio_values[-1]
            profit_percent = (final_portfolio - initial_portfolio) / initial_portfolio * 100
            
            # İlerlemeyi kaydet
            self.accuracies.append(portfolio_accuracy)
            self.trend_accuracies.append(trend_accuracy)
            self.action_accuracies.append(action_accuracy)
            self.rewards.append(np.mean(episode_rewards))
            self.timesteps.append(self.num_timesteps)
            self.profits.append(profit_percent)
            
            # Sonuçları log'a kaydet
            self.logger.record('eval/portfolio_accuracy', portfolio_accuracy)
            self.logger.record('eval/trend_accuracy', trend_accuracy)
            self.logger.record('eval/action_accuracy', action_accuracy)
            self.logger.record('eval/profit_percent', profit_percent)
            
            # Kombine doğruluk (portföy bazlı ve trend bazlı)
            combined_accuracy = (portfolio_accuracy + trend_accuracy) / 2
            
            if self.verbose > 0:
                print(f"\n----- Eğitim İlerlemesi: {self.num_timesteps} adım -----")
                print(f"Portföy Doğruluğu: %{portfolio_accuracy:.2f}")
                print(f"Trend Doğruluğu: %{trend_accuracy:.2f}")
                print(f"Aksiyon Doğruluğu: %{action_accuracy:.2f}")
                print(f"Ortalama Ödül: {np.mean(episode_rewards):.2f}")
                print(f"Kar/Zarar: %{profit_percent:.2f}")
                print(f"Kombine Doğruluk: %{combined_accuracy:.2f}")
                print("-" * 40)
            
            # Belirli bir doğruluk seviyesine ulaşıldığında modeli kaydet
            if len(self.trend_accuracies) > 0:
                if trend_accuracy >= 55.0 and trend_accuracy > max(self.trend_accuracies[:-1], default=0):
                    model_path = f"models/bitcoin_trader_ppo_trend{trend_accuracy:.0f}_step{self.num_timesteps}"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.model.save(model_path)
                    print(f"Yüksek trend doğruluk modeli kaydedildi: {model_path}")
                
                if profit_percent > 5.0 and profit_percent > max(self.profits[:-1], default=0):
                    model_path = f"models/bitcoin_trader_ppo_profit{profit_percent:.0f}_step{self.num_timesteps}"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.model.save(model_path)
                    print(f"Yüksek karlılık modeli kaydedildi: {model_path}")
            
            # Eğer doğruluk eşiği belirlenmişse ve bu eşiğe ulaşıldıysa eğitimi durdur
            if self.accuracy_threshold and combined_accuracy >= self.accuracy_threshold:
                print(f"\n{'='*50}")
                print(f"Doğruluk eşiğine ulaşıldı: %{combined_accuracy:.2f} >= %{self.accuracy_threshold:.2f}")
                print("Eğitim başarıyla sonlandırılıyor...")
                print(f"{'='*50}\n")
                
                # Son modeli kaydet
                model_path = f"models/bitcoin_trader_ppo_final_acc{combined_accuracy:.0f}"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save(model_path)
                print(f"Final modeli kaydedildi: {model_path}")
                
                # Eğitimi durdur
                return False
        
        return True
    
    def plot_accuracy(self):
        """
        Eğitim sırasında doğruluk ve ödüllerin grafiğini çiz
        """
        if len(self.timesteps) == 0:
            return
        
        plt.figure(figsize=(14, 14))
        
        # Doğruluk grafikleri
        plt.subplot(4, 1, 1)
        plt.plot(self.timesteps, self.accuracies, 'g-', label='Portföy Doğruluğu')
        plt.plot(self.timesteps, self.trend_accuracies, 'b-', label='Trend Doğruluğu')
        plt.plot(self.timesteps, self.action_accuracies, 'r-', label='Aksiyon Doğruluğu')
        plt.title('Eğitim Sırasında Model Doğrulukları')
        plt.xlabel('Timesteps')
        plt.ylabel('Doğruluk (%)')
        plt.axhline(y=50, color='k', linestyle='--', alpha=0.3, label='Rastgele Tahmin')
        plt.legend()
        plt.grid(True)
        
        # Ödül grafiği
        plt.subplot(4, 1, 2)
        plt.plot(self.timesteps, self.rewards, 'b-')
        plt.title('Eğitim Sırasında Ortalama Ödül')
        plt.xlabel('Timesteps')
        plt.ylabel('Ortalama Ödül')
        plt.grid(True)
        
        # Kar grafiği
        plt.subplot(4, 1, 3)
        plt.plot(self.timesteps, self.profits, 'g-')
        plt.title('Eğitim Sırasında Portföy Kar/Zarar (%)')
        plt.xlabel('Timesteps')
        plt.ylabel('Kar/Zarar (%)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Breakeven')
        plt.grid(True)
        
        # Doğruluk ve kar ilişkisi
        plt.subplot(4, 1, 4)
        plt.scatter(self.trend_accuracies, self.profits, alpha=0.7, label='Trend Doğruluğu vs Kar')
        plt.scatter(self.accuracies, self.profits, alpha=0.7, label='Portföy Doğruluğu vs Kar')
        
        # Eğilim çizgisi (Trend Doğruluğu vs Kar)
        if len(self.trend_accuracies) > 2:
            z = np.polyfit(self.trend_accuracies, self.profits, 1)
            p = np.poly1d(z)
            plt.plot(sorted(self.trend_accuracies), p(sorted(self.trend_accuracies)), "r--", alpha=0.7)
        
        plt.title('Doğruluk ve Karlılık İlişkisi')
        plt.xlabel('Doğruluk (%)')
        plt.ylabel('Kar/Zarar (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_accuracy.png')
        plt.show()

class BitcoinTraderAI:
    """
    Reinforcement Learning tabanlı Bitcoin Day Trading Botu
    """
    def __init__(self, symbol='BTC/USDT', timeframe='1h', train_start_date=None, train_end_date=None, test_start_date=None, test_end_date=None):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Veri aralıkları
        self.train_start_date = train_start_date or (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        self.train_end_date = train_end_date or (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')
        self.test_start_date = test_start_date or (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        self.test_end_date = test_end_date or datetime.now().strftime('%Y-%m-%d')
        
        # PPO model
        self.model = None
        
        # Veri setleri
        self.train_df = None
        self.test_df = None
        
        # Ortamlar
        self.train_env = None
        self.test_env = None
        
        logger.info(f"BitcoinTraderAI başlatıldı - {symbol} {timeframe}")
    
    def fetch_data(self):
        """
        CCXT kullanarak Binance'den Bitcoin fiyat verilerini çek
        """
        try:
            logger.info(f"Veri çekiliyor: {self.symbol} {self.timeframe} period: {self.train_start_date} - {self.test_end_date}")
            
            # Binance bağlantısı
            exchange = ccxt.binance({
                'enableRateLimit': True
            })
            
            # Tüm veri aralığını çek
            since = int(datetime.strptime(self.train_start_date, '%Y-%m-%d').timestamp() * 1000)
            end_time = int(datetime.strptime(self.test_end_date, '%Y-%m-%d').timestamp() * 1000)
            
            all_data = []
            
            # CCXT 500 kayıtlık limit koyduğu için, verileri kısım kısım çekeriz
            current_since = since
            while current_since < end_time:
                # Verileri çek
                ohlcv = exchange.fetch_ohlcv(self.symbol, self.timeframe, current_since)
                
                if len(ohlcv) == 0:
                    break
                
                # Verileri ana listeye ekle
                all_data.extend(ohlcv)
                
                # Son kaydın zaman damgasını al ve ilerlet
                last_timestamp = ohlcv[-1][0]
                current_since = last_timestamp + 1
                
                # Hızlıca çok fazla istek göndermeyi önlemek için biraz bekle
                time.sleep(exchange.rateLimit / 1000)
            
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Veri kalitesini kontrol et
            if len(df) < 100:
                logger.warning("Çok az veri çekildi, model yeterince öğrenemeyebilir.")
                if len(df) < 50:
                    logger.error(f"Yetersiz veri: Sadece {len(df)} kayıt çekildi. En az 50 kayıt gerekli.")
                    logger.info("Sentetik veri oluşturuluyor...")
                    df = self._generate_synthetic_data(300)
            
            # Veri kalitesini kontrol et
            self._check_data_quality(df)
            
            # Eğitim ve test setlerini ayır
            train_mask = (df.index >= self.train_start_date) & (df.index <= self.train_end_date)
            test_mask = (df.index >= self.test_start_date) & (df.index <= self.test_end_date)
            
            self.train_df = df.loc[train_mask].copy()
            self.test_df = df.loc[test_mask].copy()
            
            # Maskelere göre veri bulunmuyorsa 
            if len(self.train_df) < 50:
                logger.warning(f"Eğitim için çok az veri: {len(self.train_df)} satır. Sentetik veri kullanılacak.")
                self.train_df = self._generate_synthetic_data(200)
            
            if len(self.test_df) < 20:
                logger.warning(f"Test için çok az veri: {len(self.test_df)} satır. Sentetik veri kullanılacak.")
                self.test_df = self._generate_synthetic_data(100)
            
            logger.info(f"Veri başarıyla çekildi - Eğitim: {len(self.train_df)} kayıt, Test: {len(self.test_df)} kayıt")
            
            return True
        
        except Exception as e:
            logger.error(f"Veri çekme hatası: {e}")
            logger.info("Sentetik veri oluşturuluyor...")
            
            # Hata durumunda sentetik veri kullan
            self.train_df = self._generate_synthetic_data(200)
            self.test_df = self._generate_synthetic_data(100)
            
            logger.info(f"Sentetik veri oluşturuldu - Eğitim: {len(self.train_df)} kayıt, Test: {len(self.test_df)} kayıt")
            return True
    
    def _check_data_quality(self, df):
        """
        Veri kalitesini kontrol et ve sorunları çöz
        """
        # NaN değerleri kontrol et
        if df.isnull().values.any():
            logger.warning("Veride NaN değerleri bulundu. Düzeltiliyor...")
            df.fillna(method='ffill', inplace=True)  # Forward fill
            df.fillna(method='bfill', inplace=True)  # Backward fill
            df.fillna(0, inplace=True)  # Kalan NaN değerlerini sıfırla
        
        # Negatif değerleri kontrol et
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if (df[col] < 0).any():
                logger.warning(f"{col} sütununda negatif değerler bulundu. Düzeltiliyor...")
                df[col] = df[col].abs()
        
        # Aykırı değerleri (outliers) kontrol et
        for col in ['open', 'high', 'low', 'close']:
            if len(df) > 20:  # Yeterli veri varsa
                mean = df[col].mean()
                std = df[col].std()
                threshold = 5  # 5 standart sapma
                outliers = df[col][(df[col] > mean + threshold * std) | (df[col] < mean - threshold * std)]
                if len(outliers) > 0:
                    logger.warning(f"{col} sütununda {len(outliers)} aykırı değer bulundu. Kırpılıyor...")
                    df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
        
        # Ters fiyat ilişkilerini kontrol et (high < low gibi)
        invalid_rows = df[df['high'] < df['low']].index
        if len(invalid_rows) > 0:
            logger.warning(f"{len(invalid_rows)} satırda high < low durumu bulundu. Düzeltiliyor...")
            for idx in invalid_rows:
                # high ve low değerlerini değiştir
                df.at[idx, 'high'], df.at[idx, 'low'] = df.at[idx, 'low'], df.at[idx, 'high']
        
        return df
    
    def _generate_synthetic_data(self, n_samples=200):
        """
        Sentetik Bitcoin fiyat verileri oluştur
        """
        logger.info(f"{n_samples} satırlık sentetik veri oluşturuluyor...")
        
        # Başlangıç fiyatı
        initial_price = 30000.0
        
        # Zaman indeksi
        start_date = datetime.now() - timedelta(days=n_samples//24)  # Saatlik veri için gün sayısını ayarla
        dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')
        
        # Rastgele yürüyüş modeli ile fiyat oluştur
        np.random.seed(42)  # Tekrarlanabilirlik için
        random_walk = np.random.normal(0, 0.02, n_samples).cumsum()
        price_multiplier = np.exp(random_walk)
        
        # OHLCV verileri oluştur
        df = pd.DataFrame(index=dates)
        df['close'] = initial_price * price_multiplier
        df['open'] = df['close'].copy()
        df['high'] = df['close'].copy()
        df['low'] = df['close'].copy()
        df['volume'] = initial_price * np.random.exponential(1, n_samples) * 10
        
        # Her mum için rastgele açılış, yüksek, düşük ve hacim oluştur
        for i in range(n_samples):
            volatility = 0.02 * price_multiplier[i]  # Volatiliteyi fiyatla ilişkilendir
            df.iloc[i, df.columns.get_indexer(['open'])[0]] = df.iloc[i]['close'] * (1 + np.random.normal(0, 0.01))
            df.iloc[i, df.columns.get_indexer(['high'])[0]] = df.iloc[i]['close'] * (1 + abs(np.random.normal(0, volatility)))
            df.iloc[i, df.columns.get_indexer(['low'])[0]] = df.iloc[i]['close'] * (1 - abs(np.random.normal(0, volatility)))
        
        # Veri tutarlılığını sağla
        for i in range(n_samples):
            # High, Low, Open ve Close ilişkisini düzelt
            row = df.iloc[i]
            high = max(row['open'], row['close'], row['high'])
            low = min(row['open'], row['close'], row['low'])
            df.iloc[i, df.columns.get_indexer(['high'])[0]] = high
            df.iloc[i, df.columns.get_indexer(['low'])[0]] = low
        
        return df
    
    def create_environments(self):
        """
        Eğitim ve test ortamlarını oluştur
        """
        if self.train_df is None or self.test_df is None:
            logger.error("Ortamlar oluşturulamadı: Veri yok")
            return False
        
        # Eğitim ortamını oluştur
        self.train_env = DummyVecEnv([lambda: BitcoinTradingEnv(self.train_df)])
        
        # Test ortamını oluştur
        self.test_env = DummyVecEnv([lambda: BitcoinTradingEnv(self.test_df)])
        
        logger.info("Eğitim ve test ortamları başarıyla oluşturuldu")
        return True
    
    def train_model(self, total_timesteps=10000, accuracy_threshold=None):
        """
        PPO modelini eğit
        
        Parametreler:
        - total_timesteps: Toplam eğitim adım sayısı
        - accuracy_threshold: Belirli bir doğruluk oranına ulaşılınca eğitimi durdur (None: durdurma)
        """
        if self.train_env is None:
            logger.error("Model eğitilemedi: Ortam yok")
            return False
        
        try:
            logger.info(f"Model eğitimi başlıyor - {total_timesteps} adım")
            
            # Eğitim için geri çağırma fonksiyonları oluştur
            save_path = "./model_checkpoints"
            os.makedirs(save_path, exist_ok=True)
            
            # Eğitim sırasında performans değerlendirme için geri çağırma
            eval_callback = EvalCallback(
                self.test_env,
                best_model_save_path=save_path,
                log_path="./logs/",
                eval_freq=1000,
                deterministic=True,
                render=False
            )
            
            # Doğruluk izleme için özel callback
            accuracy_callback = AccuracyCallback(
                eval_env=self.test_env,
                verbose=1,
                eval_freq=2000,
                accuracy_threshold=accuracy_threshold
            )
            
            # PPO modelini oluştur ve eğit (en iyi finansal performans için optimize edilmiş parametreler)
            self.model = PPO(
                "MlpPolicy",
                self.train_env,
                verbose=1,
                learning_rate=0.0002,     # Daha düşük öğrenme oranı - daha kararlı
                n_steps=2048,            # Her güncellemeden önce toplanan adım sayısı
                batch_size=256,          # Daha büyük batch size - daha iyi genelleme
                gamma=0.995,             # Daha yüksek gamma - uzun vadeli getiri
                ent_coef=0.01,           # Daha yüksek entropy - daha fazla keşif
                vf_coef=0.7,             # Değer fonksiyonu katsayısını artır
                max_grad_norm=0.7,       # Daha esnek gradient clipping
                gae_lambda=0.98,         # Yüksek lambda - daha iyi avantaj tahmini
                clip_range=0.25,         # Daha geniş clip aralığı
                n_epochs=10,             # Her batch için daha fazla eğitim epoch
                target_kl=0.015,         # KL ıraksaması için hedef değer
                policy_kwargs={
                    'net_arch': [
                        {'pi': [256, 128, 64], 'vf': [256, 128, 64]}  # Daha derin ve geniş ağ mimarisi
                    ],
                    'activation_fn': torch.nn.ReLU,  # ReLU aktivasyon fonksiyonu
                    'log_std_init': -2.0,  # Başlangıçta daha düşük rastgelelik
                    'ortho_init': True     # Daha iyi ağırlık başlatma
                }
            )
            
            # Modeli eğit
            self.model.learn(total_timesteps=total_timesteps, callback=[eval_callback, accuracy_callback])
            
            # Doğruluk grafiğini çiz
            accuracy_callback.plot_accuracy()
            
            # Modeli kaydet
            self.model.save("bitcoin_trader_ppo_model")
            
            logger.info("Model eğitimi tamamlandı")
            return True
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
            return False
    
    def load_model(self, model_path="bitcoin_trader_ppo_model"):
        """
        Kaydedilmiş modeli yükle
        """
        try:
            logger.info(f"Model yükleniyor: {model_path}")
            self.model = PPO.load(model_path)
            logger.info("Model başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False
    
    def backtest(self):
        """
        Test veri seti üzerinde modeli değerlendir
        """
        if self.model is None or self.test_env is None:
            logger.error("Backtest yapılamadı: Model veya test ortamı yok")
            return None
        
        try:
            logger.info("Backtest başlıyor")
            
            # Test ortamını sıfırla
            obs = self.test_env.reset()
            
            # Test sonuçlarını tut
            done = False
            portfolio_values = []
            actions = []
            profitable_actions = 0
            total_actions = 0
            correct_trend_predictions = 0
            total_trend_predictions = 0
            trades = []
            rewards = []
            
            # Test ortamındaki BitcoinTradingEnv sınıfını al
            test_env_unwrapped = self.test_env.envs[0].unwrapped
            initial_portfolio = test_env_unwrapped.portfolio_value
            prev_portfolio_value = initial_portfolio
            
            step = 0
            while not done:
                # Modelin aksiyonunu al
                action, _states = self.model.predict(obs, deterministic=True)
                actions.append(action[0])
                
                # Mevcut adımı kaydet (tahmin yapmadan önce)
                current_step = test_env_unwrapped.current_step
                
                # Fiyat trendini belirle (şu anki fiyat ve bir sonraki fiyat karşılaştırması)
                if current_step + 1 < len(test_env_unwrapped.df):
                    current_price = test_env_unwrapped.df.iloc[current_step]['close']
                    next_price = test_env_unwrapped.df.iloc[current_step + 1]['close']
                    price_trend = 1 if next_price > current_price else (-1 if next_price < current_price else 0)
                    
                    # Tahmin doğruluğunu hesapla
                    if action[0] != 1:  # Sadece alım/satım aksiyonları
                        total_trend_predictions += 1
                        predict_up = action[0] == 2  # Al aksiyonu için yukarı tahmin
                        actual_up = price_trend > 0
                        
                        if (predict_up and actual_up) or (not predict_up and not actual_up):
                            correct_trend_predictions += 1
                
                # Aksiyonu uygula
                obs, reward, done, info = self.test_env.step(action)
                rewards.append(reward)
                
                # Portföy değerini kaydet
                current_portfolio_value = info[0]['portfolio_value']
                portfolio_values.append(current_portfolio_value)
                
                # İşlemi kaydet
                if 'successful_action' in info[0] and info[0]['successful_action']:
                    trades.append({
                        'step': step,
                        'date': test_env_unwrapped.df.index[current_step],
                        'action': 'BUY' if action[0] == 2 else 'SELL',
                        'price': info[0]['current_price'],
                        'success': info[0].get('action_success', False)
                    })
                
                # Aksiyon doğruluğunu hesapla
                if action[0] != 1:  # Eğer aksiyon "tut" değilse (al veya sat ise)
                    total_actions += 1
                    portfolio_change = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
                    
                    if portfolio_change > 0:
                        profitable_actions += 1
                
                prev_portfolio_value = current_portfolio_value
                
                # Her 10 adımda bir durumu göster
                if step % 10 == 0:
                    test_env_unwrapped.render()
                
                step += 1
            
            # Sonuçları hesapla
            final_portfolio = portfolio_values[-1]
            profit = final_portfolio - initial_portfolio
            profit_percent = (profit / initial_portfolio) * 100
            
            # Doğruluk oranlarını hesapla
            portfolio_accuracy = profitable_actions / total_actions * 100 if total_actions > 0 else 0
            trend_accuracy = correct_trend_predictions / total_trend_predictions * 100 if total_trend_predictions > 0 else 0
            combined_accuracy = (portfolio_accuracy + trend_accuracy) / 2 if total_actions > 0 and total_trend_predictions > 0 else 0
            
            # Günlük getiri
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Sharpe oranı (yıllıklaştırılmış)
            sharpe_ratio = np.sqrt(len(daily_returns)) * np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
            
            # Sortino oranı (sadece negatif getiriler için risk)
            downside_returns = [r for r in daily_returns if r < 0]
            sortino_ratio = np.sqrt(len(daily_returns)) * np.mean(daily_returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
            
            # Maksimum düşüş
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = 1 - np.array(portfolio_values) / running_max
            max_drawdown = np.max(drawdowns) * 100
            
            # İşlem sayısı
            n_trades = len(test_env_unwrapped.trades)
            
            # Kazançlı/Zararlı işlem oranı
            profitable_trades = sum(1 for trade in trades if trade['success'])
            profit_ratio = profitable_trades / n_trades if n_trades > 0 else 0
            
            # Sonuçları göster
            logger.info("Backtest Sonuçları:")
            logger.info(f"Başlangıç Portföy Değeri: ${initial_portfolio:.2f}")
            logger.info(f"Final Portföy Değeri: ${final_portfolio:.2f}")
            logger.info(f"Kar/Zarar: ${profit:.2f} (%{profit_percent:.2f})")
            logger.info(f"Portföy Doğruluk Oranı: %{portfolio_accuracy:.2f}")
            logger.info(f"Trend Doğruluk Oranı: %{trend_accuracy:.2f}")
            logger.info(f"Kombine Doğruluk Oranı: %{combined_accuracy:.2f}")
            logger.info(f"Sharpe Oranı: {sharpe_ratio:.2f}")
            logger.info(f"Sortino Oranı: {sortino_ratio:.2f}")
            logger.info(f"Maksimum Düşüş: %{max_drawdown:.2f}")
            logger.info(f"İşlem Sayısı: {n_trades}")
            logger.info(f"Kazançlı İşlem Oranı: %{profit_ratio*100:.2f}")
            
            # Grafik oluştur
            self._plot_results(portfolio_values, test_env_unwrapped, trend_accuracy, portfolio_accuracy, combined_accuracy)
            
            return {
                'initial_portfolio': initial_portfolio,
                'final_portfolio': final_portfolio,
                'profit': profit,
                'profit_percent': profit_percent,
                'portfolio_accuracy': portfolio_accuracy,
                'trend_accuracy': trend_accuracy,
                'combined_accuracy': combined_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'n_trades': n_trades,
                'profit_ratio': profit_ratio,
                'portfolio_values': portfolio_values,
                'actions': actions,
                'trades': trades,
                'rewards': rewards
            }
        
        except Exception as e:
            logger.error(f"Backtest hatası: {e}")
            return None
    
    def _plot_results(self, portfolio_values, env, trend_accuracy=None, portfolio_accuracy=None, combined_accuracy=None):
        """
        Backtest sonuçlarını görselleştir
        """
        try:
            # Tarihler
            dates = env.df.index[env.window_size:env.window_size+len(portfolio_values)]
            
            # Grafiği ayarla
            plt.figure(figsize=(14, 16))
            
            # 1. Portföy değeri ve Bitcoin fiyatı
            plt.subplot(5, 1, 1)
            plt.plot(dates, portfolio_values, 'b-', label='Portföy Değeri')
            
            # Bitcoin fiyatını ikinci eksende göster
            ax2 = plt.twinx()
            ax2.plot(dates, env.df['close'].values[env.window_size:env.window_size+len(portfolio_values)], 'r--', label='BTC Fiyatı')
            
            plt.title('Portföy Değeri ve Bitcoin Fiyatı')
            plt.xlabel('Tarih')
            plt.ylabel('Değer ($)')
            
            # İşlemleri işaretle
            for trade in env.trades:
                idx = trade['step'] - env.window_size
                if 0 <= idx < len(dates):
                    color = 'g' if trade['type'] == 'buy' else 'r'
                    marker = '^' if trade['type'] == 'buy' else 'v'
                    plt.plot(dates[idx], portfolio_values[idx], marker, color=color, markersize=10)
            
            # Her iki eksendeki etiketleri birleştir
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
            
            # 2. Getiriler
            plt.subplot(5, 1, 2)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            plt.plot(dates[1:], returns, 'g-')
            plt.title('Günlük Getiriler')
            plt.xlabel('Tarih')
            plt.ylabel('Getiri (%)')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 3. Drawdown
            plt.subplot(5, 1, 3)
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = 1 - np.array(portfolio_values) / running_max
            plt.plot(dates, drawdowns * 100, 'r-')
            plt.title('Drawdown')
            plt.xlabel('Tarih')
            plt.ylabel('Drawdown (%)')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 4. Kümülatif Kâr/Zarar
            plt.subplot(5, 1, 4)
            cumulative_returns = np.array(portfolio_values) / portfolio_values[0] - 1
            plt.plot(dates, cumulative_returns * 100, 'b-')
            plt.title('Kümülatif Kâr/Zarar')
            plt.xlabel('Tarih')
            plt.ylabel('Kâr/Zarar (%)')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # 5. Model Doğruluk Bilgisi
            plt.subplot(5, 1, 5)
            plt.axis('off')  # Eksen çizgilerini kapatır
            
            # Sonuçları metin olarak göster
            profit = portfolio_values[-1] - portfolio_values[0]
            profit_percent = (profit / portfolio_values[0]) * 100
            
            result_text = f"""
            BITCOIN TRADER AI - MODEL PERFORMANSI
            ===================================
            Başlangıç Portföy: ${portfolio_values[0]:.2f}
            Final Portföy: ${portfolio_values[-1]:.2f}
            Kar/Zarar: ${profit:.2f} (%{profit_percent:.2f})
            İşlem Sayısı: {len(env.trades)}
            
            Doğruluk Metrikleri:
            - Portföy Doğruluk: %{portfolio_accuracy:.2f}
            - Trend Doğruluk: %{trend_accuracy:.2f}
            - Kombine Doğruluk: %{combined_accuracy:.2f}
            
            Risk Metrikleri:
            - Sharpe Oranı: {np.sqrt(len(returns)) * np.mean(returns) / np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0:.2f}
            - Maksimum Düşüş: %{np.max(drawdowns) * 100:.2f}
            """
            plt.text(0.5, 0.5, result_text, horizontalalignment='center', 
                    verticalalignment='center', transform=plt.gca().transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('bitcoin_trader_ai_results.png')
            plt.show()
            
            logger.info("Sonuç grafiği kaydedildi: bitcoin_trader_ai_results.png")
        
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {e}")
    
    def run(self, mode='train_and_test', total_timesteps=10000, load_existing=False, accuracy_threshold=None):
        """
        Tüm işlemi çalıştır: veri çekme, model eğitimi ve backtest
        
        Parametreler:
        - mode: 'train_and_test', 'train_only', or 'test_only'
        - total_timesteps: Eğitim adım sayısı
        - load_existing: True ise mevcut modeli yükle, False ise yeni model eğit
        - accuracy_threshold: Belirli bir doğruluk oranına ulaşılınca eğitimi durdur
        """
        try:
            # Verileri çek
            if not self.fetch_data():
                return False
            
            # Ortamları oluştur
            if not self.create_environments():
                return False
            
            # Modu kontrol et
            if mode in ['train_and_test', 'train_only']:
                if load_existing:
                    # Mevcut modeli yükle
                    if not self.load_model():
                        # Yükleme başarısız olursa yeni model eğit
                        if not self.train_model(total_timesteps, accuracy_threshold):
                            return False
                else:
                    # Yeni model eğit
                    if not self.train_model(total_timesteps, accuracy_threshold):
                        return False
            else:
                # Test modunda mevcut modeli yükle
                if not self.load_model():
                    logger.error("Test için model yüklenemedi")
                    return False
            
            # Backtest
            if mode in ['train_and_test', 'test_only']:
                results = self.backtest()
                if results is None:
                    return False
                
                # Sonuçları göster
                self._show_accuracy_report(results)
            
            logger.info("İşlem başarıyla tamamlandı")
            return True
            
        except Exception as e:
            logger.error(f"İşlem hatası: {e}")
            return False
    
    def _show_accuracy_report(self, results):
        """
        Eğitim ve test sonuçlarını detaylı olarak gösteren rapor
        """
        try:
            # Terminal raporu
            print("\n" + "="*50)
            print("       BITCOIN TRADER AI - MODEL DOĞRULUK RAPORU       ")
            print("="*50)
            print(f"Sembol: {self.symbol}  Zaman Dilimi: {self.timeframe}")
            print(f"Eğitim Dönemi: {self.train_start_date} - {self.train_end_date}")
            print(f"Test Dönemi: {self.test_start_date} - {self.test_end_date}")
            print("-"*50)
            print("TEST SONUÇLARI:")
            print(f"Doğruluk Oranı: %{results['portfolio_accuracy']:.2f}")
            print(f"Kar/Zarar: ${results['profit']:.2f} (%{results['profit_percent']:.2f})")
            print(f"Sharpe Oranı: {results['sharpe_ratio']:.2f}")
            print(f"Maksimum Düşüş: %{results['max_drawdown']:.2f}")
            print(f"İşlem Sayısı: {results['n_trades']}")
            print("-"*50)
            
            # Doğruluk ile ilgili ek bilgiler
            actions = results.get('actions', [])
            if actions:
                action_counts = {
                    'sell': actions.count('SELL'),
                    'hold': actions.count('BUY'),
                    'buy': actions.count('BUY')
                }
                print("Aksiyon Dağılımı:")
                print(f"Satış (SELL): {action_counts['sell']} (%{action_counts['sell']/len(actions)*100:.1f})")
                print(f"Tut (BUY): {action_counts['hold']} (%{action_counts['hold']/len(actions)*100:.1f})")
                print(f"Alış (BUY): {action_counts['buy']} (%{action_counts['buy']/len(actions)*100:.1f})")
            
            print("="*50)
            print("\nGrafiksel sonuçlar 'bitcoin_trader_ai_results.png' dosyasına kaydedildi.")
            
            # Doğruluk raporu grafiği
            plt.figure(figsize=(10, 8))
            
            # Model doğruluk özeti
            plt.subplot(1, 1, 1)
            plt.axis('off')  # Eksen çizgilerini kapatır
            
            # Başlık ve sonuçları içeren kapsamlı metin
            report_text = f"""
            BITCOIN TRADER AI - DOĞRULUK RAPORU
            ===================================
            Model: PPO (Proximal Policy Optimization)
            
            Veri Özeti:
            - Sembol: {self.symbol}
            - Zaman Dilimi: {self.timeframe}
            - Eğitim Verileri: {len(self.train_df)} kayıt ({self.train_start_date} - {self.train_end_date})
            - Test Verileri: {len(self.test_df)} kayıt ({self.test_start_date} - {self.test_end_date})
            
            Doğruluk Metrikleri:
            - Model Doğruluk Oranı: %{results['portfolio_accuracy']:.2f}
            - Kar/Zarar: ${results['profit']:.2f} (%{results['profit_percent']:.2f})
            - Sharpe Oranı: {results['sharpe_ratio']:.2f}
            - Maksimum Düşüş: %{results['max_drawdown']:.2f}
            
            İşlem İstatistikleri:
            - Toplam İşlem Sayısı: {results['n_trades']}
            - Alım İşlemleri: {action_counts.get('buy', 0)}
            - Satım İşlemleri: {action_counts.get('sell', 0)}
            - Tut İşlemleri: {action_counts.get('hold', 0)}
            
            Model Başarı Değerlendirmesi:
            - Doğruluk: {"Mükemmel (>80%)" if results['portfolio_accuracy'] > 80 else "İyi (>70%)" if results['portfolio_accuracy'] > 70 else "Orta (>60%)" if results['portfolio_accuracy'] > 60 else "Geliştirilebilir (<60%)"}
            - Kar Performansı: {"Mükemmel (>20%)" if results['profit_percent'] > 20 else "İyi (>10%)" if results['profit_percent'] > 10 else "Orta (>0%)" if results['profit_percent'] > 0 else "Geliştirilebilir (<0%)"}
            - Risk Değerlendirmesi: {"Düşük Risk (DD<10%)" if results['max_drawdown'] < 10 else "Orta Risk (DD<20%)" if results['max_drawdown'] < 20 else "Yüksek Risk (DD>20%)"}
            
            Geliştirilecek Alanlar:
            - {"Daha az işlem yapılabilir" if results['n_trades'] > 50 else "Daha fazla işlem yapılabilir" if results['n_trades'] < 10 else "İşlem sayısı uygun"}
            - {"Risk yönetimi iyileştirilebilir" if results['max_drawdown'] > 15 else "Risk yönetimi iyi durumda"}
            - {"Doğruluk oranı artırılabilir" if results['portfolio_accuracy'] < 65 else "Doğruluk oranı iyi durumda"}
            """
            
            plt.text(0.5, 0.5, report_text, horizontalalignment='center', 
                    verticalalignment='center', transform=plt.gca().transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.9), 
                    linespacing=1.5)
            
            plt.savefig('bitcoin_trader_ai_accuracy_report.png')
            logger.info("Doğruluk raporu kaydedildi: bitcoin_trader_ai_accuracy_report.png")
            
        except Exception as e:
            logger.error(f"Rapor oluşturma hatası: {e}")

if __name__ == "__main__":
    # Komut satırı argümanlarını ayarla
    parser = argparse.ArgumentParser(description='Bitcoin Trader AI - Reinforcement Learning Tabanlı Trading Bot')
    parser.add_argument('--mode', type=str, default='train_and_test', choices=['train_and_test', 'train_only', 'test_only'],
                        help='Çalıştırma modu: train_and_test, train_only, test_only')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='İşlem çifti (örn. BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Zaman dilimi (örn. 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--train_days', type=int, default=60, help='Eğitim için geçmiş gün sayısı')
    parser.add_argument('--test_days', type=int, default=14, help='Test için geçmiş gün sayısı')
    parser.add_argument('--timesteps', type=int, default=50000, help='Eğitim adım sayısı')
    parser.add_argument('--load_existing', action='store_true', help='Mevcut modeli yükle')
    parser.add_argument('--accuracy_threshold', type=float, default=65.0, 
                       help='Hedef doğruluk eşiği (belirtilen değere ulaşırsa eğitim erken sonlandırılır)')
    
    args = parser.parse_args()
    
    # Tarih aralıklarını hesapla
    current_date = datetime.now()
    train_start_date = (current_date - timedelta(days=args.train_days)).strftime('%Y-%m-%d')
    train_end_date = (current_date - timedelta(days=args.test_days)).strftime('%Y-%m-%d')
    test_start_date = (current_date - timedelta(days=args.test_days-1)).strftime('%Y-%m-%d')
    test_end_date = current_date.strftime('%Y-%m-%d')
    
    print(f"\n{'='*50}")
    print(f"BITCOIN TRADER AI - RL TABANLI TRADING BOT")
    print(f"{'='*50}")
    print(f"Mod: {args.mode}")
    print(f"Sembol: {args.symbol}")
    print(f"Zaman Dilimi: {args.timeframe}")
    print(f"Eğitim: {train_start_date} - {train_end_date} ({args.train_days} gün)")
    print(f"Test: {test_start_date} - {test_end_date} ({args.test_days} gün)")
    print(f"Adım Sayısı: {args.timesteps}")
    print(f"Mevcut Model: {'Evet' if args.load_existing else 'Hayır'}")
    print(f"{'='*50}\n")
    
    # Accuracy callback için eşiği ayarla
    os.environ['ACCURACY_THRESHOLD'] = str(args.accuracy_threshold)
    
    try:
        # Botu oluştur
        bot = BitcoinTraderAI(
            symbol=args.symbol,
            timeframe=args.timeframe,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date
        )
        
        # Botu çalıştır
        success = bot.run(
            mode=args.mode, 
            total_timesteps=args.timesteps, 
            load_existing=args.load_existing,
            accuracy_threshold=args.accuracy_threshold
        )
        
        # Sonuç durumunu göster
        if success:
            print(f"\n{'='*50}")
            print("İşlem başarıyla tamamlandı!")
            print(f"{'='*50}")
        else:
            print(f"\n{'='*50}")
            print("İşlem tamamlanamadı. Lütfen hata günlüğünü kontrol edin.")
            print(f"{'='*50}")
            
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"HATA: {str(e)}")
        print("İşlem sırasında bir hata oluştu. Lütfen hata günlüğünü kontrol edin.")
        print(f"{'='*50}")
        logger.error(f"Ana program hatası: {str(e)}", exc_info=True)
        sys.exit(1)
    
    # Örnek kullanım:
    # python bitcoin_trader_ai.py --mode train_and_test --symbol BTC/USDT --timeframe 1h --train_days 60 --test_days 14 --timesteps 50000
    # python bitcoin_trader_ai.py --mode test_only --load_existing
