#!/usr/bin/env python
# Hızlı demo - CCXT ile gerçek veriler ve 5 saniyelik aralıklarla çalışan bitcoin ticaret botu

import os
import sys
import time
import logging
from datetime import datetime

# Ana modülü içe aktar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bitcoin_trader_bot import BitcoinTraderBot, logger

if __name__ == "__main__":
    print("=" * 60)
    print("BITCOIN TRADER BOTU HIZLI DEMO")
    print("=" * 60)
    print("Gerçek BTC verileri ile simülasyon yapılıyor (CCXT kütüphanesi kullanılarak)")
    print("5 iterasyon, her biri 5 saniye aralıklarla")
    print("=" * 60)
    
    # Botu oluştur
    bot = BitcoinTraderBot(
        exchange_id='binance',  # Veri için Binance kullan
        symbol='BTC/USDT',
        timeframe='1h',
        initial_balance=1000  # 1000 USDT başlangıç bakiyesi
    )
    
    # 5 iterasyon çalıştır, 5 saniye aralıklarla
    print("Başlangıç Portföy Değeri: $1000.00")
    print("-" * 50)
    
    # İlk iterasyonu hemen çalıştır
    bot.run_single_iteration()
    
    # 4 kere daha 5 saniye aralıklarla çalıştır
    for i in range(4):
        print(f"\nSonraki iterasyon bekleniyor ({i+2}/5)...")
        time.sleep(5)  # 5 saniye bekle (demo için kısaltılmış)
        bot.run_single_iteration()
    
    # Final portföy sonuçları
    final_value = bot.portfolio_value
    profit_loss_amount = final_value - 1000
    profit_loss_percent = (profit_loss_amount / 1000) * 100
    
    print("\n" + "=" * 50)
    print("TİCARET SİMÜLASYONU SONUÇLARI")
    print("=" * 50)
    print(f"Başlangıç Portföy Değeri: $1000.00")
    print(f"Final Portföy Değeri:     ${final_value:.2f}")
    print(f"Kâr/Zarar:                ${profit_loss_amount:.2f} ({profit_loss_percent:.2f}%)")
    print("=" * 50)
    print(f"Ticaret aktivitesi: {len(bot.trade_history)} işlem")
    
    # İşlem detaylarını yazdır
    if bot.trade_history:
        print("\nİŞLEM GEÇMİŞİ:")
        print("-" * 50)
        for i, trade in enumerate(bot.trade_history):
            print(f"İşlem #{i+1} - {trade['type'].upper()} - ${trade['price']:.2f}")
            print(f"  Miktar: {trade['amount']:.6f} BTC")
            print(f"  Değer: ${trade['value']:.2f}")
            print(f"  Sinyaller: {', '.join(trade['signals'])}")
            print(f"  Zaman: {trade['timestamp']}")
            print("-" * 30)
    
    print("\nDetaylı grafikler için trading_performance.png dosyasını kontrol edin") 