@echo off
echo Bitcoin Trader AI başlatılıyor...
echo Gerekli kütüphaneler yükleniyor...
pip install numpy pandas matplotlib ccxt gym stable-baselines3 python-dotenv torch
echo.
echo Geliştirilmiş Bitcoin Trader AI çalıştırılıyor...
echo Doğruluk odaklı eğitim için daha az adım ve doğruluk eşiği kullanılıyor.
echo Log dosyasını kontrol etmek için bitcoin_trader_ai.log dosyasını izleyin.
echo.
cd C:\bitcoin_trader_bot
python bitcoin_trader_ai.py --mode train_and_test --symbol BTC/USDT --timeframe 1h --train_days 30 --test_days 7 --timesteps 15000 --accuracy_threshold 55
pause 