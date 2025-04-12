# Custom Bitcoin Trading Bot

Gerçek Bitcoin verilerini kullanan ve teknik analiz göstergeleri ile alım-satım kararları veren otomatik bir ticaret botudur.

## Özellikler

- CCXT kütüphanesi üzerinden gerçek BTC fiyat verilerini çeker
- RSI, EMA ve MACD gibi teknik analiz göstergelerini hesaplar
- Göstergelere dayalı alım-satım sinyalleri üretir
- Stop-loss ve take-profit fonksiyonlarını destekler
- Portföy değeri takibi yapar
- Performans grafiği oluşturur
- Demo modunda gerçek para riskine girmeden çalışır

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/[KULLANICI_ADINIZ]/CustomBitcoinTradeBot.git
cd CustomBitcoinTradeBot
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. (İsteğe Bağlı) Gerçek alım-satım için `.env` dosyasını düzenleyin:
```
API_KEY=your_api_key
API_SECRET=your_api_secret
```

## Kullanım

### Hızlı Demo

5 iterasyonu hızlıca çalıştırmak için:

```bash
python quick_demo.py
```

### 20 Dakikalık Simülasyon

20 dakika boyunca her dakika bir iterasyon çalıştırmak için:

```bash
python bitcoin_trader_bot.py
```

## Ticaret Stratejisi

Bot şu sinyallere göre işlem yapar:

- **Alım Sinyalleri**:
  - RSI aşırı satış seviyesinin altında (30)
  - EMA kısa vadeli çizgisi uzun vadeli çizgiyi yukarı doğru kesiyor
  - MACD sinyal çizgisinin üzerinde

- **Satım Sinyalleri**:
  - RSI aşırı alım seviyesinin üzerinde (70)
  - EMA kısa vadeli çizgisi uzun vadeli çizgiyi aşağı doğru kesiyor
  - MACD sinyal çizgisinin altında

## Çıktılar

- `trading_bot.log`: İşlem kayıtları
- `trading_performance.png`: Performans grafiği ve teknik göstergeler

## Dikkat

Bu bot eğitim amaçlıdır ve gerçek para ile ticaret yapılması tavsiye edilmez. Kripto para ticareti yüksek risk içerir.

## Lisans

MIT 