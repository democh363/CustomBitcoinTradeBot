# Custom Bitcoin Trading Bot

Gerçek Bitcoin verilerini kullanan ve gelişmiş teknik analiz göstergeleri ile alım-satım kararları veren otomatik bir ticaret botudur.

## Özellikler

- CCXT kütüphanesi üzerinden gerçek BTC fiyat verilerini çeker
- Birden çok borsayı destekler (Binance, KuCoin, vb.)
- Çoklu teknik analiz göstergeleri ile alım-satım sinyalleri üretir
- Ağırlıklı karar verme sistemi ile akıllı ticaret stratejisi
- Stop-loss ve take-profit fonksiyonlarını destekler
- Portföy değeri takibi yapar
- Kapsamlı performans grafikleri oluşturur
- Demo modunda gerçek para riskine girmeden çalışır

## Gelişmiş Analiz Özellikleri

### 1. Temel Teknik Göstergeler
- **RSI (Göreceli Güç Endeksi)**: Alım-satım baskısını ölçer
  - RSI > 70: Aşırı alım bölgesi, satış sinyali
  - RSI < 30: Aşırı satım bölgesi, alım sinyali
- **EMA (Üssel Hareketli Ortalama)**: Fiyat trendini belirler
  - Kısa EMA (9) uzun EMA'yı (21) yukarı keserse: Alım sinyali
  - Kısa EMA (9) uzun EMA'yı (21) aşağı keserse: Satım sinyali
- **MACD (Hareketli Ortalama Yakınsama/Iraksama)**: Trend değişimlerini ve momentumu tespit eder
  - MACD > Sinyal çizgisi: Alım sinyali
  - MACD < Sinyal çizgisi: Satım sinyali

### 2. Gelişmiş Volatilite Analizi
- **Bollinger Bantları**: Fiyat volatilitesini analiz eder
  - Fiyat alt bandın altına düşerse: Aşırı satım, potansiyel alım fırsatı
  - Fiyat üst bandın üstüne çıkarsa: Aşırı alım, potansiyel satım fırsatı
  - Bant genişliği artarsa: Volatilite artışı, büyük hareket beklentisi

### 3. Hacim Analizi
- **Hacim Değişimi**: Fiyat hareketlerinin güvenilirliğini değerlendirir
  - Yüksek hacimli fiyat artışı: Güçlü alım sinyali
  - Düşük hacimli fiyat artışı: Zayıf trend, dikkat edilmeli
  - Hacim trendleri ve fiyat uyumsuzlukları: Olası trend dönüşü

### 4. Trend Gücü Analizi
- **ADX (Ortalama Yön Endeksi)**: Trend gücünü ölçer
  - ADX > 25: Güçlü trend varlığı
  - ADX < 20: Zayıf veya trend yok
  - +DI > -DI ve ADX yükseliyorsa: Güçlü yükseliş trendi
  - -DI > +DI ve ADX yükseliyorsa: Güçlü düşüş trendi

### 5. Gelişmiş Momentum Göstergeleri
- **Para Akışı Endeksi (MFI)**: RSI'ya benzer ama hacmi de hesaba katar
  - MFI < 20: Aşırı satım
  - MFI > 80: Aşırı alım
- **Stokastik RSI**: RSI'nın hassasiyetini artıran stokastik versiyonu
  - Çift teyit için RSI ile birlikte kullanılır

### 6. Destek ve Direnç Analizi
- **Otomatik Seviye Tespiti**: Önemli fiyat seviyelerini otomatik belirler
- **Fibonacci Geri Çekilme Seviyeleri**: Potansiyel destek ve direnç noktaları
  - Anahtar Fibonacci seviyeleri: 0.236, 0.382, 0.5, 0.618, 0.786

### 7. Piyasa Duyarlılığı Analizi
- **Fear & Greed Index**: Kripto piyasasındaki korku ve açgözlülük seviyesi
  - Extreme Fear (0-25): Potansiyel alım fırsatı
  - Extreme Greed (75-100): Potansiyel satım fırsatı

### 8. Çoklu Zaman Dilimi Analizi
- Farklı zaman dilimlerindeki (1s, 5dk, 15dk, 1s, 4s, 1g) sinyalleri birleştirir
- Güçlü alım/satım sinyalleri için birden fazla zaman diliminde teyit arar

## Ağırlıklı Karar Verme Sistemi

Bot, farklı göstergelere farklı ağırlıklar vererek daha akıllı kararlar alır:

| Gösterge   | Ağırlık | Açıklama                                      |
|------------|---------|-----------------------------------------------|
| RSI        | 1.5     | Güçlü aşırı alım/satım göstergesi             |
| EMA        | 1.0     | Temel trend belirleme                         |
| MACD       | 1.2     | Trend değişimi ve momentum                    |
| Bollinger  | 1.3     | Volatilite ve fiyat aşırılıkları              |
| ADX        | 1.1     | Trend gücü değerlendirmesi                    |
| MFI        | 1.2     | Hacim destekli momentum                       |
| Sentiment  | 0.8     | Piyasa duyarlılığı                            |
| Volume     | 0.9     | İşlem hacmi doğrulaması                       |

Bot, ağırlıklı puanlamaya göre alım, satım veya bekleme kararı verir:
- Alım için minimum eşik: 2.0 puan
- Satım için minimum eşik: 2.0 puan

## İşleyiş

1. **Veri Toplama**: CCXT kütüphanesi üzerinden gerçek zamanlı BTC verilerini çeker
2. **Temel Analiz**: RSI, EMA, MACD gibi temel göstergeleri hesaplar
3. **Gelişmiş Analiz**: Bollinger Bantları, ADX, MFI, Hacim Analizi gibi ileri göstergeleri hesaplar
4. **Piyasa Duyarlılığı**: Fear & Greed Index ile piyasa duyarlılığını analiz eder
5. **Ağırlıklı Puanlama**: Tüm göstergeleri ağırlıklarına göre değerlendirir
6. **Karar Verme**: Ağırlıklı puanlamaya göre alım, satım veya bekleme kararı verir
7. **İşlem Yapma**: Verilen kararı uygular (demo modunda sanal olarak)
8. **Portföy Takibi**: İşlemlerden sonra portföy değerini günceller
9. **Stop-Loss/Take-Profit**: Aşırı zarar veya kâr durumlarında otomatik satış yapar
10. **Performans Grafiği**: Tüm göstergeleri ve portföy performansını görselleştirir

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

## Çıktılar

- **Log Dosyası**: `trading_bot.log` içinde tüm işlemler ve analizler kaydedilir
- **Performans Grafiği**: `trading_performance.png` dosyasında:
  - Fiyat ve EMA göstergeleri
  - RSI göstergesi
  - MACD göstergesi
  - Portföy değeri değişimi
- **Konsol Çıktısı**: İşlemler, kararlar ve performans metrikleri

## Dikkat

Bu bot eğitim amaçlıdır ve gerçek para ile ticaret yapılması tavsiye edilmez. Kripto para ticareti yüksek risk içerir.

## Lisans

MIT 