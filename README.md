# E-Commerce Churn Prediction (PyTorch + Gradio)

Bu proje, bir e-ticaret platformunda yer alan müşterilerin aboneliklerini
iptal etme (churn) olasılığını tahmin etmek amacıyla geliştirilmiştir.
Proje kapsamında tabular (tablo) veri üzerinde çalışan derin öğrenme
tabanlı bir model oluşturulmuştur.

---

## 1. Proje Konusu ve Seçilme Gerekçesi

Churn tahmini, e-ticaret platformları için müşteri kaybını önceden
öngörmek ve müşteri elde tutma (customer retention) stratejileri
geliştirmek açısından kritik bir problemdir.

Bu çalışmada kullanılan veri seti tablo (tabular) yapıdadır.
Bu nedenle görüntü verileri için kullanılan CNN mimarileri yerine,
tabular veriler üzerinde daha etkili sonuçlar veren
Çok Katmanlı Algılayıcı (MLP – Multi Layer Perceptron) tabanlı
bir derin öğrenme modeli tercih edilmiştir.

---

## 2. Veri Seti

Bu projede **E Commerce Customer Insights and Churn Dataset**
adlı veri seti kullanılmıştır.

Veri seti CSV formatındadır ve aşağıdaki bilgileri içermektedir:

- Demografik bilgiler (yaş, cinsiyet, ülke)
- Satın alma davranışları (alışveriş sıklığı, sipariş bilgileri)
- Ürün bilgileri (kategori, ürün adı)
- Abonelik durumu

### Churn Etiketi

- `subscription_status = cancelled` → churn = 1  
- `subscription_status = active` → churn = 0  

Veri seti, projenin çalıştırılabilmesi için GitHub reposu içerisinde
yer almaktadır.

---

## 3. Veri Ön İşleme ve Özellik Mühendisliği

Veri seti üzerinde aşağıdaki işlemler uygulanmıştır:

- Tarih alanları `datetime` formatına dönüştürülmüştür
- Eksik veriler doldurulmuştur
- Yeni özellikler türetilmiştir:
  - `total_order_value = unit_price × quantity`
  - `days_since_signup`
  - `days_since_last_purchase`
- Sayısal veriler standartlaştırılmıştır
- Kategorik veriler one-hot encoding ile dönüştürülmüştür

---

## 4. Kullanılan Model ve Yöntem

Model, PyTorch kullanılarak geliştirilmiş bir
**MLP (Multi Layer Perceptron)** mimarisidir.

Model özellikleri:

- Girdi katmanı: Sayısal + kategorik özellikler
- Gizli katmanlar: ReLU aktivasyon fonksiyonu
- Çıkış katmanı: Binary sınıflandırma (churn / not churn)
- Kayıp fonksiyonu: `BCEWithLogitsLoss`
- Optimizasyon algoritması: Adam

---

## 5. Model Eğitimi ve Değerlendirme

Veri seti %80 eğitim ve %20 test olacak şekilde ayrılmıştır.

Test verisi üzerinde elde edilen performans metrikleri:

Accuracy : 0.5912
Precision : 0.2959
Recall : 0.2929
F1 Score : 0.2944


Bu sonuçlar, veri setindeki sınıf dengesizliği nedeniyle
precision ve recall değerlerinin görece düşük olduğunu göstermektedir.
Buna rağmen model churn eğilimini öğrenebilmiştir.

---

## 6. Kurulum ve Çalıştırma

6.1 Ortam Kurulumu

Python 3.x sürümü önerilmektedir.
Gerekli kütüphaneler aşağıdaki komut ile kurulabilir:

```bash
pip install torch pandas numpy scikit-learn gradio

6.2 Modeli Eğitme

Aşağıdaki komut çalıştırıldığında model eğitilir ve
churn_artifact.pt dosyası oluşturulur:

python train.py

Eğitim tamamlandığında modelin performans metrikleri
terminalde görüntülenir

6.3 Gradio Arayüzünü Başlatma

Model eğitildikten sonra tahmin yapmak için
Gradio tabanlı kullanıcı arayüzü başlatılır:

python serve.py

Terminal çıktısında aşağıdaki adrese benzer bir bağlantı oluşur:

http://127.0.0.1:7860


7. Proje Dosya Yapısı

ecomm-churn-prediction/
│
├── train.py        # Veri hazırlama ve model eğitimi
├── model.py        # PyTorch MLP modeli
├── serve.py        # Gradio arayüzü
├── README.md       # Proje dokümantasyonu
├── .gitignore
└── E Commerce Customer Insights and Churn Dataset (1).csv


8. Sonuç

Bu projede, e-ticaret müşteri verileri kullanılarak churn tahmini
problemi derin öğrenme yaklaşımı ile ele alınmıştır.
MLP tabanlı model ile hem eğitim hem de gerçek zamanlı
tahmin yapılabilmektedir.

Proje, veri ön işleme, model eğitimi, değerlendirme ve
kullanıcı arayüzü aşamalarını kapsayan
uçtan uca bir derin öğrenme uygulamasıdır.
