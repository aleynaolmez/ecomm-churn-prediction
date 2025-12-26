# E-Commerce Churn Prediction (PyTorch + Gradio)

Bu proje, bir e-ticaret platformunda yer alan müşterilerin aboneliklerini
iptal etme (churn) olasılığını tahmin etmek amacıyla geliştirilmiştir.
Derin öğrenme tabanlı bir sınıflandırma modeli kullanılarak müşterilerin
davranışsal ve demografik özelliklerinden yararlanılmıştır.

Proje kapsamında PyTorch kullanılarak Çok Katmanlı Algılayıcı (MLP – Multi Layer Perceptron)
tabanlı bir model eğitilmiş, ardından Gradio ile kullanıcı dostu bir web arayüzü
oluşturulmuştur.

---

## 1. Proje Konusu ve Seçilme Gerekçesi

Churn tahmini, işletmeler için müşteri kaybını önceden tespit ederek
müşteri elde tutma (customer retention) stratejileri geliştirilmesi açısından
kritik öneme sahiptir.

Bu projede ele alınan problem, gerçek hayatta sıkça karşılaşılan bir iş
problemidir. Veri seti tablo (tabular) yapıda olduğu için görüntü verilerine
özgü CNN tabanlı modeller yerine, bu tür verilerde etkili sonuçlar veren
MLP tabanlı derin öğrenme yaklaşımı tercih edilmiştir.

---

## 2. Veri Seti

Bu projede **E Commerce Customer Insights and Churn Dataset** adlı veri seti
kullanılmıştır.

Veri seti aşağıdaki bilgileri içermektedir:

- Yaş (age)
- Ülke (country)
- Cinsiyet (gender)
- Satın alma sıklığı (purchase_frequency)
- İptal sayısı (cancellations_count)
- Ürün fiyatı ve miktarı
- Tercih edilen kategori
- Abonelik durumu (active / cancelled)

Hedef değişken (label), müşterinin aboneliğini iptal edip etmediğini
gösteren **churn** bilgisidir.

---

## 3. Kullanılan Yöntem ve Model Mimarisi

Projede PyTorch kullanılarak Çok Katmanlı Algılayıcı (MLP) tabanlı bir
derin öğrenme modeli geliştirilmiştir.

Modelin genel özellikleri:

- Girdi: Sayısal ve kategorik özelliklerin birleştirilmiş hali
- Kategorik veriler: One-Hot Encoding
- Sayısal veriler: Standartlaştırma (Standardization)
- Kayıp Fonksiyonu: Binary Cross Entropy Loss
- Optimizasyon Algoritması: Adam Optimizer

Bu yapı, ikili sınıflandırma (churn / not churn) problemleri için
uygun ve yaygın olarak kullanılan bir yaklaşımdır.

---

## 4. Model Eğitimi ve Değerlendirme Süreci

Model, veri setinin %80’i eğitim, %20’si test olacak şekilde bölünerek
eğitilmiştir.

Eğitim tamamlandıktan sonra model, test veri seti üzerinde
değerlendirilmiştir.

---

## 5. Model Performansı ve Değerlendirme

Modelin başarımı aşağıdaki sınıflandırma metrikleri kullanılarak ölçülmüştür:

- **Accuracy:** Genel doğruluk oranı
- **Precision:** Churn olarak tahmin edilen müşterilerin gerçekten churn olma oranı
- **Recall:** Gerçek churn olan müşterilerin ne kadarının yakalandığı
- **F1-Score:** Precision ve Recall’un dengeli ölçümü

### Test Sonuçları

| Metrik     | Değer  |
|------------|--------|
| Accuracy   | 0.5912 |
| Precision  | 0.2959 |
| Recall     | 0.2929 |
| F1-Score   | 0.2944 |

### Değerlendirme ve Yorum

Elde edilen sonuçlar, churn tahmin probleminin zorlu yapısını
yansıtmaktadır. Veri setinde sınıflar arasında dengesizlik bulunması
(churn olmayan müşterilerin daha fazla olması), özellikle Precision ve
Recall değerlerinin görece düşük çıkmasına neden olmuştur.

Buna rağmen model, churn davranışını belirli ölçüde öğrenmiş ve
anlamlı tahminler üretebilmiştir. Gerçek dünya senaryolarında churn
problemi karmaşık müşteri davranışlarına dayandığından, elde edilen
sonuçlar kabul edilebilir düzeydedir.

Gelecekte veri dengeleme teknikleri (oversampling / class weighting),
ek özellik mühendisliği veya farklı model mimarileri ile performansın
artırılması mümkündür.

---

## 6. Kurulum ve Çalıştırma

### 6.1 Ortam Kurulumu

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


## 7. Proje Dosya Yapısı

ecomm-churn-prediction/
│
├── train.py        # Veri hazırlama ve model eğitimi
├── model.py        # PyTorch MLP modeli
├── serve.py        # Gradio arayüzü
├── README.md       # Proje dokümantasyonu
├── .gitignore
└── E Commerce Customer Insights and Churn Dataset (1).csv


## 8. Sonuç

Bu projede, e-ticaret müşteri verileri kullanılarak churn tahmini
problemi derin öğrenme yaklaşımı ile ele alınmıştır.
MLP tabanlı model ile hem eğitim hem de gerçek zamanlı
tahmin yapılabilmektedir.

Proje, veri ön işleme, model eğitimi, değerlendirme ve
kullanıcı arayüzü aşamalarını kapsayan
uçtan uca bir derin öğrenme uygulamasıdır.
