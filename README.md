# 🩺 Makine Öğrenmesi ile Meme Tümörü Sınıflandırma

Bu projede, hücre çekirdeğine ait çeşitli ölçümler kullanılarak meme tümörlerinin **Benign (iyi huylu)** veya **Malignant (kötü huylu)** olarak sınıflandırılması amaçlanmıştır.

Proje, makine öğrenmesi sürecinin temel aşamalarını uçtan uca uygulamaktadır. Veri inceleme, veri ön işleme, öznitelik mühendisliği, öznitelik seçimi, model eğitimi, model karşılaştırması, çapraz doğrulama, hiperparametre ayarlama ve test değerlendirmesi gerçekleştirilmiştir.

> **Not:** Bu proje eğitim amacıyla hazırlanmıştır. Tıbbi teşhis amacıyla kullanılamaz ve doktor kararının yerine geçmez.

---

## 📌 Projenin Amacı

Amaç, hücre çekirdeğine ait ölçümlerden yararlanarak tümör sınıfını tahmin edebilen bir makine öğrenmesi modeli geliştirmektir.

Bu kapsamda farklı sınıflandırma algoritmaları karşılaştırılmış ve validation sonuçlarına göre en başarılı model belirlenmiştir.

---

## 📊 Veri Seti

Projede Scikit-learn içerisinde bulunan **Breast Cancer Wisconsin (Diagnostic)** veri seti kullanılmıştır.

Veri setinde:

- **569 örnek**
- **30 temel sayısal öznitelik**
- **2 sınıf** bulunmaktadır.

Hedef değişken:

- `0` → Benign (İyi huylu)
- `1` → Malignant (Kötü huylu)

Veri setindeki sınıf dağılımı:

| Sınıf | Örnek Sayısı |
|---|---:|
| Benign | 357 |
| Malignant | 212 |

Öznitelikler; yarıçap, doku, çevre, alan, konkavite ve benzeri hücre çekirdeği ölçümlerinden oluşmaktadır.

---

## 🛠️ Kullanılan Teknolojiler

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## ⚙️ Uygulanan Makine Öğrenmesi Süreci

1. Veri İnceleme

İlk olarak veri setinin:

- İlk satırları
- Veri setinin boyutu
- Veri tipleri
- Temel istatistikleri
- Sınıf dağılımı

incelenmiştir.

Veri setinin boyutu:

```text
569 satır
31 sütun

2. Eksik Değer Kontrolü

Veri setindeki eksik değerler kontrol edilmiştir.

Sonuç:

Eksik değer sayısı: 0

Bu nedenle herhangi bir eksik değer doldurma işlemi yapılmamıştır.

3. Aykırı Değer İncelemesi

Sayısal değişkenlerde IQR yöntemi kullanılarak potansiyel aykırı değerler incelenmiştir.

Toplam 608 potansiyel aykırı gözlem tespit edilmiştir.

Bu değerler doğrudan silinmemiştir. Bunun nedeni, veri setindeki ölçümlerin gerçek biyolojik farklılıkları temsil edebilmesi ve aykırı görünen değerlerin her zaman hatalı veri anlamına gelmemesidir.

4. Öznitelik Mühendisliği

Modelin kullanabileceği iki yeni öznitelik oluşturulmuştur:

radius_texture_ratio

Ortalama yarıçapın ortalama dokuya oranıdır.

perimeter_area_ratio

Ortalama çevrenin ortalama alana oranıdır.

Bu özelliklerin oluşturulmasında sıfıra bölme ihtimaline karşı küçük bir epsilon değeri kullanılmıştır.

5. Veri Ayrımı

Veri üç bölüme ayrılmıştır:

Veri	Oran	Örnek
Train	%70	398
Validation	%15	85
Test	%15	86

Sınıfların veri kümelerine dengeli şekilde dağılması için stratify kullanılmıştır.

Aynı sonuçların tekrar elde edilebilmesi için random_state=42 kullanılmıştır.

6. Öznitelik Seçimi

Öznitelik sayısını azaltmak ve modelin daha anlamlı değişkenlerle çalışmasını sağlamak amacıyla:

SelectKBest
f_classif

kullanılmıştır.

En iyi 10 öznitelik seçilmiştir.

Öznitelik seçimi model pipeline'ı içerisinde gerçekleştirildiği için cross-validation sırasında veri sızıntısının önüne geçilmiştir.

7. Ölçekleme

Özellikle Logistic Regression ve KNN gibi modeller için StandardScaler kullanılmıştır.

Ölçekleme yalnızca eğitim verisi üzerinde öğrenilmiş, validation ve test verilerine daha sonra uygulanmıştır.

Bu şekilde veri sızıntısının önlenmesi amaçlanmıştır.

🤖 Kullanılan Modeller

Projede dört farklı sınıflandırma modeli karşılaştırılmıştır:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest

Modeller validation verisi üzerinde Accuracy, Precision, Recall ve F1-Score metrikleri kullanılarak karşılaştırılmıştır.

📈 Model Karşılaştırması

Validation sonuçları:

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.9529	0.9667	0.9062	0.9355
Decision Tree	0.9529	0.9667	0.9062	0.9355
Random Forest	0.9412	0.9655	0.8750	0.9180
KNN	0.9059	0.9286	0.8125	0.8667

Validation sonucuna göre en başarılı model Logistic Regression olarak belirlenmiştir.

🔄 5-Fold Cross Validation

Modellerin farklı veri bölümlerindeki performansını görmek amacıyla 5-Fold Cross Validation uygulanmıştır.

F1 sonuçları:

Model	Ortalama F1
Logistic Regression	0.9162
KNN	0.9198
Decision Tree	0.8823
Random Forest	0.9151

Cross-validation sonucunda modellerin performanslarının birbirine yakın olduğu görülmüştür.

🔧 Hiperparametre Ayarlama

Validation sonucunda seçilen Logistic Regression modeli için GridSearchCV kullanılmıştır.

Denenen C değerleri:

0.1
1.0
10.0
50.0

En iyi sonuç:

C = 10.0

En iyi Cross Validation F1 sonucu:

0.9212
🧪 Test Sonuçları

Optimize edilen Logistic Regression modeli daha önce kullanılmamış test verisi üzerinde değerlendirilmiştir.

Sonuçlar:

Metrik	Sonuç
Accuracy	%98.84
Precision	%100.00
Recall	%96.88
F1-Score	%98.41
Confusion Matrix
[[54  0]
 [ 1 31]]

Buna göre:

TN: 54
FP: 0
FN: 1
TP: 31

Model test verisindeki 32 malignant örneğin 31'ini doğru sınıflandırmış, 1 örneği yanlış sınıflandırmıştır.

🔍 Özniteliklerin Etkisi

Logistic Regression modelinin katsayılarının mutlak büyüklükleri incelenmiştir.

En yüksek değerlere sahip öznitelikler:

Öznitelik	Önem
worst area	5.0733
mean perimeter	2.7683
worst radius	2.7570
mean radius	2.3320
worst perimeter	1.7383
mean concave points	1.7156
worst concave points	0.9329
mean area	0.5576
mean concavity	0.2003
perimeter_area_ratio	0.0982

Bu sonuçlara göre özellikle tümörün alan, çevre ve yarıçap gibi ölçümleri modelin kararında daha fazla etkili görünmektedir.

📊 Oluşturulan Grafikler

Proje çalıştırıldığında aşağıdaki grafikler oluşturulur:

Model Karşılaştırması

model_karsilastirma.png

Modellerin validation F1-Score değerlerini karşılaştırır.

Confusion Matrix

confusion_matrix.png

Test verisindeki doğru ve yanlış sınıflandırmaları gösterir.

Öznitelik Etkileri

feature_importance.png

Logistic Regression modelinde seçilen özniteliklerin katsayı büyüklüklerini gösterir.

⚠️ Projenin Sınırlılıkları

Bu proje eğitim amacıyla hazırlanmıştır.

Kullanılan veri seti sınırlı sayıda örnek içermektedir ve gerçek bir klinik uygulamayı tamamen temsil etmez.

Modelin yüksek test başarısı elde etmesi, gerçek hayattaki tüm hastalarda aynı performansın elde edileceği anlamına gelmez.

Bu nedenle model herhangi bir tıbbi teşhis veya tedavi kararında tek başına kullanılmamalıdır.

🚀 Kurulum

Öncelikle gerekli kütüphaneleri yükleyin:

pip install -r requirements.txt

Daha sonra projeyi çalıştırın:

python main.py

Program çalıştırıldığında analiz sonuçları terminalde gösterilir ve grafikler proje klasörüne kaydedilir.

📁 Proje Dosya Yapısı
meme-tumoru-siniflandirma/
│
├── main.py
├── requirements.txt
├── README.md
│
├── model_karsilastirma.png
├── confusion_matrix.png
└── feature_importance.png
🎯 Sonuç

Bu projede makine öğrenmesi sürecinin temel aşamaları uçtan uca uygulanmıştır.

Dört farklı sınıflandırma modeli karşılaştırılmış, validation sonucunda Logistic Regression seçilmiş ve Grid Search ile hiperparametre ayarlaması yapılmıştır.

Optimize edilen model test verisinde:

%98.84 Accuracy ve %98.41 F1-Score elde etmiştir.

Proje sonucunda hücre çekirdeğine ait ölçümlerin tümör sınıflandırmasında kullanılabilecek anlamlı bilgiler taşıdığı görülmüştür.

Ancak elde edilen sonuçlar yalnızca bu veri seti ve bu çalışma kapsamında değerlendirilmelidir.