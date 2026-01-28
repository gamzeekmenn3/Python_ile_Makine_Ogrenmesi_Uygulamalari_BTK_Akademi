'''
Bu program, Olivetti Faces veri setini kullanarak 40 farklı kişiyi yüz görüntülerinden ayırt edebilen bir Random Forest Classifier modeli oluşturur. Yüksek boyutlu piksel verileri üzerinde çalışan bu model, 
bir çoklu sınıflandırma (multi-class classification) görevidir.

Kullanılan Kütüphaneler
- sklearn.datasets: 400 gri tonlamalı yüz görüntüsü içeren Olivetti Faces setini yükler.
- sklearn.ensemble: Kolektif öğrenme (ensemble learning) temelli RandomForestClassifier modelini sağlar.
- sklearn.model_selection: Veriyi %80 Eğitim ve %20 Test olarak böler.
- sklearn.metrics: Modelin doğru kişi eşleştirme oranını (Accuracy) hesaplar.
- matplotlib.pyplot: Ham yüz verilerini ekrana yansıtır.
'''

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

oli = fetch_olivetti_faces()
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[i], cmap = "gray")
    plt.axis("off")
plt.show()

X = oli.data
y = oli.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Acc: ", accuracy)

'''
Kod Akışı ve Model Analizi
- Veri Seti Yapısı: Olivetti Faces veri seti; 40 farklı kişiye ait, her biri 64x64 piksel (toplam 4096 özellik) boyutunda 400 gri tonlamalı yüz görüntüsü içerir.
- Veri Hazırlığı: Görüntüler 4096 boyutlu vektörler halinde hazır gelir. Modelin başarısını objektif ölçmek için veri seti %80 eğitim ve %20 test olarak ikiye ayrılır.
- Topluluk Öğrenmesi (Ensemble Learning): Rastgele Orman (RF), birbirinden bağımsız 100 farklı karar ağacını aynı anda eğitir. Her ağacın kararı oylanır; bu sayede tek bir karar ağacının hata yapma payı düşürülerek daha istikrarlı bir sonuç elde edilir.
- Tahmin ve Doğruluk (Accuracy): Model, test setindeki pikselleri analiz ederek görüntünün hangi kişiye ait olduğunu tahmin eder. Elde edilen Accuracy Skoru, modelin doğru eşleştirme yüzdesini temsil eder.
- Hiperparametre Kontrolü (n_estimators): Modelde kullanılan 100 ağaç, hem işlem hızı hem de doğruluk performansı arasında en verimli dengeyi (optimum) sağlar. Ağaç sayısının artması genelde performansı iyileştirir ancak donanım maliyetini ve süreyi artırır.
'''

# %%
'''
Bu program, California Housing veri setini kullanarak bölge bazlı konut fiyatlarını tahmin eden bir Random Forest Regressor modeli oluşturur. Topluluk (ensemble) öğrenme yöntemini kullanarak karmaşık özellikler ile konut fiyatları arasındaki ilişkiyi analiz eder.
Kullanılan Kütüphaneler:
•	sklearn.datasets: California Housing veri setini yüklemek.
•	sklearn.ensemble: Birden fazla karar ağacını birleştirerek daha kararlı tahminler üreten RandomForestRegressor modelini sağlar.
•	sklearn.model_selection: Veriyi %80 Eğitim ve %20 Test kümelerine ayırır.
•	sklearn.metrics: Modelin hata payını ölçmek için Mean Squared Error (MSE) fonksiyonunu sunar.
•	numpy: MSE değerinden RMSE'ye geçiş için karekök hesaplaması yapar.
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
rf_reg = RandomForestRegressor(random_state = 42)
rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("rmse: ",rmse)

'''
Kod Akışı ve Model Analizi
- Veri Hazırlığı: California Housing veri setindeki 8 temel özellik (gelir, oda sayısı, ev yaşı vb.) bağımsız değişkenler (X) olarak, konut fiyatları ise hedef değişken (y) olarak belirlenir. Verinin %20'si 
modelin performansını objektif test etmek için ayrılır.
- Topluluk Tabanlı Eğitim (Ensemble Learning): Rastgele Orman Regresyonu modeli yapılandırılır. Bu model, tek bir karar ağacı yerine çok sayıda ağacın tahminlerini birleştirip ortalamasını alarak çok daha kararlı 
ve genellenebilir sonuçlar üretir.
- Tahmin ve Test: Eğitimini tamamlayan model, test kümesindeki konutların fiyatlarını tahmin eder. Tahminlerin başarısı, konut fiyatları gibi sayısal veriler için en kritik ölçüt olan RMSE ile analiz edilir.

Performans Ölçütü: Karesel Ortalama Hata Kökü (RMSE)
- Doğrudan Yorumlanabilirlik: RMSE, modelin yaptığı hataların büyüklüğünü doğrudan konut fiyatı birimiyle ifade eder. Bu, modelin tahminlerinde ortalama kaç dolar/birim saptığını anlamamızı sağlar.
- Hata Duyarlılığı: Karesel işlem yapıldığı için büyük hataları daha çok cezalandırır; bu da modelin büyük sapmalardan ne kadar kaçındığını gösterir.
- Kıyaslama Standartı: Daha düşük bir RMSE değeri, modelin gerçeğe daha yakın tahminler ürettiğini kanıtlar. Bu metrik, Random Forest modelini Lineer Regresyon veya Gradient Boosting gibi diğer algoritmalarla 
kıyaslarken kullanılan temel referans noktasıdır.

Özetle: Elde edilen RMSE değeri, modelin mevcut haliyle başarısını belgelerken; hiperparametre ayarlaması (ağaç sayısı, derinlik vb.) gibi optimizasyon süreçleri için iyileştirilecek temel hedefi temsil eder.
'''
