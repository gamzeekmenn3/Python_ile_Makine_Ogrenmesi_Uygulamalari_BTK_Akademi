"""
Bu çalışma, Scikit-learn kütüphanesindeki "Breast Cancer" veri setini kullanarak, tümörlerin klinik özelliklerine göre iyi huylu (benign) veya kötü huylu(malignant) olup 
olmadığını tahmin eden bir KNN (K-En Yakın Komşu) modelidir.

Proje İçeriği:
- Veri Analizi: Veri setinin yapısal incelenmesi ve hedef değişken dağılımının belirlenmesi.
- Ön İşleme (Preprocessing): Modelin mesafe tabanlı çalışması nedeniyle verilerin StandardScaler ile normalize edilmesi.
- Model ve Eğitim: K=3 parametresi ile KNN modelinin eğitilmesi.
- Performans Metrikleri: Model başarısının Accuracy Score ve Confusion Matrix ile ölçülmesi.
- Hiperparametre Optimizasyonu: Modelin başarısını etkileyen en önemli değer olan $K$ parametresinin (1-20 arası) test edilerek en iyi doğruluk skorunu veren K değerinin 
  bulunması ve görselleştirilmesi.

Kullanılan Kütüphaneler:
•	sklearn: Makine öğrenmesi algoritmaları ve araçları.
•	pandas: Veri manipülasyonu ve analizi.
•	matplotlib: Grafik ve görselleştirme.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dogruluk:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("confusion_matrix:")
print(conf_matrix)

"""
    KNN: Hyperparameter = K
        K: 1, 2, 3, ... N
        Accuracy: %A, %B, %C ....
"""
accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.figure()
plt.plot(k_values, accuracy_values, marker = "o", linestyle ="-")
plt.title("K degerine göre dogruluk")
plt.xlabel("K degeri")
plt.ylabel("Dogruluk")   
plt.xticks(k_values)
plt.grid(True)

'''
Kod Akışı:
1.	Veri seti yüklenir ve bağımsız değişkenler (X) ile bağımlı değişken (y) olarak ayrılır.
2.	Veri %70 eğitim, %30 test olacak şekilde bölünür ve ölçeklendirilir.
3.	KNN sınıflandırıcısı eğitilir ve test seti üzerinden tahminler üretilir.
4.	Elde edilen doğruluk sonuçları ve hata matrisi ekrana yazdırılır.
5.	Döngü yardımıyla farklı $K$ değerleri için doğruluk grafiği oluşturulur; bu grafik, modelin hangi $K$ değerinde "en dengeli" sonucu verdiğini gösterir
'''


# %%
'''
Bu bölüm, KNN algoritmasının sadece sınıflandırmada değil, regresyon (sayısal değer tahmini) problemlerinde de nasıl çalıştığını gürültülü bir sinüs dalgası üzerinden 
simüle eder.

Proje İçeriği
- Sentetik Veri Üretimi: Kontrollü bir deney alanı oluşturmak için sinüs dalgasına rastgele gürültü (noise) eklenmesi.
- Regresyon Analizi: KNeighborsRegressor kullanımı.
- Ağırlık Fonksiyonu Karşılaştırması: uniform (eşit ağırlık) ve distance (mesafe bazlı ağırlık) yöntemlerinin tahmin çizgisi üzerindeki etkisinin analizi.

Kullanılan Kütüphaneler
•	numpy: Matematiksel işlemler ve matris yönetimi.
•	sklearn.neighbors.KNeighborsRegressor: Sayısal tahmin modeli.
•	matplotlib.pyplot: Tahmin eğrilerinin görsel kıyaslaması.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

X = np.sort(5 * np.random.rand(40, 1), axis = 0)
y = np.sin(X).ravel() 
# plt.scatter(X, y)

y[::5] += 1 * (0.5 - np.random.rand(8))
# plt.scatter(X, y)

T = np.linspace(0, 5, 500)[:, np.newaxis]

for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors = 5, weights = weight)
    y_pred = knn.fit(X, y).predict(T)
    
    plt.subplot(2, 1, i+1)
    plt.scatter(X, y, color = "green", label = "data")
    plt.plot(T, y_pred, color = "blue", label = "prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN Regressor weights = {}".format(weight))
plt.tight_layout()
plt.show()

'''
Kod Akışı:
- Numpy ile 0-5 aralığında rastgele noktalar üretilir ve sinüs fonksiyonuna göre hedefleri belirlenir.
- Veriye kasıtlı olarak gürültü eklenerek gerçek hayat verisi simüle edilir.
- Model, iki farklı ağırlıklandırma stratejisiyle (uniform vs distance) eğitilir.
Sonuç Analizi:
- Uniform: Komşuların aritmetik ortalamasını aldığı için gürültüye karşı daha katı ve basamaklı bir yapı sergiler.
- Distance: Yakın olan komşuya daha fazla güvenildiği için orijinal veri trendine (sinüs dalgasına) daha sadık ve esnek bir çizgi çizer.
- İki modelin farkı subplot yapısıyla görselleştirilerek "overfitting" ve "smoothness" kavramları vurgulanır.
'''
