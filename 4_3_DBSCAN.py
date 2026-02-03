'''
Bu proje, karmaşık geometrik yapılara sahip verilerin (iç içe iki halka) yoğunluk farklarına göre nasıl kümelendiğini gösterir. Geleneksel algoritmalar (K-Means gibi) bu tarz dairesel verilerde merkez odaklı
çalıştığı için başarısız olurken; DBSCAN, noktaların birbirine ne kadar yakın (yoğun) olduğuna bakarak küre şeklinde olmayan grupları başarıyla tespit eder.

* Kullanılan Kütüphaneler:
- sklearn.datasets (make_circles): İç içe geçmiş iki daire şeklinde sentetik veri seti oluşturmak için kullanılır.
- sklearn.cluster (DBSCAN): Yoğunluk tabanlı kümeleme algoritmasının ana motorudur.
- matplotlib.pyplot: Verinin başlangıçtaki ham halini ve kümeleme sonrası renkli halini görselleştirmek için kullanılır.
'''
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples = 1000, factor = 0.5, noise = 0.08, random_state = 42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])

dbscan = DBSCAN(eps = 0.15, min_samples = 15)
cluster_labels = dbscan.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
plt.title("DBSCAN sonuclari")

'''
Kod Akışı
1. Veri Seti Oluşturma: make_circles fonksiyonu ile biri diğerinin içinde olan iki halka oluşturulur. noise = 0.08 parametresi verilere biraz rastgelelik (gürültü) katarak gerçek hayat verisine benzetilir.
2. İlk Görselleştirme: Verilerin herhangi bir işlem yapılmadan önceki hali ekrana basılır.
3. Model Yapılandırma: DBSCAN algoritması iki kritik parametre ile tanımlanır:
  - eps = 0.15: İki nokta arasındaki maksimum komşuluk mesafesi.
  - min_samples = 15: Bir bölgenin "yoğun" sayılabilmesi için gereken minimum nokta sayısı.
4.Kümeleme (Fit & Predict): Algoritma veriyi tarar; yoğun bölgeleri küme olarak işaretler, yoğunluk dışında kalan noktaları ise "aykırı değer" (gürültü) olarak tanımlar.
5. Sonuç Görselleştirme: c = cluster_labels parametresi ile her küme farklı bir renkle boyanır. DBSCAN'in en büyük avantajı olan "iç içe halkaları birbirinden ayırma" yeteneği bu aşamada gözlemlenir.
'''
