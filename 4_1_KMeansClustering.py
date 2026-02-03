'''
Bu çalışma, rastgele oluşturulmuş veri kümesi üzerinde kümeleme analizi yapmayı amaçlar. Algoritma, birbirine yakın noktaları tespit ederek veriyi 4 farklı gruba ayırır ve her grubun merkez noktasını (centroid)
belirler. Temel hedef, verideki gizli yapıları keşfetmektir.

* Kullanılan Kütüphaneler:
- sklearn.datasets (make_blobs): Test amaçlı, yapay "nokta bulutları" (kümeler) oluşturmak için kullanılır.
- sklearn.cluster (KMeans): K-Means algoritmasını uygulayan ana makine öğrenmesi sınıfıdır.
- matplotlib.pyplot: Verileri ve algoritma sonuçlarını grafik üzerinde görselleştirmek için kullanılan kütüphanedir.
'''
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.6, random_state = 42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = labels, cmap = "viridis")

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = "red", marker = "X")
plt.title("K-Means")

'''
* Kod Akışı:
- Veri Üretimi: make_blobs ile 300 örnekten oluşan, 4 merkez etrafında toplanmış yapay bir veri seti oluşturulur.
- İlk Görselleştirme: Herhangi bir işlem yapılmadan önce verinin ham hali (tek renk) ekrana basılır.
- Algoritmanın Eğitilmesi:
  - KMeans(n_clusters=4) ile model tanımlanır.
  - fit(X) komutuyla algoritma verideki merkezleri aramaya başlar.
- Sonuçların Görselleştirilmesi:
  - labels değişkeni ile her noktanın hangi kümeye ait olduğu belirlenir ve farklı renklerle çizilir.
  - cluster_centers_ kullanılarak, algoritmanın bulduğu küme merkezleri kırmızı bir "X" işareti ile grafiğe eklenir.
'''
