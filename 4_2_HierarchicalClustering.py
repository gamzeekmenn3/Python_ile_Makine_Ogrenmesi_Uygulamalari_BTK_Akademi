'''
Bu projenin amacı, rastgele oluşturulmuş veri noktalarını (blobs) birbirine olan uzaklıklarına göre gruplara ayırmak ve dört farklı bağlantı yönteminin (Ward, Single, Average, Complete) kümeleme sonuçları 
üzerindeki etkisini hem dendrogramlar hem de dağılım grafikleri üzerinden karşılaştırmaktır.

* Kullanılan Kütüphaneler:
- Scikit-learn (sklearn): Veri seti üretimi (make_blobs) ve kümeleme modelinin (AgglomerativeClustering) çalıştırılması için kullanılmıştır.
- SciPy (scipy): Hiyerarşik yapının görselleştirilmesi için kritik olan matematiksel hesaplamaları ve dendrogram çizimini sağlar.
- Matplotlib (plt): Elde edilen verilerin ve model sonuçlarının görselleştirilmesi, grafiklerin düzenlenmesi için kullanılmıştır.
'''
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.6, random_state = 42)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")

linkage_methods = ["ward", "single", "average", "complete"]

plt.figure()
for i, linkage_method in enumerate(linkage_methods, 1):
    
    model = AgglomerativeClustering(n_clusters = 4, linkage = linkage_method)
    cluster_labels = model.fit_predict(X)
    
    plt.subplot(2, 4, i)
    plt.title(f"{linkage_method.capitalize()} Linkage Dendrogram")
    dendrogram(linkage(X, method = linkage_method), no_labels = "True")
    plt.xlabel("Veri Noktları")
    plt.ylabel("Uzaklık")
    
    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c = cluster_labels, cmap = "viridis")
    plt.title(f"{linkage_method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
'''
* Kod Akışı: 
- Veri Hazırlama: make_blobs ile 4 merkezli, toplam 300 noktadan oluşan yapay bir veri seti oluşturulur ve ilk grafikte ham hali gösterilir.
- Yöntem Belirleme: Algoritmanın kümeleri birleştirirken kullanacağı dört temel strateji (linkage_methods) bir liste içinde tanımlanır.
- Döngü ve Model Eğitimi: Her bir bağlantı yöntemi için sırayla:
    - AgglomerativeClustering modeli tanımlanan yöntemle kurulur ve veriye uygulanır.
    - Verilerin hangi kümeye ait olduğu (cluster_labels) hesaplanır.
- Görselleştirme (Dendrogram): scipy kullanılarak her yöntemin oluşturduğu hiyerarşik yapı (dendrogram) üst satırdaki grafiklere çizilir. Bu, verilerin adım adım nasıl birleştiğini gösterir.
- Görselleştirme (Kümeleme): Alt satırdaki grafiklerde ise noktalar, modelin atadığı renklerle (küme etiketleri) 2 boyutlu düzlemde gösterilir.
'''
