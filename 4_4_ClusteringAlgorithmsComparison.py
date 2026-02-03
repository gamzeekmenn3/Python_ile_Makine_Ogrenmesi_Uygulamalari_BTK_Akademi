'''
Bu proje, Scikit-Learn kütüphanesi tarafından sağlanan sentetik veri setleri (daireler, aylar, kümeler ve rastgele dağılım) üzerinde, 6 farklı kümeleme algoritmasının veriyi nasıl gruplandırdığını test eder. 
Temel amaç, her algoritmanın geometrik olarak farklı şekillere sahip verilerde ne kadar başarılı olduğunu görsel olarak analiz etmektir.

* Kullanılan Kütüphaneler: 
- sklearn (scikit-learn): Projenin kalbidir. Kümeleme algoritmalarını (cluster), veri seti oluşturma araçlarını (datasets) ve veriyi ölçeklendirme modülünü (preprocessing) sağlar.
- numpy: Sayısal hesaplamalar, veri manipülasyonu ve rastgele veri üretimi için kullanılır.
- matplotlib.pyplot: Elde edilen sonuçların 2 boyutlu grafikler (scatter plot) üzerinde görselleştirilmesini sağlar.
'''
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples = n_samples, factor = 0.5, noise = 0.05)
noisy_moons = datasets.make_moons(n_samples = n_samples, noise = 0.05)
blobs = datasets.make_blobs(n_samples = n_samples)
no_structure = np.random.rand(n_samples, 2), None

clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward",
                    "AgglomerativeClustering", "DBSCAN", "Birch"]

colors = np.array(["b", "g", "r", "c", "m", "y"])
datasets = [noisy_circles, noisy_moons, blobs, no_structure]

plt.figure()
i = 1
for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    
    two_means = cluster.MiniBatchKMeans(n_clusters = 2)
    ward = cluster.AgglomerativeClustering(n_clusters = 2, linkage = "ward")
    spectral = cluster.SpectralClustering(n_clusters = 2)
    dbscan = cluster.DBSCAN(eps = 0.2)
    average_linkage = cluster.AgglomerativeClustering(n_clusters = 2, linkage = "average")
    birch = cluster.Birch(n_clusters = 2)
    
    clustering_algorithms = [two_means, ward, spectral, dbscan, average_linkage, birch]
    
    for name, algo in zip(clustering_names, clustering_algorithms):
        algo.fit(X)
        if hasattr(algo, "labels_"):
            y_pred = algo.labels_.astype(int)
        else:
            y_pred = algo.predict(X)
        
        plt.subplot(len(datasets), len(clustering_algorithms), i)
        if i_dataset == 0:
            plt.title(name)
        plt.scatter(X[:,0], X[:, 1], color = colors[y_pred].tolist(), s = 10)
        i += 1
'''
* Kod Akışı:
- Veri Setlerinin Hazırlanması: make_circles (içe içe halkalar), make_moons (ay şekilleri), make_blobs (standart kümeler) ve tamamen rastgele (no_structure) olmak üzere 4 farklı veri yapısı oluşturulur.
- Ön İşleme (Scaling): Her veri seti StandardScaler ile standartlaştırılır. Bu, algoritmaların değişkenlerin ölçeğinden etkilenmemesini sağlar.
- Algoritmaların Tanımlanması: MiniBatchKMeans, SpectralClustering, DBSCAN gibi farklı çalışma prensiplerine sahip 6 algoritma bir liste içinde yapılandırılır.
- Döngü ve Eğitim:
  - Dış döngü veri setlerini sırayla alır.
  - İç döngü her veri seti için 6 algoritmayı tek tek çalıştırır (fit).
- Tahmin ve Görselleştirme: Algoritmaların atadığı küme etiketleri (labels_) alınır ve plt.subplot kullanılarak her sonuç devasa bir ızgara (grid) üzerinde ilgili sütuna çizdirilir.
'''
