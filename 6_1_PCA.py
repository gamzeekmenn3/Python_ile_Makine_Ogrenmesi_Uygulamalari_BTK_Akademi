'''
Bu proje, yüksek boyutlu verilerin (Iris veri setinde 4 özellik bulunur) bilgisini mümkün olduğunca koruyarak daha düşük boyutlara (2D ve 3D) indirilmesini ve görselleştirilmesini amaçlar. 
Bu işlem, verideki karmaşıklığı azaltarak sınıflar arasındaki ayrımı gözle görebilmemizi sağlar.

* Kullanılan Kütüphaneler
- Scikit-learn (sklearn): Veri kümesini yüklemek (load_iris) ve PCA algoritmasını uygulamak (PCA) için kullanıldı.
- Matplotlib: Elde edilen temel bileşenleri 2 boyutlu ve 3 boyutlu grafiklere dökmek için kullanıldı.
'''
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components = 2) # 2 adet temel bileşen
X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y ==i, 1], label = iris.target_names[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend()

#%%
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components = 3) # 3 adet temel bileşen
X_pca = pca.fit_transform(X)
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev = 150, azim = 110)

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y, s = 40)
ax.set_title("First three PCA Dimensions of Iris Dataset")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

'''
* Kod Akışı:
1. Bölüm: 2 Boyutlu (2D) Görselleştirme:
- Veri Yükleme: Iris verisindeki 4 özellik (çanak yaprak uzunluğu/genişliği, taç yaprak uzunluğu/genişliği) X değişkenine, çiçek türleri y değişkenine atanır.
- PCA Uygulaması: n_components = 2 parametresiyle verinin boyutunu 4'ten 2'ye düşürür. Bu, verideki en yüksek varyansı (bilgiyi) temsil eden iki yeni eksen (PC1 ve PC2) oluşturur.
- Çizim: plt.scatter ile her bir çiçek türü farklı bir renk/etiketle 2 boyutlu düzlemde gösterilir.

2. Bölüm: 3 Boyutlu (3D) Görselleştirme:
- Boyut Artırımı: Bu kez n_components = 3 seçilerek verinin %99'una yakınını temsil eden ilk üç bileşen hesaplanır.
- 3D Projeksiyon: fig.add_subplot(111, projection="3d") komutuyla üç boyutlu bir uzay oluşturulur.
- Görselleştirme: Veriler 3. boyutta (derinlik) dağıtılarak, sınıfların (Setosa, Versicolor, Virginica) birbirlerinden nasıl ayrıştığı daha net bir perspektifle sunulur.
'''
