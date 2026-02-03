'''
Bu proje, Gözetimsiz Öğrenme (PCA) ve Gözetimli Öğrenme (LDA) tekniklerini kıyaslar.
* PCA (Temel Bileşenler Analizi): Sınıf etiketlerine bakmadan, verideki en büyük değişimi (varyansı) bulmaya çalışır.
* LDA (Doğrusal Ayrıştırma Analizi): Sınıf etiketlerini kullanarak, farklı sınıflar arasındaki ayrımı maksimize etmeye çalışır. Kod, bu iki yöntemi hem karmaşık MNIST (el yazısı rakamlar) 
hem de klasik Iris veri kümesi üzerinde test eder.

* Kullanılan Kütüphaneler
- fetch_openml (Scikit-learn): İnternet üzerinden MNIST gibi büyük veri kümelerini çekmek için kullanılır.
- LinearDiscriminantAnalysis: Sınıfları birbirinden en iyi ayıracak bileşenleri bulmak için kullanılan algoritma.
- PCA: Boyut indirgemek için kullanılan istatistiksel yöntem.
- Matplotlib: Sonuçları renkli grafiklerle (scatter plot) görselleştirmek için kullanılır.
'''
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target.astype(int)

lda  = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c = y, cmap = 'tab10', alpha = 0.6)
plt.title("LDA of MNIST Dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")  
plt.colorbar(label='Digits')

#%% PCA & LDA comparison
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

colors = ['red', 'green', 'blue']

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha = 0.8, color = color, label = target_name)
plt.legend()
plt.title('PCA of Iris Dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha = 0.8, color = color, label = target_name)
plt.legend()
plt.title('LDA of Iris Dataset')

'''
* Kod Akışı: 
1. Bölüm: MNIST Üzerinde LDA Uygulaması
- Veri Çekme: 70,000 adet el yazısı rakam (0-9 arası) görseli indirilir. Her görsel 784 pikselden (boyuttan) oluşur.
- LDA ile Boyut İndirme: 784 boyutlu devasa veri, sınıflar arasındaki farkı en iyi temsil eden 2 boyuta indirilir.
- Görselleştirme: 10 farklı rakamın 2 boyutlu düzlemde nasıl kümelendiği (tab10 renk haritası ile) gösterilir.

2. Bölüm: PCA ve LDA Karşılaştırması (Iris Verisi)
- PCA Uygulanışı: Iris verisi, sınıflara (çiçek türlerine) bakılmaksızın en geniş yayıldığı 2 eksene izdüşürülür.
- LDA Uygulanışı: Iris verisi, sınıfların birbirine en az karıştığı 2 eksene izdüşürülür.
- Kıyaslama: İki ayrı grafik oluşturulur. Genelde LDA grafiğinde çiçek türlerinin (nokta kümelerinin) birbirinden daha net ve uzak şekilde ayrıldığı gözlemlenir, çünkü LDA'nın asıl amacı budur.
'''
