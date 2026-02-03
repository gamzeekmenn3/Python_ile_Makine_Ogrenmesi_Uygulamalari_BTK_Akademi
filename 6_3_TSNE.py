'''
Bu projenin temel amacı, 784 boyutlu (28x28 piksel) MNIST el yazısı rakamları veri kümesini, aralarındaki benzerlikleri koruyarak 2 boyutlu bir düzleme taşımaktır. t-SNE, birbirine benzeyen 
rakamları (örneğin tüm 1'leri veya tüm 0'ları) aynı kümede toplamaya çalışarak verinin topolojik yapısını görselleştirir.

* Kullanılan Kütüphaneler:
- fetch_openml (Scikit-learn): MNIST veri kümesini (70,000 örnek) bulut üzerinden indirmek için kullanılır.
- TSNE (Scikit-learn): "t-Distributed Stochastic Neighbor Embedding" algoritmasını uygulamak için kullanılır. Bu algoritma, yüksek boyutlu uzaydaki yakın noktaların düşük boyutlu uzayda da 
yakın kalmasını sağlar.
- Matplotlib: Hesaplanan t-SNE koordinatlarını renkli bir harita üzerinde çizdirmek için kullanılır.
'''
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = y, cmap = 'tab10', alpha = 0.6)
plt.title("TSNE of MNIST Dataset")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")

'''
* Kod Akışı
- Veri Hazırlığı: MNIST verisi indirilir. X pikselleri (girdi), y ise bu piksellerin hangi rakama (0-9) ait olduğu bilgisini (etiket) temsil eder.
- t-SNE Yapılandırması: TSNE(n_components=2) ile hedef boyut iki olarak belirlenir.
- Dönüştürme (Fit-Transform): Kodun en yoğun çalıştığı kısımdır. t-SNE, her bir veri noktası için olasılıksal benzerlikler hesaplar.
    Not: t-SNE iteratif bir algoritma olduğu için PCA'ya göre çok daha yavaş çalışır ve işlemcinin gücüne göre tamamlanması birkaç dakika sürebilir.
- Görselleştirme: plt.scatter fonksiyonuyla, her rakam farklı bir renkle (cmap='tab10') ekrana basılır. Sonuçta, her rakam grubunun (0'dan 9'a) kendi adacıklarını oluşturduğu bulutsu bir 
yapı ortaya çıkar.
'''
