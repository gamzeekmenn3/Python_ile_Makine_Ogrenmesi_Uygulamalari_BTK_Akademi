'''
Bu çalışma, farklı geometrik yapılara sahip üç sentetik veri seti üzerinde, beş temel makine öğrenmesi algoritmasının performansını ve karar sınırlarını görsel bir matris üzerinde karşılaştırmaktadır.
Analizin Temel Amacı: Algoritmaların (doğrusal, doğrusal olmayan ve topluluk tabanlı) veri dağılımlarına nasıl tepki verdiğini ve hangi durumlarda daha iyi genelleme sağladığını görselleştirmektir.
* Kullanılan Teknolojiler ve Kütüphaneler
- Veri Üretimi: sklearn.datasets ile sentetik küme, hilal ve daire veri setleri oluşturulmuştur.
- Modelleme: * KNeighborsClassifier (Mesafe tabanlı)
             * SVC (Destek Vektör Makineleri)
             * DecisionTreeClassifier & RandomForestClassifier (Ağaç tabanlı)
             * GaussianNB (Olasılıksal)
- Önişleme: StandardScaler ile veri ölçekleme yapılmış, işlemler make_pipeline ile birleştirilmiştir.
- Görselleştirme: DecisionBoundaryDisplay kullanılarak modellerin tahmin alanları çizilmiştir.
'''

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

X, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 2, n_clusters_per_class = 1,random_state = 42)
X += 1.2 * np.random.uniform(size = X.shape)
Xy = (X, y)
# plt.scatter(X[:, 0], X[:, 1], c = y)

# X, y = make_moons(noise = 0.2, random_state = 42)
# plt.scatter(X[:, 0], X[:, 1], c = y)
# X, y = make_circles(noise = 0.1, factor = 0.3, random_state = 42)
# plt.scatter(X[:, 0], X[:, 1], c = y) 

datasets = [Xy,
            make_moons(noise = 0.2, random_state = 42),
            make_circles(noise = 0.35, factor = 0.3, random_state = 42)]    

fig = plt.figure(figsize = (6, 9))
i = 1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    
    ax = plt.subplot(len(datasets), 1, i)
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.coolwarm, edgecolors = "black")
    i += 1
plt.show()

names = ["Nearest Neighbors", "linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [KNeighborsClassifier(),  SVC(),  DecisionTreeClassifier(),  RandomForestClassifier(),  GaussianNB()]

fig = plt.figure(figsize = (6, 9))
i = 1
for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
    cm_bright = ListedColormap(["darkred", "darkblue"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
        
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cm_bright, edgecolors = "black")
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cm_bright, edgecolors = "black", alpha = 0.6)    
    i += 1

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)  # accuracy
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap = plt.cm.RdBu, alpha = 0.7, ax = ax, eps = 0.5)
        
        ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cm_bright, edgecolors = "black")
        ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train, cmap = cm_bright, edgecolors = "black", alpha = 0.6)
        
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(
            X[:, 0].max() - 0.15,
            X[:, 1].min() - 0.35,
            str(score))
        i += 1

'''
* Karşılaştırma Matrisi:
Aşağıdaki tablo, modellerin hangi mantıkla çalıştığını ve görsel çıktıdaki davranışlarını özetler:

| Algoritma | Tür | Karar Sınırı Yapısı | Notlar |
| :--- | :--- | :--- | :--- |
| **KNN (Nearest Neighbors)** | Mesafe Tabanlı | Esnek ve Veriye Duyarlı | Ölçeklemeye karşı çok hassastır. |
| **Linear SVM** | Geometrik | Doğrusal (RBF ile Esnek) | Karmaşık yapılarda çekirdek hilesi (kernel trick) gerekir. |
| **Decision Tree** | Ağaç Tabanlı | Eksenlere Paralel / Basamaklı | Veriyi dik açılarla böler, yorumlanabilirliği yüksektir. |
| **Random Forest** | Topluluk (Ensemble) | Karmaşık ve Pürüzsüz | Birçok ağacın ortalamasını alarak aşırı öğrenmeyi (overfitting) engeller. |
| **Naive Bayes** | Olasılıksal | Pürüzsüz ve Geniş | Özelliklerin bağımsız ve normal dağıldığını varsayar. |

* Kod Akışı ve Uygulama Mimarisi:
1. Veri Setlerinin Hazırlanması: sklearn.datasets kullanılarak üç farklı senaryo oluşturulmuştur:
- make_classification: Doğrusal ayrım testi için.
- make_moons: Doğrusal olmayan kavisli yapılar için.
- make_circles: İç içe geçmiş dairesel yapılar için.
2. Önişleme Hattı (Pipeline): Mesafe tabanlı algoritmaların (KNN ve SVM) özellik ölçeklerinden etkilenmemesi için StandardScaler ve sınıflandırıcılar make_pipeline fonksiyonu ile birleştirilmiştir. 
Bu, her modelin adil bir şekilde karşılaştırılmasını sağlar.
3. Model Eğitimi ve Testi: Her veri seti için veriler %80 eğitim ve %20 test olarak ayrılmıştır. Modeller eğitim setinde fit edilmiş, doğruluk skorları ise test seti üzerinden hesaplanmıştır.
4.Karar Sınırı Görselleştirme: DecisionBoundaryDisplay aracı kullanılarak her modelin tahmin uzayı renklendirilmiş; bu sayede algoritmanın veriyi "nasıl gördüğü" görselleştirilmiştir.
5. Sonuç Matrisi: Final çıktısı, 3 farklı veri seti (satırlar) ve 5 farklı algoritmanın (sütunlar) performansını tek bir figür üzerinde sunarak modellerin geometrik yaklaşımlarını kıyaslamaya olanak tanır.
'''
