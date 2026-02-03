'''
Bu proje, Iris veri kümesi üzerinde üç farklı sınıflandırma algoritmasını (KNN, Karar Ağaçları ve SVM) eğitir. Projenin asıl odağı, bu modellerin en iyi ayarlarını (hiperparametrelerini) 
bulmak için kullanılan GridSearchCV (Izgara Araması) ve RandomizedSearchCV (Rastgele Arama) yöntemlerini kıyaslamaktır.

* Kullanılan Kütüphaneler: 
- sklearn.model_selection: Veriyi eğitim/test olarak ayırmak (train_test_split) ve en iyi parametreleri aramak (GridSearchCV, RandomizedSearchCV) için.
- sklearn.neighbors, tree, svm: Kullanılan temel algoritmalar: En Yakın Komşu (KNN), Karar Ağacı (DT) ve Destek Vektör Makineleri (SVM).
- numpy (np): Parametre aralıklarını sayısal diziler (array) olarak oluşturmak için.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier()
knn_param_grid = {"n_neighbors": np.arange(2, 31)}
knn_grid_search = GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train, y_train)
print("KNN Grid Search Best Parameters: ", knn_grid_search.best_params_)
print("KNN Grid Search Best Accuracy: ", knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn, knn_param_grid, n_iter = 10)
knn_random_search.fit(X_train, y_train)
print("KNN Random Search Best Parameters: ", knn_random_search.best_params_)
print("KNN Random Search Best Accuracy: ", knn_random_search.best_score_)
print()

tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

tree_random_search = RandomizedSearchCV(tree, tree_param_grid, n_iter = 10)
tree_random_search.fit(X_train, y_train)
print("DT Random Search Best Parameters: ", tree_random_search.best_params_)
print("DT Random Search Best Accuracy: ", tree_random_search.best_score_)
print()
# SVM
svm = SVC()
svm_param_grid = {"C":[0.1, 1, 10, 100],
                  "gamma":[0.1, 0.01, 0.001, 0.0001]}
svm_grid_search = GridSearchCV(svm, svm_param_grid)
svm_grid_search.fit(X_train, y_train)
print("SVM Grid Search Best Parameters: ", svm_grid_search.best_params_)
print("SVM Grid Search Best Accuracy: ", svm_grid_search.best_score_)

svm_random_search = RandomizedSearchCV(svm, svm_param_grid, n_iter = 10)
svm_random_search.fit(X_train, y_train)
print("SVM Random Search Best Parameters: ", svm_random_search.best_params_)
print("SVM Random Search Best Accuracy: ", svm_random_search.best_score_)

'''
* Kod Akışı:
1. Hazırlık Aşaması: Iris verisi yüklenir ve %80 eğitim, %20 test olacak şekilde bölünür. Bu, modellerin hiç görmediği veriler üzerindeki başarısını ölçmek için standart bir adımdır.
2. Model Optimizasyonları (3 Farklı Algoritma): Her bir algoritma için şu iki yöntem uygulanır:
- Grid Search: Belirlenen tüm parametre kombinasyonlarını tek tek dener (Garanti sonuç ama yavaştır).
- Random Search: Belirlenen parametreler arasından rastgele seçimler yaparak deneme yapar (Daha hızlıdır, büyük verilerde tercih edilir).
KNN -> Komşu sayısı (n_neighbors) 2'den 30'a kadar taranır.
Decision Tree -> Ağacın derinliği (max_depth) ve yaprak sayısı (max_leaf_nodes) optimize edilir.
SVM -> Ceza katsayısı (C) ve eğrilik katsayısı (gamma) için en iyi ikili aranır.
3. Sonuçların Raporlanması: Kodun sonunda her bir yöntem için "En İyi Parametreler" ve bu parametrelerle elde edilen "En Yüksek Doğruluk (Accuracy)" skorları ekrana yazdırılır. 
Bu sayede hangi modelin ve hangi yöntemin Iris verisi için daha verimli olduğu analiz edilir.
'''
