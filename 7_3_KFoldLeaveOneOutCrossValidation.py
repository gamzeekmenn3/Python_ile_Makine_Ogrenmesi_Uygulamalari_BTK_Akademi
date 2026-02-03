'''
Bu proje, veri biliminde Cross-Validation yöntemlerinin model seçimi üzerindeki etkisini inceler. Karar Ağacı modelinin derinlik parametresi (max_depth), iki farklı teknikle test edilir:
1. K-Fold: Veriyi K adet parçaya bölerek yapılan dengeli bir test.
2. Leave-One-Out (LOO): Her seferinde sadece bir örneği test için ayırıp geri kalanını eğitimde kullanan kapsamlı bir test.

* Kullanılan Kütüphaneler:
- load_iris (sklearn.datasets): Analiz için kullanılan klasik Iris çiçek veri kümesi.
- KFold (sklearn.model_selection): Veriyi belirlenen sayıda (burada 10) bloğa bölmek için.
- LeaveOneOut (sklearn.model_selection): Veri sayısı kadar iterasyon yaparak en yüksek varyanslı doğrulama yapmak için.
- GridSearchCV: Belirlenen parametreleri (max_depth) bu doğrulama yöntemlerini kullanarak otomatik olarak test etmek için.
- DecisionTreeClassifier: Sınıflandırma işlemini yapan temel model.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris= load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

tree = DecisionTreeClassifier()
tree_param_dist = {"max_depth": [3, 5, 7]}

# KFold Grid Search
kf = KFold(n_splits = 10)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv = kf)
tree_grid_search_kf.fit(X_train, y_train)
print("KF En iyi parameter: ", tree_grid_search_kf.best_params_)
print("KF En iyi acc: ",  tree_grid_search_kf.best_score_)

# LOO
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv = loo)
tree_grid_search_loo.fit(X_train, y_train)
print("LOO En iyi parameter: ", tree_grid_search_loo.best_params_)
print("LOO En iyi acc: ",  tree_grid_search_loo.best_score_)

'''
Kod Akışı:
1. Veri Hazırlığı: Iris veri seti yüklenir ve eğitim/test olarak ayrılır. Modelin hiperparametre araması X_train üzerinde gerçekleştirilir.
2. K-Fold ile Izgara Araması (Grid Search):
- n_splits = 10 seçilerek eğitim verisi 10 parçaya bölünür.
- Model, her derinlik değeri (3, 5, 7) için 10 farklı eğitim-test döngüsü yapar ve bunların ortalamasını alır. Daha hızlıdır ve genellikle genel performans hakkında iyi bir fikir verir.
3. Leave-One-Out (LOO) ile Izgara Araması: Her bir veri noktası tek başına bir test seti gibi davranır. Eğer eğitim setinde 120 örnek varsa, model her parametre için 120 kez eğitilir.
- Avantajı: Verinin neredeyse tamamı eğitimde kullanılır.
- Dezavantajı: Hesaplama maliyeti çok yüksektir ve büyük veri setlerinde çok yavaş çalışır.
4. Sonuçların Kıyaslanması: Her iki yöntem için de bulunan "En İyi Parametre" ve "En İyi Doğruluk Skoru" ekrana yazdırılır. Genellikle iki yöntem benzer sonuçlar verse de, LOO daha hassas 
(fakat bazen yanıltıcı) sonuçlar üretebilir.
'''
