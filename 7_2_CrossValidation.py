'''
Bu proje, Karar Ağacı algoritmasının en kritik iki parametresi olan ağaç derinliği (max_depth) ve maksimum yaprak sayısı (max_leaf_nodes) üzerine odaklanır. Kodun temel amacı, sadece en iyi 
sonucu bulmak değil, aynı zamanda K-Fold Cross Validation (Çapraz Doğrulama) sürecindeki her bir adımın başarısını tek tek incelemektir.

* Kullanılan Kütüphaneler:
- load_iris (sklearn.datasets): Analiz edilecek çiçeği özelliklerini içeren veri kümesini sağlar.
- train_test_split (sklearn.model_selection): Veriyi ana eğitim ve test bloklarına ayırmak için kullanılır.
- GridSearchCV (sklearn.model_selection): Belirlenen tüm parametre kombinasyonlarını çapraz doğrulama yaparak test eden temel araçtır.
- DecisionTreeClassifier (sklearn.tree): Veri sınıflandırmasında kullanılacak olan ana algoritmadır.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth": [3, 5, 7], "max_leaf_nodes": [None, 5, 10, 20, 30, 50]}

nb_cv = 3
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv = nb_cv)
tree_grid_search.fit(X_train, y_train)
print("DT Grid Search Best Parameters: ", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

for mean_score, params in zip(tree_grid_search.cv_results_["mean_test_score"], tree_grid_search.cv_results_["params"]):
    print(f"Ortalama test skoru: {mean_score}, Parametreler: {params}")
    
cv_result = tree_grid_search.cv_results_
for i, params in enumerate(cv_result["params"]):
    print(f"Parametreler: {params}")
    for j in range(nb_cv):
        accuracy = cv_result[f"split{j}_test_score"][i]
        print(f"\tFold (j+1) - Accuracy: {accuracy}") 

'''
* Kod Akışı:
1. Veri Hazırlığı ve Model Kurulumu: 
- Iris verisi yüklenir ve %20'si final testi için ayrılır.
- Karar Ağacı için bir "parametre ızgarası" (grid) oluşturulur. Bu ızgara toplamda 18 farklı kombinasyonu (3 derinlik X 6 yaprak sayısı) test edecektir.
2. Çapraz Doğrulama (Cross Validation) Süreci:
- cv = 3 (nb_cv) parametresiyle veriler 3 parçaya bölünür. Model her seferinde bir parçayı test, diğer ikisini eğitim için kullanarak 3 kez döner.
- GridSearchCV tüm parametre kombinasyonlarını bu 3-fold yöntemiyle eğitir ve ortalama başarı puanlarını hesaplar.
3. Detaylı Sonuç Analizi (Döngüler): 
Kodun sonundaki döngüler, standart bir model eğitiminden farklı olarak "perde arkasını" gösterir:
- Ortalama Skorlar: Her parametre setinin tüm foldlardaki genel başarı ortalaması yazdırılır.
- Fold Bazlı Analiz: En alt kısımdaki iç içe döngü, her bir parametre kombinasyonunun 1., 2. ve 3. Fold adımlarında tek tek kaç puan aldığını listeler. Bu, modelin verinin farklı kısımlarında
ne kadar istikrarlı olduğunu anlamamızı sağlar.
'''
