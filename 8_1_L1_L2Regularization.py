'''
Bu proje, Diabetes (Diyabet) veri kümesini kullanarak hastaların hastalık ilerleyişini tahmin etmeyi amaçlar. Projenin odak noktası, doğrusal regresyonun geliştirilmiş versiyonları olan 
Ridge ve Lasso algoritmalarını kıyaslamaktır. Her iki yöntem de modele "ceza" (penalty) terimi ekleyerek katsayıların aşırı büyümesini engeller; ancak Lasso'nun bazı katsayıları sıfıra 
indirme (özellik seçimi) gibi bir farkı vardır.

* Kullanılan Kütüphaneler:
- load_diabetes (sklearn.datasets): On farklı değişken (yaş, cinsiyet, BMI, vb.) içeren regresyon veri setini yükler.
- Ridge & Lasso: Düzenleştirilmiş doğrusal regresyon modelleri.
- GridSearchCV: En uygun ceza katsayısını (alpha) bulmak için çapraz doğrulama yapar.
- mean_squared_error (sklearn.metrics): Tahminlerin gerçek değerlerden ne kadar saptığını ölçen hata metriği (MSE).
'''
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge
ridge = Ridge()
ridge_param_grid = {"alpha": [0.1, 1, 10, 100]}

ridge_grid_search =GridSearchCV(ridge, ridge_param_grid, cv = 5)
ridge_grid_search.fit(X_train, y_train)
print("Ridge Best Parameters: ", ridge_grid_search.best_params_)
print("Ridge Best Score: ", ridge_grid_search.best_score_)

best_ridge_model = ridge_grid_search.best_estimator_
y_pred_ridge = best_ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
print("Ridge MSE: ", ridge_mse)

# Lasso
lasso = Lasso()
lasso_param_grid = {"alpha": [0.1, 1, 10, 100]}
lasso_grid_search =GridSearchCV(lasso, lasso_param_grid, cv = 5)
lasso_grid_search.fit(X_train, y_train)  
print("Lasso Best Parameters: ", lasso_grid_search.best_params_)
print("Lasso Best Score: ", lasso_grid_search.best_score_) 

best_lasso_model = lasso_grid_search.best_estimator_
y_pred_lasso = best_lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print("Lasso MSE: ", lasso_mse)

'''
Kod Akışı:
1. Veri Hazırlığı: Diyabet verisi yüklenir ve standart eğitim/test bölünmesi yapılır. Burada hedef değişken (y), kategorik bir sınıf değil, sürekli bir sayısal değerdir.
2. Ridge Regresyon Uygulaması:
- Parametre Arama: alpha (L2 cezası) parametresi için 0.1 ile 100 arasında bir tarama yapılır.
- Model Eğitimi: 5 katlı çapraz doğrulama (cv=5) ile en iyi Ridge modeli belirlenir.
- Değerlendirme: En iyi model ile test seti üzerinde tahmin yapılır ve Hata Kareler Ortalaması (MSE) hesaplanır.
3. Lasso Regresyon Uygulaması: 
- Parametre Arama: Aynı alpha (L1 cezası) aralığı Lasso için de test edilir.
- Model Eğitimi: Lasso, Ridge'den farklı olarak gereksiz gördüğü özelliklerin katsayısını tamamen sıfırlayabilir.
- Değerlendirme: Lasso modelinin MSE skoru hesaplanarak Ridge ile kıyaslanır.
'''
