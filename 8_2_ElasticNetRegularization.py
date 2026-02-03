'''
Bu proje, diyabet hastalarına ait verileri kullanarak hastalık ilerlemesini tahmin eden bir regresyon modelidir. Projenin odak noktası olan ElasticNet, Ridge (L2) ve Lasso (L1) cezalarını 
hibrit bir şekilde kullanır. Bu sayede hem katsayıları küçültür (Ridge özelliği) hem de bazılarını sıfıra indirerek özellik seçimi yapar (Lasso özelliği).

* Kullanılan Kütüphaneler:
- load_diabetes (sklearn.datasets): Analiz için gerekli olan diyabet veri kümesini sağlar.
- ElasticNet (sklearn.linear_model): L1 ve L2 cezalarını birleştiren regresyon algoritması.
- GridSearchCV (sklearn.model_selection): En iyi hata yönetimini sağlayan parametre kombinasyonlarını bulur.
- mean_squared_error (sklearn.metrics): Modelin tahmin başarısını ölçmek için kullanılan hata metriği.
'''
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

elastic_net = ElasticNet()
elastic_net_param_grid = {"alpha": [0.1, 1, 10, 100],  "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}  # l1 or l2 penalty ratio

elastic_net_grid_search = GridSearchCV(elastic_net, elastic_net_param_grid, cv=5)
elastic_net_grid_search.fit(X_train, y_train)
print("ElasticNet Best Parameters: ", elastic_net_grid_search.best_params_)
print("ElasticNet Best Score: ", elastic_net_grid_search.best_score_)

best_elastic_net_model = elastic_net_grid_search.best_estimator_
y_pred = best_elastic_net_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)

'''
* Kod Akışı:
1. Veri Hazırlama: Diyabet verileri yüklenir; %80 eğitim ve %20 test olacak şekilde ayrılır. Bu, modelin eğitimden sonra yeni verilerde ne kadar başarılı olacağını test etmek için kritiktir.
2. Parametre Izgarasının Tanımlanması:
- alpha: Toplam ceza miktarını belirler.
- l1_ratio: Modelin ne kadar "Lasso" ne kadar "Ridge" olacağını belirler.
Örneğin l1_ratio = 0.9 ise model %90 Lasso, %10 Ridge gibi davranır.
3. Hiperparametre Optimizasyonu (Grid Search):
- GridSearchCV, tanımlanan tüm alpha ve l1_ratio kombinasyonlarını 5 katlı çapraz doğrulama (cv=5) ile tek tek dener.
- Veri seti için en düşük hatayı veren en iyi parametre setini ve en iyi skoru belirler.
4. Tahmin ve Performans Ölçümü:
- Belirlenen "En İyi Model" (best_estimator_) kullanılarak test verileri üzerinden tahminler (y_pred) yapılır.
- Son adımda, tahminlerin gerçek değerlerden ne kadar saptığını gösteren MSE (Hata Kareler Ortalaması) hesaplanarak yazdırılır.
'''
