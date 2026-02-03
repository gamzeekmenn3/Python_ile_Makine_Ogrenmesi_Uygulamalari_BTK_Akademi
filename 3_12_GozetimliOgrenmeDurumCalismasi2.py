'''
Bu proje, Kaliforniya’daki evlerin çeşitli özelliklerine (oda sayısı, gelir düzeyi vb.) bakarak fiyat tahmini yapar. Temel amaç, veriye matematiksel takla attırarak (polinom ekleyerek) tahminlerin ne kadar 
iyileştiğini görmektir.

* Kullanılan Kütüphaneler:
- sklearn.datasets: Hazır ev verilerini getirir.
- train_test_split: Veriyi öğrenme ve sınav (test) verisi olarak böler.
- PolynomialFeatures: Verideki karmaşık ilişkileri yakalamak için değişkenlerin karelerini/etkileşimlerini üretir.
- LinearRegression: Tahmin yapan ana algoritma.
- mean_squared_error: Modelin ne kadar hata yaptığını ölçer.
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

poly_feat = PolynomialFeatures(degree = 2)
X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)
print("Polynomial Regression rmse: ", mean_squared_error(y_test, y_pred, squared = False))

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Multi Variable Linear Regression rmse: ", mean_squared_error(y_test, y_pred, squared = False))

'''
* Kod Akışı:
- Veri Hazırlığı: Veriler çekilir ve modelin hiç görmeyeceği bir test grubu ayrılır.
- Polinom Dönüşümü: Mevcut özellikler 2. dereceye yükseltilir (örn: x -> x^2).
- Karmaşık Model: Genişletilmiş veriyle bir regresyon eğitilir ve hata payı (RMSE) ölçülür.
- Basit Model: Hiçbir değişiklik yapmadan ham veriyle doğrusal bir model eğitilir.
- Kıyaslama: İki modelin hata oranları ekrana yazdırılarak hangisinin daha başarılı olduğu görülür.
'''
