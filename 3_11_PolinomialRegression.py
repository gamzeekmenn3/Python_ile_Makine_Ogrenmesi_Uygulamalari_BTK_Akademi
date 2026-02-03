'''
Bu çalışma, doğrusal olmayan (eğrisel) bir veri setine Polinom Regresyon uygulayarak verideki trendi yakalamayı amaçlar. Amaç, düz bir çizgi yerine veriye en uygun eğriyi çizmektir.

* Kullanılan Kütüphaneler:
- NumPy: Veri setini oluşturmak ve matematiksel işlemler (matris işlemleri) için.
- Matplotlib: Verileri ve oluşturulan tahmin eğrisini görselleştirmek için.
- Scikit-Learn: Veriyi polinomsal özelliklere dönüştürmek (PolynomialFeatures) ve modeli eğitmek (LinearRegression) için.
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = 4 * np.random.rand(100, 1)
y = 2 + 3 * X ** 2 + 2 * np.random.rand(100, 1)

# plt.scatter(X, y)

poly_feat = PolynomialFeatures(degree = 2)
X_poly = poly_feat.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

plt.scatter(X, y, color = "blue")

X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color = "red")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polinom Regresyon Modeli")

'''
* Kod Akışı:
- Veri Üretimi: İkinci dereceden bir denklem (3x^2 + 2) kullanılarak rastgele gürültülü veriler oluşturulur.
- Özellik Dönüşümü: degree=2 parametresiyle veriye kare değerleri eklenir; böylece model "eğri" çizebilir hale gelir.
- Model Eğitimi: Genişletilmiş veri seti üzerinden lineer regresyon algoritması çalıştırılır.
- Görselleştirme: Gerçek veriler mavi nokta, modelin tahmini ise kırmızı çizgi olarak grafiğe dökülür.
'''
