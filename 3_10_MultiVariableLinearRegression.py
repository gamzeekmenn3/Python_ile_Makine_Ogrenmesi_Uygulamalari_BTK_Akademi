'''
Bu çalışma iki ana bölümden oluşmaktadır:
* Sentetik Veri ile Görselleştirme: Rastgele oluşturulan 3 boyutlu bir veri kümesi üzerinde model eğitilerek, regresyon düzleminin veriye nasıl oturduğu görselleştirilir.
* Gerçek Dünya Veri Kümesi (Diabetes): Scikit-learn kütüphanesindeki şeker hastalığı verileri kullanılarak, hastaların özelliklerine göre hastalık ilerlemesini tahmin eden bir model kurulur ve 
başarısı (RMSE) ölçülür.

* Kullanılan Kütüphaneler
- numpy (np): Çok boyutlu diziler ve matris işlemleri için kullanılır. Veri oluşturma ve şekillendirme aşamasında temel rol oynar.
- matplotlib.pyplot (plt): Verilerin ve regresyon düzleminin 3 boyutlu grafiklerini çizmek için kullanılır.
- sklearn (Scikit-learn):
  - LinearRegression: Regresyon modelini kuran ve eğiten ana algoritma.
  - load_diabetes: Hazır bir veri kümesi sağlar.
  - train_test_split: Veriyi eğitim ve test olarak ikiye bölmeye yarar.
  - mean_squared_error: Modelin tahmin hatasını ölçmek için kullanılan metrik fonksiyonudur.
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# y = a0 + a1x -> linear regression
# y = a0 + a1x1 + a2x2 + ... + anxn -> multi variable linear regression
# y = a0 + a1x1 + a2x2

X = np.random.rand(100, 2)
coef = np.array([3, 5])
# y = 0 + np.dot(X, coef)
y = np.random.rand(100) + np.dot(X, coef)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X[:, 0], X[:, 1], y)
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("y")

lin_reg = LinearRegression()
lin_reg.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha = 0.3)
plt.title("multi variable linear regression")

print("Katsayilar:", lin_reg.coef_)
print("Kesisim:",lin_reg.intercept_)

#%%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared = False)
print("rmse: ", rmse)

'''
Kodun çalışma mantığını üç ana aşamada inceleyebiliriz:
A. Sentetik Veri Oluşturma ve Formül Mantığı:
- Kodun başında şu matematiksel modelleme yapılır: y = a0 + a1x1 + a2x2 + epsilon
- Burada x1 ve x2 bağımsız değişkenleri temsil ederken, y hedef değişkendir. np.dot(X, coef) işlemiyle katsayılar uygulanır ve üzerine rastgele gürültü eklenerek gerçekçi bir senaryo oluşturulur.
B. Görselleştirme ve Model Eğitimi (3D Plot):
- LinearRegression() nesnesi oluşturulur ve fit(X, y) ile veriye en uygun katsayılar bulunur.
- np.meshgrid kullanılarak bir koordinat düzlemi oluşturulur.
- Model, bu düzlem üzerindeki her nokta için bir $y$ değeri tahmin eder (y_pred).
- Sonuç: Veri noktaları (scatter) ve modelin öğrendiği "tahmin düzlemi" (surface) aynı grafik üzerinde gösterilir.
C. Diabetes Veri Seti ile Uygulama:
- Gerçek bir veri seti üzerinde şu adımlar izlenir:
1. Veriyi Yükleme: Şeker hastalığı verileri çekilir.
2. Bölme: Verinin %80'i eğitim, %20'si test için ayrılır. Bu, modelin görmediği veriler üzerindeki performansını ölçmek için kritiktir.
3. Performans Ölçümü: mean_squared_error ile RMSE (Kök Ortalama Kare Hata) hesaplanır.
* Not: RMSE değeri ne kadar düşükse, modelin tahminleri gerçek değerlere o kadar yakındır.
'''
