'''
Bu çalışma, Iris veri seti üzerinden çiçek türlerini tahmin etmek için Basit Bayes Teoremi’ne dayanan Gaussian Naive Bayes (GNB) modelini kullanır. Özelliklerin birbirinden bağımsız olduğunu ve her birinin 
sınıflar içinde normal (Gaussian) dağılım sergilediğini varsayarak olasılıksal bir sınıflandırma yapar. Modelin başarısı; her sınıf için hesaplanan olasılık dağılımları üzerinden, detaylı bir performans raporu ile 
analiz edilir.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
iris = load_iris()

X= iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
print(classification_report(y_test, y_pred))

'''
Kod Akışı:
- Iris veri setindeki morfolojik özellikler (X) ve hedef sınıflar (y) ayrıştırılır. Modelin genelleme yeteneğini tarafsız bir şekilde ölçebilmek için verinin %20'si test kümesi olarak ayrılır.
- Model, her özelliğin sınıflar bazında Gaussian dağılım sergilediği varsayımıyla eğitilir. Bu aşamada algoritma, her sınıf için özelliklerin ortalama ve standart sapma değerlerini hesaplayarak sınıflara 
ait olasılık yoğunluk fonksiyonlarını öğrenir.
- Eğitilen model, test verileri üzerinde Bayes Teoremi'ni kullanarak olasılıksal tahminler yapar. Tahminlerin başarısı, Sınıflandırma Raporu üzerinden detaylıca analiz edilir.

Sonuç: Naive Bayes; düşük hesaplama maliyeti ve hızı sayesinde, özellikle özelliklerin birbirinden bağımsız olduğu varsayılan durumlarda çok güçlü ve etkili bir başlangıç modelidir.
'''
