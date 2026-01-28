'''
Bu program, Iris veri setini kullanarak bir Decision Tree sınıflandırma modelini uygulamakta, eğitmekte ve modelin sonuçlarını detaylıca görselleştirmektedir. Projenin temel amacı, bir çiçeğin taç ve çanak yaprağı
ölçümleri gibi özelliklerine bakarak hangi Iris türüne (Setosa, Versicolor, Virginica) ait olduğunu yüksek doğrulukla tahmin edebilen bir model oluşturmaktır.

Proje İçeriği:
1. Veri Hazırlama: Iris veri setinin %80'i eğitim, %20'si test için ayrılmıştır.
2. Model Eğitimi: Gini katsayısı ve maksimum 5 derinlik (max_depth) ile bir DecisionTreeClasifier eğitilmiştir.
3. Değerlendirme: Doğruluk skoru ve Karmaşıklık Matrisi ile performans ölçümü.
4. Ağaç Görselleştirme: Karar mekanizmasının ve özellik önem düzeylerinin analizi.

Kullanılan Kütüphaneler
- sklearn.datasets: Iris veri setini sisteme yükler.
- sklearn.model_selection: Veriyi %80 Eğitim ve %20 Test olarak ayırır.
- sklearn.tree: Karar ağacı modelini kurar (DecisionTreeClassifier) ve yapısını çizer (plot_tree).
-sklearn.metrics: Modelin başarısını Doğruluk (Accuracy) ve Hata Matrisi (Confusion Matrix) ile ölçer.
-matplotlib: Model sonuçlarını ve ağaç yapısını görselleştirir.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

tree_clf = DecisionTreeClassifier(criterion = "gini", max_depth = 5, random_state = 42) # criterion = "entropy" 
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("iris veri seti ile eğitilen DT modeli dogrulugu: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("conf_matrix: ")
print(conf_matrix)

plt.figure(figsize = (15,10))
plot_tree(tree_clf, filled = True, feature_names = iris.feature_names, class_names = list(iris.target_names))
plt.show()

feature_importances = tree_clf.feature_importances_ 
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse = True)
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")

'''
Kod Akışı
1.Veri Hazırlama: Iris veri setindeki özellikler (X) ve hedef sınıflar (y) ayrıştırılarak model eğitimi için hazır hale getirilir.
2.Model Eğitimi: Gini katsayısı ve max_depth=5 parametresiyle, aşırı öğrenmeyi (overfitting) engelleyecek şekilde model eğitilir.
3.Performans Testi: Modelin doğruluğu, daha önce görmediği test verileri üzerinden ölçülür.
4.Ağaç Görselleştirme: Karar mekanizması grafiksel olarak çizilir; düğümlerdeki karar kuralları ve örnek dağılımları analiz edilir.
5.Özellik Önem Analizi: Modelin sınıflandırma yaparken en çok hangi verilere güvendiği belirlenir. Bu modelde petal length (taç yaprak uzunluğu) ve petal width (taç yaprak genişliği) en kritik özelliklerdir.
'''

#%%
'''
Bu program, Iris veri setindeki özelliklerin ikili kombinasyonlarını kullanarak Decision Tree sınıflandırıcısının karar sınırlarını görselleştirmeyi amaçlar. Temel hedef farklı özellik çiftlerinin sınıflandırma 
uzayını nasıl böldüğünü ve hangi özelliklerin (SL, SW, PL, PW) tür ayrımında en etkili olduğunu görsel olarak belirlemek. Karar ağaçlarının karakteristik özelliği olan "eksenlere paralel" sınırlar burada netçe 
görülür.

Kullanılan Kütüphaneler:
• sklearn.datasets: Iris veri setini yüklemek.
• sklearn.tree: Karar Ağacı sınıflandırma modelini kullanmak.
• sklearn.inspection: Karar Sınırlarını (DecisionBoundaryDisplay) çizmek.
• matplotlib: Görselleştirme ve alt grafikler oluşturmak.
• numpy: Veri manipülasyonu ve indis işlemleri.
'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()

n_classes = len(iris.target_names)
plot_colors = "ryb"

for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]): 
    X = iris.data[:, pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X,y)

    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad = 2.5)
    DecisionBoundaryDisplay.from_estimator(clf,  X,  cmap = plt.cm.RdYlBu,  response_method = "predict",  ax = ax,
                                           xlabel = iris.feature_names[pair[0]],  ylabel = iris.feature_names[pair[1]])
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c = color, label = iris.target_names[i],
                   cmap = plt.cm.RdYlBu,  edgecolors = "Black")
plt.legend()

'''
Kod Akışı:
1. Kod, SL, SW, PL ve PW özelliklerinin tüm ikili varyasyonlarını dener.
2. Her grafik, modelin ilgili bölgede hangi sınıfı tahmin edeceğini (renk alanları) ve gerçek verinin dağılımını (noktalar) gösterir.
3. Analiz: Petal Length (PL) ve Petal Width (PW) özelliklerinin kullanıldığı grafiklerde (genellikle sağ alt köşedeki grafikler) karar sınırlarının en basit ve en net olduğu görülür. 
4. Bu durum, taç yaprak özelliklerinin Iris türlerini birbirinden ayırmak için en güçlü ayırıcılar olduğunu deneysel olarak kanıtlar.
'''

#%%
'''
Bu program, tıbbi özelliklere dayanarak diyabet hastalığının ilerleme seviyesini tahmin eden bir Decision Tree Regressor modeli oluşturur ve performansını ölçer.Modelin temel amacı, hastaların tıbbi özelliklerine 
(BMI, kan basıncı, vb.) dayanarak diyabet hastalığının ilerleme seviyesini tahmin etmek. Bu, sayısal (sürekli) bir değer tahmin etme problemi olduğu için bir Regresyon görevidir.

Kullanılan Kütüphaneler:
• sklearn.datasets: Diyabet veri setini yüklemek.
• sklearn.tree: Karar Ağacı Regresyon modelini (DecisionTreeRegressor) kullanmak.
• sklearn.model_selection: Veriyi eğitim (%80) ve test (%20) kümelerine ayırmak.
• sklearn.metrics: Regresyon performansını (MSE) değerlendirmek.
• numpy: Matematiksel işlemler (karekök alma) için.
'''
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)
rmse = np.sqrt(mse)
print("rmse: ",rmse)

'''
Kod Akışı ve Model Değerlendirmesi
- Veri Hazırlığı: Diyabet veri setindeki 10 klinik özellik (X) ve hedef değişken olan hastalık ilerleme düzeyi (y) ayrıştırılır. Verinin %20'si modelin hiç görmediği bir test kümesi olarak saklanır.
- Model Eğitimi: DecisionTreeRegressor modeli eğitim verisiyle yapılandırılır. Algoritma, dallanma kararlarını alırken her düğümdeki varyansı veya MSE (Ortalama Karesel Hata) değerini en aza indirecek eşik 
noktalarını belirler.
- Tahmin ve Test: Eğitilen model, test kümesindeki veriler için tahminler üretir. Bu tahminlerin gerçek değerlerden ne kadar saptığı stratejik hata metrikleri ile analiz edilir.

Hata Metrikleri Analizi
- Ortalama Karesel Hata (MSE): Hata paylarının karesinin ortalamasıdır. Kare alma işlemi nedeniyle büyük hataları çok daha sert cezalandırır. Modelin genel başarısını gösterse de birimi, hedef değişkenin birimiyle 
(hastalık ilerleme düzeyi) aynı değildir.
- Karesel Ortalama Hata Kökü (RMSE): MSE'nin kareköküdür. Regresyon analizlerinde en çok tercih edilen metriktir. En büyük avantajı, hata biriminin tahmin edilen değerle aynı olmasıdır; bu da sonucun tıbbi açıdan 
yorumlanmasını kolaylaştırır. Değerin sıfıra yakın olması, modelin mükemmele yakın tahminler yaptığını gösterir.

Sonuç: Elde edilen RMSE değeri, modelin tahminlerinde gerçek değerlerden ortalama ne kadar saptığını temsil eder. Bu metrik, Karar Ağacı modelinizin başarısını Lineer Regresyon gibi farklı algoritmalarla 
kıyaslamanızı sağlayan en temel performans göstergesidir.
'''
#%%
'''
Bu program, Karar Ağacı Regresyon modellerinde maksimum derinlik parametresinin modelin genelleme yeteneği üzerindeki kritik rolünü görsel olarak inceler. Temel hedef, yapay bir sinüs dalgası üzerinde sığ ve derin
ağaçların tahmin eğrilerini karşılaştırarak Aşırı Öğrenme (Overfitting) ve Eksik Öğrenme (Underfitting) kavramlarını somutlaştırmaktır.

Kullanılan Kütüphaneler:
• sklearn.tree: Karar Ağacı Regresyon modelini (DecisionTreeRegressor) kullanmak.
• numpy: Sinüs dalgası üretimi ve rastgele gürültü ekleme işlemleri için.
• matplotlib: Tahmin eğrilerini ve veri noktalarını görselleştirmek için.
'''
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()
y[::5] += 0.5 * (0.5 - np.random.rand(16))
# plt.scatter(X,y)

regr_1 = DecisionTreeRegressor(max_depth = 2)
regr_2 = DecisionTreeRegressor(max_depth = 15)
regr_1.fit(X,y)
regr_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.plot(X, y, c = "red", label = "data")
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "Max Depth: 2", linewidth = 2)
plt.plot(X_test, y_pred_2, color = "green", label = "Max Depth: 15", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
'''
Kod Akışı
- 0 ile 5 arasında 80 noktadan oluşan, sinüs fonksiyonu ile üretilmiş ve rastgele gürültü eklenmiş yapay bir veri seti (X, y) oluşturulur.
- Farklı derinliklerde iki ayrı "DecisionTreeRegressor" modeli aynı veri seti ile eğitilir.,
- 0-5 aralığındaki yoğun test noktaları üzerinden modellerin davranışları görselleştirilir.

Sonuçların Özeti:
- Mavi Eğri (Max Depth = 2) | Underfitting: Model çok basit kalmıştır. Verideki ana yapıyı (sinüs formunu) tam yakalayamaz, detayları kaçırır.
- Yeşil Eğri (Max Depth = 15) | Overfitting: Model veriyi ezberlemiştir. Gürültüleri bile takip ettiği için zikzaklı bir eğri çizer; yeni verilerde hata payı yüksek olur.
! Kritik Not: En iyi model, verideki ana sinyali yakalayan ancak gürültüyü görmezden gelen ideal derinlikteki modeldir.
'''
