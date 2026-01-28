'''
Bu program, 8x8 piksellik düşük çözünürlüklü görüntüleri analiz ederek 0'dan 9'a kadar olan rakamları tahmin eden bir Support Vector Classifier (SVC) modeli oluşturur. 64 boyutlu özellik uzayında sınıfları 
birbirinden ayıran optimum hiperdüzlemi bulmayı amaçlayan bir çoklu sınıflandırma uygulamasıdır.

Kullanılan Kütüphaneler
- sklearn.datasets: 1797 adet el yazısı rakam görüntüsü içeren Digits veri setini yükler.
- sklearn.svm: Sınıflar arası marjı maksimize eden Destek Vektör Sınıflandırıcıyı (SVC) sağlar.
- sklearn.model_selection: Veriyi %80 Eğitim ve %20 Test kümesi olarak ayırır.
- sklearn.metrics: Modelin başarısını Precision, Recall ve F1-Score bazında detaylandıran Classification Report sunar.
- matplotlib.pyplot: Rakamların piksel yoğunluklarını görselleştirir.
'''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

digits = load_digits()
fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10,5), subplot_kw = {"xticks":[], "yticks":[]})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(digits.target[i])
plt.show()
  
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
 
svm_clf = SVC(kernel = "linear", random_state = 42)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test) 
print(classification_report(y_test, y_pred))

'''
Kod Akışı:
- Digits veri setindeki her bir örnek, 8x8 piksel boyutunda düşük çözünürlüklü görüntülerden oluşur. Model, bu 64 farklı piksel yoğunluk değerini bağımsız değişken (özellik) olarak kabul eder.
- Orijinal 8x8 matris yapısındaki görüntüler, modelin işleyebileceği 64 boyutlu tek boyutlu vektörlere dönüştürülür. Verinin %20'si modelin genel başarısını doğrulamak için "test kümesi" olarak ayrılır.
- Model, kernel="linear" (doğrusal çekirdek) parametresiyle eğitilir. SVM'nin temel felsefesi, sınıfları birbirinden ayıran en geniş boşluğu (maximum margin) bularak, 64 boyutlu uzayda rakamları birbirinden 
ayıran en ideal karar sınırını (hiperdüzlemi) inşa etmektir.
- Performans Denetimi: Eğitim sonrası modelin tahmin gücü, Sınıflandırma Raporu üzerinden 0-9 arası her bir rakam için ayrı ayrı analiz edilir.

Performans Metriklerinin Teknik Analizi
- Precision (Kesinlik): Modelin "Bu bir 5 rakamıdır" dediği tahminlerin ne kadarının gerçekten doğru olduğunu ölçer. Hatalı pozitifleri (False Positives) cezalandırır.
- Recall (Duyarlılık): Veri setindeki gerçek rakamların (örneğin tüm 5'lerin) model tarafından yüzde kaçının yakalanabildiğini gösterir. Gözden kaçan verileri (False Negatives) denetler.
- F1-Score: Kesinlik ve duyarlılık arasındaki dengeyi temsil eden harmonik ortalamadır.
- Linear Kernel Avantajı: Yüksek boyutlu piksel uzayında rakamlar genellikle doğrusal bir düzlemle net bir şekilde ayrılabilir. Bu nedenle basit bir doğrusal çekirdek, işlem yükünü azaltırken oldukça yüksek 
doğruluk sağlar.
- Genel Doğruluk (Accuracy): Raporun alt kısmındaki ağırlıklı ortalamalar, modelin tüm rakam sınıfları genelinde ne kadar tutarlı çalıştığını özetleyen nihai başarı göstergesidir.
'''
