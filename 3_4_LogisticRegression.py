'''
Bu program, UCI Kalp Hastalığı veri setini kullanarak hastaların klinik verilerine göre hastalık varlığını tahmin eden bir Lojistik Regresyon modeli oluşturur. İkili sınıflandırma (binary classification) problemlerinde olasılıksal yaklaşımıyla güçlü bir temel oluşturur.

Kullanılan Kütüphaneler
- ucimlrepo: UCI makine öğrenmesi havuzundan gerçek zamanlı veri çekmek için kullanılır.
- pandas: Veri çerçevesi (DataFrame) oluşturma ve eksik verilerin (NaN) temizlenmesi için tercih edilir.
- sklearn.linear_model: Lojistik Regresyon algoritmasını sağlar.
- sklearn.model_selection: Veriyi eğitim ve test (Validation) aşamaları için böler.
'''
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

heart_disease = fetch_ucirepo(id = 45)

df = pd.DataFrame(data = heart_disease.data.features)
df["target"] = heart_disease.data.targets

if df.isna().any().any():
    df.dropna(inplace = True)
    print("nan")
    
X = df.drop(["target"],axis = 1).values
y = df.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

log_reg = LogisticRegression(penalty = "l2", C = 1, solver = "lbfgs", max_iter = 100)
log_reg.fit(X_train, y_train)
accuracy = log_reg.score(X_test, y_test)
print("Logistic Regression Acc: ", accuracy)

'''
Kod Akışı ve Model Analizi
- Veri seti UCI havuzundan çekilerek analiz için Pandas yapısına aktarılır. Modelin matematiksel kararlılığını korumak adına, eksik veri (NaN) içeren satırlar dropna yöntemiyle hiyerarşiden çıkarılır.
- Modelin başarısını doğrulamak amacıyla veri setinin %10'u "test kümesi" olarak ayrılır; kalan büyük kısım eğitim için kullanılır.
- Model, katsayıların aşırı büyümesini engelleyen L2 regülarizasyonu ile eğitilir. Bu yöntem, modelin eğitim verisini ezberlemesini (overfitting) önleyerek, gerçek hayattaki farklı hasta verilerine uyum sağlama 
(genelleme) yeteneğini artırır.
- Eğitim tamamlandıktan sonra, modelin hiç görmediği test verileri üzerindeki tahmin başarısı Accuracy (Doğruluk) skoru ile raporlanır.

Model Parametreleri ve Metrik Yorumu
- Accuracy Skoru: Bu değer, modelin test setindeki hastaların yüzde kaçını "hasta" veya "sağlıklı" olarak doğru teşhis ettiğini gösterir. Tıbbi teşhislerde yüksek doğruluk, erken tanı için kritik önem taşır.
- Hiperparametre Kontrolü: Modelde tanımlanan penalty = "l2" ve C = 1 değerleri, bir denge mekanizması görevi görür. Katsayıların aşırı değerler alıp modeli karmaşıklaştırmasını engelleyerek, modelin daha sade 
ve yorumlanabilir kalmasını sağlar.
- Neden Lojistik Regresyon? Bu algoritma, ikili sınıflandırma (binary classification) problemlerinde sonuçları bir olasılık değeri (0 ile 1 arası) olarak sunduğu için tıbbi karar destek sistemlerinde basit ama 
oldukça güçlü bir başlangıç modelidir.
'''
