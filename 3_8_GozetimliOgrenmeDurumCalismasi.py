'''
Bu çalışma; beş farklı makine öğrenmesi algoritmasının performansını optimize eden, çapraz doğrulama (cross-validation) ile modelleri kıyaslayan ve nihayetinde bir Topluluk Oylaması (Voting Classifier) ile en 
yüksek doğruluğa ulaşmayı hedefleyen kapsamlı bir uçtan uca iş akışını kapsamaktadır. Temel amaç, aynı sentetik veri seti üzerinde farklı algoritmaların potansiyelini keşfetmek ve en iyi sonuçları elde etmek için
en güçlü modelleri birleştirerek Topluluk Öğrenmesi yöntemini uygulamaktır.

*  Teknolojik Altyapı ve Kütüphaneler
Analiz süreci, veri manipülasyonundan karmaşık modellemeye kadar modern kütüphaneler üzerine inşa edilmiştir:
- Veri ve Görselleştirme: pandas, numpy, seaborn, matplotlib.
- Modelleme: LogisticRegression, SVC, KNeighborsClassifier, RandomForestClassifier.
- Optimizasyon: GridSearchCV, StratifiedKFold.
- Topluluk Öğrenmesi: VotingClassifier.
'''
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Survived': np.random.randint(0, 2, 891),
    'Pclass': np.random.randint(1, 4, 891),
    'Age': np.random.uniform(18, 80, 891),
    'Fare': np.random.uniform(7, 512, 891),
    'Sex_male': np.random.randint(0, 2, 891)
}

train_df = pd.DataFrame(data)
train_df_len = len(train_df)

train = train_df[:train_df_len].copy()
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)
print("X_train", len(X_train))
print("X_test", len(X_test))
print("y_train", len(y_train))
print("y_test", len(y_test))

logreg = LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train) * 100, 2)
acc_log_test = round(logreg.score(X_test,y_test) * 100, 2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))

random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
              SVC(random_state = random_state, probability = True),
              RandomForestClassifier(random_state = random_state),
              LogisticRegression(random_state = random_state, solver = "liblinear"),
              KNeighborsClassifier()]

dt_param_grid = {"min_samples_split": range(10,500,20),
                 "max_depth": range(1, 20, 2)}

svc_param_grid = {"kernel": {"rbf"},
                  "gamma": {0.001, 0.01, 0.1, 1},
                  "C": [1, 10, 50, 100, 200, 300, 1000]}

rf_param_grid = {"max_features": [1, 3, 10],
                 "min_samples_split": [2, 3, 10],
                 "min_samples_leaf":[1, 3, 10],
                 "bootstrap": [False],
                 "n_estimators": [100, 300],
                 "criterion": ["gini"]}

logreg_param_grid = {"C": np.logspace(-3, 3, 7),
                     "penalty":["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1, 19, 10, dtype = int).tolist(),
                  "weights": ["uniform", "distance"],
                  "metric": ["euclidean","manhattan"]}

classifier_param = [dt_param_grid,
                    svc_param_grid,
                    rf_param_grid,
                    logreg_param_grid,
                    knn_param_grid]

cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv = StratifiedKFold(n_splits = 10),scoring ="accuracy", n_jobs = -1, verbose = 1)
    clf.fit(X_train, y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
    
cv_results = pd.DataFrame({"Cross Validation Accuracy Means": cv_result, "ML Models": ["DecisionTreeClassifier", "SVM", "RandomForestClassifier",
                                                                                       "LogisticRegression","KNeighborsClassifier"]})
cv_results

plt.figure(figsize=(10, 6))
s = sns.barplot(x="ML Models", y="Cross Validation Accuracy Means", data=cv_results)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title("ML Modelleri Çapraz Doğrulama Başarımı")
plt.tight_layout()

s = sns.barplot(cv_results, x = "ML Models", y = "Cross Validation Accuracy Means")

votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),
                                         ("svc", best_estimators[1]),
                                         ("rfc", best_estimators[2]),
                                         ("lr", best_estimators[3])],
                                         voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test),y_test))
'''
Kod Akışı:
1.	Veri Hazırlığı: Titanic veri seti boyutlarında, tamamen rastgele özelliklerden oluşan sentetik bir DataFrame oluşturulur.
2.	Baseline Belirleme: Optimizasyon öncesi başarı kriterini belirlemek amacıyla Lojistik Regresyon ile bir temel skor alınır.
3.	Model ve Hiperparametre Tanımlama: Beş farklı algoritma (Decision Tree, SVC, Random Forest, Logistic Regression, KNN) için geniş hiperparametre aralıkları belirlenir.
4.	GridSearchCV ile Optimizasyon: Her model için çapraz doğrulama (CV) yapılarak en iyi parametre kombinasyonları tespit edilir.
5.	Performans Karşılaştırma: Optimize edilmiş modellerin en iyi CV skorları, karşılaştırmalı bir çubuk grafik ile görselleştirilir.
6.	Voting Classifier (Ensemble): En başarılı modeller (DT, SVC, RF, LR), soft voting yöntemiyle birleştirilir.
o	Neden Soft Voting? Sadece etiketleri değil, modellerin tahmin olasılıklarını da hesaba katarak daha dengeli ve yüksek performanslı bir sonuç hedeflenir.
7.	Final Değerlendirme: Oluşturulan topluluk modelinin nihai başarısı, test seti üzerinde doğruluk (accuracy) skoru ile ölçülür.
'''
