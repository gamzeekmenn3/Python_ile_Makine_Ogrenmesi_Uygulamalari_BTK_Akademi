3_1_KNN.py dosyasında ilk örnekte Scikit-learn kütüphanesinden alınan Breast Cancer veri setini kullanarak KNN sınıflandırma modelini uygulamaktadır. Amacı, tümör özelliklerine göre tümörün iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin etmektir.

Proje İçeriği:
 * Veri İncelemesi: "load_breast_cancer" veri setinin yüklenmesi ve Pandas DataFrame'e dönüştürülmesi.
 * Ön İşleme: Verilerin train ve test kümelerine ayrılması ve StandardScaler ile ölçeklenmesi.
 * Model Eğitimi: Başlangıçta K = 3 komşu kullanılarak KNN modelinin eğitilmesi.
 * Model Değerlendirmesi: Doğruluk (Accuracy) skoru ve Karmaşıklık Matrisi (Confusion Matrix) ile performansın ölçülmesi.
 * Hiperparametre Optimizasyonu: Farklı K değerleri denenerek en iyi doğruluk skorunu veren K değerinin bulunması ve görselleştirilmesi.

Kullanılan Kütüphaneler:
- sklearn: Makine öğrenmesi algoritmaları ve araçları.
- pandas: Veri manipülasyonu ve analizi.
- matplotlib: Grafik ve görselleştirme.

Kod Akışı:
1. Gerekli tüm kütüphaneler içeri aktarılır ve "load_breast_cancer" fonksiyonu ile veri seti yüklenir.
2. Özellikler (X) ve hedef değişken (y) ayrıldıktan sonra veri, modelin daha iyi performans göstermesi için ölçeklenir.
3. K = 3 komşu sayısıyla bir KNN modeli oluşturulur ve eğitim verisi ile eğitilir.
4. Modelin performansını doğruluk skoru ve karmaşıklık matrisi ile değerlendirilir.
5. K hiperparametresinin 1 ile 20 arasındaki değerleri denenerek en iyi K değeri bulunur ve sonuçlar grafiklenir.

3_1_KNN.py dosyasında ikinci yani regresyon örneğinde KNN algoritmasını kullanarak yapay olarak oluşturulmuş gürültülü bir sinüs dalgası üzerindeki tahmin yeteneğini incelemektedir. Temel odak noktası, farklı ağırlıklandırma (weights) parametrelerinin modelin tahmin eğrisi üzerindeki etkisini görselleştirmektir.

Proje İçeriği:
* Veri Seti: 0 ile 5 arasında 40 adet noktadan oluşan, sinüs fonksiyonu ile üretilmiş ve rastgele gürültü eklenmiş yapay bir veri seti kullanılır.
* Model: Scikit-learn'den "KNeighborsRegressor"
* Hiperparametre: Komşu sayısı sabit 5 olarak ayarlanmıştır.
* Karşılaştırma: İki farklı ağırlıklandırma yöntemi denenir:
    1. uniform"(Tek Tip): Tüm komşuların tahmine eşit katkısı vardır.
    2. "distance" (Mesafeye Göre): Tahmin noktasına daha yakın komşular daha yüksek ağırlığa sahiptir.

Kullanılan Kütüphaneler:
- numpy: Sayısal işlemler ve veri seti oluşturma.
- matplotlib: Grafik çizimi ve görselleştirme.
- sklearn.neighbors: KNN Regresyon modelini kullanmak.
  
Kod Akışı:
1. X değerleri (0 ile 5 arasında 40 nokta) oluşturulur ve y değerleri (y = sin(X)) hesaplanır. Daha sonra gerçekçi simülasyon için her 5. noktaya gürültü eklenir.
2. Modelin 0 ile 5 aralığındaki tüm tahmin eğrisini detaylıca çizebilmek için yoğun bir test aralığı (T) oluşturulur.
3. Bir döngü ile iki farklı ağırlıklandırma yöntemi denenir. Her döngüde model eğitilir ve test noktaları üzerinde tahmin yapılır.
4. "matplotlib" kullanılarak her bir ağırlıklandırma yönteminin tahmin eğrisi, orijinal gürültülü veri noktaları ile birlikte ayrı alt grafiklerde (subplot) gösterilir.
  * "uniform": Tahmin eğrisi, komşuların ortalamasını aldığı için daha basamaklı görünme eğilimindedir.
  * "distance": Daha yakın komşular daha fazla etkili olduğu için, tahmin eğrisi genellikle orijinal verilere daha yakındır.
Bu karşılaştırma, KNN Regresyon algoritmasında mesafe bazlı ağırlıklandırmanın (distance) genellikle temel sinyal eğilimini takip etmede, basit ortalama ağırlıklandırmaya (uniform) kıyasla daha iyi ve daha az basamaklı tahmin eğrileri ürettiğini gösterir. Bu, makine öğrenmesinde hiperparametre seçiminin önemini vurgular.

---------------------------------------------------------------------------------------------------------------------------------------------------------

3_2_DecisionTree.py dosyası, Iris veri setini kullanarak bir Decision Tree-DT sınıflandırma modelini uygulamakta, eğitmekte ve modelin sonuçlarını detaylıca görselleştirmektedir. Projenin temel amacı, bir çiçeğin taç ve çanak yaprağı ölçümleri gibi özelliklerine bakarak hangi Iris türüne (Setosa, Versicolor, Virginica) ait olduğunu yüksek doğrulukla tahmin edebilen bir model oluşturmaktır.

Proje İçeriği:
* Veri Hazırlama: Iris veri setinin %80'i eğitim, %20'si test için ayrılmıştır.
* Model Eğitimi: Gini katsayısı (criterion = "gini") ve maksimum 5 derinlik (max_depth = 5) ile bir DecisionTreeClassifier eğitilmiştir.
* Değerlendirme: Doğruluk skoru ve Karmaşıklık Matrisi ile model performansı ölçülmüştür.
* Ağaç Görselleştirme: Eğitilmiş Karar Ağacının yapısı ve aldığı kararlar görselleştirilmiştir.

Kullanılan Kütüphaneler:
- sklearn.datasets: Iris veri setini yüklemek.
- sklearn.tree: Karar Ağacı modelini (DecisionTreeClassifier) ve çizimini (plot_tree) sağlamak. 
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak.
- sklearn.metrics: Model performansını değerlendirmek (Doğruluk, Karmaşıklık Matrisi).
- matplotlib: Grafik çizimi ve görselleştirme.

Kod Akışı:
1. Veri setindeki özellikler (X) ve hedef sınıflar (y) ayrılır, ardından model eğitimi için hazır hale getirilir.
2. Model, Gini katsayısı ve 5 maksimum derinlik (overfitting'i önlemek için) kullanılarak eğitilir.
3. Modelin, daha önce görmediği test verisi üzerindeki performansı ölçülür.
4. Eğitilen modelin iç yapısı grafiksel olarak çizilerek modelin karar verme mantığı anlaşılır hale getirilir. Her düğümdeki kutu, karar kuralını, örnek sayısını ve sınıf dağılımını gösterir.
5. Modelin nihai kararlarında hangi özelliklerin en kritik olduğunu gösterir.
Bu çıktıya göre, modelin Iris türünü sınıflandırmada en çok petal length ve petal width özelliklerine güvendiği anlaşılır.

3_2_DecisionTree.py dosyası, Iris veri setini kullanarak bir Decision Tree sınıflandırıcısının karar sınırlarını incelemektedir. Iris veri setindeki dört özelliğin tüm olası ikişerli kombinasyonları için modeller eğitilir ve karar bölgeleri görselleştirilir. Modelin, farklı özellik çiftlerini kullandığında sınıflandırma uzayını nasıl böldüğünü ve hangi özelliklerin en net ayrımı sağladığını görsel olarak belirlemektir. Karar Ağacının temel özelliği olan eksenlere paralel karar sınırları bu grafikte açıkça görülür.

Kullanılan Kütüphaneler:
- sklearn.datasets: Iris veri setini yüklemek. 
- sklearn.tree: Karar Ağacı sınıflandırma modelini kullanmak.
- sklearn.inspection: Karar Sınırlarını (DecisionBoundaryDisplay) çizmek.
- matplotlib: Grafik çizimi ve görselleştirme.
- numpy: Sayısal işlemler ve veri manipülasyonu.

Kod Akışı:
1. Iris veri setinde dört özellik bulunur: Sepal Length (SL), Sepal Width (SW), Petal Length (PL) ve Petal Width (PW). Kod, bu 4 özelliğin tüm olası ikili kombinasyonunu dener.
2. Bir döngü, 6 özellik çiftini de dolaşır. Her döngüde:
    * Veri, o anki iki özellik kullanılarak filtre edilir (X = iris.data[:, pair).
    * Yeni bir DecisionTreeClassifier eğitilir (clf.fit(X, y)).
3. Her bir modelin sonucu, 2x3 düzeninde bir alt grafiğe çizilir:
    * Renk Alanları: Modelin ilgili bölgede hangi sınıfı tahmin edeceğini gösterir.
    * Noktalar: Orijinal veri noktalarıdır ve modelin ne kadarının doğru bölgelerde yer aldığını gösterir.
4. Grafikler incelendiğinde, genellikle Petal Length (PL) ve Petal Width (PW) özelliklerinin kullanıldığı grafiklerde (sağ alt köşe), Karar Sınırlarının en basit ve en net olduğu görülür. Bu, bu iki özelliğin Iris türlerini sınıflandırmak için en güçlü ayırıcılar olduğunu doğrular.

3_2_DecisionTree.py dosyası, Scikit-learn kütüphanesinden alınan Diyabet veri setini kullanarak bir Decision Tree Regressor modeli oluşturmayı ve bu modelin performansını Ortalama Karesel Hata (MSE) ve Karesel Ortalama Hata Kökü (RMSE) metrikleriyle değerlendirmeyi amaçlamaktadır. Modelin temel amacı, hastaların tıbbi özelliklerine (BMI, kan basıncı, vb.) dayanarak diyabet hastalığının ilerleme seviyesini tahmin etmek. Bu, sayısal (sürekli) bir değer tahmin etme problemi olduğu için bir Regresyon görevidir.

Kullanılan Kütüphaneler:
- sklearn.datasets: Diyabet veri setini yüklemek.
- sklearn.tree: Karar Ağacı Regresyon modelini (DecisionTreeRegressor) kullanmak.
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak. 
- sklearn.metrics: Regresyon performansını değerlendirmek (mean_squared_error).
- numpy: Matematiksel işlemler (özellikle karekök alma için).

Kod Akışı:
1. Diyabet veri setindeki 10 adet özellik (X) ve hedef ilerleme değeri (y) ayrılır. Veri setinin %20'si test için ayrılmıştır.
2. Model oluşturulur ve eğitim verisi ile eğitilir. Regresyon Ağaçları, dallanma kararlarını alırken her düğümdeki varyansı (veya MSE'yi) en aza indirmeyi hedefler.
3. Eğitilen model, test verisi üzerinde tahminler yapar ve bu tahminlerin kalitesi hata metrikleri ile ölçülür.
    * Hata Metrikleri:
        - Ortalama Karesel Hata (Mean Squared Error - MSE): Hata paylarının karesinin ortalamasını alır. Büyük hataları daha çok cezalandırır. Modelin ne kadar              başarılı olduğunu sayısal olarak gösterir, ancak hedef değişkenin birimiyle aynı değildir.
        - Karesel Ortalama Hata Kökü (Root Mean Squared Error - RMSE): MSE'nin kareköküdür. En yaygın kullanılan regresyon metriğidir. Hata birimi, tahmin edilen            değerin (hastalık ilerlemesi) birimiyle aynıdır, bu da sonucu yorumlamayı kolaylaştırır. Değer ne kadar düşükse, model o kadar iyidir.

Elde edilen RMSE değeri, modelin yaptığı tahminlerin, gerçek hastalık ilerleme seviyelerinden ortalama olarak ne kadar sapma gösterdiğini ifade eder. Bu değer, modeli diğer regresyon algoritmalarıyla (örneğin Lineer Regresyon) karşılaştırmak için kullanılacak temel ölçüttür.

3_2_DecisionTree.py dosyası, Decision Tree Regressor modelinde maksimum derinliğin (max_depth) modelin genelleme yeteneği üzerindeki kritik rolünü görsel olarak incelemektedir. Yapay bir sinüs dalgası veri seti kullanılarak, sığ (shallow) ve derin (deep) ağaçların tahmin eğrileri karşılaştırılmıştır. Makine öğrenmesinde sıkça karşılaşılan Aşırı Öğrenme (Overfitting) ve Eksik Öğrenme (Underfitting) kavramlarını, "max_depth" hiperparametresi üzerinden somutlaştırmaktır.
  * Sığ Ağaç (Max Depth = 2): Basit ve genelleştirilmiş bir eğri çizer. "Underfitting" riski taşır.
  * Derin Ağaç (Max Depth = 15): Eğitim verisindeki gürültüyü dahi ezberleyerek karmaşık, zikzaklı bir eğri çizer. "Overfitting" riski taşır.

Kullanılan Kütüphaneler:
- sklearn.tree: Karar Ağacı Regresyon modelini kullanmak.
- numpy: Sayısal işlemler ve veri seti oluşturma (sinüs ve gürültü).
- matplotlib: Grafik çizimi ve görselleştirme.

Kod Akışı:
1. 0 ile 5 arasında 80 noktadan oluşan, sinüs fonksiyonu ile üretilmiş ve rastgele gürültü eklenmiş yapay bir veri seti (X, y) oluşturulur.
2. Farklı derinliklerde iki ayrı "DecisionTreeRegressor" modeli aynı veri seti ile eğitilir.
3. Tüm 0 ile 5 aralığında yoğun test noktaları (X_test) oluşturulur ve her iki modelden tahminler alınır.

Sonuçların Yorumlanması:
- Mavi Eğri | Max Depth = 2 | Model çok basit kalmıştır. Verideki gürültüyü görmezden gelirken, sinüs dalgasının ince detaylarını da yakalayamaz. | Underfitting |
- Yeşil Eğri | Max Depth = 15 | Model, eğitim verisindeki her noktaya (gürültü dahil) uymak için çok karmaşık kararlar almıştır. Yeni veride başarısız olması muhtemeldir. | Overfitting |

En iyi performans genellikle, eğitim ve test verisi arasında en iyi dengeyi sağlayan orta bir "max_depth" değeri seçilerek elde edilir.

---------------------------------------------------------------------------------------------------------------------------------------------------------

 3_3_RandomForest.py dosyası, Olivetti Faces veri setini kullanarak bir Random Forest Classifier modeli oluşturmayı ve bu modelin performansını yüz tanıma görevinde değerlendirmeyi amaçlamaktadır. Modelin temel amacı, bir yüz görüntüsünün hangi 40 kişiden birine ait olduğunu tahmin etmektir. Bu, yüksek boyutlu özelliklere (piksel değerleri) sahip bir çoklu sınıflandırma (multi-class classification) problemidir.

Kullanılan Kütüphaneler:
- sklearn.datasets: Olivetti Faces veri setini yüklemek.
- sklearn.ensemble: Rastgele Orman modelini (RandomForestClassifier) kullanmak.
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak.
- sklearn.metrics: Model performansını değerlendirmek (Accuracy)
- matplotlib: Görüntüleri görselleştirmek. 

Kod Akışı:
1. Olivetti veri seti 40 farklı kişiye ait toplam 400 adet 64x64 piksel (4096 özellik) gri tonlamalı görüntü içerir.
2. Görüntüler zaten 4096 boyutlu vektörler halinde düzleştirilmiştir (oli.data). Veri seti %20'si test, %80'i eğitim için ayrılır.
3. Rastgele Orman (RF), tahmin yapmak için 100 farklı karar ağacının sonuçlarını bir araya getirir. Bu ensemble learning yöntemi, tek bir karar ağacına göre daha istikrarlı ve doğru sonuçlar verir.
4. Model, test setindeki yüz görüntülerinin hangi kişiye ait olduğunu tahmin eder ve bu tahminlerin doğruluğu ölçülür.

* Accuracy Skoru: Elde edilen doğruluk değeri, modelin test setindeki yüzlerin yüzde kaçını doğru kişiyle eşleştirdiğini gösterir. Rastgele Orman gibi algoritmalarla yüz tanıma görevinde genellikle oldukça yüksek doğruluk skorları elde edilir.
* n_estimators: Ağaç sayısının artırılması performansı genellikle artırır ancak eğitim süresini uzatır. Bu modelde 100 ağaç optimum dengeyi sağlamıştır.

3_3_RandomForest.py dosyası, California Housing veri setini kullanarak bir Random Forest Regressor modeli oluşturmayı ve bu modelin performansını Karesel Ortalama Hata Kökü (RMSE) metriğiyle değerlendirmeyi amaçlamaktadır. Modelin temel amacı, bir konut bölgesinin medyan hane geliri, evlerin yaşı, ortalama oda sayısı gibi özelliklerine dayanarak, o bölgedeki konut fiyatlarının medyan değerini tahmin etmektir. Bu, Regresyon problemine bir topluluk (ensemble) öğrenme çözümü uygulamaktır.

Kullanılan Kütüphaneler:
- sklearn.datasets: California Housing veri setini yüklemek.
- sklearn.ensemble: Rastgele Orman Regresyon modelini kullanmak.
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak. 
- sklearn.metrics: Regresyon performansını değerlendirmek (mean_squared_error).
- numpy: Matematiksel işlemler (özellikle karekök alma). 

Kod Akışı:
1. Veri setindeki özellikler (`X`) ve hedef konut fiyatı değerleri (`y`) ayrılır. Veri setinin %20'si test için ayrılmıştır.
2. Rastgele Orman Regresyonu modeli oluşturulur ve eğitim verisi ile eğitilir. Bu model, birçok Karar Ağacının tahminlerini toplayarak (ortalama alarak) daha genellenebilir ve doğru sonuçlar üretir.
3. Model, test verisi üzerinde tahminler yapar ve bu tahminlerin kalitesi "RMSE" ile ölçülür.
    * Hata Metriği: Karesel Ortalama Hata Kökü (RMSE): 
      - RMSE, modelin tahminlerinin gerçek konut fiyatlarından ortalama olarak ne kadar sapma gösterdiğini hedef değişkenin birimiyle aynı birimde ifade eder.
      - Daha düşük RMSE, modelin konut fiyatlarını o kadar doğru tahmin ettiği anlamına gelir.
Elde edilen RMSE değeri, modelin performansını ölçmek ve modeli diğer regresyon algoritmalarıyla karşılaştırmak için kullanılacak temel ölçüttür. Bu değer, optimizasyon (hiperparametre ayarlaması) yoluyla düşürülmeye çalışılmalıdır.

---------------------------------------------------------------------------------------------------------------------------------------------------------

3_4_LogisticRegression.py dosyası, UCI Kalp Hastalığı (Heart Disease) veri setini kullanarak bir Logistic Regression sınıflandırma modeli oluşturmayı amaçlamaktadır. Model, hastaların tıbbi özelliklerine dayanarak kalp hastalığı riskini tahmin etmektedir. Modelin temel amacı, bir hastanın tıbbi verilerine (yaş, cinsiyet, kolesterol, kan basıncı vb.) dayanarak kalp hastalığı varlığını (ikili sınıflandırma) tahmin etmektir. Lojistik Regresyon, tahminlerini 0 ile 1 arasında bir olasılık olarak sunar.

Kullanılan Kütüphaneler:
- ucimlrepo: UCI Machine Learning Repository'den veri setini indirmek.
- pandas: Veri manipülasyonu ve eksik değer yönetimi.
- sklearn.linear_model: Lojistik Regresyon modelini kullanmak.
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak.

Kod Akışı:
1. Veri seti, UCI deposundan doğrudan çekilir ve Pandas DataFrame'e dönüştürülür. Modelin düzgün çalışması için eksik değerler (`NaN`) içeren satırlar veri setinden silinir (`dropna`).
2. Veri setinin %10'u test için ayrılmıştır (`test_size = 0.1`).
3. Model, L2 regülarizasyonu kullanılarak eğitilir. Regülarizasyon, modelin aşırı karmaşıklaşmasını (overfitting) engelleyerek genelleme yeteneğini artırmayı hedefler.
4. Modelin, test verisi üzerindeki performansı Accuracy skoru ile ölçülür.

  * Accuracy Skoru: Elde edilen doğruluk değeri, modelin test setindeki hastaların yüzde kaçını doğru şekilde sınıflandırdığını (kalp hastası olup olmadığını doğru tahmin etme) gösterir.
  * Regülarizasyon: Modelde kullanılan `penalty = "l2"` ve `C = 1` hiperparametreleri, modelin katsayılarının çok büyük değerler almasını engelleyerek modelin genelleme yeteneğini korumaya yardımcı olur.
  * Lojistik Regresyon bu tür ikili sınıflandırma (binary classification) problemleri için güçlü, basit ve yorumlanması kolay bir başlangıç modelidir.

---------------------------------------------------------------------------------------------------------------------------------------------------------

3_5_SVM.py dosyası, Digits veri setini kullanarak bir Support Vector Classifier - SVC modeli oluşturmayı, eğitip değerlendirmeyi amaçlamaktadır. Model, 8X8 piksellik görüntü vektörlerine dayanarak el yazısı rakamları 0’dan 9’a) tanır. Modelin temel amacı, düşük çözünürlüklü görüntü verilerinden rakamları doğru bir şekilde tanımaktır. Bu bir çoklu sınıflandırma (multi-class classification) problemidir.

Kullanılan Kütüphaneler:
- sklearn.datasets: Digits veri setini yüklemek. 
- sklearn.svm: Destek Vektör Sınıflandırıcı (SVC) modelini kullanmak. 
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak.
- sklearn.metrics: Model performansını değerlendirmek (`classification_report`). 
- matplotlib: Görüntüleri görselleştirmek.

Kod Akışı:
1. Digits veri seti, 8x8 (yani 64 piksel) boyutunda görüntüler içerir. Model bu 64 piksel değerini özellik olarak kullanır.
2. 8x8 matris şeklindeki görüntüler 64 boyutlu vektörlere dönüştürülmüş (`digits.data`) olarak kullanılır. Veri setinin %20'si test için ayrılır.
3. Model, doğrusal (linear) bir karar sınırı oluşturmayı hedefleyen bir çekirdek (`kernel`) ile eğitilir. SVM'nin temel amacı, sınıflar arasındaki en büyük marjı bularak optimum ayırma hiperdüzlemini oluşturmaktır.
4. Modelin, test verisi üzerindeki performansı Sınıflandırma Raporu ile detaylıca incelenir.

  * Sınıflandırma Raporu çıktısı, modelin her bir rakam (sınıf) için Precision, Recall ve F1-Score metriklerini gösterir.
      * Yüksek Precision, modelin bir rakamı X olarak tahmin ettiğinde ne kadar sıklıkla haklı olduğunu gösterir.
      * Yüksek Recall, gerçekte X olan rakamların ne kadarının model tarafından doğru yakalandığını gösterir.
  * kernel = "linear": Doğrusal çekirdek, SVM'nin 64 boyutlu uzayda rakamları ayırmak için hiperdüzlem kullandığı anlamına gelir. Bu veri seti için basit bir doğrusal çekirdek genellikle yüksek doğruluk sağlar.
  * Genel Doğruluk (Accuracy): Raporun altındaki makro/ağırlıklı ortalamalar, modelin tüm rakamlar üzerindeki genel başarı seviyesini özetler.

---------------------------------------------------------------------------------------------------------------------------------------------------------

3_6_NaiveBayes.py dosyası, Iris veri setini kullanarak bir Gaussian Naive Bayes (GNB) sınıflandırma modeli oluşturmayı ve bu modelin performansını detaylı bir sınıflandırma raporuyla değerlendirmeyi amaçlamaktadır. Modelin temel amacı, bir çiçeğin taç ve çanak yaprağı ölçümleri gibi özelliklerine bakarak hangi Iris türüne (Setosa, Versicolor, Virginica) ait olduğunu tahmin etmektir. Naive Bayes, Basit Bayes Teoremi'ne dayanır ve özelliklerin birbirinden bağımsız olduğunu varsayar.

Kullanılan Kütüphaneler:
- sklearn.datasets: Iris veri setini yüklemek. 
- sklearn.naive_bayes: Gaussian Naive Bayes modelini kullanmak. 
- sklearn.model_selection: Veriyi eğitim ve test kümelerine ayırmak.
- sklearn.metrics: Model performansını değerlendirmek (`classification_report`). 

Kod Akışı:
1. Iris veri setindeki özellikler (`X`) ve hedef sınıflar (`y`) ayrılır. Veri setinin %20'si test için ayrılmıştır.
2. Model, Gaussian (Normal) dağılım varsayımı ile eğitilir. Bu model, her sınıf için her bir özelliğin ortalama ve standart sapma değerlerini hesaplayarak olasılık dağılımlarını öğrenir.
3. Model, test verisi üzerinde tahminler yapar ve bu tahminlerin kalitesi Sınıflandırma Raporu ile detaylıca incelenir.

Sınıflandırma Raporu çıktısı, modelin her bir sınıf (Iris türü) üzerindeki performansını özetler:
  * Precision: Modelin bir türü X olarak tahmin ettiğinde ne kadar haklı olduğunu gösterir.
  * Recall: Gerçekte X olan türlerin ne kadarının model tarafından yakalandığını gösterir.
  * F1-Score: Precision ve Recall'un denge ölçüsüdür.
  * Accuracy: Genel başarı skoru.
Naive Bayes, basitliği ve yüksek hesaplama hızı nedeniyle, özellikle özelliklerin bağımsız olduğu (veya bağımsızlığa yakın olduğu) durumlarda çok etkili bir başlangıç modelidir.

---------------------------------------------------------------------------------------------------------------------------------------------------------

3_7_ClassificationModelComparison.py dosyası, farklı geometrik yapıdaki üç sentetik veri seti üzerinde beş temel makine öğrenmesi sınıflandırma algoritmasının performansını, karar sınırlarını ve doğruluk skorlarını görsel olarak karşılaştırmayı amaçlamaktadır. Temel amacımız, her bir algoritmanın (doğrusal, doğrusal olmayan, topluluk tabanlı) farklı veri yapılarına nasıl tepki verdiğini ve hangi durumlarda doğrusal olmayan sınırlar oluşturarak daha iyi genelleme sağladığını görsel bir matris üzerinde göstermektir.

Kullanılan Kütüphaneler:
- sklearn.datasets: Sentetik veri setlerini (küme, hilal, daire) oluşturmak.
- sklearn.neighbors: K En Yakın Komşu (`KNeighborsClassifier`).
- sklearn.svm: Destek Vektör Makineleri (`SVC`). 
- sklearn.tree / sklearn.ensemble: Karar Ağaçları ve Rastgele Orman.
- sklearn.naive_bayes: Gaussian Naive Bayes. 
- sklearn.preprocessing: Veri ölçekleme (`StandardScaler`).
- sklearn.pipeline: İşlem hatları (`make_pipeline`) oluşturmak.
- sklearn.inspection: Karar sınırlarını çizmek (`DecisionBoundaryDisplay`).

Veri Setleri ve Yapıları:
* Küme Tipi (`make_classification`): Doğrusal bir sınırla ayrılabilir, ancak rastgele gürültü eklenmiştir.
* Hilal Tipi (`make_moons`): Doğrusal olmayan bir sınır gerektirir (örneğin, SVM veya KNN).
* Daire Tipi (`make_circles`): İç içe yapısından dolayı güçlü doğrusal olmayan yeteneklere sahip modeller gerektirir.

Karşılaştırılan Algoritmalar: 
**Nearest Neighbors (KNN)**: Doğrusal Olmayan | Mesafeye dayalıdır. Ölçeklemeye karşı hassastır.
**Linear SVM**: Doğrusal/Çekirdek | Doğrusal olmayan bir çekirdek (`RBF`) ile test edilmiştir.
**Decision Tree**: Doğrusal Olmayan | Eksenlere paralel, kademeli sınırlar oluşturur. 
**Random Forest**: Topluluk (Ensemble) | Birçok Karar Ağacının ortalamasıyla daha düzgün sınırlar oluşturur.
**Naive Bayes**: Olasılıksal | Özelliklerin normal dağıldığını varsayar. 

Kod Akışı:
1. Tüm algoritmaların adil bir şekilde karşılaştırılması ve özellikle mesafe tabanlı algoritmaların doğru çalışması için, StandardScaler ve sınıflandırıcı bir işlem hattında (`make_pipeline`) zincirlenmiştir.
2. Kodun ana çıktısı, 3 satır (veri setleri) ve 6 sütundan (giriş verisi + 5 model) oluşan büyük bir Matplotlib figürüdür.

| | **Giriş Verisi** | **KNN** | **SVM** | **Decision Tree** | **Random Forest** | **Naive Bayes** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Satır 1** | Küme Tipi | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı |
| **Satır 2** | Hilal Tipi | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı |
| **Satır 3** | Daire Tipi | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı | Karar Sınırı |

Her bir alt grafik, modelin test verisi üzerindeki doğruluk skorunu içerir ve renkli alanlar, modelin hangi bölgeyi hangi sınıfa atadığını gösteren karar sınırlarını temsil eder.

- Doğrusal Veri: Çoğu model (basit olanlar dahil) yüksek doğruluk skorları elde edecektir.
- Hilal/Daire Veri: Lineer SVM ve Gaussian Naive Bayes gibi doğrusal modellerin, Hilal ve Daire setlerinde düşük performans göstermesi ve düz/basit sınırlar çizmesi beklenir.
- Doğrusal Olmayan Başarı: Random Forest ve KNN gibi güçlü doğrusal olmayan modellerin, Daire ve Hilal setlerinde yüksek skorlar ve karmaşık, veri yapısına uygun sınırlar çizmesi beklenir.
- Karar Ağacı Sınırları: Karar Ağacı modelleri, her zaman eksenlere paralel, basamaklı sınırlar oluşturacaktır.

---------------------------------------------------------------------------------------------------------------------------------------------------------

