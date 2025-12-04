3_1_KNN.py dosyası Scikit-learn kütüphanesinden alınan Meme Kanseri (Breast Cancer) veri setini kullanarak KNN sınıflandırma modelini uygulamaktadır.
Amacı, tümör özelliklerine göre tümörün iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin etmektir.

Proje İçeriği:
 * Veri İncelemesi: "load_breast_cancer" veri setinin yüklenmesi ve Pandas DataFrame'e dönüştürülmesi.
 * Ön İşleme: Verilerin eğitim (train) ve test (test) kümelerine ayrılması ve StandardScaler ile ölçeklenmesi.
 * Model Eğitimi: Başlangıçta K=3 komşu kullanılarak KNN modelinin eğitilmesi.
 * Model Değerlendirmesi: Doğruluk (Accuracy) skoru ve Karmaşıklık Matrisi (Confusion Matrix) ile performansın ölçülmesi.
 * Hiperparametre Optimizasyonu:Farklı K değerleri denenerek en iyi doğruluk skorunu veren K değerinin bulunması ve görselleştirilmesi.

Kullanılan Kütüphaneler:
- sklearn: Makine öğrenmesi algoritmaları ve araçları.
- pandas: Veri manipülasyonu ve analizi.
- matplotlib: Grafik ve görselleştirme.

Kod Akışı:
1. Gerekli tüm kütüphaneler içeri aktarılır ve "load_breast_cancer" fonksiyonu ile veri seti yüklenir.
2. Özellikler (X) ve hedef değişken (y) ayrıldıktan sonra veri, modelin daha iyi performans göstermesi için ölçeklenir.
3. K=3 komşu sayısıyla bir KNN modeli oluşturulur ve eğitim verisi ile eğitilir.
4. Modelin performansını doğruluk skoru ve karmaşıklık matrisi ile değerlendirilir.
5. K hiperparametresinin 1 ile 20 arasındaki değerleri denenerek en iyi K değeri bulunur ve sonuçlar grafiklenir.

3_1_KNN.py dosyası regresyon örneğinde KNN algoritmasını kullanarak yapay olarak oluşturulmuş gürültülü bir sinüs dalgası üzerindeki tahmin yeteneğini incelemektedir. Temel odak noktası, farklı ağırlıklandırma (weights) parametrelerinin modelin tahmin eğrisi üzerindeki etkisini görselleştirmektir.

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

1. X değerleri (0 ile 5 arasında 40 nokta) oluşturulur ve y değerleri (y = sin(X)) hesaplanır. Daha sonra gerçekçi simülasyon için her 5. noktaya gürültü eklenir.
2. Modelin 0 ile 5 aralığındaki tüm tahmin eğrisini detaylıca çizebilmek için yoğun bir test aralığı ("T") oluşturulur.
3. Bir döngü ile iki farklı ağırlıklandırma yöntemi denenir. Her döngüde model eğitilir ve test noktaları üzerinde tahmin yapılır.
4. "matplotlib" kullanılarak her bir ağırlıklandırma yönteminin tahmin eğrisi, orijinal gürültülü veri noktaları ile birlikte ayrı alt grafiklerde ("subplot") gösterilir.
  * "uniform": Tahmin eğrisi, komşuların ortalamasını aldığı için daha basamaklı görünme eğilimindedir.
  * "distance": Daha yakın komşular daha fazla etkili olduğu için, tahmin eğrisi genellikle orijinal verilere daha yakındır.
Bu karşılaştırma, KNN Regresyon algoritmasında mesafe bazlı ağırlıklandırmanın (distance) genellikle temel sinyal eğilimini takip etmede, basit ortalama ağırlıklandırmaya (uniform) kıyasla daha iyi ve daha az basamaklı tahmin eğrileri ürettiğini gösterir. Bu, makine öğrenmesinde hiperparametre seçiminin önemini vurgular.
