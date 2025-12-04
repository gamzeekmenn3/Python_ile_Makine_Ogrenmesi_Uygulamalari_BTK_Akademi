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

1. Gerekli tüm kütüphaneler içeri aktarılır ve "load_breast_cancer" fonksiyonu ile veri seti yüklenir.
2. Özellikler (X) ve hedef değişken (y) ayrıldıktan sonra veri, modelin daha iyi performans göstermesi için ölçeklenir.
3. K=3 komşu sayısıyla bir KNN modeli oluşturulur ve eğitim verisi ile eğitilir.
4. Modelin performansını doğruluk skoru ve karmaşıklık matrisi ile değerlendirilir.
5. K hiperparametresinin 1 ile 20 arasındaki değerleri denenerek en iyi K değeri bulunur ve sonuçlar grafiklenir.
