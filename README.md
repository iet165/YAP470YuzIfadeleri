# YAP470YuzIfadeleri
Yüz İfadelerinin Sınıflandırılması ve Gerçek Zamanlı Tahmini


Veri Hazırlama
Orijinal veri seti çok büyük olduğu için GitHub reposuna sadece her sınıftan birkaç örnek içeren küçük bir alt küme yükledim. Kodda, veri setinin mevcut olup olmadığı ilk adımda kontrol ediliyor. Eğer işlenmiş veri seti klasörü boşsa ya da yoksa, preprocess.py adlı script otomatik olarak çalıştırılıyor. Bu script ile görüntüler yeniden boyutlandırılıyor, etiketleniyor ve uygun klasör yapısında kaydediliyor. Dosya yolları doğrudan bilgisayar dizinine göre sabitlenmediği için, bu kod başka bir bilgisayarda da sorunsuz çalışabilir.

Split ve Generator Yaratımı
Verileri eğitim, doğrulama ve test olarak üçe böldüm. Bu işlemi yaparken sınıf dengesi korunması için stratify parametresini kullandım. Görüntüleri modele uygun hale getirmek ve çeşitlendirmek için TensorFlow’un ImageDataGenerator sınıfını kullandım. Eğitim sırasında veri artırma (data augmentation) uygulandı. Eğitim setinde veriler karıştırılıyor, doğrulama ve test setlerinde ise sıralı şekilde kullanılıyor.

Model Tasarımı ve Eğitimi
Evrişimli Sinir Ağı (CNN) modelimi sıfırdan kendim tasarladım ve Sequential API kullanarak oluşturdum. Modelin içinde birden fazla Conv2D ve BatchNormalization katmanı var. Aşırı öğrenmeyi (overfitting) önlemek için Dropout katmanları ekledim. Modeli Adam optimizer ile derledim ve sınıflandırma problemi olduğu için categorical_crossentropy kayıp fonksiyonu kullandım. Eğitim süresince erken durdurma (EarlyStopping) ve en iyi modeli kaydetme (ModelCheckpoint) gibi callback’lerden faydalandım. Böylece eğitim sırasında modelin en iyi ağırlıkları otomatik olarak kayıt altına alındı.

Gerçek Zamanlı Demo
Web kamerası üzerinden yüz tanıma ve duygu tahmini yapan bir demo hazırladım. OpenCV kütüphanesini kullanarak görüntü akışını aldım, yüzleri Haar Cascade yöntemiyle tespit ettim. Algılanan yüzler modele uygun hale getirilip sınıflandırıldı ve tahmin edilen duygu etiketi ekran üzerinde gösterildi.

Notebook'taki Çıktılarla Yeni Output'lar Neden Birebir Aynı Olmayabilir?
Kodun bazı kısımlarında rastgelelik olduğu için (örneğin verilerin karıştırılması, örnek resimlerin gösterilmesi) notebook çıktılarındaki görsellerle yeni çalıştırmada elde edilen görseller birebir aynı olmayabilir. Ancak bu durum modelin performansını veya doğruluğunu etkilemez. Kodun çalışması ve modelin tahmin gücü açısından bir sorun teşkil etmiyor.

Bu projenin ikinci kısmında,FANE veri setindeki ham görüntüler, öncelikle dlib ile yüz tespiti, CLAHE ile kontrast artırımı ve standart bir boyuta getirme gibi adımları içeren kapsamlı bir ön işleme sürecinden geçirilmiştir. Bu standartlaştırılmış görüntülerden, şekil ve kenar bilgisini etkin bir şekilde yakalayan HOG (Histogram of Oriented Gradients) öznitelikleri çıkarılarak, modellerin anlayabileceği sayısal bir veri seti oluşturulmuştur. Model eğitiminden önce, farklı girdi boyutlarının performans üzerindeki etkisini ölçmek amacıyla sistematik bir deney yürütülmüş ve en verimli boyut belirlenmiştir. Ayrıca, yüksek boyutlu HOG verileri, StandardScaler ve PCA kullanılarak hem ölçeklendirilmiş hem de daha yönetilebilir bir boyuta indirgenmiştir.

Proje gereksinimleri doğrultusunda, hazırlanan bu veri üzerinde üç farklı makine öğrenmesi modeli eğitilmiş ve GridSearchCV ile en iyi hiperparametreleri bulunmuştur. Bu modeller; yüksek boyutlu verilerdeki gücüyle bilinen Support Vector Machine (SVM), karar ağaçlarından oluşan ve aşırı öğrenmeye karşı dayanıklı olan Random Forest ve önceki ağaçların hatalarından öğrenerek kendini geliştiren yüksek performanslı bir boosting algoritması olan XGBoost'tur. Her bir model, kendi doğasına uygun parametre setleri ile optimize edilerek, veri setimiz için mümkün olan en iyi performansı sergilemesi hedeflenmiştir. Bu çoklu model yaklaşımı, problemin farklı açılardan ele alınmasını ve en sağlam çözümün bulunmasını sağlamıştır.

Eğitim sürecinde, zorunlu modellerden biri olan Multi-Layer Perceptron (MLP) ile çalışılırken önemli bir teknik zorlukla karşılaşılmıştır. HOG özniteliklerinin "seyrek" ve "sıfır ağırlıklı" yapısı, MLP modelinde sayısal kararsızlığa yol açarak "gradient explosion" (patlayan gradyanlar) sorununu tetiklemiştir. Bu durum, modelin anlamlı bir öğrenme gerçekleştiremeden çökmesine veya eğitimi anında sonlandırmasına neden olmuştur. Bu sorunu aşmak ve projenin ilerleyişini sağlamak adına, geçici bir strateji olarak MLP modeli, veri yapısıyla daha uyumlu olan Random Forest ile değiştirilmiştir. Demo aşamasına kadar, MLP'deki bu sorunu çözmek için alternatif veri normalleştirme teknikleri ve daha kararlı optimizer yapılandırmaları üzerinde çalışmalara devam edilmesi planlanmaktadır.



