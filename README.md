# Farklı Evrişimli Sinir Ağı (CNN) ve Hibrit Modellerin Görüntü Sınıflandırma Performanslarının Karşılaştırılması
## Özet

Bu çalışma, farklı mimarilere ve yaklaşımlara sahip dört adet görüntü sınıflandırma modelinin performansını karşılaştırmayı amaçlamaktadır. Kullanılan modeller arasında iki adet MNIST veri seti üzerinde eğitilmiş özel Evrişimli Sinir Ağları (CNN), bir adet CIFAR-10 veri seti üzerinde transfer öğrenme yöntemiyle ince ayar yapılmış hazır CNN (ResNet-18) ve bir adet CNN'in özellik çıkarıcı katmanlarından elde edilen vektörler ile geleneksel bir Makine Öğrenmesi (ML) modelinin eğitildiği hibrit bir model bulunmaktadır. Modellerin tanımları Python kodları incelenerek çıkarılmış, performansları ise eğitim ve test aşaması çıktıları analiz edilerek değerlendirilmiştir. Sonuçlar, farklı veri setleri ve mimari yaklaşımların sınıflandırma başarısı üzerindeki etkisini açıkça göstermektedir.

## 1. Giriş (Introduction)
Görüntü sınıflandırma, bilgisayarlı görü alanının temel problemlerinden biridir ve son yıllarda Derin Öğrenme, özellikle Evrişimli Sinir Ağları (CNN) sayesinde büyük ilerleme kaydetmiştir. CNN'ler, görüntülerden hiyerarşik özellikler öğrenme yetenekleri sayesinde bu görevde üstün başarı göstermektedir. Ancak, farklı problemler ve veri setleri için en uygun mimariyi veya yaklaşımı seçmek önemli bir konudur.

Bu rapor, üç farklı veri seti ve model kombinasyonunu (MNIST üzerinde iki özel CNN, CIFAR-10 üzerinde bir hazır CNN) ve ek olarak bir hibrit yaklaşımı (CNN özellik çıkarımı + Geleneksel ML) inceleyerek performanslarını karşılaştırmayı hedeflemektedir. Çalışma kapsamında kullanılan modeller ve veri setleri şunlardır:

Model 1: MNIST veri seti üzerinde eğitilmiş özel bir CNN.
Model 2: MNIST veri seti üzerinde eğitilmiş farklı bir özel CNN.
Model 3: CIFAR-10 veri seti üzerinde ImageNet ağırlıklarıyla ince ayar yapılmış ResNet-18.
Model 4 (Hibrit): CIFAR-10 üzerinde ImageNet ağırlıklarıyla eğitilmiş ResNet-18'in özellik çıkarıcı katmanlarından elde edilen özellikler üzerinde eğitilmiş bir Destek Vektör Makinesi (SVC).

## 2. Yöntemler (Methods)
Çalışmada kullanılan dört modelin her biri farklı bir yaklaşımı temsil etmektedir. Tekrarlanabilirliği sağlamak amacıyla her modelin temel bileşenleri aşağıda ayrıntılı olarak açıklanmıştır.

### 2.1 Veri Setleri
MNIST: 28x28 piksel boyutunda, el yazısı rakamlardan (0-9) oluşan 10 sınıflı bir veri setidir. Eğitim seti 60.000, test seti 10.000 görüntü içerir. Model 1 ve Model 2 bu veri seti üzerinde kullanılmıştır. Görüntüler transforms.Pad(2) ile 32x32'ye boyutlandırılmış, ToTensor() ile tensor'a çevrilmiş ve Normalize((0.5,), (0.5,)) ile normalize edilmiştir.
CIFAR-10: 32x32 piksel boyutunda, 10 farklı nesne kategorisine ait renkli (RGB) görüntülerden oluşan 10 sınıflı bir veri setidir. Eğitim seti 50.000, test seti 10.000 görüntü içerir. Model 3 ve Model 4 bu veri seti üzerinde kullanılmıştır.
### 2.2 Model Mimarileri ve Eğitim Stratejileri
#### 2.2.1 Model 1 (MNIST üzerinde Özel CNN)
Mimari: İki Evrişim katmanı (Conv2d) ve iki Tam Bağlantılı katmandan (Linear) oluşan basit bir CNN mimarisidir. Evrişim katmanları ReLU aktivasyon fonksiyonu kullanır. İlk evrişim katmanından sonra max pooling uygulanmıştır.
Conv1: Giriş 1 kanal (gri skala), Çıkış 10 kanal, Kernel 5x5
MaxPool1: Kernel 2x2
Conv2: Giriş 10 kanal, Çıkış 20 kanal, Kernel 5x5
MaxPool2: Kernel 2x2
Ardından düzleştirme (flatten).
Linear1: Giriş boyutu evrişim katmanlarının çıktısına göre belirlenir, Çıkış 50.
Linear2: Giriş 50, Çıkış 10 (sınıf sayısı).
Eğitim: Model sıfırdan (pretrained=False) eğitilmiştir.
Kayıp Fonksiyonu: nn.CrossEntropyLoss()
Optimizatör: Adam, Öğrenme Oranı (LR): 0.001
Epoch Sayısı: 1
Batch Boyutu: 64
#### 2.2.2 Model 2 (MNIST üzerinde Özel CNN)
Mimari: Model 1'e benzer şekilde iki Evrişim katmanı ve iki Tam Bağlantılı katmandan oluşur.
Conv1: Giriş 1 kanal, Çıkış 10 kanal, Kernel 5x5
MaxPool1: Kernel 2x2
Conv2: Giriş 10 kanal, Çıkış 20 kanal, Kernel 5x5
Dropout: p=0.5 (Eğitim sırasında aşırı uydurmayı azaltmaya yardımcı olur)
MaxPool2: Kernel 2x2
Ardından düzleştirme (flatten).
Linear1: Giriş boyutu evrişim katmanlarının çıktısına göre belirlenir, Çıkış 50.
Linear2: Giriş 50, Çıkış 10 (sınıf sayısı).
Eğitim: Model sıfırdan (pretrained=False) eğitilmiştir. Dropout katmanı hariç mimari Model 1 ile aynıdır.
Kayıp Fonksiyonu: nn.CrossEntropyLoss()
Optimizatör: Adam, Öğrenme Oranı (LR): 0.001
Epoch Sayısı: 1
Batch Boyutu: 64
#### 2.2.3 Model 3 (CIFAR-10 üzerinde ResNet-18)
Mimari: torchvision.models kütüphanesinden hazır ResNet-18 mimarisi kullanılmıştır. Model, ImageNet veri seti üzerinde önceden eğitilmiş ağırlıklarla başlatılmıştır (pretrained=True). CIFAR-10'un 10 sınıfına uyarlanması için modelin son tam bağlantılı katmanı (model.fc), giriş boyutu korunarak 10 çıkışlı yeni bir nn.Linear katmanı ile değiştirilmiştir.
Veri Hazırlık: CIFAR-10 görüntüleri, ImageNet'in beklediği 224x224 boyutuna yeniden boyutlandırılmış ve ImageNet'in ortalama ve standart sapma değerleri kullanılarak normalize edilmiştir.
Eğitim: Bu model, transfer öğrenme yöntemiyle eğitilmiştir. Kod çıktılarından anlaşıldığı üzere, modelin önceden eğitilmiş (ImageNet) katmanları dondurulmamış, tüm model daha düşük bir öğrenme oranıyla (0.001) CIFAR-10 veri seti üzerinde fine-tune edilmiştir.
Kayıp Fonksiyonu: nn.CrossEntropyLoss()
Optimizatör: Adam, Öğrenme Oranı (LR): 0.001
Epoch Sayısı: 1
Batch Boyutu: 64
#### 2.2.4 Model 4 (CIFAR-10 üzerinde Hibrit CNN + Geleneksel ML)
Mimari: Bu model iki aşamalıdır.
CNN Özellik Çıkarıcı: ImageNet üzerinde önceden eğitilmiş torchvision.models.resnet18(pretrained=True) modelinin son tam bağlantılı katmanı (fc) çıkarılmıştır. Geriye kalan kısım (evrişimsel ve pooling katmanları), görüntülerden özellik vektörleri çıkarmak için kullanılır.
Geleneksel ML Sınıflandırıcı: CNN özellik çıkarıcıdan elde edilen özellik vektörleri üzerinde eğitilen ayrı bir modeldir. Çalışmada Destek Vektör Makinesi (SVC) kullanılmıştır. SVC(kernel='rbf', C=10, gamma='scale') varsayılan parametrelerle kullanılmıştır.
Veri Hazırlık: CIFAR-10 görüntüleri, CNN özellik çıkarıcının beklediği 224x224 boyutuna yeniden boyutlandırılmış ve ImageNet normalizasyonu uygulanmıştır. CNN'den çıkarılan özellikler ve karşılık gelen etiketler .npy dosyalarına kaydedilmiştir.
Eğitim: Sadece geleneksel ML modeli (SVC) eğitilmiştir. Bu eğitim, CNN'den çıkarılmış sabit özellik vektörleri üzerinde gerçekleşir. CNN'in ağırlıkları bu aşamada güncellenmez.
ML Modeli: sklearn.svm.SVC
Eğitim Verisi: CNN'den çıkarılan eğitim özellikleri ve etiketleri (train_features.npy, train_labels.npy).
Test Verisi: CNN'den çıkarılan test özellikleri ve etiketleri (test_features.npy, test_labels.npy).
### 2.3 Değerlendirme Metrikleri
Modellerin performansı, test veri seti üzerinde doğruluk (Accuracy) metrikleri kullanılarak değerlendirilmiştir. Hibrit model için ayrıca scikit-learn kütüphanesinin classification_report fonksiyonu ile Precision, Recall ve F1-Score gibi ek metrikler sınıf bazında hesaplanmıştır.

## 3. Sonuçlar (Results)
Dört farklı modelin çalıştırılması sonucunda elde edilen test doğruluğu değerleri aşağıda sunulmuştur:

Tablo 1: Modellerin Test Doğruluğu Karşılaştırması

Model	Veri Seti	Mimari Yaklaşımı	Test Doğruluğu
Model 1	MNIST	Özel Basit CNN	97.82%
Model 2	MNIST	Özel Basit CNN (Dropout ile)	98.58%
Model 3	CIFAR-10	ResNet-18 (ImageNet Pretrained, Fine-tuned)	80.46%
Model 4 (Hibrit)	CIFAR-10	ResNet-18 Özellik Çıkarıcı + SVC	91.08%*

* Hibrit model çıktısında SVC sınıflandırıcısı için elde edilen doğruluktur.

Karmaşıklık Matrisi (Model 4 Hibrit - SVC)

Hibrit modelin (SVC) test sonuçlarına ait sınıflandırma raporu (çıktılardan alınmıştır):

              precision    recall  f1-score   support

    airplane       0.89      0.89      0.89       1000
  automobile       0.95      0.94      0.95       1000
        bird       0.89      0.88      0.89       1000
         cat       0.80      0.81      0.80       1000
        deer       0.90      0.90      0.90       1000
         dog       0.84      0.83      0.84       1000
        frog       0.95      0.94      0.95       1000
       horse       0.94      0.95      0.95       1000
        ship       0.95      0.95      0.95       1000
       truck       0.96      0.97      0.96       1000

    accuracy                           0.91       10000
   macro avg       0.91      0.91      0.91       10000
weighted avg       0.91      0.91      0.91       10000

## 4. Analiz ve Tartışma (Analysis and Discussion)
Elde edilen sonuçlar, farklı veri setlerinin zorluğu ve mimari seçimlerin model performansı üzerindeki etkisini belirgin bir şekilde ortaya koymaktadır.

MNIST Modelleri (Model 1 ve Model 2): Her iki özel CNN modeli de MNIST veri setinde oldukça yüksek doğruluk elde etmiştir (sırasıyla %97.82 ve %98.58). MNIST, basit gri skala ve iyi ayrılmış sınıflara sahip olduğu için nispeten kolay bir veri setidir. Model 2'nin Dropout katmanı eklemesi, Model 1'e kıyasla hafif bir performans artışı sağlamıştır (%0.76). Bu, Dropout'un küçük ölçekli modellerde bile aşırı uydurmayı azaltarak veya farklı özellik kombinasyonlarını öğrenerek performansı iyileştirebileceğini göstermektedir. Özel ve basit CNN'ler bile MNIST gibi nispeten basit görevler için yeterince güçlü olabilir.

CIFAR-10 Modeli (Model 3 - ResNet-18 Fine-tuned): CIFAR-10 veri setinde ImageNet üzerinde önceden eğitilmiş ResNet-18 modelinin fine-tuning sonrası elde ettiği doğruluk %80.46'dır. Bu doğruluk, MNIST modellerinden belirgin şekilde düşüktür. Bu durum, CIFAR-10'un daha karmaşık (renkli, daha az belirgin özellikler, daha fazla arka plan çeşitliliği) bir veri seti olmasından kaynaklanmaktadır. ResNet-18 gibi güçlü bir mimarinin ve transfer öğrenmenin kullanılmasına rağmen, tek epoch fine-tuning ve muhtemelen tüm katmanların dondurulmaması (çıktılardan anlaşıldığı üzere) bu doğruluk seviyesini açıklamaktadır. Daha fazla epoch fine-tuning ve öğrenme oranı ayarı ile bu doğruluk artırılabilir.

Hibrit Model (Model 4 - ResNet-18 Özellik Çıkarıcı + SVC): En dikkat çekici sonuç hibrit modelden gelmiştir. ResNet-18'in (ImageNet pretrained) özellik çıkarıcı katmanlarından elde edilen vektörler üzerinde eğitilen basit bir SVC modelinin test doğruluğu %91.08 gibi oldukça yüksek bir seviyeye ulaşmıştır. Bu sonuç, End-to-End CNN olarak kullanılan Model 3'ten (%80.46) çok daha yüksektir. Bu durumun birkaç nedeni olabilir:

SVC'nin Etkinliği: SVM gibi geleneksel modeller, yüksek boyutlu ve iyi ayrılmış özellik vektörleri üzerinde güçlü sınıflandırma yeteneğine sahip olabilir. CNN'in ImageNet'ten öğrendiği özellikler, CIFAR-10 sınıflarını ayırt etmek için SVC'ye çok iyi bir girdi sağlamıştır.
CNN'in Sabit Özellikleri: Hibrit modelde, özellik çıkarıcı CNN'in ağırlıkları sabittir. Bu, eğitim sırasında sadece SVC'nin eğitildiği anlamına gelir. Model 3'te ise tüm ResNet fine-tune edilmiştir. Tek epoch fine-tuning, tüm model için yeterli olmamış olabilir veya küçük veri setinde tüm modeli fine-tune etmek (özellikle çok fazla epoch yapılırsa) aşırı uydurmaya daha yatkın olabilir. Hibrit yaklaşımda sadece ML modeli eğitildiği için bu risk azalır.
SVC'nin Farklı Karar Sınırları: SVC, CNN'in doğrusal sınıflandırıcılarının aksine doğrusal olmayan (kernel trick ile) karar sınırları oluşturarak özellik uzayında daha karmaşık ayrımlar yapabilmiş olabilir.
Hibrit modelin sınıflandırma raporuna bakıldığında, farklı sınıflar için Precision, Recall ve F1-Score değerlerinin genellikle dengeli ve yüksek olduğu görülmektedir. "cat" sınıfı diğerlerine göre biraz daha düşük performans göstermiştir (Precision: 0.80, Recall: 0.81), bu da kedilerin diğer nesnelerle karıştırılmaya daha yatkın olduğunu düşündürebilir.

Modellerin Karşılaştırması:

Veri Seti Zorluğu: MNIST modellerinin yüksek başarısı, veri setinin basitliğini; CIFAR-10 modellerinin nispeten daha düşük başarısı (özellikle End-to-End CNN'de), veri setinin daha karmaşık olduğunu göstermektedir.
Özel CNN vs Hazır CNN/Transfer Learning: MNIST gibi basit veri setlerinde özel, basit CNN'ler yeterli olabilirken, CIFAR-10 gibi karmaşık veri setlerinde hazır, büyük mimarilerin (ResNet) ve transfer öğrenmenin kullanılması genellikle gereklidir (Model 3).
End-to-End CNN vs Hibrit: CIFAR-10 özelinde, tek epoch fine-tuning yapılmış End-to-End ResNet-18'e kıyasla, aynı mimarinin özellik çıkarıcı olarak kullanılıp geleneksel bir SVC ile birleştirildiği hibrit model belirgin şekilde daha iyi performans göstermiştir (%91.08 vs %80.46). Bu durum, özellikle sınırlı fine-tuning yapıldığında veya veri seti hibrit yaklaşım için çıkarılan özellikleri iyi bir şekilde temsil ettiğinde, geleneksel sınıflandırıcıların güçlü bir seçenek olabileceğini ortaya koymaktadır. Hibrit yaklaşımın başarısı, CNN'in ImageNet'ten öğrendiği güçlü özelliklerin kalitesine ve SVC'nin bu özellikleri etkin şekilde kullanabilme yeteneğine dayanmaktadır.
## 5. Sonuç (Conclusion)
Bu çalışma, görüntü sınıflandırma problemlerine yaklaşımda farklı mimari ve stratejilerin (özel CNN, transfer öğrenme, hibrit yaklaşım) performans üzerindeki etkilerini araştırmıştır. MNIST gibi basit veri setlerinde özel ve nispeten küçük CNN'lerin bile yüksek doğruluk elde edebileceği görülmüştür. Daha karmaşık CIFAR-10 veri setinde ise, ImageNet üzerinde önceden eğitilmiş büyük bir mimarinin (ResNet-18) kullanılması gerekliliği ortaya çıkmıştır.

En dikkat çekici bulgu, CIFAR-10 veri setinde End-to-End fine-tuned ResNet-18'e kıyasla, aynı mimarinin özellik çıkarıcı olarak kullanıldığı ve elde edilen özellikler üzerinde bir SVC'nin eğitildiği hibrit modelin belirgin derecede daha yüksek doğruluk (%91.08 vs %80.46) elde etmesidir. Bu sonuç, özellikle transfer öğrenme senaryolarında fine-tuning sürecinin sınırlı tutulduğu durumlarda veya çıkarılan özelliklerin geleneksel ML modelleri tarafından etkin bir şekilde kullanılabildiği senaryolarda, hibrit yaklaşımların güçlü bir alternatif olabileceğini göstermektedir. Hibrit modelin başarısı, ImageNet'ten öğrenilen güçlü genel özellikleri alıp, bu sabit özellikler üzerinde CIFAR-10'a özgü karmaşık karar sınırlarını çizebilen SVC'nin gücüne bağlanabilir.

Çalışma, probleme ve veri setinin özelliklerine göre en uygun model seçiminin ve eğitim stratejisinin (sıfırdan eğitim, fine-tuning, hibrit) kritik önem taşıdığını vurgulamaktadır. CIFAR-10 özelinde, bu çalışma kapsamında, hibrit (ResNet-18 Özellikleri + SVC) model en yüksek performansı sergilemiştir. Gelecekteki çalışmalarda, farklı geleneksel ML modelleri denenerek veya End-to-End CNN modeli için daha kapsamlı fine-tuning (daha fazla epoch, öğrenme oranı çizelgeleri) yapılarak karşılaştırma detaylandırılabilir.

