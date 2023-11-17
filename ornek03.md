# Scikit-Learn kütüphanesi ile kümeleme/sınıflandırma çalışması

## Aşağıdaki kodu kullanarak gerekli kütüphaneleri import ediniz
```
from sklearn import datasets
from sklearn import cluster

import matplotlib.pyplot as plt
import numpy as np
```

## SKLearn, içerisinde yapay veri setleri barındırır:
```
datasets.make_circles()
```

* Yukarıdaki kod ile hem X (noktaların koordinatları) hem de Y (noktaların etiketleri) değerlerine sahip bir veri seti üretilir.
* Sınıflandırma (classification) işlemlerinde kullanılan veri setlerinde X ve Y varken, kümeleme (clustering) işlemlerinde kullanılan veri setleri sadece X içerir.
* Bu sebeple yukarıda oluşturulan [X, Y] formatındaki veri setinin sadece ilk elemanı (X) alınır.

```
dataset = datasets.make_circles()[0]
```

* Veri setimizin neye benzediğini görmek için çeşitli çizim araçları kullanırız. Bu veri setimiz noktaların koordinatlarından oluştuğu için "scatter" tipinde bir grafik çizdirmek işimize yarayacaktır.

```
plt.scatter(dataset[:,0], dataset[:,1])
```
* X içerisinde bulunan iki elemanlı dizilerin ilk değeri x, ikinci değeri y eksenine tekabül eder. (Dikkat: bu veri setinin büyük harflerle gösterilen X ve Y değerleri ile, verilerin koordinatlarını veren x ve y değerlerini karıştırmayınız.)

## KMeans

* KMeans algoritması bir kümeleme algoritmasıdır. Bununla eldeki veri setini kümeleyiniz:
```
algo = cluster.KMeans(n_clusters=3)
```
* n_clusters parametresini 3 belirleyerek, kümeleme işlemi sonucunda 3 farklı küme oluşmasını istediğimizi belirtmiş olduk.
* Not: Kümeleme algoritmalarında nihai küme sayısını, veriler arasındaki maksimum uzaklığı veya bunlar gibi başka bir parametreyi vererek kümeleme işleminin yapılmasını isteyebiliriz (hangi parametrelere göre işlemin gerçekleşeceği algoritmadan algoritmaya göre değişir).

* Yukarıdaki gibi hazırladığınız algoritmayı fit fonksiyonu ile kullanınız:
```
clustering = algo.fit(dataset)
```

* Makine öğrenmesi yöntemlerinde işlemi başlatan fonksiyon genellikle "fit()" fonksiyonudur

* "predict()" fonksiyonu ise fit edilmiş (eğitilmiş/hesaplanmış) modele göre yeni verilerin ait olabilecekleri kümenin/sınıfın belirlenmesi için kullanılır. (Bu verilerin setine "test seti" adı verilir. Bir veri setinin eğitim için kullanılan kısmına ise "eğitim seti" -train set- denir.)

## Sonuç

* Elde ettiğimiz kümeleme sonuçları bir nesne içerisinde bulunur. Bu nesnenin "labels_" elemanı, tüm verilerin tahmin edilmiş etiket verilerini sırayla tutar.

* Tekrar bir scatter tipinde grafik çizdirmek fakat bu sefer noktaların renklerini kümeye göre belirlemek için "c" parametresine etiketlerin bulunduğu listeyi verebiliriz.

* Aşağıdaki kodu kullanarak sonucu inceleyiniz.

```
plt.scatter(dataset[:,0], dataset[:,1], c=clustering.labels_)
```

## Ödev 1

* Bir başka kümeleme algoritmaları kullanarak aynı verileri kümeleyiniz ve sonuçları karşılaştırınız.
* SKLearn kütüphanesinde bulunan kümeleme algoritmalarını incelemek için: (https://scikit-learn.org/stable/modules/clustering.html)[https://scikit-learn.org/stable/modules/clustering.html]


## Ödev 2

* Aşağıdaki kod ile 3 çeşit türde çiçeklerin 4 boyutlu verilerini içeren "iris" veri setini bir değişkene atayınız.
```
from sklearn import datasets
iris = datasets.load_iris()
```

* "iris" adlı nesnenin data adlı dizisi 150 satır ve 4 sütundan oluşmaktadır:
```
print(iris.data.shape)
```

* Hangi verinin hangi sınıfa ait olduğu bilgisi "target" elemanında bulunur:
```
print(iris.target)
```

* iris.data verilerini X, iris.target verilerini Y olarak ele alıp bir sınıflandırma algoritmasında kullanınız. Bkz: (https://scikit-learn.org/stable/supervised_learning.html)[https://scikit-learn.org/stable/supervised_learning.html]

* Veri setinizi [ [X1,Y1] , [X2,Y2] ] şeklinde bir bütün haline getirip "shuffle" ediniz.

* Karıştırılmış dizinizin ilk %60'lık kısmını eğitim, kalan %40'lık kısmını test seti olarak ayarlayınız (train_set ve test_set adında farklı dizilere atayınız).

* Kullandığınız sınıflandırma işlemi bir "gözetimli öğrenme" (supervised learning) yöntemidir. Çünkü sonuçları eğitim esnasında vererek bir model geliştirmiş olursunuz. Eğitim aşamasında gösterilmemiş sonuçları ise eğitilmiş model üzerinde kullanarak modelinizin performansını ölçebilirsiniz.

* Kümeleme algoritmaları "gözetimsiz öğrenme" yöntemleridir. Çünkü model oluşturulurken elimizde, verilerin etiketleri/sınıfları/kümeleri yoktur. Uygun bir yöntem ile verileri bölümlere ayrıştırmış oluruz.
