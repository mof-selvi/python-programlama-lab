# Deep Learning Examples for Python Programming Practice Course

# Python Programlama Uygulama Dersi için Derin Öğrenme Örnekleri

* Örnekler *.ipynb formatında verilmiştir.
* Çalıştırmadan da kodların çıktıları incelenebilir.
* Örnekleri açabilmek için:
  * Google Colab veya Kaggle gibi bir platforma kodları yükleyebilirsiniz.
  * Bilgisayarınızda Anaconda Navigator uygulamasını çalıştırıp Jupyter Notebook veya Jupyter Lab programlarını yükleyebilirsiniz.
  * VS Code içerisinden Jupyter eklentilerini kurarak direkt VS Code'da görüntüleyebilirsiniz.
* Örnekleri bilgisayarınızda çalıştırabilmek için:
  * Anaconda üzerinde yeni bir "environment" oluşturunuz.
  * Environment içerisinde şu kütüphanelerin yüklü olduğundan emin olunuz:
    * ipykernel
    * ipywidgets
    * matplotlib
    * numpy
    * keras
    * tensorflow
    * skimage
  * GPU'nuz varsa daha hızlı eğitim aşaması için şunları yüklemelisiniz:
    * tensorflow-gpu (Tensorflow'un son sürümlerinde sadece tensorflow'un yüklenmesi yeterli olmaktadır.)
    * cudatoolkit
    * cudnn
    * GPU üretinizin resmi sitesinden sinir ağı kütüphaneleri ve sürücüler
  * Gerekli sürücüleri yükledikten sonra GPU'nuzun Tensorflow tarafından görünüp görünmediğini şu kod ile test edebilirsiniz:
  ```
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
  ```
* Not: GPU'nuzun algılanmaması durumunda Tensorflow kütüphanelerini Anaconda yerine pip üzerinden conda environment'ınıza yüklemeniz tavsiye edilir.
