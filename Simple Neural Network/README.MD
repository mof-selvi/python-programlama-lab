## Tensorflow, Pytorch vb. herhangi bir hazır "Neural Network" kütüphanesi kullanılmadan Python dilinde yazılmış basit bir yapay sinir ağı kodu

### *.py ve *.ipynb dosyaları aynı kodları içermektedir. Notebook dosyası (ipynb) hücre çıktılarını da barındırdığı için paylaşılmıştır. Kodları çalıştırmadan incelemek isteyenler Jupyter Notebook kurulumu yaparak veya VS Code kullanarak ipynb uzantılı dosyayı açabilirler.

### Kod iki adet ağ içermektedir:
* Network1: Çıkışta aktivasyon fonksiyonu yok, Loss fonksiyonu: Mean Squared Error (MSE)
* Network2: Çıkışta Sigmoid aktivasyonu var, Loss fonksiyonu: Binary Cross Entropy (BCE)

### Her iki ağ da bir gizli katmana (hidden layer) bir de çıkış katmanına (output layer) sahiptir.

### Not: Kod içerisinde, fonksiyonların türevlerine ek olarak Autodiff alan bir fonksiyon da bulunmaktadır. Fakat bu fonksiyon BCE için kullanıldığında "Exploding Gradient" problemine sebep olmaktadır.
