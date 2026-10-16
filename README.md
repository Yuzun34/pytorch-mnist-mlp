# pytorch-mnist-mlp

MNIST veri seti ile **CNN kullanmadan**, PyTorch tabanlı bir **MLP (Multi-Layer Perceptron)** modeli eğitmek ve model yapılandırmalarını karşılaştırmak için hazırlanmıştır.

## Gereksinimler

```bash
pip install -r requirements.txt
```

## Deney 1 - Temel MLP Eğitimi

Varsayılan model:
- Giriş: `28x28=784`
- Hidden layers: `256 -> 128` (**en az 2 hidden layer**)
- Aktivasyon: `ReLU`
- Çıkış: `10 sınıf`
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` (opsiyonel `SGD`)

Eğitim komutu:

```bash
python train_mnist_mlp.py --epochs 10 --optimizer adam --learning-rate 0.001 --hidden-dims 256 128
```

Bu komut:
- MNIST'i indirir
- Train/Validation/Test ayrımını kurar
- Her epoch için train/validation loss ve accuracy hesaplar
- Test accuracy hesaplar
- Eğitim süresini ölçer
- Sonuçları `outputs/` klasörüne kaydeder

> Not: İnternet erişimi olmayan ortamlarda sadece kod akışını test etmek için
> `--use-fake-data` kullanılabilir. Ödev sonuçları için gerçek MNIST ile eğitim yapılmalıdır.

## Model Yapılandırması Karşılaştırma

Birden fazla MLP yapılandırması ile deney yapmak için:

```bash
python train_mnist_mlp.py --run-experiments --epochs 10
```

Bu modda örnek olarak şu konfigürasyonlar çalışır:
- `baseline`: `[256, 128]`
- `wide`: `[512, 256]`
- `small_lr`: `[256, 128]` + daha küçük öğrenme oranı

Karşılaştırma sonuçları:
- `outputs/experiment_summary.json`

## Üretilen Çıktılar

Her deney için (`outputs/<config_name>/`):
- `model.pt`: eğitilmiş model ağırlıkları
- `metrics.json`:
  - training loss
  - validation loss
  - training accuracy
  - validation accuracy
  - test accuracy
  - eğitim süresi
- `loss_accuracy.png`: eğitim sürecindeki loss/accuracy grafikleri

## Kendi El Yazısı Görüntüleri ile Test

1. Öğrenci numaranızı el yazınızla yazın (ayrı rakam görüntüleri önerilir).
2. Görselleri bir klasöre koyun (örn: `custom_digits/`).
3. Aşağıdaki komutu çalıştırın:

```bash
python predict_custom_digits.py \
  --model-path outputs/single_run/model.pt \
  --image-dir custom_digits \
  --hidden-dims 256 128
```

Ön işleme adımları:
- Grayscale'e çevirme
- Renkleri MNIST formatına yaklaştırmak için invert + autocontrast
- `28x28` yeniden boyutlandırma
- MNIST normalizasyonu

## Önemli Notlar

- Bu projede CNN yoktur; yalnızca MLP kullanılır.
- Hazır eğitilmiş model kullanılmaz, model sıfırdan eğitilir.
- Test verisi eğitim/iyileştirme döngüsüne dahil edilmez.
- Hiperparametreler komut satırı argümanlarıyla açıkça belirtilir.

## Teslim için Öneri

Repo içine şunları ekleyin:
- `custom_digits/` klasöründe öğrenci numaranızın el yazısı görselleri
- `outputs/` altındaki deney sonuçları (özellikle `metrics.json`, `loss_accuracy.png`)
- Kısa bir genel değerlendirme (README içine eklenebilir)
