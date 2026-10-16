# pytorch-mnist-mlp

Bu proje, **CNN kullanmadan** PyTorch ile MNIST üzerinde sıfırdan eğitilen bir **MLP** çözümüdür.
Ödev gereksinimlerine uygun olarak train/validation/test ayrımı, epoch bazlı metrik takibi, grafik üretimi, hiperparametre karşılaştırması ve kendi el yazısı görselleri ile tahmin akışı içerir.

## 1) Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Reproducibility ve cihaz seçimi

- Tohum (seed) parametresi varsayılan `42` olup `--seed` ile değiştirilebilir.
- Cihaz seçimi otomatik yapılır: CUDA varsa `cuda`, yoksa `cpu`.

## 3) Mimari

- Model: `mnist_mlp.model.MLPClassifier`
- Girdi: `28x28` görüntü (flatten: 784)
- Hidden katmanlar: varsayılan `[256, 128]` (**en az iki hidden layer**)
- Aktivasyon: `ReLU`
- Çıkış: `10` sınıf (`0..9`)

## 4) Veri bölme yöntemi

- `train=True` MNIST seti train+validation için kullanılır.
- Varsayılan `val_split=0.1` ile train/validation ayrımı yapılır.
- `train=False` MNIST seti yalnızca test için kullanılır.
- Test verisi eğitim/hiperparametre geliştirme sürecine dahil edilmez.

## 5) Eğitim (sıfırdan)

> Hazır eğitilmiş model kullanılmaz. Her çalıştırma sıfırdan eğitir.

```bash
python -m mnist_mlp.train \
  --data-dir ./data \
  --output-dir ./outputs/run \
  --epochs 5 \
  --batch-size 128 \
  --optimizer adam \
  --lr 0.001 \
  --hidden-sizes 256 128 \
  --val-split 0.1 \
  --seed 42 \
  --save-checkpoint
```

### Eğitim çıktıları

`outputs/run/` altında:
- `training_summary.json`
  - epoch bazlı `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
  - `training_time_seconds`
  - `test_accuracy` ve `test_loss`
- `loss_curve.png`
- `accuracy_curve.png`
- (opsiyonel) `checkpoint.pt`

Not: `checkpoint.pt`, bu çalıştırmada sıfırdan eğitilen modelin ağırlıklarıdır; önceden eğitilmiş model değildir.

## 6) Hiperparametre / yapılandırma karşılaştırması

Varsayılan 2 deney:
- `adam_baseline`
- `sgd_wider`

Çalıştırma:

```bash
python -m mnist_mlp.experiments --epochs 3 --experiments-output-dir ./outputs/experiments
```

Sonuçlar:
- `outputs/experiments/experiment_results.json`
- Her deney klasöründe eğitim özeti ve grafikler

İsterseniz JSON dosyası ile kendi deneylerinizi verin (`--experiments-json`).

## 7) Kendi el yazısı görselleri ile tahmin akışı

Gerçek kişiye ait örnek görseller bu repoya eklenmemiştir. Bunun yerine şablon klasör sağlanmıştır:

- `data/handwritten_digits/README.md`

Ön işleme akışı:
- gri tonlama (`L`)
- 28x28 yeniden boyutlandırma
- MNIST'e uyumlu şekilde gerekirse renk tersleme (açık zemin/koyu rakam)
- MNIST normalizasyonu (`mean=0.1307`, `std=0.3081`)

Tahmin komutu:

```bash
python -m mnist_mlp.handwriting \
  --checkpoint outputs/run/checkpoint.pt \
  data/handwritten_digits/0.png data/handwritten_digits/1.png
```

## 8) Distribution shift yorumu

MNIST temiz, ortalanmış ve standartlaştırılmış bir veri dağılımına sahiptir. Gerçek dünyadaki el yazısı görselleri ise kalem kalınlığı, eğiklik, arka plan gürültüsü ve kadraj açısından farklılaşır. Bu nedenle kendi el yazısı testlerinde düşüş beklenebilir; bu fark, train dağılımı ile gerçek dünya dağılımı arasındaki **distribution shift** göstergesidir.

## 9) Testler

Aşağıdaki kontroller otomatik test edilir:
- Modelin MLP yapısı (en az iki hidden layer + ReLU)
- Çıkış boyutu `10`
- El yazısı ön işleme çıktısının MNIST uyumlu tensor üretmesi
- Eğitim geçmişinin her epoch için train/validation metrikleri üretmesi

```bash
pytest -q
```
