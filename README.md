# Stage 1 — Liver Localization (3D U-Net)

Bu paket, Volumetric Liver Tumor Segmentation projesinin **Week 2 — Stage 1 (Liver Localization)** kısmıdır. Week 1 preprocessing pipeline'ınızdan çıkan processed NIfTI dosyaları üzerinde çalışır.

## Klasör Yapısı

```
liver_project/
├── requirements.txt
├── kaggle_stage1.ipynb          # Kaggle starter notebook
├── README.md                    # Bu dosya
└── src/
    ├── cv_split.py              # 5-fold patient-level split
    ├── dataset.py               # Patch-based 3D dataset (FG-biased sampling)
    ├── losses.py                # Dice+BCE / Dice+CE hybrid
    ├── metrics.py               # DSC, IoU accumulator
    ├── inference.py             # Sliding-window full-volume inference
    ├── train_stage1.py          # Training loop (AMP + checkpoint)
    └── models/
        └── unet3d.py            # Custom 3D U-Net (InstanceNorm+LeakyReLU)
```

## Önemli Tasarım Kararları

| Karar | Sebep |
|---|---|
| 96×96×96 patch | P100 16 GB'da batch=2 ile rahat sığar; 64³'ten daha geniş bağlam |
| Foreground-biased sampling (%66) | Tumor %0.01, random patch %99 boş çıkar |
| InstanceNorm + LeakyReLU(0.01) | BatchNorm batch=1-2'de patlar; LeakyReLU dying ReLU'yu önler |
| Dice + BCE/CE hybrid loss | Pure Dice küçük FG'de unstable, pure CE arka planda boğulur |
| Patient-level 5-fold | Slice-level split = data leakage |
| Sliding-window inference (gaussian blend) | Tüm volume eğitim sırasında değil val'de kullanılır, kenar artifact yok |
| AMP (FP16) | VRAM yarı yarıya azalır, P100'de hız ~1.6x |
| Her epoch checkpoint | Kaggle 12h timeout'una karşı sigorta |

## Lokalde Hızlı Test

```bash
pip install -r requirements.txt

# Önce: Week 1 preprocessing'iniz tamamlanmış olmalı
# liver_<id>_processed.nii.gz ve liver_<id>_label_processed.nii.gz dosyaları olmalı

# 1) 5-fold split oluştur
python -m src.cv_split --processed_dir /path/to/processed --out folds.json

# 2) Stage 1 training, fold 0
python -m src.train_stage1 \
    --processed_dir /path/to/processed \
    --folds_json folds.json \
    --fold 0 \
    --out_dir runs/stage1_fold0 \
    --epochs 40 \
    --batch_size 2
```

CPU'da çalışır ama çok yavaş — GPU şart.

## Kaggle Workflow

1. **Kaggle hesabı aç**, "Phone verify" yap (GPU için zorunlu)
2. **Notebook oluştur** → sağ panelde:
   - Accelerator: **GPU P100**
   - Internet: **On** (paket için, sonra kapatabilirsiniz)
   - Persistence: Files only
3. **Sağ panelden veri ekle**:
   - "+ Add Data" → "MSD Task03 Liver" ara, ekle
   - (kodu Kaggle Dataset olarak yüklediyseniz onu da ekle)
4. **kaggle_stage1.ipynb** dosyasının içeriğini yapıştır
5. Hücreleri sırayla çalıştır
6. Önemli: training sırasında "Save Version → Save & Run All" deyip pencereyi kapatabilirsiniz; Kaggle arka planda 12 saate kadar çalıştırır.

## Beklenen Sonuçlar (Stage 1, Fold 0)

| Epoch | Val DSC (liver) |
|---|---|
| 5 | ~0.85 |
| 15 | ~0.92 |
| 30 | ~0.95 |
| 40 (best) | ~0.95-0.96 |

40 epoch P100'de ~8-10 saat. Hedef: liver DSC ≥ 0.94.

## Sıradakiler

- [ ] Stage 1'i en az 1 fold çalıştır, DSC ≥ 0.93 olduğunu doğrula
- [ ] **Stage 2 dataset modunu aç**: `LiverPatchDataset(stage='stage2', ...)` — kod hazır
- [ ] **train_stage2.py yaz** — train_stage1.py'nin neredeyse aynısı, sadece:
  - `out_channels=3`, `DiceCELoss(n_classes=3, include_background_in_dice=False)`
  - Stage 1 modelinin tahminini girdiye eklemek (cascade input) — opsiyonel, basit cascade için sadece liver bbox crop yeter
- [ ] **Post-processing**: `scipy.ndimage.label` ile CCA, küçük tumor component'leri filtrele
- [ ] **Final evaluation script**: 5 fold'un ortalaması + std

## GitHub Workflow Hızlı Hatırlatma

```bash
# Lokal değişiklik sonrası:
git add .
git commit -m "anlamli bir mesaj"
git push

# Kaggle notebook başında:
!cd /kaggle/working/code && git pull
```

**ASLA commit etmeyin**: `*.nii.gz`, `*.pt`, `processed/`, `runs/`, `outputs/`. `.gitignore` bunları engellemeli.

## Bilinen Tuzaklar

- **`scipy.ndimage.zoom` mask interpolation**: nearest-neighbor (order=0) şart, yoksa label {0,1,2}'yi bozar (Week 1'de zaten doğru yapılmış)
- **Patient ID parsing**: `liver_<id>_processed.nii.gz` formatına bağlı. Sizin output'unuz farklı isimleniyorsa `cv_split.py` ve `dataset.py` içindeki regex'i değiştirin
- **Kaggle internet**: training sırasında kapatın, deterministik olur ve session crash riski azalır
- **Patch RAM**: `cache_volumes=True` yapmayın — 40 GB > 13 GB Kaggle RAM
