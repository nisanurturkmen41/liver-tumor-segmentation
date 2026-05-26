"""
preprocess.py
-------------
HU Windowing ve Min-Max Normalization modulu.

Bu modul:
1. CT goruntusundeki Hounsfield Unit (HU) degerlerini kirpar (windowing)
   - Default: [-150, 300] (abdominal yumusak doku window'u)
   - Bu aralik karaciger ve tumor icin standart klinik degerdir
2. Min-Max normalization ile pikselleri [0, 1] araligina getirir
3. Mask'e dokunmaz - mask zaten {0, 1, 2} olarak ayrik

Neden bu HU araligi?
- Hava (-1000), kemik (+1000) ilgisiz, kontrasti bozar
- Karaciger parankimi ~30-70 HU
- Tumor lezyonlari ~20-50 HU (hipodens) veya 80+ HU (kontrast tutarsa)
- [-150, 300] araligi tum bu yumusak dokuyu rahat kapsar

Kullanim:
    from src.preprocess import preprocess_image, preprocess_pipeline
    
    # Sadece image'i isle
    normalized = preprocess_image(raw_image, hu_min=-150, hu_max=300)
    
    # Tam pipeline (image + mask birlikte, integrity kontrolu ile)
    norm_img, mask, info = preprocess_pipeline(raw_image, raw_mask)
"""
from __future__ import annotations
from typing import Tuple, Dict

import numpy as np


def apply_hu_windowing(
    image: np.ndarray,
    hu_min: float = -150.0,
    hu_max: float = 300.0
) -> np.ndarray:
    """
    CT goruntusune HU windowing uygula.
    
    np.clip ile [hu_min, hu_max] disindaki tum degerleri sinirlara cek.
    
    Args:
        image: Ham CT goruntusu (D, H, W) - HU degerleri
        hu_min: Alt sinir (default -150, yumusak doku icin)
        hu_max: Ust sinir (default 300, kontrastli doku icin)
    
    Returns:
        Windowed image, ayni shape, ayni dtype
    """
    # np.clip: x < hu_min ise hu_min'e, x > hu_max ise hu_max'a cek
    # Aradakileri degistirme
    windowed = np.clip(image, hu_min, hu_max)
    return windowed


def min_max_normalize(
    image: np.ndarray,
    src_min: float = -150.0,
    src_max: float = 300.0,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Min-Max normalization: [src_min, src_max] -> [0.0, 1.0]
    
    Formul: normalized = (x - src_min) / (src_max - src_min)
    
    Neden FIXED src_min/src_max kullaniyoruz (image.min()/image.max() degil)?
    - Tum hastalarda AYNI olcek - karsilastirilabilir
    - Eger her hasta icin kendi min/max kullanirsak, dusuk-kontrast tarama
      ile yuksek-kontrast tarama farkli olceklendirilir, model bozulur
    - Klinik anlam korunur: 0.5 her zaman ~75 HU'ya karsilik gelir
    
    Args:
        image: Windowed CT goruntusu (D, H, W)
        src_min: Kaynak aralik alt sinir (default -150)
        src_max: Kaynak aralik ust sinir (default 300)
        dtype: Cikti tipi (default float32, neural network icin standart)
    
    Returns:
        Normalize edilmis goruntu, [0.0, 1.0] araliginda
    """
    # Once float'a cevir, sonra normalize
    image = image.astype(dtype)
    normalized = (image - src_min) / (src_max - src_min)
    
    # Guvenlik: clipping sonrasi sayisal hata olabilir, [0,1]'e kilitle
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized


def preprocess_image(
    image: np.ndarray,
    hu_min: float = -150.0,
    hu_max: float = 300.0
) -> np.ndarray:
    """
    Tam image preprocessing: Windowing + Normalization
    
    Bu fonksiyon iki adimi birlestirir, en yaygin kullanim noktasi.
    
    Args:
        image: Ham CT (D, H, W) HU degerleri
        hu_min, hu_max: Windowing araligi
    
    Returns:
        [0, 1] araliginda normalize edilmis float32 goruntu
    """
    windowed = apply_hu_windowing(image, hu_min, hu_max)
    normalized = min_max_normalize(windowed, hu_min, hu_max)
    return normalized


def preprocess_pipeline(
    image: np.ndarray,
    mask: np.ndarray,
    hu_min: float = -150.0,
    hu_max: float = 300.0
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Image ve mask icin tam preprocessing pipeline'i.
    
    Bu fonksiyon hem isler hem de detayli rapor uretir.
    
    Args:
        image: Ham CT goruntusu (D, H, W) HU degerleri
        mask: Ham segmentasyon maski (D, H, W) {0, 1, 2}
        hu_min, hu_max: Windowing araligi
    
    Returns:
        norm_image: [0, 1] araliginda normalize edilmis goruntu
        mask: Orijinal mask (degistirilmez, sadece kopyalanir)
        info: Preprocessing istatistikleri sozlugu
    """
    # Orijinal istatistikleri kaydet (rapor icin)
    original_stats = {
        "hu_min": float(image.min()),
        "hu_max": float(image.max()),
        "hu_mean": float(image.mean()),
        "shape": tuple(image.shape),
    }
    
    # Adim 1: HU windowing
    windowed = apply_hu_windowing(image, hu_min, hu_max)
    
    windowed_stats = {
        "hu_min": float(windowed.min()),
        "hu_max": float(windowed.max()),
        "hu_mean": float(windowed.mean()),
    }
    
    # Adim 2: Normalization
    normalized = min_max_normalize(windowed, hu_min, hu_max)
    
    normalized_stats = {
        "norm_min": float(normalized.min()),
        "norm_max": float(normalized.max()),
        "norm_mean": float(normalized.mean()),
    }
    
    # Mask'i kopyala (originale dokunma)
    mask_out = mask.copy()
    
    # Kapsamli rapor olustur
    info = {
        "window_range": [hu_min, hu_max],
        "original": original_stats,
        "windowed": windowed_stats,
        "normalized": normalized_stats,
        "mask_labels_preserved": (
            set(np.unique(mask).tolist()) == set(np.unique(mask_out).tolist())
        ),
        "mask_label_counts": {
            int(label): int((mask_out == label).sum())
            for label in np.unique(mask_out)
        },
    }
    
    return normalized, mask_out, info


# Hizli test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESS MODULU - HIZLI TEST")
    print("=" * 60)
    
    # Yapay test verisi olustur (gercek CT'ye benzeyen)
    np.random.seed(42)
    fake_image = np.random.uniform(-1024, 1500, size=(50, 256, 256)).astype(np.float32)
    fake_mask = np.zeros((50, 256, 256), dtype=np.uint8)
    fake_mask[20:30, 100:150, 100:150] = 1  # yapay karaciger
    fake_mask[24:26, 120:130, 120:130] = 2  # yapay tumor
    
    print(f"\nGirdi sekli: {fake_image.shape}")
    print(f"Girdi HU araligi: [{fake_image.min():.1f}, {fake_image.max():.1f}]")
    print(f"Mask unique: {np.unique(fake_mask).tolist()}")
    
    norm_img, mask, info = preprocess_pipeline(fake_image, fake_mask)
    
    print(f"\n--- PIPELINE TAMAMLANDI ---")
    print(f"Cikti sekli: {norm_img.shape}")
    print(f"Cikti dtype: {norm_img.dtype}")
    print(f"Normalize araligi: [{norm_img.min():.4f}, {norm_img.max():.4f}]")
    print(f"Normalize ortalamasi: {norm_img.mean():.4f}")
    print(f"Mask korundu mu: {info['mask_labels_preserved']}")
    print(f"Mask sinif sayilari: {info['mask_label_counts']}")
    print("=" * 60)