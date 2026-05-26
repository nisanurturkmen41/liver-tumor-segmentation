"""
resample.py
-----------
Isotropic resampling modulu.

CT taramalari anizotropik voxel'lere sahiptir:
- x-y: ~0.7 mm (yuksek cozunurluk)
- z:   ~5.0 mm (dusuk cozunurluk, slice arasi)

Bu modul tum volume'lari 1x1x1 mm kup voxel'lere yeniden orneklestirir.

Neden 1x1x1 mm?
- 3D U-Net uniform voxel bekler (asimetrik kernel'i onler)
- Tumor volume hesabi kolay: 1 voxel = 1 mm^3
- Cok kucuk lezyonlar bile korunur
- Memory ile fidelity arasinda iyi denge

KRITIK: Image cubic interpolation, mask nearest-neighbor interpolation kullanir.
Aksi takdirde mask'te {0, 0.7, 1.3, 2.1} gibi anlamsiz degerler olusur.

Kullanim:
    from src.resample import resample_to_isotropic
    
    new_img, new_mask, new_spacing = resample_to_isotropic(
        image, mask, original_spacing=(0.7, 0.7, 5.0)
    )
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def calculate_zoom_factors(
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[float, float, float]:
    """
    Orijinal ve hedef spacing'den zoom faktorlerini hesapla.
    
    Formul: zoom_factor = original_spacing / target_spacing
    
    Ornek: original=(0.7, 0.7, 5.0), target=(1.0, 1.0, 1.0)
        zoom = (0.7/1.0, 0.7/1.0, 5.0/1.0) = (0.7, 0.7, 5.0)
    
    Yorumlama:
    - x ekseni: 0.7x kucult (zaten ince, az kucult)
    - y ekseni: 0.7x kucult
    - z ekseni: 5.0x BUYUT (5 mm slice'i 1 mm yapmak icin 5 kati slice gerek)
    
    Args:
        original_spacing: NIfTI header'dan gelen (x, y, z) mm degerleri
        target_spacing: Hedef isotropic spacing, default (1, 1, 1) mm
    
    Returns:
        Zoom faktorleri (zx, zy, zz)
    """
    factors = tuple(
        orig / target
        for orig, target in zip(original_spacing, target_spacing)
    )
    return factors


def resample_image(
    image: np.ndarray,
    zoom_factors: Tuple[float, float, float],
    order: int = 3
) -> np.ndarray:
    """
    Image'i scipy.ndimage.zoom ile yeniden orneklestir.
    
    Args:
        image: 3D CT goruntusu (D, H, W)
        zoom_factors: Her eksen icin olcek faktoru
        order: Interpolasyon derecesi
            0 = nearest neighbor (mask icin)
            1 = linear
            3 = cubic (image icin onerilen)
    
    Returns:
        Yeniden orneklenmis image
    """
    # mode='nearest': sinir disinda kalan voxel'leri en yakin ile doldur
    # prefilter=True: cubic icin smoothing on-filter uygula (default)
    resampled = zoom(
        image,
        zoom=zoom_factors,
        order=order,
        mode="nearest",
        prefilter=(order > 1)  # sadece cubic+ icin gerekli
    )
    return resampled


def resample_mask(
    mask: np.ndarray,
    zoom_factors: Tuple[float, float, float]
) -> np.ndarray:
    """
    Mask'i nearest-neighbor ile yeniden orneklestir.
    
    KRITIK: Cubic kullanmiyoruz!
    Mask ayrik degerler icerir: {0=arkaplan, 1=karaciger, 2=tumor}.
    Cubic interpolation 0.7, 1.3 gibi anlamsiz degerler uretir.
    Nearest-neighbor sadece var olan degerleri kopyalar.
    
    Args:
        mask: 3D segmentasyon maski (D, H, W) {0, 1, 2}
        zoom_factors: Image ile AYNI olmali!
    
    Returns:
        Yeniden orneklenmis mask, hala {0, 1, 2}
    """
    # order=0: nearest-neighbor, ayrik etiketleri korur
    # prefilter=False: nearest icin smoothing yapilmamali, etiket kayar
    resampled = zoom(
        mask,
        zoom=zoom_factors,
        order=0,
        mode="nearest",
        prefilter=False
    )
    
    # Guvenlik: cikti tipi degisebilir, uint8'e geri donduralim
    return resampled.astype(np.uint8)


def resample_to_isotropic(
    image: np.ndarray,
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Tam resampling pipeline: image + mask -> isotropic 1x1x1 mm.
    
    Args:
        image: Ham veya windowed CT (D, H, W)
        mask: Segmentasyon maski (D, H, W) {0, 1, 2}
        original_spacing: NIfTI header'dan gelen voxel spacing (mm)
        target_spacing: Hedef spacing, default (1, 1, 1) mm
    
    Returns:
        new_image: Resampled CT (yeni shape)
        new_mask: Resampled mask (yeni shape, hala {0, 1, 2})
        new_spacing: Aslinda target_spacing, dokumantasyon icin doner
    
    Raises:
        ValueError: Image ve mask farkli shape'lere sahipse
    """
    # On kosul kontrolu
    if image.shape != mask.shape:
        raise ValueError(
            f"Image ve mask farkli shape'lere sahip!\n"
            f"  Image: {image.shape}\n"
            f"  Mask:  {mask.shape}"
        )
    
    # Zoom faktorlerini hesapla
    factors = calculate_zoom_factors(original_spacing, target_spacing)
    
    # Image: cubic interpolation
    new_image = resample_image(image, factors, order=3)
    
    # Mask: nearest-neighbor interpolation
    new_mask = resample_mask(mask, factors)
    
    # Cikti boyutlari ufak farkli olabilir (scipy yuvarlamasi)
    # Eger farkliysa, kucuk olana kirp (genelde 1-2 voxel farki)
    if new_image.shape != new_mask.shape:
        min_shape = tuple(
            min(i, m) for i, m in zip(new_image.shape, new_mask.shape)
        )
        new_image = new_image[
            :min_shape[0], :min_shape[1], :min_shape[2]
        ]
        new_mask = new_mask[
            :min_shape[0], :min_shape[1], :min_shape[2]
        ]
    
    return new_image, new_mask, target_spacing


def validate_resampling(
    original_mask: np.ndarray,
    resampled_mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> dict:
    """
    Resampling sonrasi mask butunlugunu dogrula.
    
    Fiziksel hacim korundu mu kontrol et:
    original_volume_mm3 = voxel_count * (sx * sy * sz)
    resampled_volume_mm3 = new_voxel_count * (1 * 1 * 1)
    Bunlarin oranı ~1.0 olmali (kucuk fark interpolasyon nedeniyle normal)
    
    Args:
        original_mask: Resampling oncesi mask
        resampled_mask: Resampling sonrasi mask
        original_spacing: Orijinal voxel boyutu (mm)
        target_spacing: Yeni voxel boyutu (mm)
    
    Returns:
        Validation raporu sozlugu
    """
    # Voxel hacmi (mm^3)
    orig_voxel_vol = np.prod(original_spacing)
    target_voxel_vol = np.prod(target_spacing)
    
    report = {
        "labels_preserved": (
            set(np.unique(original_mask).tolist())
            == set(np.unique(resampled_mask).tolist())
        ),
        "labels_in_resampled": sorted(np.unique(resampled_mask).tolist()),
    }
    
    # Her sinif icin fiziksel hacim koruma orani
    for label in [1, 2]:  # karaciger ve tumor
        orig_voxels = int((original_mask == label).sum())
        new_voxels = int((resampled_mask == label).sum())
        
        orig_vol_mm3 = orig_voxels * orig_voxel_vol
        new_vol_mm3 = new_voxels * target_voxel_vol
        
        # Hacim koruma orani: yeni/orijinal, ideal 1.0
        if orig_vol_mm3 > 0:
            preservation_ratio = new_vol_mm3 / orig_vol_mm3
        else:
            preservation_ratio = 1.0 if new_vol_mm3 == 0 else 0.0
        
        label_name = "liver" if label == 1 else "tumor"
        report[f"{label_name}_original_mm3"] = float(orig_vol_mm3)
        report[f"{label_name}_resampled_mm3"] = float(new_vol_mm3)
        report[f"{label_name}_preservation_ratio"] = float(preservation_ratio)
    
    return report


# Hizli test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("RESAMPLE MODULU - HIZLI TEST")
    print("=" * 60)
    
    # Yapay anizotropik veri olustur (gercek CT spacing'i taklit et)
    np.random.seed(42)
    
    # Tipik MSD volume: (75 slice, 512, 512) (z, y, x)
    fake_image = np.random.uniform(-150, 300, size=(75, 512, 512)).astype(np.float32)
    fake_mask = np.zeros((75, 512, 512), dtype=np.uint8)
    
    # Yapay karaciger (z=20-40, merkezi bolge)
    fake_mask[20:40, 200:350, 200:350] = 1
    # Yapay tumor (karaciger icinde kucuk lezyon)
    fake_mask[28:32, 250:280, 250:280] = 2
    
    original_spacing = (5.0, 0.7, 0.7)  # (z, y, x) - tipik MSD
    target_spacing = (1.0, 1.0, 1.0)
    
    print(f"\nOrijinal:")
    print(f"  Image shape: {fake_image.shape}")
    print(f"  Spacing: {original_spacing} mm")
    print(f"  Liver voxels: {(fake_mask == 1).sum():,}")
    print(f"  Tumor voxels: {(fake_mask == 2).sum():,}")
    
    print(f"\nResampling baslatildi...")
    new_img, new_mask, _ = resample_to_isotropic(
        fake_image, fake_mask,
        original_spacing=original_spacing,
        target_spacing=target_spacing
    )
    
    print(f"\nResampled:")
    print(f"  Image shape: {new_img.shape}")
    print(f"  Spacing: {target_spacing} mm")
    print(f"  Liver voxels: {(new_mask == 1).sum():,}")
    print(f"  Tumor voxels: {(new_mask == 2).sum():,}")
    print(f"  Mask unique: {np.unique(new_mask).tolist()}")
    
    print(f"\nValidation:")
    report = validate_resampling(
        fake_mask, new_mask,
        original_spacing, target_spacing
    )
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60)