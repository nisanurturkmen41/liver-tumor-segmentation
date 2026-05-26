"""
augmentations.py
----------------
Egitim sirasinda veri cesitliligini artirmak icin augmentation modulu.

Bu modul iki augmentation uygular:
1. Random Intensity Shift: Pixellere kucuk rastgele offset ekler
   -> Farkli CT cihazlarinin kalibrasyon farkini simule eder
2. Elastic Deformation: Goruntuyu yumusakca esnetir/buker
   -> Hastalar arasi anatomik cesitliligi simule eder

Augmentation egitim sirasinda HER batch'te rastgele uygulanir.
Sonuc: ayni 131 hasta, ama her epoch'ta yeni varyasyonlar gibi gorunur.

KRITIK KURAL:
- Intensity shift: SADECE image'a uygula, mask'a DOKUNMA
- Spatial transforms (elastic): Hem image hem mask AYNI sekilde, ama
  mask icin nearest-neighbor interpolation (kategorik etiketleri korur)

Kullanim:
    from src.augmentations import random_intensity_shift, elastic_deformation
    
    # Sadece image
    aug_img = random_intensity_shift(image, shift_range=0.05)
    
    # Hem image hem mask (spatial transform)
    aug_img, aug_mask = elastic_deformation(image, mask, alpha=15, sigma=3)
"""
from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def random_intensity_shift(
    image: np.ndarray,
    shift_range: float = 0.05,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Image'a kucuk rastgele intensity offset ekle.
    
    Simule edilen: Farkli CT cihazlarinin kalibrasyon variyasyonu.
    Bir hastanin scan'i Phillips cihazinda 75 HU okunurken,
    GE cihazinda 78 HU okunabilir. Network buna robust olmali.
    
    Args:
        image: Normalize edilmis goruntu [0, 1] (D, H, W)
        shift_range: Maksimum offset buyuklugu (default 0.05 = +/-%5)
        rng: numpy random generator (deterministik test icin)
    
    Returns:
        Augmented image (yine yaklasik [0, 1] araliginda)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # [-shift_range, +shift_range] araliginda rastgele bir offset sec
    shift = rng.uniform(-shift_range, shift_range)
    
    # Tum pixellere AYNI offset'i ekle (uniform shift)
    shifted = image + shift
    
    # [0, 1] dışına tasanlari kirpa (güvenlik)
    shifted = np.clip(shifted, 0.0, 1.0)
    
    return shifted.astype(image.dtype)


def random_intensity_scale(
    image: np.ndarray,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Image'i rastgele bir faktorle carpip kontrasti degistir.
    
    Simule edilen: Farkli kontrast madde dozaji.
    
    Args:
        image: Normalize edilmis goruntu [0, 1]
        scale_range: (min, max) carpan araligi
        rng: numpy random generator
    
    Returns:
        Augmented image
    """
    if rng is None:
        rng = np.random.default_rng()
    
    scale = rng.uniform(*scale_range)
    scaled = image * scale
    scaled = np.clip(scaled, 0.0, 1.0)
    
    return scaled.astype(image.dtype)


def elastic_deformation(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha: float = 15.0,
    sigma: float = 3.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Image (ve varsa mask) icin elastic deformation uygula.
    
    Algoritma:
    1. Her pixel icin rastgele displacement vector olustur (rastgele kayma)
    2. Bu vector field'i Gaussian filter ile yumusatip "elastik" yap
    3. alpha ile siddetini olcekle
    4. map_coordinates ile her pixeli yeni konumuna esle
    
    Parametre yorumu:
    - alpha: Deformation siddetı (yuksek = daha cok bukulme)
    - sigma: Yumusakligi (yuksek = daha pürüzsüz deformation, dusuk = sert)
    - Tipik degerler: alpha=15, sigma=3 (orta seviye)
    
    KRITIK: Mask varsa, AYNI displacement field ile ama nearest-neighbor
    interpolation ile donustur. Aksi takdirde mask etiketleri bozulur.
    
    Args:
        image: 3D goruntu (D, H, W)
        mask: 3D mask (D, H, W) {0, 1, 2}, optional
        alpha: Deformation siddetı
        sigma: Yumusakligi
        rng: numpy random generator
    
    Returns:
        deformed_image, deformed_mask (mask None ise sadece image)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    shape = image.shape
    
    # Adim 1: Rastgele displacement field olustur (her eksen icin)
    # rng.uniform(-1, 1, ...) -> [-1, +1] araliginda rastgele degerler
    dz = rng.uniform(-1, 1, size=shape).astype(np.float32)
    dy = rng.uniform(-1, 1, size=shape).astype(np.float32)
    dx = rng.uniform(-1, 1, size=shape).astype(np.float32)
    
    # Adim 2: Gaussian filter ile yumusat (elastic davranis)
    dz = gaussian_filter(dz, sigma=sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(dy, sigma=sigma, mode="constant", cval=0) * alpha
    dx = gaussian_filter(dx, sigma=sigma, mode="constant", cval=0) * alpha
    
    # Adim 3: Yeni koordinat haritasi olustur
    # Orijinal grid + displacement = yeni grid
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij"
    )
    
    new_coords = np.stack([
        (z_coords + dz).ravel(),
        (y_coords + dy).ravel(),
        (x_coords + dx).ravel()
    ])
    
    # Adim 4: Image'a uygula (linear interpolation, yumusak)
    deformed_image = map_coordinates(
        image,
        new_coords,
        order=1,  # linear: hizli ve kabul edilebilir kalite
        mode="reflect"  # sinir disi: aynanin yansimasi
    ).reshape(shape).astype(image.dtype)
    
    # Mask varsa, AYNI displacement ama nearest-neighbor
    if mask is not None:
        deformed_mask = map_coordinates(
            mask,
            new_coords,
            order=0,  # nearest-neighbor: kategorik etiket korunur
            mode="reflect"
        ).reshape(shape).astype(mask.dtype)
        return deformed_image, deformed_mask
    
    return deformed_image, None


def random_flip(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    axes: Tuple[int, ...] = (1, 2),
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rastgele eksen flip (yansitma).
    
    Tipik kullanim:
    - axes=(1, 2): Sadece y ve x ekseninde flip (anatomi z'de simetrik degil)
    - Sol-sag aynalama saglar
    
    Args:
        image: 3D goruntu
        mask: 3D mask (optional)
        axes: Flip uygulanabilecek eksenler
        p: Her eksen icin flip olasiligi
        rng: numpy random generator
    
    Returns:
        flipped_image, flipped_mask
    """
    if rng is None:
        rng = np.random.default_rng()
    
    out_img = image
    out_mask = mask
    
    for axis in axes:
        if rng.random() < p:
            out_img = np.flip(out_img, axis=axis)
            if out_mask is not None:
                out_mask = np.flip(out_mask, axis=axis)
    
    # np.flip "view" doner, contiguous yap (PyTorch icin gerekli)
    out_img = np.ascontiguousarray(out_img)
    if out_mask is not None:
        out_mask = np.ascontiguousarray(out_mask)
    
    return out_img, out_mask


def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    use_intensity_shift: bool = True,
    use_intensity_scale: bool = True,
    use_elastic: bool = True,
    use_flip: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tum augmentation'lari sirayla uygula.
    
    Bu fonksiyon training loop'unda her sample icin cagrilir.
    Her augmentation rastgele uygulanir (default'lari kullanilir).
    
    KRITIK SIRA: Once spatial (elastic, flip), sonra intensity.
    Cunku intensity sadece image'a uygulanir, spatial mask'a da.
    
    Args:
        image: Normalize edilmis goruntu [0, 1]
        mask: Segmentasyon maski {0, 1, 2}
        use_*: Hangi augmentation'larin aktif olacagi
        rng: numpy random generator
    
    Returns:
        aug_image, aug_mask
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Spatial transforms once (hem image hem mask)
    if use_elastic and rng.random() < 0.5:  # %50 olasilikla
        image, mask = elastic_deformation(image, mask, rng=rng)
    
    if use_flip:
        image, mask = random_flip(image, mask, rng=rng)
    
    # Intensity transforms (sadece image)
    if use_intensity_shift:
        image = random_intensity_shift(image, rng=rng)
    
    if use_intensity_scale and rng.random() < 0.5:
        image = random_intensity_scale(image, rng=rng)
    
    return image, mask


# Hizli test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("AUGMENTATIONS MODULU - HIZLI TEST")
    print("=" * 60)
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    # Yapay normalize edilmis goruntu
    fake_image = np.random.uniform(0.2, 0.8, size=(30, 128, 128)).astype(np.float32)
    fake_mask = np.zeros((30, 128, 128), dtype=np.uint8)
    fake_mask[10:20, 40:90, 40:90] = 1
    fake_mask[13:17, 55:75, 55:75] = 2
    
    print(f"\nOrijinal image araligi: [{fake_image.min():.3f}, {fake_image.max():.3f}]")
    print(f"Orijinal mask labels: {np.unique(fake_mask).tolist()}")
    print(f"Orijinal liver voxels: {(fake_mask == 1).sum()}")
    print(f"Orijinal tumor voxels: {(fake_mask == 2).sum()}")
    
    # Test 1: Intensity shift
    print("\n--- Intensity Shift ---")
    shifted = random_intensity_shift(fake_image, shift_range=0.05, rng=rng)
    print(f"Sonrasi: [{shifted.min():.3f}, {shifted.max():.3f}]")
    print(f"Ortalama fark: {(shifted - fake_image).mean():.4f}")
    
    # Test 2: Elastic deformation
    print("\n--- Elastic Deformation ---")
    deformed_img, deformed_mask = elastic_deformation(
        fake_image, fake_mask, alpha=15, sigma=3, rng=rng
    )
    print(f"Image shape: {deformed_img.shape}")
    print(f"Mask labels: {np.unique(deformed_mask).tolist()}")
    print(f"Liver voxels: {(deformed_mask == 1).sum()} (orijinal: {(fake_mask == 1).sum()})")
    print(f"Tumor voxels: {(deformed_mask == 2).sum()} (orijinal: {(fake_mask == 2).sum()})")
    print("(Kucuk fark elastic deformation'in sinir komsulugu nedeniyle normaldir)")
    
    # Test 3: Tam pipeline
    print("\n--- Full Pipeline ---")
    aug_img, aug_mask = apply_augmentations(fake_image, fake_mask, rng=rng)
    print(f"Final image araligi: [{aug_img.min():.3f}, {aug_img.max():.3f}]")
    print(f"Final mask labels: {np.unique(aug_mask).tolist()}")
    
    print("=" * 60)