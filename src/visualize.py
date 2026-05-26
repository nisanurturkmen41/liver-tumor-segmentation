"""
visualize.py
------------
CT goruntusu ve segmentasyon mask gorsellestirme modulu.

Bu modul:
1. Tek slice CT goruntusunu gri tonda goster
2. Mask'i ayri panelde discrete renklerle goster
3. CT uzerine semi-transparent mask overlay
4. Histogram comparison (preprocessing oncesi/sonrasi)
5. Otomatik "best slice" secimi (en cok bilgi iceren slice)

Tum cikti gorseller PNG olarak kaydedilebilir veya VS Code/Jupyter'da
direkt gosterilebilir.

Kullanim:
    from src.visualize import visualize_patient, plot_histograms
    
    # Bir hasta icin tam gorsellestirme
    visualize_patient(image, mask, save_path="liver_0_overlay.png")
    
    # Preprocessing histogramlari
    plot_histograms(raw_image, windowed_image, normalized_image)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def find_best_slice(mask: np.ndarray) -> int:
    """
    Bir volume'da en cok bilgi iceren axial slice'i bul.
    
    Strateji:
    1. Once tumor (label=2) en cok olan slice'a bak
    2. Tumor hic yoksa, karaciger (label=1) en cok olan slice'a bak
    3. Hic foreground yoksa, ortadaki slice'i dondur
    
    Args:
        mask: 3D segmentasyon maski (D, H, W)
    
    Returns:
        En bilgilendirici slice indeksi (z ekseninde)
    """
    # Her slice'taki tumor voxel sayisini hesapla
    tumor_per_slice = (mask == 2).sum(axis=(1, 2))
    
    if tumor_per_slice.max() > 0:
        # Tumor var, en cok tumorlu slice'i sec
        return int(tumor_per_slice.argmax())
    
    # Tumor yok, karaciger'e bak
    liver_per_slice = (mask == 1).sum(axis=(1, 2))
    
    if liver_per_slice.max() > 0:
        return int(liver_per_slice.argmax())
    
    # Hicbir foreground yok, ortayi dondur
    return mask.shape[0] // 2


def visualize_patient(
    image: np.ndarray,
    mask: np.ndarray,
    slice_idx: Optional[int] = None,
    patient_id: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Bir hastanin uc panelli gorsellestirmesi:
    1. CT slice (gri tonlama)
    2. Mask (discrete renkler: siyah=BG, yesil=liver, kirmizi=tumor)
    3. Overlay (CT uzerinde semi-transparent mask)
    
    Args:
        image: CT goruntusu (D, H, W) - normalize edilmis veya ham
        mask: Segmentasyon maski (D, H, W) {0, 1, 2}
        slice_idx: Gosterilecek slice; None ise otomatik bul
        patient_id: Baslikta gosterilecek hasta ID'si
        save_path: PNG olarak kaydetmek icin yol; None ise kaydetme
        show: Pencerede goster (Jupyter/VS Code icin True, batch icin False)
    
    Returns:
        matplotlib Figure objesi (show=False ise)
    """
    # Otomatik slice secimi
    if slice_idx is None:
        slice_idx = find_best_slice(mask)
    
    # Slice'lari cek
    img_slice = image[slice_idx]
    mask_slice = mask[slice_idx]
    
    # Sinif sayilarini hesapla (baslikta gostermek icin)
    n_liver = int((mask_slice == 1).sum())
    n_tumor = int((mask_slice == 2).sum())
    
    # 3 panelli figure olustur
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: CT slice
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title(f"CT Slice (z={slice_idx})")
    axes[0].axis("off")
    
    # Panel 2: Mask
    # Discrete colormap: siyah, yesil, kirmizi
    mask_cmap = ListedColormap(["black", "limegreen", "red"])
    axes[1].imshow(mask_slice, cmap=mask_cmap, vmin=0, vmax=2)
    axes[1].set_title(
        f"Mask (Liver={n_liver:,} px, Tumor={n_tumor:,} px)"
    )
    axes[1].axis("off")
    
    # Panel 3: Overlay
    # Once CT'yi gri arka plan olarak ciz
    axes[2].imshow(img_slice, cmap="gray")
    
    # Liver mask'i yesil overlay (sadece label=1 yerleri)
    liver_overlay = np.ma.masked_where(mask_slice != 1, mask_slice)
    axes[2].imshow(
        liver_overlay,
        cmap=ListedColormap(["limegreen"]),
        alpha=0.4,
        vmin=1, vmax=1
    )
    
    # Tumor mask'i kirmizi overlay (sadece label=2 yerleri)
    tumor_overlay = np.ma.masked_where(mask_slice != 2, mask_slice)
    axes[2].imshow(
        tumor_overlay,
        cmap=ListedColormap(["red"]),
        alpha=0.6,
        vmin=2, vmax=2
    )
    
    axes[2].set_title("Overlay (green=liver, red=tumor)")
    axes[2].axis("off")
    
    # Genel baslik
    title = f"Patient liver_{patient_id}" if patient_id else "Patient visualization"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Kaydet (istenirse)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gorsel kaydedildi: {save_path}")
    
    # Goster (istenirse)
    if show:
        plt.show()
        plt.close(fig)
        return None
    
    return fig


def plot_histograms(
    original: np.ndarray,
    windowed: np.ndarray,
    normalized: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Preprocessing asamalarinin histogramini karsilastir.
    Week 1 raporundaki Figure 1 ile birebir uyumlu.
    
    Args:
        original: Ham CT goruntusu (HU)
        windowed: HU windowing uygulanmis goruntu
        normalized: [0, 1] araliginda normalize edilmis goruntu
        save_path: PNG olarak kaydet
        show: Ekranda goster
    
    Returns:
        matplotlib Figure objesi
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Orijinal HU
    axes[0].hist(original.flatten(), bins=100, color="steelblue")
    axes[0].set_title("Original Histogram")
    axes[0].set_xlabel("HU value")
    axes[0].set_ylabel("Voxel count")
    
    # Panel 2: Windowed HU
    axes[1].hist(windowed.flatten(), bins=100, color="orange")
    axes[1].set_title("Windowed Histogram")
    axes[1].set_xlabel("HU value (clipped to [-150, 300])")
    axes[1].set_ylabel("Voxel count")
    
    # Panel 3: Normalize edilmis
    axes[2].hist(normalized.flatten(), bins=100, color="seagreen")
    axes[2].set_title("Normalized Histogram")
    axes[2].set_xlabel("Normalized value [0, 1]")
    axes[2].set_ylabel("Voxel count")
    
    fig.suptitle("Preprocessing Pipeline: HU Distribution", fontweight="bold")
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Histogram kaydedildi: {save_path}")
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    
    return fig


def plot_augmentation_comparison(
    original_slice: np.ndarray,
    augmented_slices: dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Bir slice icin augmentation efektlerini yan yana goster.
    Week 1 raporundaki Figure 4 ile uyumlu.
    
    Args:
        original_slice: 2D orijinal slice
        augmented_slices: {"isim": 2D_augmented_slice, ...} sozluk
            ornek: {"Intensity Shift": shifted, "Elastic": deformed}
        save_path: PNG yolu
        show: Ekranda goster
    """
    n_panels = 1 + len(augmented_slices)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    
    if n_panels == 1:
        axes = [axes]
    
    # Panel 1: Orijinal
    axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    # Diger paneller: augmentation'lar
    for i, (name, slice_data) in enumerate(augmented_slices.items(), start=1):
        axes[i].imshow(slice_data, cmap="gray")
        axes[i].set_title(name)
        axes[i].axis("off")
    
    fig.suptitle("Data Augmentation Comparison", fontweight="bold")
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Augmentation gorseli kaydedildi: {save_path}")
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    
    return fig


# Hizli test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZE MODULU - HIZLI TEST")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Yapay CT volume (normalize edilmis [0, 1])
    fake_image = np.random.uniform(0, 1, size=(50, 256, 256)).astype(np.float32)
    
    # Yapay mask
    fake_mask = np.zeros((50, 256, 256), dtype=np.uint8)
    fake_mask[20:35, 100:180, 80:180] = 1  # karaciger
    fake_mask[25:30, 130:160, 120:160] = 2  # tumor
    
    print("\nBest slice bulunuyor...")
    best = find_best_slice(fake_mask)
    print(f"En bilgilendirici slice: z={best}")
    
    print("\nGorsel olusturuluyor...")
    print("(NOT: Ekran yoksa, save_path verilebilir)")
    
    # Test save (ekran gosterilemiyorsa)
    # visualize_patient(
    #     fake_image, fake_mask,
    #     patient_id="test",
    #     save_path="test_output.png",
    #     show=False
    # )
    
    print("Test tamamlandi.")
    print("=" * 60)