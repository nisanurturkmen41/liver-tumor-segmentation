"""
calculate_metrics.py
--------------------
Preprocessing validation ve metrik analiz modulu.

Bu modul, ham (raw) ve islenmis (processed) volume'lari karsilastirarak
preprocessing pipeline'inin dogru calistigini sayisal olarak kanitlar.

3 ana metrik grubu:
1. Dynamic Range Enhancement: HU araliklari ve normalize sonuc
2. Physical Volume Preservation: mm^3 bazinda hacim koruma orani
3. Class Distribution: Sinif dengesizligi analizi

Week 1 raporundaki Figure 5'in (terminal output) ve quantitative results
bolumunun (Table) kaynak kodudur.

Kullanim:
    from src.calculate_metrics import generate_full_report
    
    report = generate_full_report(
        raw_image, raw_mask, raw_spacing,
        processed_image, processed_mask, processed_spacing=(1, 1, 1)
    )
    print_report(report)
"""
from __future__ import annotations
from typing import Tuple, Dict

import numpy as np


def calculate_dynamic_range(
    raw_image: np.ndarray,
    processed_image: np.ndarray
) -> Dict:
    """
    HU dynamic range enhancement metrigini hesapla.
    
    Args:
        raw_image: Ham CT (HU degerleri)
        processed_image: Islenmis CT ([0, 1] normalize)
    
    Returns:
        Dynamic range raporu
    """
    return {
        "original_hu_min": float(raw_image.min()),
        "original_hu_max": float(raw_image.max()),
        "original_hu_range": float(raw_image.max() - raw_image.min()),
        "processed_min": float(processed_image.min()),
        "processed_max": float(processed_image.max()),
        "is_normalized_to_unit": (
            processed_image.min() >= 0.0 and processed_image.max() <= 1.0
        ),
    }


def calculate_physical_volume(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    label: int
) -> float:
    """
    Belirli bir sinifin fiziksel hacmini mm^3 cinsinden hesapla.
    
    Formul: voxel_count * (sx * sy * sz)
    
    Args:
        mask: Segmentasyon maski
        spacing: Voxel boyutu (mm) - (z, y, x) veya (x, y, z) tutarli ise OK
        label: Hangi sinif (1=liver, 2=tumor)
    
    Returns:
        Fiziksel hacim (mm^3)
    """
    voxel_count = int((mask == label).sum())
    voxel_volume_mm3 = float(np.prod(spacing))
    return voxel_count * voxel_volume_mm3


def calculate_volume_preservation(
    raw_mask: np.ndarray,
    raw_spacing: Tuple[float, float, float],
    processed_mask: np.ndarray,
    processed_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict:
    """
    Resampling sonrasi fiziksel hacim koruma oranini hesapla.
    
    Her sinif (liver ve tumor) icin:
    preservation_ratio = processed_volume_mm3 / raw_volume_mm3
    
    Ideal deger: 1.0 (yani %100). Pratikte 0.95-1.05 arasi normaldir
    (cubic vs nearest interpolation arasi kucuk farklar).
    
    Args:
        raw_mask: Resampling oncesi mask
        raw_spacing: Resampling oncesi voxel boyutu (mm)
        processed_mask: Resampling sonrasi mask
        processed_spacing: Hedef spacing (default 1x1x1 mm)
    
    Returns:
        Her sinif icin orijinal hacim, yeni hacim ve oran
    """
    report = {}
    
    for label, name in [(1, "liver"), (2, "tumor")]:
        raw_vol = calculate_physical_volume(raw_mask, raw_spacing, label)
        proc_vol = calculate_physical_volume(
            processed_mask, processed_spacing, label
        )
        
        # Korumama orani: ideal 1.0
        if raw_vol > 0:
            ratio = proc_vol / raw_vol
        else:
            # Orijinalde bu sinif yoksa, islenmiste de olmamalı
            ratio = 1.0 if proc_vol == 0 else 0.0
        
        report[f"{name}_original_mm3"] = raw_vol
        report[f"{name}_processed_mm3"] = proc_vol
        report[f"{name}_preservation_ratio"] = ratio
        report[f"{name}_preservation_pct"] = ratio * 100.0
    
    return report


def calculate_class_distribution(mask: np.ndarray) -> Dict:
    """
    Mask'taki sinif dagilimini hesapla.
    
    Cikti: her sinif icin voxel sayisi ve toplam icindeki yuzde.
    
    Bu metrik EXTREME class imbalance'i belgelemek icin onemli:
    - Background: ~%97
    - Liver: ~%2-3
    - Tumor: ~%0.01-0.1  <-- bu kadar az!
    
    Bu sayede neden ozel loss fonksiyonu (Dice + CE) ve
    foreground-biased sampling kullandigimiz mantikli aciklanir.
    
    Args:
        mask: Segmentasyon maski
    
    Returns:
        Sinif dagilim raporu
    """
    total_voxels = int(mask.size)
    
    bg_count = int((mask == 0).sum())
    liver_count = int((mask == 1).sum())
    tumor_count = int((mask == 2).sum())
    
    return {
        "total_voxels": total_voxels,
        "background_voxels": bg_count,
        "liver_voxels": liver_count,
        "tumor_voxels": tumor_count,
        "background_pct": 100.0 * bg_count / total_voxels,
        "liver_pct": 100.0 * liver_count / total_voxels,
        "tumor_pct": 100.0 * tumor_count / total_voxels,
        # Imbalance ratio: background / tumor
        # Yuksek deger = ciddi imbalance
        "background_to_tumor_ratio": (
            bg_count / tumor_count if tumor_count > 0 else float("inf")
        ),
    }


def validate_mask_integrity(
    raw_mask: np.ndarray,
    processed_mask: np.ndarray
) -> Dict:
    """
    Mask'in resampling/processing sonrasi bozulmadigini kontrol et.
    
    Kontroller:
    - Ayni etiketler korundu mu? {0, 1, 2}?
    - Hicbir sinif tamamen kayboldu mu?
    
    Args:
        raw_mask: Orijinal mask
        processed_mask: Islenmis mask
    
    Returns:
        Validation raporu
    """
    raw_labels = set(np.unique(raw_mask).tolist())
    proc_labels = set(np.unique(processed_mask).tolist())
    
    return {
        "raw_labels": sorted(raw_labels),
        "processed_labels": sorted(proc_labels),
        "labels_match": raw_labels == proc_labels,
        "valid_label_set": proc_labels.issubset({0, 1, 2}),
        "tumor_preserved": (
            (2 in raw_labels) == (2 in proc_labels)
        ),
        "liver_preserved": (
            (1 in raw_labels) == (1 in proc_labels)
        ),
    }


def generate_full_report(
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    raw_spacing: Tuple[float, float, float],
    processed_image: np.ndarray,
    processed_mask: np.ndarray,
    processed_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    patient_id: str = ""
) -> Dict:
    """
    Bir hasta icin tam preprocessing validation raporu.
    
    Tum metric grouplarini birlestirip tek bir sozlukte sunar.
    
    Args:
        raw_image: Ham CT (D, H, W) HU
        raw_mask: Ham mask (D, H, W) {0, 1, 2}
        raw_spacing: Orijinal voxel boyutu (mm)
        processed_image: Islenmis CT (D, H, W) normalize
        processed_mask: Islenmis mask
        processed_spacing: Hedef spacing (default 1x1x1)
        patient_id: Hasta ID (raporda gostermek icin)
    
    Returns:
        Tam validation raporu
    """
    return {
        "patient_id": patient_id,
        "raw_shape": tuple(raw_image.shape),
        "processed_shape": tuple(processed_image.shape),
        "raw_spacing": raw_spacing,
        "processed_spacing": processed_spacing,
        "dynamic_range": calculate_dynamic_range(raw_image, processed_image),
        "volume_preservation": calculate_volume_preservation(
            raw_mask, raw_spacing, processed_mask, processed_spacing
        ),
        "class_distribution": calculate_class_distribution(processed_mask),
        "mask_integrity": validate_mask_integrity(raw_mask, processed_mask),
    }


def print_report(report: Dict) -> None:
    """
    Validation raporunu terminal-friendly formatta yazdir.
    Week 1 raporundaki Figure 5 ile birebir uyumlu.
    """
    pid = report.get("patient_id", "?")
    
    print("=" * 60)
    print(f"GANG TEAM: PREPROCESSING METRICS ANALYSIS (liver_{pid})")
    print("=" * 60)
    
    # Shape ve spacing degisikligi
    print(f"\nSHAPE & SPACING:")
    print(f"  Raw shape:        {report['raw_shape']}")
    print(f"  Processed shape:  {report['processed_shape']}")
    print(f"  Raw spacing:      {report['raw_spacing']} mm")
    print(f"  Processed spacing: {report['processed_spacing']} mm")
    
    # Dynamic range
    dr = report["dynamic_range"]
    print(f"\n[1] DYNAMIC RANGE (HU) ENHANCEMENT:")
    print(f"  -> Original HU Range:   "
          f"{dr['original_hu_min']:.1f} / {dr['original_hu_max']:.1f}")
    print(f"  -> Processed Value Range: "
          f"{dr['processed_min']:.4f} / {dr['processed_max']:.4f} (Normalized)")
    
    # Volume preservation
    vp = report["volume_preservation"]
    print(f"\n[2] PHYSICAL VOLUME PRESERVATION (Mask Integrity):")
    print(f"  -> Original Liver Volume:  {vp['liver_original_mm3']:>15,.2f} mm^3")
    print(f"  -> Processed Liver Volume: {vp['liver_processed_mm3']:>15,.2f} mm^3")
    print(f"  => LIVER PRESERVATION RATIO: {vp['liver_preservation_pct']:.2f}%")
    print(f"")
    print(f"  -> Original Tumor Volume:  {vp['tumor_original_mm3']:>15,.2f} mm^3")
    print(f"  -> Processed Tumor Volume: {vp['tumor_processed_mm3']:>15,.2f} mm^3")
    print(f"  => TUMOR PRESERVATION RATIO: {vp['tumor_preservation_pct']:.2f}%")
    
    # Class distribution
    cd = report["class_distribution"]
    print(f"\n[3] CLASS DISTRIBUTION (Class Imbalance):")
    print(f"  -> Background Ratio: {cd['background_pct']:.2f}%")
    print(f"  -> Liver Ratio:      {cd['liver_pct']:.2f}%")
    print(f"  -> Tumor Ratio:      {cd['tumor_pct']:.4f}%")
    print(f"  -> BG:Tumor ratio:   {cd['background_to_tumor_ratio']:,.0f}:1")
    
    # Mask integrity
    mi = report["mask_integrity"]
    status = "PASS" if (mi["labels_match"] and mi["valid_label_set"]) else "FAIL"
    print(f"\n[4] MASK INTEGRITY: [{status}]")
    print(f"  -> Raw labels:       {mi['raw_labels']}")
    print(f"  -> Processed labels: {mi['processed_labels']}")
    print(f"  -> Liver preserved:  {mi['liver_preserved']}")
    print(f"  -> Tumor preserved:  {mi['tumor_preserved']}")
    
    print("=" * 60)


def summarize_dataset_metrics(reports: list) -> Dict:
    """
    Coklu hasta raporlarini ozetle (mean, std).
    
    Tum dataset icin overall preprocessing quality'sini gosterir.
    
    Args:
        reports: generate_full_report ciktilari listesi
    
    Returns:
        Dataset-level istatistikler
    """
    if not reports:
        return {}
    
    liver_ratios = [
        r["volume_preservation"]["liver_preservation_ratio"]
        for r in reports
    ]
    tumor_ratios = [
        r["volume_preservation"]["tumor_preservation_ratio"]
        for r in reports
        if r["volume_preservation"]["tumor_original_mm3"] > 0
    ]
    
    bg_pcts = [r["class_distribution"]["background_pct"] for r in reports]
    liver_pcts = [r["class_distribution"]["liver_pct"] for r in reports]
    tumor_pcts = [r["class_distribution"]["tumor_pct"] for r in reports]
    
    n_passed = sum(
        1 for r in reports
        if r["mask_integrity"]["labels_match"]
        and r["mask_integrity"]["valid_label_set"]
    )
    
    return {
        "n_patients": len(reports),
        "n_passed_integrity": n_passed,
        "liver_preservation_mean": float(np.mean(liver_ratios)),
        "liver_preservation_std": float(np.std(liver_ratios)),
        "tumor_preservation_mean": float(np.mean(tumor_ratios)) if tumor_ratios else None,
        "tumor_preservation_std": float(np.std(tumor_ratios)) if tumor_ratios else None,
        "n_with_tumor": len(tumor_ratios),
        "background_pct_mean": float(np.mean(bg_pcts)),
        "liver_pct_mean": float(np.mean(liver_pcts)),
        "tumor_pct_mean": float(np.mean(tumor_pcts)),
    }


# Hizli test/demo
if __name__ == "__main__":
    print("=" * 60)
    print("CALCULATE_METRICS MODULU - HIZLI TEST")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Yapay ham veri
    raw_image = np.random.uniform(-1024, 1400, size=(75, 512, 512)).astype(np.float32)
    raw_mask = np.zeros((75, 512, 512), dtype=np.uint8)
    raw_mask[20:40, 200:350, 200:350] = 1  # karaciger
    raw_mask[28:32, 250:280, 250:280] = 2  # tumor
    raw_spacing = (5.0, 0.7, 0.7)
    
    # Yapay islenmis veri (resample edilmis gibi)
    proc_image = np.random.uniform(0, 1, size=(375, 358, 358)).astype(np.float32)
    proc_mask = np.zeros((375, 358, 358), dtype=np.uint8)
    proc_mask[100:200, 140:245, 140:245] = 1
    proc_mask[140:160, 175:196, 175:196] = 2
    proc_spacing = (1.0, 1.0, 1.0)
    
    report = generate_full_report(
        raw_image, raw_mask, raw_spacing,
        proc_image, proc_mask, proc_spacing,
        patient_id="test"
    )
    
    print_report(report)