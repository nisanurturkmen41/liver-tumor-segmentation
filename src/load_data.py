"""
load_data.py
------------
NIfTI veri yukleme ve veri butunlugu kontrol modulu.

Bu modul:
1. Task03_Liver klasorundeki tum hastalari tarar
2. Image-label esleskeni dogrular
3. Voxel spacing, sekil, HU araligi gibi metadata cikarir
4. Mask label'larinin {0, 1, 2} oldugunu kontrol eder
5. Tum dataset icin istatistiksel rapor uretir

Kullanim:
    from src.load_data import load_dataset_info, load_patient
    
    # Tum dataset bilgisi
    df = load_dataset_info("/path/to/Task03_Liver")
    
    # Tek bir hasta yukle
    image, mask, meta = load_patient("/path/to/Task03_Liver", patient_id="0")
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List
import re

import numpy as np
import nibabel as nib


def list_patients(data_dir: Path) -> List[str]:
    """
    imagesTr klasorundeki tum hasta ID'lerini listele.
    
    Hidden files (. ile baslayanlar) filtrelenir - OneDrive sync metadata'si
    NIfTI olarak yorumlanmaya calisilirsa hata verir.
    
    Returns:
        ["0", "1", "2", ...] gibi sirali hasta ID listesi
    """
    images_dir = data_dir / "imagesTr"
    
    if not images_dir.exists():
        raise FileNotFoundError(
            f"imagesTr klasoru bulunamadi: {images_dir}\n"
            f"Lutfen Task03_Liver klasorunun dogru yapida oldugunu kontrol edin."
        )
    
    # liver_<id>.nii.gz formatindaki dosyalari yakala
    pattern = re.compile(r"^liver_(\d+)\.nii\.gz$")
    
    patient_ids = []
    for file_path in images_dir.iterdir():
        # Gizli dosyalari (._liver_0.nii.gz gibi) atla
        if file_path.name.startswith("."):
            continue
        
        match = pattern.match(file_path.name)
        if match:
            patient_ids.append(match.group(1))
    
    # Sayisal sirala (string sirasi 1, 10, 11, 2 verir; biz 1, 2, 10, 11 isteriz)
    patient_ids.sort(key=lambda x: int(x))
    
    return patient_ids


def load_patient(
    data_dir: Path | str,
    patient_id: str
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Tek bir hastanin CT goruntusunu ve maskini yukle.
    
    Args:
        data_dir: Task03_Liver klasoru yolu
        patient_id: Hasta ID'si ("0", "1", vb.)
    
    Returns:
        image: (D, H, W) seklinde float32 numpy array - CT goruntusu
        mask:  (D, H, W) seklinde uint8 numpy array  - segmentasyon maski
        meta:  Metadata sozlugu (shape, spacing, HU range, vb.)
    """
    data_dir = Path(data_dir)
    
    img_path = data_dir / "imagesTr" / f"liver_{patient_id}.nii.gz"
    lbl_path = data_dir / "labelsTr" / f"liver_{patient_id}.nii.gz"
    
    # Dosya varligi kontrolu
    if not img_path.exists():
        raise FileNotFoundError(f"Goruntu dosyasi bulunamadi: {img_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Mask dosyasi bulunamadi: {lbl_path}")
    
    # NIfTI dosyalarini yukle
    img_nifti = nib.load(str(img_path))
    lbl_nifti = nib.load(str(lbl_path))
    
    # Pixel verisini numpy array'e cevir
    # CT degerleri (Hounsfield Unit) float olarak gelir
    image = img_nifti.get_fdata().astype(np.float32)
    # Mask degerleri tam sayi (0, 1, 2)
    mask = lbl_nifti.get_fdata().astype(np.uint8)
    
    # Metadata cikar
    # NIfTI header'inda voxel spacing (mm cinsinden fiziksel boyut) saklanir
    img_spacing = img_nifti.header.get_zooms()[:3]
    
    meta = {
        "patient_id": patient_id,
        "shape": tuple(image.shape),
        "spacing_mm": tuple(float(s) for s in img_spacing),
        "hu_min": float(image.min()),
        "hu_max": float(image.max()),
        "hu_mean": float(image.mean()),
        "unique_labels": sorted(np.unique(mask).tolist()),
        "image_path": str(img_path),
        "label_path": str(lbl_path),
    }
    
    return image, mask, meta


def validate_patient(image: np.ndarray, mask: np.ndarray, meta: Dict) -> Dict:
    """
    Bir hasta verisinin butunlugunu kontrol et.
    
    Kontroller:
    - Image ve mask ayni sekle sahip mi?
    - Mask sadece {0, 1, 2} degerleri iceriyor mu?
    - Mask icinde liver (1) ve tumor (2) voxel'leri var mi?
    
    Returns:
        validation: Kontrol sonuclarini iceren sozluk
    """
    validation = {
        "patient_id": meta["patient_id"],
        "shape_match": image.shape == mask.shape,
        "valid_labels": set(meta["unique_labels"]).issubset({0, 1, 2}),
        "has_liver": 1 in meta["unique_labels"],
        "has_tumor": 2 in meta["unique_labels"],
        "liver_voxels": int((mask == 1).sum()),
        "tumor_voxels": int((mask == 2).sum()),
        "background_voxels": int((mask == 0).sum()),
    }
    
    # Genel basari durumu
    validation["passed"] = (
        validation["shape_match"]
        and validation["valid_labels"]
        and validation["has_liver"]  # her hastada en az liver olmali
    )
    
    return validation


def load_dataset_info(data_dir: Path | str, verbose: bool = True) -> List[Dict]:
    """
    Tum dataset hakkinda istatistiksel rapor olustur.
    
    Her hastayi tek tek yukler (yavas ama tam tarama), validation yapar
    ve sonuclari liste olarak doner.
    
    Args:
        data_dir: Task03_Liver klasoru
        verbose: True ise her hasta icin satir basina rapor yazdir
    
    Returns:
        Tum hastalarin metadata + validation bilgilerini iceren liste
    """
    data_dir = Path(data_dir)
    patient_ids = list_patients(data_dir)
    
    if verbose:
        print(f"=" * 70)
        print(f"DATASET TARAMA RAPORU")
        print(f"=" * 70)
        print(f"Veri klasoru: {data_dir}")
        print(f"Bulunan hasta sayisi: {len(patient_ids)}")
        print(f"-" * 70)
    
    results = []
    for i, pid in enumerate(patient_ids):
        try:
            image, mask, meta = load_patient(data_dir, pid)
            validation = validate_patient(image, mask, meta)
            
            # Metadata + validation birlestir
            combined = {**meta, **validation}
            results.append(combined)
            
            if verbose:
                status = "OK" if validation["passed"] else "FAIL"
                print(
                    f"[{i+1:3d}/{len(patient_ids)}] liver_{pid}: "
                    f"shape={meta['shape']}, "
                    f"spacing={tuple(round(s, 2) for s in meta['spacing_mm'])}, "
                    f"liver={validation['liver_voxels']:,}, "
                    f"tumor={validation['tumor_voxels']:,} "
                    f"[{status}]"
                )
        except Exception as e:
            if verbose:
                print(f"[{i+1:3d}/{len(patient_ids)}] liver_{pid}: HATA - {e}")
    
    if verbose:
        print(f"-" * 70)
        passed = sum(1 for r in results if r.get("passed", False))
        print(f"Basarili yuklenen: {passed}/{len(patient_ids)}")
        print(f"=" * 70)
    
    return results


# Bu dosya direkt calistirilirsa hizli bir test yap
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Kullanim: python -m src.load_data <Task03_Liver klasor yolu>")
        print("Ornek:  python -m src.load_data C:/Users/nisan/Desktop/Task03_Liver")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    results = load_dataset_info(data_dir, verbose=True)
    print(f"\nToplam {len(results)} hasta yuklendi.")