"""
preprocess_pipeline.py
----------------------
Tum preprocessing modullerini ardisik calistiran ana orkestrator.

Bu modul, Task03_Liver klasorunden ham NIfTI dosyalarini alir,
6 adimli preprocessing pipeline'i uygular ve islenmis NIfTI dosyalarini
output klasorune kaydeder.

Pipeline akisi:
    1. load_data:        Ham NIfTI yukle, metadata cikar
    2. preprocess:       HU windowing [-150, 300] + min-max normalize [0, 1]
    3. resample:         Isotropic 1x1x1 mm yeniden ornekleme
    4. calculate_metrics: Validation raporu uret
    5. Save:             Islenmis volume'lari .nii.gz olarak kaydet
    6. Summary:          Tum dataset icin ozet rapor

Kullanim:
    # Komut satirindan (tum dataset)
    python -m src.preprocess_pipeline \\
        --input  C:/Users/nisan/Desktop/Task03_Liver \\
        --output C:/Users/nisan/Desktop/processed
    
    # Python'dan (tek hasta)
    from src.preprocess_pipeline import process_single_patient
    process_single_patient("0", input_dir, output_dir)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import nibabel as nib

# Bizim modullerimizi import et
from src.load_data import list_patients, load_patient, validate_patient
from src.preprocess import preprocess_pipeline
from src.resample import resample_to_isotropic
from src.calculate_metrics import (
    generate_full_report,
    print_report,
    summarize_dataset_metrics,
)


def save_processed_volume(
    array: np.ndarray,
    output_path: Path,
    spacing: tuple = (1.0, 1.0, 1.0),
    reference_nifti: Optional[nib.Nifti1Image] = None
) -> None:
    """
    Islenmis numpy array'i NIfTI olarak diske kaydet.
    
    Args:
        array: Kaydedilecek 3D array
        output_path: .nii.gz dosya yolu
        spacing: Voxel boyutu (mm)
        reference_nifti: Orijinal NIfTI (header bilgisi icin opsiyonel)
    """
    # Affine matrix olustur (voxel-to-world coordinate)
    # Basit yaklasim: spacing'i diagonal'a koy
    if reference_nifti is not None:
        # Orijinal orientation/origin'i kullan ama spacing'i guncelle
        affine = reference_nifti.affine.copy()
        # Spacing degisikligini affine'e yansit
        for i in range(3):
            norm = np.linalg.norm(affine[:3, i])
            if norm > 0:
                affine[:3, i] = affine[:3, i] / norm * spacing[i]
    else:
        # Sifirdan basit diagonal affine
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    
    nifti_img = nib.Nifti1Image(array, affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(output_path))


def process_single_patient(
    patient_id: str,
    input_dir: Path,
    output_dir: Path,
    hu_min: float = -150.0,
    hu_max: float = 300.0,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    save_outputs: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Tek bir hasta icin full preprocessing pipeline.
    
    Args:
        patient_id: Hasta ID ("0", "1", ...)
        input_dir:  Task03_Liver klasoru
        output_dir: Cikti klasoru
        hu_min, hu_max: Windowing araligi
        target_spacing: Hedef voxel boyutu (mm)
        save_outputs: True ise NIfTI olarak kaydet
        verbose: Detayli log yazdir
    
    Returns:
        Validation raporu sozlugu
    """
    if verbose:
        print(f"\n>>> Patient liver_{patient_id} isleniyor...")
    
    t0 = time.time()
    
    # ADIM 1: Yukleme
    image_raw, mask_raw, meta = load_patient(input_dir, patient_id)
    raw_spacing = meta["spacing_mm"]
    
    if verbose:
        print(f"    [1/4] Yuklendi: shape={image_raw.shape}, "
              f"spacing={tuple(round(s, 2) for s in raw_spacing)} mm")
    
    # ADIM 2: HU windowing + normalize
    image_norm, mask_copy, prep_info = preprocess_pipeline(
        image_raw, mask_raw, hu_min=hu_min, hu_max=hu_max
    )
    
    if verbose:
        print(f"    [2/4] Windowed [-{abs(hu_min)}, {hu_max}] + normalized [0, 1]")
    
    # ADIM 3: Isotropic resample
    image_proc, mask_proc, proc_spacing = resample_to_isotropic(
        image_norm, mask_copy,
        original_spacing=raw_spacing,
        target_spacing=target_spacing
    )
    
    if verbose:
        print(f"    [3/4] Resampled: shape={image_proc.shape}, "
              f"spacing={proc_spacing} mm")
    
    # ADIM 4: Validation raporu
    report = generate_full_report(
        raw_image=image_raw,
        raw_mask=mask_raw,
        raw_spacing=raw_spacing,
        processed_image=image_proc,
        processed_mask=mask_proc,
        processed_spacing=proc_spacing,
        patient_id=patient_id,
    )
    
    if verbose:
        liver_pct = report["volume_preservation"]["liver_preservation_pct"]
        tumor_pct = report["volume_preservation"]["tumor_preservation_pct"]
        print(f"    [4/4] Validation: liver={liver_pct:.2f}%, "
              f"tumor={tumor_pct:.2f}%")
    
    # ADIM 5: Kaydet
    if save_outputs:
        # Reference NIfTI'yi yeniden yukle (affine icin)
        ref_path = input_dir / "imagesTr" / f"liver_{patient_id}.nii.gz"
        ref_nifti = nib.load(str(ref_path))
        
        img_out_path = output_dir / f"liver_{patient_id}_processed.nii.gz"
        lbl_out_path = output_dir / f"liver_{patient_id}_label_processed.nii.gz"
        
        save_processed_volume(
            image_proc, img_out_path,
            spacing=proc_spacing, reference_nifti=ref_nifti
        )
        save_processed_volume(
            mask_proc, lbl_out_path,
            spacing=proc_spacing, reference_nifti=ref_nifti
        )
        
        if verbose:
            print(f"    [Kaydedildi] {img_out_path.name}, {lbl_out_path.name}")
    
    elapsed = time.time() - t0
    report["elapsed_seconds"] = elapsed
    
    if verbose:
        print(f"    Sure: {elapsed:.1f} saniye")
    
    return report


def process_all_patients(
    input_dir: Path | str,
    output_dir: Path | str,
    hu_min: float = -150.0,
    hu_max: float = 300.0,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    save_outputs: bool = True,
    verbose: bool = True,
    save_report: bool = True
) -> List[Dict]:
    """
    Tum dataset icin preprocessing pipeline.
    
    Her hastayi tek tek isler, hatalar yakalanir ve devam edilir.
    
    Args:
        input_dir: Task03_Liver klasoru
        output_dir: Cikti klasoru (otomatik olusturulur)
        hu_min, hu_max: Windowing araligi
        target_spacing: Hedef voxel boyutu
        save_outputs: Islenmis NIfTI dosyalarini diske yaz
        verbose: Detayli log
        save_report: JSON formatta rapor kaydet
    
    Returns:
        Her hasta icin rapor listesi
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hastalari listele
    patient_ids = list_patients(input_dir)
    print(f"\n{'=' * 70}")
    print(f"PREPROCESSING PIPELINE")
    print(f"{'=' * 70}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Patients: {len(patient_ids)}")
    print(f"Windowing: [{hu_min}, {hu_max}] HU")
    print(f"Target spacing: {target_spacing} mm")
    print(f"{'=' * 70}")
    
    reports = []
    failed = []
    t_start = time.time()
    
    for i, pid in enumerate(patient_ids):
        try:
            print(f"\n[{i+1}/{len(patient_ids)}]", end=" ")
            report = process_single_patient(
                patient_id=pid,
                input_dir=input_dir,
                output_dir=output_dir,
                hu_min=hu_min,
                hu_max=hu_max,
                target_spacing=target_spacing,
                save_outputs=save_outputs,
                verbose=verbose,
            )
            reports.append(report)
        except Exception as e:
            print(f"\n    !!! HATA (liver_{pid}): {e}")
            failed.append({"patient_id": pid, "error": str(e)})
    
    # Ozet
    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE TAMAMLANDI")
    print(f"{'=' * 70}")
    print(f"Basarili: {len(reports)}/{len(patient_ids)}")
    print(f"Basarisiz: {len(failed)}")
    print(f"Toplam sure: {total_time:.1f} saniye ({total_time/60:.1f} dakika)")
    
    if reports:
        summary = summarize_dataset_metrics(reports)
        print(f"\nDATASET OZETI:")
        print(f"  Liver volume preservation: "
              f"{summary['liver_preservation_mean']*100:.2f}% "
              f"+/- {summary['liver_preservation_std']*100:.2f}%")
        if summary['tumor_preservation_mean'] is not None:
            print(f"  Tumor volume preservation: "
                  f"{summary['tumor_preservation_mean']*100:.2f}% "
                  f"+/- {summary['tumor_preservation_std']*100:.2f}% "
                  f"({summary['n_with_tumor']} hastada tumor var)")
        print(f"  Background ratio:   {summary['background_pct_mean']:.2f}%")
        print(f"  Liver ratio:        {summary['liver_pct_mean']:.2f}%")
        print(f"  Tumor ratio:        {summary['tumor_pct_mean']:.4f}%")
    
    # Rapor kaydet
    if save_report and reports:
        report_path = output_dir / "preprocessing_report.json"
        # numpy/tuple gibi tipleri JSON-friendly hale getir
        clean_reports = json.loads(
            json.dumps(reports, default=lambda x: list(x) if hasattr(x, '__iter__') else str(x))
        )
        with open(report_path, "w") as f:
            json.dump({
                "summary": summary if reports else {},
                "patients": clean_reports,
                "failed": failed,
            }, f, indent=2)
        print(f"\nRapor kaydedildi: {report_path}")
    
    print(f"{'=' * 70}\n")
    return reports


def main():
    """Komut satiri entry point."""
    parser = argparse.ArgumentParser(
        description="Task03_Liver preprocessing pipeline"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Task03_Liver klasoru yolu"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Cikti klasoru yolu"
    )
    parser.add_argument(
        "--hu-min", type=float, default=-150.0,
        help="HU windowing alt sinir (default: -150)"
    )
    parser.add_argument(
        "--hu-max", type=float, default=300.0,
        help="HU windowing ust sinir (default: 300)"
    )
    parser.add_argument(
        "--target-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="Hedef voxel spacing (z y x), default 1.0 1.0 1.0"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="NIfTI dosyalarini diske yazma (sadece test)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Az log yazdir"
    )
    
    args = parser.parse_args()
    
    process_all_patients(
        input_dir=args.input,
        output_dir=args.output,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        target_spacing=tuple(args.target_spacing),
        save_outputs=(not args.no_save),
        verbose=(not args.quiet),
    )


if __name__ == "__main__":
    main()