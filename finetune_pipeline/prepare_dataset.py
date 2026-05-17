"""
prepare_dataset.py
Usa: nielsr/ade20k-demo  (Parquet puro, sin loading script, funciona en Windows)
"""
import numpy as np
import cv2
from pathlib import Path

OUTPUT_DIR  = Path("dataset_interior")
TRAIN_RATIO = 0.85
MAX_IMAGES  = 3000

# Nombres de clase ADE20K que nos interesan → id nuestro
TARGET_MAP = {
    "wall":    0,
    "door":    1,
    "window":  2,
    "fence":   3,
    "glass":   2,
    "mirror":  2,
}

def create_dirs():
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[OK] Carpetas creadas")

def mask_to_yolo(mask, class_id, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 400:
            continue
        eps    = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        coords = []
        for x, y in pts:
            coords.append(f"{x/img_w:.6f}")
            coords.append(f"{y/img_h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines

def download_and_process():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] pip install datasets")
        return

    print("[INFO] Descargando dataset...")

    # Dataset ADE20K en formato Parquet (sin loading script)
    # Tiene columnas: image (PIL), annotation (PIL con indices de clase)
    # y label_map (dict nombre->indice)
    try:
        ds = load_dataset("scene_parse_150", split="train",
                          download_mode="force_redownload",
                          ignore_verifications=True)
    except Exception:
        # Fallback: dataset alternativo en Parquet puro
        ds = load_dataset("EduardoPacheco/ADE20K", split="train")

    create_dirs()

    # Detectar formato del dataset
    first = ds[0]
    has_label_map = "label_map" in first or "labels" in first
    print(f"[INFO] Columnas disponibles: {list(first.keys())}")

    processed = 0
    skipped   = 0

    for sample in ds:
        if processed >= MAX_IMAGES:
            break

        try:
            img_np = np.array(sample["image"].convert("RGB"))[:, :, ::-1]
            ann_np = np.array(sample["annotation"])
        except Exception:
            skipped += 1
            continue

        img_h, img_w = img_np.shape[:2]
        yolo_lines   = []

        # Obtener mapa de labels si existe
        label_map = sample.get("label_map", {}) or {}

        if label_map:
            # Buscar indices de nuestras clases por nombre
            for name, idx in label_map.items():
                name_lower = str(name).lower()
                our_id = None
                for target, cid in TARGET_MAP.items():
                    if target in name_lower:
                        our_id = cid
                        break
                if our_id is None:
                    continue
                binary = ((ann_np == int(idx)) * 255).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                yolo_lines += mask_to_yolo(binary, our_id, img_w, img_h)
        else:
            # Fallback: usar indices fijos de SceneParse150
            SP150 = {0: 0, 4: 1, 8: 2, 15: 2, 32: 3, 43: 1}
            for sp_idx, our_id in SP150.items():
                binary = ((ann_np == sp_idx) * 255).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                yolo_lines += mask_to_yolo(binary, our_id, img_w, img_h)

        if not yolo_lines:
            skipped += 1
            continue

        split = "train" if (processed / MAX_IMAGES) < TRAIN_RATIO else "val"
        stem  = f"img_{processed:05d}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / split / f"{stem}.jpg"), img_np)
        (OUTPUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines))
        processed += 1

        if processed % 100 == 0:
            print(f"  {processed}/{MAX_IMAGES} procesadas...")

    print(f"\n[OK] Procesadas : {processed}")
    print(f"     Descartadas : {skipped}")

    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_DIR.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: wall\n"
        "  1: door\n"
        "  2: window\n"
        "  3: fence\n"
    )
    print("[OK] data.yaml generado")
    print("\n>>> Siguiente paso: python train.py")

if __name__ == "__main__":
    download_and_process()