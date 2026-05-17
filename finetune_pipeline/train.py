"""
train.py
--------
Fine-tune de YOLOv8s-seg sobre el dataset de interiores.

Parte desde yolov8s-seg.pt (preentrenado en COCO) y
añade las 4 clases nuevas: wall, door, window, fence

Uso:
  python train.py
  python train.py --epochs 80 --batch 8 --device cpu

El modelo final queda en:
  runs/segment/interior_v1/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="dataset_interior/data.yaml")
    p.add_argument("--model",   default="yolov8s-seg.pt",
                   help="Modelo base (yolov8n-seg | yolov8s-seg | yolov8m-seg)")
    p.add_argument("--epochs",  type=int,   default=60)
    p.add_argument("--batch",   type=int,   default=8,
                   help="Reducir a 4 si hay OOM en GPU / usar CPU")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="0",
                   help="'0' para GPU, 'cpu' para CPU")
    p.add_argument("--name",    default="interior_v1")
    p.add_argument("--resume",  action="store_true",
                   help="Reanudar entrenamiento interrumpido")
    return p.parse_args()


def main():
    args = parse_args()

    # Verificar que el dataset existe
    if not Path(args.data).exists():
        print(f"[ERROR] No se encontró {args.data}")
        print("        Ejecuta primero: python prepare_dataset.py")
        return

    print("=" * 55)
    print("  Fine-tune YOLOv8-seg → Interiores")
    print(f"  Modelo base : {args.model}")
    print(f"  Dataset     : {args.data}")
    print(f"  Épocas      : {args.epochs}")
    print(f"  Batch       : {args.batch}")
    print(f"  Dispositivo : {args.device}")
    print("=" * 55)

    model = YOLO(args.model)

    results = model.train(
        data      = args.data,
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device,
        name      = args.name,
        resume    = args.resume,

        # ── Hiperparámetros optimizados para fine-tune ──────────
        lr0       = 0.001,      # LR inicial bajo (ya está preentrenado)
        lrf       = 0.01,       # LR final = lr0 * lrf
        warmup_epochs = 3,
        weight_decay  = 0.0005,
        dropout       = 0.1,    # regularización ligera

        # ── Augmentación para interiores ────────────────────────
        hsv_h     = 0.015,
        hsv_s     = 0.5,
        hsv_v     = 0.3,
        degrees   = 5.0,        # rotación leve (cámara puede estar inclinada)
        translate = 0.1,
        scale     = 0.4,
        flipud    = 0.0,        # no voltear verticalmente (paredes siempre arriba)
        fliplr    = 0.5,
        mosaic    = 0.8,
        copy_paste= 0.1,        # útil para segmentación

        # ── Guardar ─────────────────────────────────────────────
        save      = True,
        save_period = 10,       # guardar checkpoint cada 10 épocas
        patience  = 20,         # early stopping si no mejora en 20 épocas
        plots     = True,       # generar gráficas de métricas
        val       = True,
    )

    best_path = f"C:/Users/josec/Documents/TTv1/Modelo Visión/finetune_pipeline/segment/{args.name}best.pt/weights/"
    print("\n" + "=" * 55)
    print(f"  Entrenamiento completo")
    print(f"  Mejor modelo: {best_path}")
    print(f"\n  Para usar en el sistema de visión:")
    print(f"  MODEL_PATH = '{best_path}'  en main.py")
    print("=" * 55)

    # Validación final automática
    print("\n[INFO] Ejecutando validación final...")
    metrics = model.val()
    print(f"  mAP50      : {metrics.seg.map50:.3f}")
    print(f"  mAP50-95   : {metrics.seg.map:.3f}")


if __name__ == "__main__":
    main()