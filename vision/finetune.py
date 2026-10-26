"""
finetune.py — afina un yolo con un dataset propio (formato YOLO) pa
detectar clases que nos faltan, p.ej. puertas abiertas/cerradas.

el dataset tiene que venir en formato YOLO (carpetas images/ + labels/
y un data.yaml con las clases). opciones pa conseguirlo:
  - roboflow universe: busca "door detection open closed" y exporta en
    formato "YOLOv8" (ya trae data.yaml, cero talacha)
  - el de kaggle (dataclusterlabs/doors-doors) son fotos SIN cajas:
    sirve de materia prima pero hay que etiquetarlo (roboflow/labelimg)
  - fotos propias etiquetadas (lo ideal pa ESCOM: sus puertas reales)

uso:
    python -m vision.finetune datasets/puertas/data.yaml
    python -m vision.finetune datasets/puertas/data.yaml --epochs 60 \
        --base vision/yolov8s-seg.pt --out vision/puertas.pt

despues se usa asi: el modelo resultante se pasa a los mains con
--model (si reemplaza al general) o se agrega como detector extra en
vision_bridge (EXTRA_OIV7 + extraModelPath).
"""
import argparse
import shutil


def main():
    p = argparse.ArgumentParser(description="fine-tune de yolo con dataset propio")
    p.add_argument("data", help="ruta al data.yaml del dataset (formato YOLO)")
    p.add_argument("--base", default="yolov8s.pt",
                   help="modelo base (yolov8s.pt, vision/yolov8s-seg.pt...)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", default="vision/puertas.pt",
                   help="a donde copiar el mejor peso")
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.base)
    res = model.train(data=args.data, epochs=args.epochs,
                      imgsz=args.imgsz, batch=args.batch,
                      project="runs_finetune", name="puertas")
    mejor = str(res.save_dir / "weights" / "best.pt")
    shutil.copy(mejor, args.out)
    print(f"\nlisto: {args.out}")
    print("metricas en", res.save_dir)
    print("pruebalo con: python main_lap.py --model", args.out)


if __name__ == "__main__":
    main()
