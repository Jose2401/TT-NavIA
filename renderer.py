"""
renderer.py
"""

import cv2
import numpy as np
from obstacle_logic import CATEGORY_COLORS



# MASK
def draw_mask(frame: np.ndarray, mask: np.ndarray | None,
              color: tuple[int, int, int], alpha: float = 0.08) -> np.ndarray:

    if mask is None:
        return frame

    mask = (mask > 0.5).astype(np.uint8)

    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # LIMPIEZA
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if mask.sum() < 200:
        return frame

    # overlay
    colored = np.zeros_like(frame)
    colored[mask == 1] = color

    frame = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1)

    return frame

# Detection frame en UI
def draw_detection(frame: np.ndarray,
                   detection: dict,
                   category: str,
                   is_moving: bool,
                   dims: str,
                   extra_labels=None) -> np.ndarray:

    x1, y1, x2, y2 = detection["bbox"]
    base_color = CATEGORY_COLORS.get(category, CATEGORY_COLORS["desconocido"])

    color = base_color

    if extra_labels:
        if "SALIDA" in extra_labels:
            color = (0, 255, 255)  # amarillo
        elif "SEÑAL" in extra_labels:
            color = (255, 200, 0)  # naranja

    if is_moving:
        color = (0, 0, 255)  # rojo domina si hay movimiento

    # DATOS BASE
    movement = "MOVIL" if is_moving else "ESTATICO"
    conf_pct = int(detection.get("confidence", 0) * 100)
    label_obj = detection.get("label", "obj")

    thickness = 3 if is_moving else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # LABEL PRINCIPAL
    label_main = f"{label_obj} [{category}] {movement} {conf_pct}%"

    # LABELS EXTRA
    if extra_labels:
        for lab in extra_labels:
            label_main += f" [{lab}]"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    font_thick = 1

    (tw, th), baseline = cv2.getTextSize(label_main, font, font_scale, font_thick)
    label_y = max(y1 - 6, th + 4)

    # Fondo sólido
    cv2.rectangle(frame,
                  (x1, label_y - th - 4),
                  (x1 + tw + 6, label_y + baseline),
                  color, -1)

    cv2.putText(frame, label_main,
                (x1 + 3, label_y),
                font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

    # ── DIMENSIONES (MEJORADAS) ────────────────────────────
    dim_y = y2 + th + 6

    if dim_y < frame.shape[0]:
        dim_text = f"{dims}"

        if is_moving:
            dim_text += " | mov"

        (dw, dh), _ = cv2.getTextSize(dim_text, font, 0.42, 1)

        cv2.rectangle(frame,
                      (x1, dim_y - dh - 3),
                      (x1 + dw + 6, dim_y + 3),
                      (20, 20, 20), -1)

        cv2.putText(frame, dim_text,
                    (x1 + 3, dim_y),
                    font, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


    if is_moving:
        cv2.circle(frame, (x2 - 8, y1 + 8), 5, (0, 0, 255), -1)

    return frame


# HUD
def draw_hud(frame: np.ndarray, n_objects: int,
             motion_ratio: float, fps: float,
             category_counts: dict[str, int]) -> np.ndarray:

    lines = [
        f"FPS: {fps:.1f}",
        f"Objetos: {n_objects}",
        f"Movimiento escena: {motion_ratio:.3f}",
    ]

    for cat, count in category_counts.items():
        if count > 0:
            lines.append(f"{cat}: {count}")

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.5
    fthick = 1
    pad   = 8
    lh    = 20

    hud_h = len(lines) * lh + pad * 2
    hud_w = 260

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        color = (255, 255, 255)

        if "Movimiento" in line:
            color = (0, 255, 255)

        cv2.putText(frame, line,
                    (pad, pad + (i + 1) * lh),
                    font, fscale, color, fthick, cv2.LINE_AA)

    return frame