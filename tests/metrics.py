"""
metrics.py
Métricas simples para evaluar el módulo SLAM (pensadas para usarse sobre
todo en simulación, donde se conoce la trayectoria/mapa "real").
"""
import math
import numpy as np


def compute_ate(est_traj, gt_traj):
    """Absolute Trajectory Error promedio: distancia euclidiana (x,y)
    entre la trayectoria estimada y la trayectoria real, promediada en el
    tiempo. Es la métrica estándar para evaluar SLAM/localización."""
    errs = [math.hypot(e[0] - g[0], e[1] - g[1]) for e, g in zip(est_traj, gt_traj)]
    return float(np.mean(errs)) if errs else 0.0


def compute_orientation_error(est_traj, gt_traj):
    """Error medio absoluto de orientación (rad), con manejo correcto del
    wrap-around en +-pi."""
    errs = []
    for e, g in zip(est_traj, gt_traj):
        d = (e[2] - g[2] + math.pi) % (2 * math.pi) - math.pi
        errs.append(abs(d))
    return float(np.mean(errs)) if errs else 0.0


def uncertainty_series(covariances):
    """Traza de la matriz de covarianza a lo largo del tiempo: sirve para
    graficar cómo evoluciona la incertidumbre de la pose."""
    return [float(np.trace(np.array(c))) for c in covariances]


def map_occupancy_accuracy(grid_est, grid_gt_occupied_cells, threshold=1.0,
                           tolerance_cells=2):
    """Compara las celdas que el mapa estimado marca como ocupadas contra
    un conjunto de celdas 'verdaderas' (útil en simulación, donde se conoce
    el mapa real). Devuelve precisión y exhaustividad (recall).

    tolerance_cells: una celda estimada cuenta como acierto si hay una
    celda verdadera a esa distancia (en celdas). Con resolución de 5 cm y
    ruido de rango de varios cm, exigir coincidencia exacta de celda
    castigaría errores menores que el propio ruido del sensor."""
    est_occ = set(map(tuple, np.argwhere(grid_est.log_odds > threshold)))
    gt_occ = set(grid_gt_occupied_cells)
    if not gt_occ:
        return None

    t = tolerance_cells
    offsets = [(dy, dx) for dy in range(-t, t + 1) for dx in range(-t, t + 1)]

    def near(cell, target_set):
        cy, cx = cell
        return any((cy + dy, cx + dx) in target_set for dy, dx in offsets)

    tp_est = sum(1 for c in est_occ if near(c, gt_occ))
    covered_gt = sum(1 for c in gt_occ if near(c, est_occ))
    precision = tp_est / len(est_occ) if est_occ else 0.0
    recall = covered_gt / len(gt_occ)
    return {"precision": precision, "recall": recall}
