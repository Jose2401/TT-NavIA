"""
navigation_feed.py
Punto ÚNICO de armado de la entrada del módulo de navegación.

El módulo de navegación (agente DRL con PPO, por implementar en TT2)
recibirá por ciclo un NavigationFrame (interfaces.py) construido aquí a
partir de las salidas de los tres módulos ya integrados:

    SLAM (EKFSlam)          -> pose + covarianza + confianza + landmarks
    Mapeo/Ray Casting       -> costmap egocéntrico + d_front/d_left/d_right
    Visión (vision_bridge)  -> detecciones + riesgos r_front/r_left/r_right
    (Ultrasonido            -> se integrará aquí mismo cuando exista)

El vector de estado documentado (TT_2026_B045 §7.3.1) es
    s_t = [d_front, d_left, d_right, phi_goal, d_goal,
           r_front, r_left, r_right]
phi_goal y d_goal dependen del módulo de rutas (waypoints), que no
existe aún; todo lo demás sale de aquí.

También implementa la capa de ALERTAS determinística (independiente del
RL, según el diseño): umbral crítico < 0.80 m -> "deténgase", umbral de
precaución < 1.50 m -> "precaución". En la Raspberry estas alertas irán
a la salida de voz; en la laptop se muestran en la interfaz.
"""
import math

from interfaces import NavigationFrame
from vision_bridge import detection_risk

# Umbrales de alerta del proyecto (alg. 7 del documento; configurables)
CRITICAL_DISTANCE_M = 0.80
CAUTION_DISTANCE_M = 1.50

# Sectores angulares (rad) respecto a la orientación del usuario
FRONT_HALF_ANGLE = math.radians(30)   # frente = +-30 grados
SIDE_MAX_ANGLE = math.radians(90)     # izquierda/derecha hasta +-90


def _sector_of(bearing):
    """'front' / 'left' / 'right' / None (fuera de los sectores)."""
    if abs(bearing) <= FRONT_HALF_ANGLE:
        return "front"
    if FRONT_HALF_ANGLE < bearing <= SIDE_MAX_ANGLE:
        return "left"
    if -SIDE_MAX_ANGLE <= bearing < -FRONT_HALF_ANGLE:
        return "right"
    return None


def direction_distances(grid, pose, max_range=4.0, rays_per_sector=5):
    """d_front, d_left, d_right por ray casting de consulta sobre el mapa
    (OccupancyGrid o ChunkedMapManager: misma interfaz cast_distance).
    Cada sector lanza varios rayos y se queda con el MÍNIMO (el obstáculo
    más cercano del sector es el que importa para la seguridad)."""
    x, y, theta = pose

    def sweep(a0, a1):
        best = max_range
        for k in range(rays_per_sector):
            a = a0 + (a1 - a0) * k / max(1, rays_per_sector - 1)
            best = min(best, grid.cast_distance(x, y, theta + a, max_range))
        return best

    d_front = sweep(-FRONT_HALF_ANGLE, FRONT_HALF_ANGLE)
    d_left = sweep(FRONT_HALF_ANGLE, SIDE_MAX_ANGLE)
    d_right = sweep(-SIDE_MAX_ANGLE, -FRONT_HALF_ANGLE)
    return d_front, d_left, d_right


def sector_risks(detections, max_distance=3.0):
    """r_front, r_left, r_right: máximo riesgo (0-3) de las detecciones
    del ciclo por sector, ignorando lo que está lejos (> max_distance)."""
    risks = {"front": 0, "left": 0, "right": 0}
    for det in detections:
        if det.range_est is not None and det.range_est > max_distance:
            continue
        sector = _sector_of(det.bearing)
        if sector is not None:
            risks[sector] = max(risks[sector], detection_risk(det))
    return risks["front"], risks["left"], risks["right"]


def build_navigation_frame(slam, grid, detections, costmap_size_m=4.0,
                           max_range=4.0, chunk=None, room=None,
                           ultrasonic=None):
    """Arma el NavigationFrame del ciclo. `grid` puede ser OccupancyGrid
    o ChunkedMapManager (misma interfaz de consulta)."""
    output = slam.get_output()
    pose = (output.x, output.y, output.theta)
    costmap = grid.local_costmap(pose, size_m=costmap_size_m)
    d_front, d_left, d_right = direction_distances(grid, pose, max_range)
    r_front, r_left, r_right = sector_risks(detections)

    return NavigationFrame(
        slam=output,
        costmap=costmap,
        costmap_resolution=grid.resolution,
        costmap_size_m=costmap_size_m,
        detections=list(detections),
        landmarks=slam.get_landmarks(),
        d_front=d_front, d_left=d_left, d_right=d_right,
        risk_front=r_front, risk_left=r_left, risk_right=r_right,
        chunk=chunk, room=room,
        ultrasonic=list(ultrasonic) if ultrasonic else [],
    )


def safety_alerts(frame: NavigationFrame):
    """Capa de seguridad determinística (independiente del RL): mensajes
    de alerta según los umbrales del proyecto. Devuelve lista de
    (prioridad, mensaje); prioridad 0 = crítica (interrumpe el audio en
    curso), 1 = precaución."""
    alerts = []

    def nearest_label(sector):
        best = None
        for det in frame.detections:
            if _sector_of(det.bearing) != sector or det.range_est is None:
                continue
            if best is None or det.range_est < best.range_est:
                best = det
        return best.label if best is not None else "obstaculo"

    sectores = (("front", frame.d_front, "al frente"),
                ("left", frame.d_left, "a la izquierda"),
                ("right", frame.d_right, "a la derecha"))
    for sector, dist, donde in sectores:
        # La distancia efectiva es la menor entre el mapa y la detección
        # visual más próxima del sector (visión ve antes de que el mapa
        # confirme la celda).
        det_d = min((d.range_est for d in frame.detections
                     if _sector_of(d.bearing) == sector
                     and d.range_est is not None), default=float("inf"))
        eff = min(dist, det_d)
        if eff < CRITICAL_DISTANCE_M:
            alerts.append((0, f"Atencion: {nearest_label(sector)} a "
                              f"{int(eff * 100)} centimetros {donde}. "
                              "Detengase."))
        elif eff < CAUTION_DISTANCE_M:
            alerts.append((1, f"Precaucion: {nearest_label(sector)} "
                              f"cerca {donde}."))
    return alerts
