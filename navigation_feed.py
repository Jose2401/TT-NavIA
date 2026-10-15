"""
navigation_feed.py — arma el NavigationFrame que consume el navegador
(slam + mapa + vision + ultrasonido en un solo paquete por ciclo).
tambien trae la capa de alertas deterministica: usa la MISMA distancia
frontal y umbral que el Navigator, pa que alerta e instruccion nunca
se contradigan.
"""
import math

from interfaces import NavigationFrame
from vision_bridge import detection_risk

# umbrales del proyecto: frente < 0.80 m critico, < 1.50 m precaucion
CRITICAL_DISTANCE_M = 0.80
CAUTION_DISTANCE_M = 1.50
# laterales mas permisivos: caminar a 60 cm de una pared es un pasillo
# normal, no una emergencia; critico solo si ya casi roza
LATERAL_CRITICAL_M = 0.45
LATERAL_CAUTION_M = CRITICAL_DISTANCE_M

# sectores angulares respecto a la orientacion del usuario
FRONT_HALF_ANGLE = math.radians(30)
SIDE_MAX_ANGLE = math.radians(90)

# ojo: nav_env entrena el riesgo con distancias a todo (paredes incluidas)
# y aqui r_* sale solo de detecciones. se probo alinear y el modelo
# entregado empeoro (llegadas 10/10 -> 9/10, vueltas 55 -> 193); hasta
# reentrenar se queda asi, las paredes las cubren d_* y las alertas.


def _sector_of(bearing):
    """'front' / 'left' / 'right' / None (fuera de sectores)."""
    if abs(bearing) <= FRONT_HALF_ANGLE:
        return "front"
    if FRONT_HALF_ANGLE < bearing <= SIDE_MAX_ANGLE:
        return "left"
    if -SIDE_MAX_ANGLE <= bearing < -FRONT_HALF_ANGLE:
        return "right"
    return None


def direction_distances(grid, pose, max_range=4.0, rays_per_sector=5):
    """d_front/left/right por ray casting; cada sector se queda con el
    minimo (el obstaculo mas cercano es el que importa)."""
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
    """max riesgo (0-3) por sector, ignorando lo lejano."""
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
    """arma el NavigationFrame del ciclo (`grid` puede ser OccupancyGrid
    o ChunkedMapManager, misma interfaz)."""
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


def effective_front_distance(frame: NavigationFrame):
    """lo libre al frente: el menor entre el mapa y la deteccion visual
    mas cercana (vision ve el objeto antes de que el mapa confirme).
    el Navigator usa ESTE mismo valor pa anular un 'avanzar'."""
    det_d = min((d.range_est for d in frame.detections
                 if _sector_of(d.bearing) == "front"
                 and d.range_est is not None), default=float("inf"))
    return min(frame.d_front, det_d)


def safety_alerts(frame: NavigationFrame):
    """alertas deterministicas (independientes del RL). devuelve lista de
    (prioridad, mensaje); 0 = critica, 1 = precaucion. frente: 80/150 cm;
    laterales: 45/80 cm, salvo riesgo alto (persona, hoyo) que usa los
    umbrales frontales."""
    alerts = []

    def nearest_det(sector):
        best = None
        for det in frame.detections:
            if _sector_of(det.bearing) != sector or det.range_est is None:
                continue
            if best is None or det.range_est < best.range_est:
                best = det
        return best

    sectores = (("front", frame.d_front, "al frente"),
                ("left", frame.d_left, "a la izquierda"),
                ("right", frame.d_right, "a la derecha"))
    for sector, dist, donde in sectores:
        det = nearest_det(sector)
        det_d = det.range_est if det is not None else float("inf")
        eff = min(dist, det_d)
        # solo se nombra la clase si la deteccion ES lo mas cercano; si
        # gana el mapa se dice "obstaculo" (antes salia "planta a 33 cm"
        # por una pared)
        det_is_nearest = det is not None and det_d <= dist
        label = det.label if det_is_nearest else "obstaculo"
        high_risk_det = det_is_nearest and detection_risk(det) >= 3

        if sector == "front" or high_risk_det:
            crit, caut = CRITICAL_DISTANCE_M, CAUTION_DISTANCE_M
        else:
            crit, caut = LATERAL_CRITICAL_M, LATERAL_CAUTION_M

        if eff < crit:
            # redondeo a 10 cm pa que la voz no repita a cada frame
            cm = int(round(eff * 10.0)) * 10
            alerts.append((0, f"Atencion: {label} a {cm} centimetros "
                              f"{donde}. Detengase."))
        elif eff < caut:
            alerts.append((1, f"Precaucion: {label} cerca {donde}."))
    return alerts


def needs_replan(alerts):
    """True si hay alerta critica: el Navigator recalcula de inmediato."""
    return any(prio == 0 for prio, _msg in alerts)
