import math

from interfaces import NavigationFrame
from vision_bridge import detectionRisk

CRITICAL_DISTANCE_M = 0.80
CAUTION_DISTANCE_M = 1.50
LATERAL_CRITICAL_M = 0.45
LATERAL_CAUTION_M = CRITICAL_DISTANCE_M

FRONT_HALF_ANGLE = math.radians(30)
SIDE_MAX_ANGLE = math.radians(90)


def _sectorOf(bearing):
    if abs(bearing) <= FRONT_HALF_ANGLE:
        return "front"
    if FRONT_HALF_ANGLE < bearing <= SIDE_MAX_ANGLE:
        return "left"
    if -SIDE_MAX_ANGLE <= bearing < -FRONT_HALF_ANGLE:
        return "right"
    return None


def directionDistances(grid, pose, maxRange=4.0, raysPerSector=5):
    x, y, theta = pose

    def barrer(a0, a1):
        mejor = maxRange
        for k in range(raysPerSector):
            a = a0 + (a1 - a0) * k / max(1, raysPerSector - 1)
            mejor= min(mejor, grid.castDistance(x, y, theta + a, maxRange))
        return mejor

    distFrente = barrer(-FRONT_HALF_ANGLE, FRONT_HALF_ANGLE)
    distIzq =barrer(FRONT_HALF_ANGLE, SIDE_MAX_ANGLE)
    distDer = barrer(-SIDE_MAX_ANGLE, -FRONT_HALF_ANGLE)
    return distFrente, distIzq, distDer


def sectorRisks(detections, maxDistance=3.0):
    riesgos = {"front": 0, "left": 0, "right": 0}
    for det in detections:
        if det.range_est is not None and det.range_est > maxDistance:
            continue
        sector = _sectorOf(det.bearing)
        if sector is not None:
            riesgos[sector] = max(riesgos[sector], detectionRisk(det))
    return riesgos["front"], riesgos["left"], riesgos["right"]


# arma el frame del ciclo pal navigator
def buildNavigationFrame(slam, grid, detections, costmapSizeM=4.0,
                         maxRange=4.0, chunk=None, room=None,
                         ultrasonic=None):
    output = slam.getOutput()
    pose = (output.x, output.y, output.theta)
    costmap = grid.localCostmap(pose, sizeM=costmapSizeM)
    dFront, dLeft, dRight = directionDistances(grid, pose, maxRange)
    rFront, rLeft, rRight = sectorRisks(detections)

    return NavigationFrame(
        slam=output,
        costmap=costmap,
        costmap_resolution=grid.resolution,
        costmap_size_m=costmapSizeM,
        detections=list(detections),
        landmarks=slam.getLandmarks(),
        d_front=dFront, d_left=dLeft, d_right=dRight,
        risk_front=rFront, risk_left=rLeft, risk_right=rRight,
        chunk=chunk, room=room,
        ultrasonic=list(ultrasonic) if ultrasonic else [],
    )


# lo libre al frente, mapa vs vision
def effectiveFrontDistance(frame: NavigationFrame):
    detD = min((d.range_est for d in frame.detections
                if _sectorOf(d.bearing) == "front"
                and d.range_est is not None), default=float("inf"))
    return min(frame.d_front, detD)


# alertas fijas, 0 critica 1 precaucion
def safetyAlerts(frame: NavigationFrame):
    alertas = []

    def detMasCerca(sector):
        papu = None
        for det in frame.detections:
            if _sectorOf(det.bearing) != sector or det.range_est is None:
                continue
            if papu is None or det.range_est < papu.range_est:
                papu = det
        return papu

    sectores = (("front", frame.d_front, "al frente"),
                ("left", frame.d_left, "a la izquierda"),
                ("right", frame.d_right, "a la derecha"))
    for sector, dist, donde in sectores:
        det = detMasCerca(sector)
        detD = det.range_est if det is not None else float("inf")
        eff = min(dist, detD)
        detGana = det is not None and detD <= dist
        label = det.label if detGana else "obstaculo"
        riesgoAlto= detGana and detectionRisk(det) >= 3

        if sector == "front" or riesgoAlto:
            crit, caut = CRITICAL_DISTANCE_M, CAUTION_DISTANCE_M
        else:
            crit, caut = LATERAL_CRITICAL_M, LATERAL_CAUTION_M

        if eff < crit:
            cm =int(round(eff * 10.0)) * 10
            alertas.append((0, f"Atencion: {label} a {cm} centimetros "
                               f"{donde}. Detengase."))
        elif eff < caut:
            alertas.append((1, f"Precaucion: {label} cerca {donde}."))
    return alertas


def needsReplan(alerts):
    return any(prio == 0 for prio, _msg in alerts)
