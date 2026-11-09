"""
route_planner.py — dibuja la RUTA completa que el modelo tomaria de
aqui al destino: simula al agente sobre el mapa actual (sin mover al
usuario) y junta los puntos. si aparece un obstaculo nuevo en el mapa,
la siguiente planeada sale distinta solita — eso es lo que se ve
cambiar en la interfaz. es ADVISORY: la decision real sigue siendo
paso a paso del navigator; esto es pa visualizar/anticipar.
"""
import math
import threading
import time

import numpy as np

from navigation.nav_env import (EXT_RAY_ANGLES, MAX_RANGE, RISK_RANGE,
                                FRONT_HALF, SIDE_MAX, FORWARD_STEP,
                                TURN_STEP)

ARRIVAL_RADIUS = 0.6
MAX_PASOS = 240
MAX_ATASCO = 14
# no es el medio ancho real de una persona (~0.2): los mapas reales traen
# celdas sueltas de ruido y con 0.20 cerraba pasillos que si existen
# (23/32 metas vs 29/32). 0.12 tapa los huecos de 5-10 cm entre ecos
# de una pared y deja pasar los pasillos reales
RADIO_CUERPO_M = 0.12


def _cabe(grid, x, y, radio=RADIO_CUERPO_M):
    """el usuario ocupa espacio: los muros del mapa son puntos de eco
    con huecos y un rayo solo se cuela entre ellos. aqui se pide que el
    disco del cuerpo este libre."""
    gx, gy = grid.worldToCell(x, y)
    rc = max(1, int(round(radio / grid.resolution)))
    for dy in range(-rc, rc + 1):
        for dx in range(-rc, rc + 1):
            if dx * dx + dy * dy <= rc * rc and grid.isBlockedCell(gx + dx, gy + dy):
                return False
    return True


def _goalAlcanzable(grid, goal):
    """un landmark tipo puerta vive SOBRE la pared: el punto navegable
    es el mas cercano enfrente DONDE QUEPA la persona (no basta una
    celda libre pegada a la pared). busca en anillos crecientes y si
    nada cabe en 1 m se queda con la celda libre mas cercana."""
    if _cabe(grid, *goal):
        return goal
    gx, gy = grid.worldToCell(goal[0], goal[1])
    res = grid.resolution
    rc = int(round(1.0 / res))
    candidatos = sorted(((dx * dx + dy * dy, dx, dy)
                         for dy in range(-rc, rc + 1)
                         for dx in range(-rc, rc + 1)
                         if dx * dx + dy * dy <= rc * rc))
    libre = None
    for _d2, dx, dy in candidatos[::2]:       # salto 1 de 2: 4x mas rapido
        if grid.isBlockedCell(gx + dx, gy + dy):
            continue
        p = grid.cellToWorld(gx + dx, gy + dy)
        if libre is None:
            libre = p
        if _cabe(grid, *p):
            return p
    return libre if libre is not None else goal


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sectorIdx(bearing):
    if abs(bearing) <= FRONT_HALF:
        return 0
    if FRONT_HALF < bearing <= SIDE_MAX:
        return 1
    if -SIDE_MAX <= bearing < -FRONT_HALF:
        return 2
    return None


def _hazardSectores(grid, pose):
    cm = grid.localCostmap(pose, sizeM=2 * RISK_RANGE)
    half = cm.shape[1] // 2
    res = grid.resolution
    out = [RISK_RANGE, RISK_RANGE, RISK_RANGE]
    ys, xs = np.nonzero(cm[1] > 0.5)
    for yy, xx in zip(ys, xs):
        dx, dy = (xx - half) * res, (yy - half) * res
        d = math.hypot(dx, dy)
        if d > RISK_RANGE:
            continue
        idx = _sectorIdx(_wrap(math.atan2(dy, dx) - pose[2]))
        if idx is not None:
            out[idx] = min(out[idx], d)
    return out


def _obsEnPose(grid, pose, goal, prevAction):
    """el mismo vector de 24 que ve el modelo, pero calculado en una
    pose HIPOTETICA sobre el mapa (sin movers: el plan es estatico)."""
    x, y, th = pose
    rays = np.array([grid.castDistance(x, y, th + a, MAX_RANGE)
                     for a in EXT_RAY_ANGLES])
    hz = _hazardSectores(grid, pose)
    sector = lambda lo, hi: rays[(EXT_RAY_ANGLES >= lo)
                                 & (EXT_RAY_ANGLES <= hi)].min()
    dSec = [sector(-FRONT_HALF, FRONT_HALF),
            sector(FRONT_HALF, SIDE_MAX),
            sector(-SIDE_MAX, -FRONT_HALF)]
    risks = []
    for i in range(3):
        if hz[i] < RISK_RANGE:
            risks.append(3)
        elif dSec[i] < 1.5:
            risks.append(2)
        elif dSec[i] < RISK_RANGE:
            risks.append(1)
        else:
            risks.append(0)
    phi = _wrap(math.atan2(goal[1] - y, goal[0] - x) - th)
    dGoal = math.hypot(goal[0] - x, goal[1] - y)
    onehot = np.zeros(4, dtype=np.float32)
    onehot[prevAction] = 1.0
    obs = np.concatenate([
        rays / MAX_RANGE,
        np.array(hz) / RISK_RANGE,
        np.array(risks) / 3.0,
        [math.sin(phi), math.cos(phi), min(dGoal, 10.0) / 10.0],
        onehot]).astype(np.float32)
    return np.clip(obs, -1.0, 1.0), dSec


def planear(policy, grid, inicio, goal, maxPasos=MAX_PASOS):
    """simula la politica sobre el mapa. -> dict con points (polilinea),
    estado ('llega'|'bloqueada'|'larga') y pasos."""
    x, y, th = inicio
    goal = _goalAlcanzable(grid, goal)
    puntos = [(x, y)]
    prev = 3
    atasco = 0
    for paso in range(maxPasos):
        if math.hypot(goal[0] - x, goal[1] - y) < ARRIVAL_RADIUS:
            return {"points": puntos, "estado": "llega", "pasos": paso}
        obs, dSec = _obsEnPose(grid, (x, y, th), goal, prev)
        a = policy.predict(obs)
        # misma capa de seguridad que el navigator: no avanzar a lo wey
        if a == 0 and dSec[0] < 0.6:
            a = 1 if dSec[1] >= dSec[2] else 2
        if a == 0:
            libre = grid.castDistance(x, y, th, FORWARD_STEP + 0.3)
            nx = x + FORWARD_STEP * math.cos(th)
            ny = y + FORWARD_STEP * math.sin(th)
            if libre < FORWARD_STEP + 0.2 or not _cabe(grid, nx, ny):
                atasco += 1
            else:
                x += FORWARD_STEP * math.cos(th)
                y += FORWARD_STEP * math.sin(th)
                puntos.append((x, y))
                atasco = 0
        elif a == 1:
            th = _wrap(th + TURN_STEP)
            atasco += 1
        elif a == 2:
            th = _wrap(th - TURN_STEP)
            atasco += 1
        else:
            atasco += 2
        prev = a
        if atasco > MAX_ATASCO:
            dFin = math.hypot(goal[0] - x, goal[1] - y)
            estado = "llega" if dFin < 1.0 else "bloqueada"
            return {"points": puntos, "estado": estado, "pasos": paso}
    return {"points": puntos, "estado": "larga", "pasos": maxPasos}


class PlanificadorRuta:
    """corre planear() en un hilo cada cadaS segundos (o al instante si
    le piden replan); la gui solo lee .ruta."""

    def __init__(self, policy, cadaS=2.0):
        self.policy = policy
        self.cadaS = cadaS
        self.ruta = None          # dict de planear() o None
        self._pedido = None       # (grid, pose, goal)
        self._urgente = False
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def solicitar(self, grid, pose, goal, urgente=False):
        with self._lock:
            self._pedido = (grid, tuple(pose), tuple(goal))
            self._urgente = self._urgente or urgente

    def limpiar(self):
        with self._lock:
            self._pedido = None
            self.ruta = None

    def _worker(self):
        ultima = 0.0
        while True:
            time.sleep(0.1)
            with self._lock:
                pedido = self._pedido
                urgente = self._urgente
            if pedido is None:
                continue
            if not urgente and time.time() - ultima < self.cadaS:
                continue
            try:
                r = planear(self.policy, *pedido)
            except Exception:
                continue
            with self._lock:
                self.ruta = r
                self._urgente = False
            ultima = time.time()
