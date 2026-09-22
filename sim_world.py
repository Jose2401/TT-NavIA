"""
sim_world.py
Entorno simulado coherente para probar el SLAM sin hardware.

Sustituye a los generadores aleatorios anteriores (que devolvían distancias
sin relación con ningún cuarto real, y por lo tanto producían mapas basura):
aquí existe un cuarto "de verdad" (paredes como segmentos + obstáculos como
círculos con etiqueta) y todos los sensores simulados se derivan de él por
geometría:

- SimulatedUltrasonicArray: cada sensor lanza un rayo desde la pose real y
  devuelve la distancia al primer choque (pared u obstáculo) + ruido.
- simulated_vision(): detecciones tipo cámara (bearing + distancia
  monocular gruesa + etiqueta) de los obstáculos dentro del campo de
  visión y no ocluidos.
- SimulatedMotion: el usuario recorre waypoints dentro del cuarto; entrega
  la pose real (ground truth) y un MotionEstimate con ruido (lo que en el
  sistema real entregará la odometría/IMU).

TODO: todo este archivo desaparece cuando existan los módulos reales de
odometría/IMU y de sensores ultrasónicos; las interfaces que produce
(MotionEstimate, UltrasonicReading, VisionDetection) son las mismas.
"""
import math
import random

from interfaces import MotionEstimate, UltrasonicReading, VisionDetection
from ekf_slam import wrap_angle


class SimulatedWorld:
    """Cuarto de referencia: paredes (segmentos) + obstáculos (círculos)."""

    def __init__(self, walls, obstacles):
        self.walls = walls          # [(x1, y1, x2, y2), ...]
        self.obstacles = obstacles  # [(cx, cy, radio, etiqueta), ...]

    @staticmethod
    def default_room():
        """Cuarto de 6 x 4 m con muebles típicos."""
        walls = [
            (0.0, 0.0, 6.0, 0.0),
            (6.0, 0.0, 6.0, 4.0),
            (6.0, 4.0, 0.0, 4.0),
            (0.0, 4.0, 0.0, 0.0),
        ]
        obstacles = [
            (2.0, 1.0, 0.40, "table"),
            (4.6, 1.1, 0.25, "chair"),
            (1.0, 3.1, 0.45, "sofa"),
            (5.3, 3.3, 0.20, "potted plant"),
            (3.3, 2.9, 0.30, "chair"),
        ]
        return SimulatedWorld(walls, obstacles)

    # ---------------- geometría de rayos ----------------
    def ray_distance(self, x, y, angle, max_range=4.0):
        """Distancia desde (x, y) en la dirección `angle` hasta el primer
        choque con una pared u obstáculo; inf si no hay nada en max_range."""
        c, s = math.cos(angle), math.sin(angle)
        best = float("inf")

        for (ax, ay, bx, by) in self.walls:
            ex, ey = bx - ax, by - ay
            denom = c * ey - s * ex
            if abs(denom) < 1e-12:
                continue
            t = ((ax - x) * ey - (ay - y) * ex) / denom      # a lo largo del rayo
            u = ((ax - x) * s - (ay - y) * c) / -denom       # a lo largo del muro
            if t > 0.0 and 0.0 <= u <= 1.0:
                best = min(best, t)

        for (cx, cy, r, _label) in self.obstacles:
            fx, fy = x - cx, y - cy
            b = fx * c + fy * s
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc < 0.0:
                continue
            t = -b - math.sqrt(disc)
            if t > 0.0:
                best = min(best, t)

        return best if best <= max_range else float("inf")

    def occupied_cells(self, grid, samples_per_m=40):
        """Celdas 'realmente' ocupadas del cuarto, en índices de la grilla
        dada. Sirve como referencia para metrics.map_occupancy_accuracy."""
        cells = set()
        for (ax, ay, bx, by) in self.walls:
            n = max(2, int(math.hypot(bx - ax, by - ay) * samples_per_m))
            for i in range(n + 1):
                t = i / n
                gx, gy = grid.world_to_grid(ax + t * (bx - ax), ay + t * (by - ay))
                if grid.in_bounds(gx, gy):
                    cells.add((gy, gx))
        for (cx, cy, r, _label) in self.obstacles:
            n = max(8, int(2 * math.pi * r * samples_per_m))
            for i in range(n):
                a = 2 * math.pi * i / n
                gx, gy = grid.world_to_grid(cx + r * math.cos(a), cy + r * math.sin(a))
                if grid.in_bounds(gx, gy):
                    cells.add((gy, gx))
        return cells


class SimulatedUltrasonicArray:
    """Arreglo de sensores fijos (frente, izquierda, derecha) que miden por
    ray casting contra el cuarto simulado.
    TODO: sustituir por lectura real de hardware (HC-SR04 vía GPIO)."""

    def __init__(self, world: SimulatedWorld,
                 bearings=(0.0, math.radians(40), math.radians(-40)),
                 max_range=4.0, sigma_r=0.02):
        self.world = world
        self.bearings = list(bearings)
        self.max_range = max_range
        self.sigma_r = sigma_r

    def read(self, true_pose):
        x, y, theta = true_pose
        readings = []
        for b in self.bearings:
            d = self.world.ray_distance(x, y, theta + b, self.max_range)
            if math.isinf(d):
                # Sin eco dentro del alcance: la lectura no sirve para el
                # EKF, pero sí informa "espacio libre" al mapa.
                readings.append(UltrasonicReading(
                    range_m=self.max_range, sensor_bearing=b,
                    max_range=self.max_range, valid=False))
            else:
                readings.append(UltrasonicReading(
                    range_m=max(0.05, d + random.gauss(0.0, self.sigma_r)),
                    sensor_bearing=b, max_range=self.max_range, valid=True))
        return readings


def simulated_vision(world: SimulatedWorld, true_pose,
                     fov=math.radians(70), max_range=4.5,
                     sigma_bearing=math.radians(2), sigma_range_pct=0.25,
                     drop_prob=0.0):
    """Detecciones tipo cámara de los obstáculos del cuarto: bearing con
    ruido pequeño (la cámara es buena midiendo dirección) y distancia
    monocular con ruido grande (la cámara es mala midiendo distancia),
    imitando el comportamiento del módulo real vision/TT-NavIA."""
    x, y, theta = true_pose
    detections = []
    for (cx, cy, r, label) in world.obstacles:
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        if dist > max_range or dist < 1e-6:
            continue
        bearing = wrap_angle(math.atan2(dy, dx) - theta)
        if abs(bearing) > fov / 2.0:
            continue
        # Oclusión: si una pared u otro objeto tapa el centro del obstáculo
        hit = world.ray_distance(x, y, theta + bearing, max_range=dist + r)
        if hit < dist - r - 1e-6:
            continue
        if random.random() < drop_prob:
            continue
        # Distancia a la SUPERFICIE visible (dist - r), no al centro: es lo
        # que mide también el ultrasonido, y así ambas fuentes describen el
        # mismo punto físico (si no, el landmark oscila hasta 2r entre
        # fuentes y sesga la corrección).
        surf = max(0.1, dist - r)
        detections.append(VisionDetection(
            bearing=bearing + random.gauss(0.0, sigma_bearing),
            label=label,
            range_est=max(0.1, surf * (1.0 + random.gauss(0.0, sigma_range_pct))),
            moving=False,
        ))
    return detections


class SimulatedMotion:
    """El usuario camina siguiendo waypoints dentro del cuarto: primero
    gira hacia el objetivo (giro limitado por ciclo) y luego avanza. Expone
    la pose real y entrega el MotionEstimate ruidoso que en el sistema real
    producirá el módulo de odometría/IMU.
    TODO: sustituir por el módulo real de estimación de movimiento."""

    def __init__(self, start=(0.7, 0.7, 0.0), waypoints=None,
                 step_dist=0.06, max_rot=math.radians(9),
                 sigma_trans=0.006, sigma_rot=math.radians(0.5),
                 dt=1 / 15.0, loop=True):
        self.true_pose = list(start)
        self.waypoints = list(waypoints) if waypoints else [
            (4.0, 0.7), (5.3, 2.0), (4.4, 3.3), (2.4, 2.2),
            (0.8, 2.4), (0.7, 0.7),
        ]
        self.wp_index = 0
        self.step_dist = step_dist
        self.max_rot = max_rot
        self.sigma_trans = sigma_trans
        self.sigma_rot = sigma_rot
        self.dt = dt
        self.loop = loop
        self.finished = False

    def get(self):
        """Avanza un ciclo: actualiza la pose real y devuelve el
        MotionEstimate ruidoso correspondiente."""
        x, y, theta = self.true_pose

        if self.finished:
            return MotionEstimate(0.0, 0.0, self.dt)

        wx, wy = self.waypoints[self.wp_index]
        if math.hypot(wx - x, wy - y) < 0.25:
            self.wp_index += 1
            if self.wp_index >= len(self.waypoints):
                if self.loop:
                    self.wp_index = 0
                else:
                    self.finished = True
                    return MotionEstimate(0.0, 0.0, self.dt)
            wx, wy = self.waypoints[self.wp_index]

        desired = math.atan2(wy - y, wx - x)
        err = wrap_angle(desired - theta)
        drot = max(-self.max_rot, min(self.max_rot, err))
        d = self.step_dist if abs(err) < math.radians(25) else 0.0

        # Pose real (ground truth), con el mismo modelo de arco del EKF
        mid = theta + drot / 2.0
        self.true_pose[0] = x + d * math.cos(mid)
        self.true_pose[1] = y + d * math.sin(mid)
        self.true_pose[2] = wrap_angle(theta + drot)

        return MotionEstimate(
            delta_trans=d + random.gauss(0.0, self.sigma_trans),
            delta_rot=drot + random.gauss(0.0, self.sigma_rot),
            dt=self.dt,
        )
