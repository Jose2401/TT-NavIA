"""
nav_env.py
Entorno Gymnasium para ENTRENAR el agente de navegación local (DRL).

Diseño (TT_2026_B045 §6.4.3 / §7.3.1):
- Acciones discretas (4): 0 avanzar, 1 girar izquierda, 2 girar derecha,
  3 detenerse.
- Estado documentado ("doc"):
      s_t = [d_front, d_left, d_right, phi_goal, d_goal,
             r_front, r_left, r_right]
  con d_* por ray casting (obstáculo más cercano por sector), phi/d_goal
  relativos al siguiente waypoint y r_* el riesgo máximo por sector
  (0 libre, 1 bajo, 2 medio, 3 alto).
- Estado "extended" (variante a comparar en el entrenamiento): más rayos
  (perfil fino del entorno), distancias a peligros de piso por sector
  (¡los hoyos no bloquean rayos!), goal como (sin, cos, dist) y la
  acción previa. Mismo espacio de acciones.
- Recompensa (doc, con escalas ajustables):
      +100 llegar, -100 colisión/caída, +k*Δd_goal por acercarse,
      -0.1 por paso, penalización por proximidad a riesgo alto,
      bonificación por DETENERSE cuando un obstáculo móvil está encima
      (esa es la función de la acción 3).

El simulador NO ejecuta YOLO/SLAM (sería inviable para RL): genera los
mismos rasgos que produce navigation_feed.build_navigation_frame() a
partir de la geometría real del mundo + ruido (randomización de dominio),
de modo que el modelo entrenado consume en producción exactamente el
mismo vector que vio en entrenamiento.

Mundos procedurales: cuarto simple, dos cuartos con puerta, exterior
abierto con postes/coladeras; obstáculos dinámicos (personas) que
rebotan dentro del área.
"""
import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:                      # pragma: no cover
    raise ImportError(
        "Falta gymnasium. Instalar con: pip install gymnasium "
        "stable-baselines3") from exc

# ----------------------------------------------------------------------
# Parámetros del agente (una persona caminando con el dispositivo)
# ----------------------------------------------------------------------
FORWARD_STEP = 0.30          # m por indicación "avanza"
TURN_STEP = math.radians(20)  # giro por indicación
CLEARANCE = 0.22             # m: más cerca que esto de un obstáculo = colisión
GOAL_RADIUS = 0.45           # m: llegar al destino
MAX_RANGE = 4.0              # m: alcance de los "sensores" (como el mapa)
RISK_RANGE = 3.0             # m: distancia a la que el riesgo cuenta

# Sectores (idénticos a navigation_feed)
FRONT_HALF = math.radians(30)
SIDE_MAX = math.radians(90)

# Recompensa (doc §7.3.1, escalas ajustadas y documentadas):
# el documento da +100/-100/±Δd/-0.1/-5·riesgo; Δd crudo (~0.3 m) apenas
# supera el costo de paso y -5·riesgo por ciclo aplasta el aprendizaje,
# así que Δd se escala x4 y el riesgo se aplica como -0.15·riesgo_frontal
# por paso (mismo orden relativo, gradiente aprendible).
REW_GOAL = 100.0
REW_COLLISION = -100.0
REW_STEP = -0.1
REW_PROGRESS = 4.0           # * Δd_goal (m)
REW_RISK = -0.15             # * riesgo frontal (0-3) por paso
REW_STOP_GOOD = +0.5         # detenerse con un móvil encima (correcto)
REW_STOP_BAD = -0.3          # detenerse sin motivo (pierde tiempo)

EPISODE_STEPS = 300


# ----------------------------------------------------------------------
# Mundo: segmentos (paredes) + círculos (obstáculos, peligros, móviles)
# ----------------------------------------------------------------------
class NavWorld:
    """Geometría del episodio. Todo numpy para ray casting vectorizado."""

    def __init__(self, segments, statics, hazards, movers, bounds):
        # segments: (M,4) x1,y1,x2,y2   paredes
        self.segments = np.asarray(segments, dtype=np.float64).reshape(-1, 4)
        # statics: (K,3) cx,cy,r        obstáculos fijos (bloquean rayos)
        self.statics = np.asarray(statics, dtype=np.float64).reshape(-1, 3)
        # hazards: (H,3) cx,cy,r        hoyos/coladeras (NO bloquean rayos)
        self.hazards = np.asarray(hazards, dtype=np.float64).reshape(-1, 3)
        # movers: (D,5) cx,cy,r,vx,vy   personas/móviles (bloquean rayos)
        self.movers = np.asarray(movers, dtype=np.float64).reshape(-1, 5)
        self.bounds = bounds            # (xmin, ymin, xmax, ymax)

    # ---------------- ray casting vectorizado ----------------
    def ray_distances(self, x, y, angles, max_range=MAX_RANGE):
        """Distancia al primer choque (pared/obstáculo/móvil) para cada
        ángulo. Vectorizado: (N rayos) x (M segmentos + círculos)."""
        angles = np.asarray(angles, dtype=np.float64)
        c, s = np.cos(angles), np.sin(angles)          # (N,)
        best = np.full(angles.shape, max_range)

        if len(self.segments):
            ax, ay = self.segments[:, 0], self.segments[:, 1]
            ex = self.segments[:, 2] - ax
            ey = self.segments[:, 3] - ay
            denom = c[:, None] * ey[None, :] - s[:, None] * ex[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax[None, :] - x) * ey[None, :]
                     - (ay[None, :] - y) * ex[None, :]) / denom
                u = ((x - ax[None, :]) * s[:, None]
                     - (y - ay[None, :]) * c[:, None]) / -denom
            ok = (np.abs(denom) > 1e-12) & (t > 1e-9) & (u >= 0.0) & (u <= 1.0)
            t = np.where(ok, t, np.inf)
            best = np.minimum(best, t.min(axis=1, initial=np.inf))

        circles = [self.statics[:, :3]]
        if len(self.movers):
            circles.append(self.movers[:, :3])
        circ = np.concatenate(circles, axis=0) if circles else None
        if circ is not None and len(circ):
            fx = x - circ[:, 0]
            fy = y - circ[:, 1]
            b = c[:, None] * fx[None, :] + s[:, None] * fy[None, :]
            cc = (fx * fx + fy * fy - circ[:, 2] ** 2)[None, :]
            disc = b * b - cc
            with np.errstate(invalid="ignore"):
                t = -b - np.sqrt(np.maximum(disc, 0.0))
            ok = (disc > 0.0) & (t > 1e-9)
            t = np.where(ok, t, np.inf)
            best = np.minimum(best, t.min(axis=1, initial=np.inf))

        return np.minimum(best, max_range)

    # ---------------- consultas puntuales ----------------
    def clearance_at(self, x, y):
        """Distancia libre mínima a paredes/obstáculos/móviles."""
        d = np.inf
        if len(self.segments):
            ax, ay = self.segments[:, 0], self.segments[:, 1]
            bx, by = self.segments[:, 2], self.segments[:, 3]
            ex, ey = bx - ax, by - ay
            ll = ex * ex + ey * ey
            t = np.clip(((x - ax) * ex + (y - ay) * ey) / np.maximum(ll, 1e-12),
                        0.0, 1.0)
            px, py = ax + t * ex, ay + t * ey
            d = min(d, float(np.min(np.hypot(px - x, py - y))))
        for circ in (self.statics, self.movers[:, :3] if len(self.movers)
                     else np.zeros((0, 3))):
            if len(circ):
                dd = np.hypot(circ[:, 0] - x, circ[:, 1] - y) - circ[:, 2]
                d = min(d, float(np.min(dd)))
        return d

    def on_hazard(self, x, y, margin=0.05):
        if not len(self.hazards):
            return False
        dd = np.hypot(self.hazards[:, 0] - x, self.hazards[:, 1] - y)
        return bool(np.any(dd < self.hazards[:, 2] + margin))

    def hazard_sector_distances(self, x, y, theta, max_d=RISK_RANGE):
        """Distancia al peligro de piso más cercano por sector
        (front/left/right); max_d si no hay. Los peligros NO aparecen en
        los rayos (no sobresalen), igual que en el sistema real."""
        out = [max_d, max_d, max_d]
        for hx, hy, hr in self.hazards:
            d = math.hypot(hx - x, hy - y) - hr
            if d > max_d:
                continue
            bearing = _wrap(math.atan2(hy - y, hx - x) - theta)
            idx = _sector_index(bearing)
            if idx is not None:
                out[idx] = min(out[idx], max(0.0, d))
        return out

    def mover_threat(self, x, y, theta):
        """(riesgo_por_sector[3], amenaza_frontal_cercana: bool).
        Un móvil dentro de RISK_RANGE pone riesgo 3 en su sector; si está
        a <1.2 m al frente, detenerse es la acción correcta."""
        risks = [0, 0, 0]
        close_front = False
        for mx, my, mr, _vx, _vy in self.movers:
            d = math.hypot(mx - x, my - y) - mr
            if d > RISK_RANGE:
                continue
            bearing = _wrap(math.atan2(my - y, mx - x) - theta)
            idx = _sector_index(bearing)
            if idx is not None:
                risks[idx] = 3
                if idx == 0 and d < 1.2:
                    close_front = True
        return risks, close_front

    def step_movers(self, rng):
        """Los móviles avanzan y rebotan dentro de los límites."""
        if not len(self.movers):
            return
        xmin, ymin, xmax, ymax = self.bounds
        self.movers[:, 0] += self.movers[:, 3]
        self.movers[:, 1] += self.movers[:, 4]
        for m in self.movers:
            if m[0] < xmin + m[2] or m[0] > xmax - m[2]:
                m[3] = -m[3]
            if m[1] < ymin + m[2] or m[1] > ymax - m[2]:
                m[4] = -m[4]
            if rng.random() < 0.02:      # cambio de rumbo ocasional
                ang = rng.uniform(0, 2 * math.pi)
                sp = math.hypot(m[3], m[4])
                m[3], m[4] = sp * math.cos(ang), sp * math.sin(ang)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sector_index(bearing):
    """0=front, 1=left, 2=right, None=fuera (igual que navigation_feed)."""
    if abs(bearing) <= FRONT_HALF:
        return 0
    if FRONT_HALF < bearing <= SIDE_MAX:
        return 1
    if -SIDE_MAX <= bearing < -FRONT_HALF:
        return 2
    return None


# ----------------------------------------------------------------------
# Generadores procedurales de mundos
# ----------------------------------------------------------------------
def _rect_walls(x0, y0, x1, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1),
            (x1, y1, x0, y1), (x0, y1, x0, y0)]


def _scatter_circles(rng, n, bounds, r_lo, r_hi, keepout, min_gap=0.9):
    """Círculos aleatorios que no se enciman con keepout ni entre sí."""
    xmin, ymin, xmax, ymax = bounds
    out = []
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        r = rng.uniform(r_lo, r_hi)
        cx = rng.uniform(xmin + r + 0.3, xmax - r - 0.3)
        cy = rng.uniform(ymin + r + 0.3, ymax - r - 0.3)
        if any(math.hypot(cx - kx, cy - ky) < kr + r + min_gap
               for kx, ky, kr in keepout):
            continue
        if any(math.hypot(cx - ox, cy - oy) < orr + r + 0.4
               for ox, oy, orr in out):
            continue
        out.append((cx, cy, r))
    return out


def gen_room(rng):
    """Cuarto rectangular con muebles, quizá un peligro y un móvil."""
    w = rng.uniform(4.0, 9.0)
    h = rng.uniform(3.5, 7.0)
    bounds = (0.0, 0.0, w, h)
    walls = _rect_walls(*bounds)
    start, goal = _sample_start_goal(rng, bounds, [])
    keep = [(start[0], start[1], 0.4), (goal[0], goal[1], 0.5)]
    statics = _scatter_circles(rng, rng.integers(2, 6), bounds, 0.15, 0.45, keep)
    keep += statics
    hazards = _scatter_circles(rng, rng.integers(0, 2), bounds, 0.15, 0.35, keep)
    movers = _make_movers(rng, rng.integers(0, 2), bounds, keep + hazards)
    return NavWorld(walls, statics, hazards, movers, bounds), start, goal


def gen_two_rooms(rng):
    """Dos cuartos unidos por una puerta: obliga a rodear la pared."""
    w1 = rng.uniform(3.5, 6.0)
    w2 = rng.uniform(3.5, 6.0)
    h = rng.uniform(3.5, 6.0)
    w = w1 + w2
    bounds = (0.0, 0.0, w, h)
    walls = _rect_walls(*bounds)
    door_y = rng.uniform(0.9, h - 0.9)
    door_half = rng.uniform(0.45, 0.65)
    # pared divisoria con hueco (la puerta)
    walls.append((w1, 0.0, w1, max(0.0, door_y - door_half)))
    walls.append((w1, min(h, door_y + door_half), w1, h))

    start = (rng.uniform(0.6, w1 - 0.6), rng.uniform(0.6, h - 0.6),
             rng.uniform(-math.pi, math.pi))
    goal = (rng.uniform(w1 + 0.6, w - 0.6), rng.uniform(0.6, h - 0.6))
    keep = [(start[0], start[1], 0.4), (goal[0], goal[1], 0.5),
            (w1, door_y, 0.8)]                     # despejar la puerta
    statics = _scatter_circles(rng, rng.integers(1, 5), bounds, 0.15, 0.40, keep)
    keep += statics
    hazards = _scatter_circles(rng, rng.integers(0, 2), bounds, 0.15, 0.30, keep)
    movers = _make_movers(rng, rng.integers(0, 2), bounds, keep + hazards)
    return NavWorld(walls, statics, hazards, movers, bounds), start, goal


def gen_outdoor(rng):
    """Exterior abierto: postes/autos dispersos y coladeras/hoyos."""
    w = rng.uniform(10.0, 18.0)
    h = rng.uniform(8.0, 14.0)
    bounds = (0.0, 0.0, w, h)
    walls = _rect_walls(*bounds)                 # perímetro lejano
    start, goal = _sample_start_goal(rng, bounds, [], min_dist=5.0)
    keep = [(start[0], start[1], 0.4), (goal[0], goal[1], 0.5)]
    statics = _scatter_circles(rng, rng.integers(4, 10), bounds, 0.15, 0.6, keep)
    keep += statics
    hazards = _scatter_circles(rng, rng.integers(1, 4), bounds, 0.20, 0.45, keep)
    movers = _make_movers(rng, rng.integers(0, 3), bounds, keep + hazards)
    return NavWorld(walls, statics, hazards, movers, bounds), start, goal


def _sample_start_goal(rng, bounds, keepout, min_dist=2.0):
    xmin, ymin, xmax, ymax = bounds
    for _ in range(100):
        sx = rng.uniform(xmin + 0.6, xmax - 0.6)
        sy = rng.uniform(ymin + 0.6, ymax - 0.6)
        gx = rng.uniform(xmin + 0.6, xmax - 0.6)
        gy = rng.uniform(ymin + 0.6, ymax - 0.6)
        if math.hypot(gx - sx, gy - sy) >= min_dist:
            return (sx, sy, rng.uniform(-math.pi, math.pi)), (gx, gy)
    return ((xmin + 0.7, ymin + 0.7, 0.0),
            (xmax - 0.7, ymax - 0.7))


def _make_movers(rng, n, bounds, keepout):
    movers = []
    for (cx, cy, r) in _scatter_circles(rng, n, bounds, 0.25, 0.30, keepout):
        sp = rng.uniform(0.05, 0.12)
        ang = rng.uniform(0, 2 * math.pi)
        movers.append((cx, cy, r, sp * math.cos(ang), sp * math.sin(ang)))
    return movers


WORLD_GENERATORS = (gen_room, gen_two_rooms, gen_outdoor)
WORLD_WEIGHTS = (0.4, 0.35, 0.25)


# ----------------------------------------------------------------------
# El entorno Gymnasium
# ----------------------------------------------------------------------
# Rayos del modo extendido: 11 direcciones de -100° a +100°
EXT_RAY_ANGLES = np.radians(np.linspace(-100, 100, 11))
# Rayos con los que se derivan d_front/left/right (como navigation_feed)
DOC_RAY_ANGLES = np.radians(np.array(
    [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]))


class NavEnv(gym.Env):
    """obs_mode:
        'doc'      -> vector de 8 (el documentado, §7.3.1)
        'extended' -> 24: 11 rayos + 3 dist. de peligro por sector +
                      3 riesgos + goal (sin, cos, dist) + acción previa
                      one-hot(4)
    """
    metadata = {"render_modes": []}

    ACTIONS = ("avanzar", "girar_izquierda", "girar_derecha", "detenerse")

    def __init__(self, obs_mode="extended", seed=None, sensor_noise=0.05,
                 max_steps=EPISODE_STEPS):
        super().__init__()
        assert obs_mode in ("doc", "extended")
        self.obs_mode = obs_mode
        self.sensor_noise = sensor_noise
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        dim = 8 if obs_mode == "doc" else 24
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(dim,),
                                            dtype=np.float32)
        self.world = None
        self.pose = None
        self.goal = None
        self.prev_d_goal = None
        self.prev_action = 3
        self.steps = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        gen = self.rng.choice(len(WORLD_GENERATORS), p=WORLD_WEIGHTS)
        self.world, start, goal = WORLD_GENERATORS[gen](self.rng)
        self.pose = list(start)
        self.goal = goal
        self.prev_d_goal = self._d_goal()
        self.prev_action = 3
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        self.steps += 1
        x, y, theta = self.pose
        reward = REW_STEP
        terminated = False
        info = {}

        risks, close_front = self.world.mover_threat(x, y, theta)

        if action == 0:                                   # avanzar
            nx = x + FORWARD_STEP * math.cos(theta)
            ny = y + FORWARD_STEP * math.sin(theta)
            # colisión: chequear el punto medio y el final del paso
            mid = (x + nx) / 2.0, (y + ny) / 2.0
            if (self.world.clearance_at(nx, ny) < CLEARANCE
                    or self.world.clearance_at(*mid) < CLEARANCE):
                reward += REW_COLLISION
                terminated = True
                info["collision"] = True
            elif self.world.on_hazard(nx, ny) or self.world.on_hazard(*mid):
                reward += REW_COLLISION                    # caída en hoyo
                terminated = True
                info["fell"] = True
            else:
                self.pose[0], self.pose[1] = nx, ny
        elif action == 1:                                 # girar izquierda
            self.pose[2] = _wrap(theta + TURN_STEP)
        elif action == 2:                                 # girar derecha
            self.pose[2] = _wrap(theta - TURN_STEP)
        else:                                             # detenerse
            reward += REW_STOP_GOOD if close_front else REW_STOP_BAD

        # progreso hacia el goal
        d_goal = self._d_goal()
        if not terminated:
            reward += REW_PROGRESS * (self.prev_d_goal - d_goal)
        self.prev_d_goal = d_goal

        # proximidad a riesgo alto (frontal pesa)
        risks2, _ = self.world.mover_threat(*self.pose)
        hz = self.world.hazard_sector_distances(*self.pose)
        risk_front = max(risks2[0], 3 if hz[0] < 1.0 else 0)
        reward += REW_RISK * risk_front

        if d_goal < GOAL_RADIUS and not terminated:
            reward += REW_GOAL
            terminated = True
            info["success"] = True

        self.world.step_movers(self.rng)
        # un móvil puede alcanzar al usuario aunque esté quieto
        if not terminated and self.world.clearance_at(
                self.pose[0], self.pose[1]) < CLEARANCE * 0.6:
            reward += REW_COLLISION
            terminated = True
            info["collision"] = True

        truncated = self.steps >= self.max_steps
        self.prev_action = action
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _d_goal(self):
        return math.hypot(self.goal[0] - self.pose[0],
                          self.goal[1] - self.pose[1])

    def _phi_goal(self):
        return _wrap(math.atan2(self.goal[1] - self.pose[1],
                                self.goal[0] - self.pose[0]) - self.pose[2])

    def _noisy(self, d):
        if self.sensor_noise <= 0:
            return d
        return d * (1.0 + self.rng.normal(0.0, self.sensor_noise,
                                          size=np.shape(d)))

    def _sector_min(self, angles, dists):
        """d_front/left/right = mínimo de los rayos de cada sector."""
        front = dists[np.abs(angles) <= FRONT_HALF + 1e-9].min()
        left = dists[(angles > FRONT_HALF) & (angles <= SIDE_MAX)].min()
        right = dists[(angles < -FRONT_HALF) & (angles >= -SIDE_MAX)].min()
        return front, left, right

    def _risks(self):
        x, y, theta = self.pose
        risks, _ = self.world.mover_threat(x, y, theta)
        hz = self.world.hazard_sector_distances(x, y, theta)
        rays = self.world.ray_distances(x, y, theta + DOC_RAY_ANGLES)
        d_f, d_l, d_r = self._sector_min(DOC_RAY_ANGLES, rays)
        out = []
        for i, d_static in enumerate((d_f, d_l, d_r)):
            r = risks[i]
            if hz[i] < RISK_RANGE:
                r = 3                          # peligro de piso: alto
            elif r == 0 and d_static < 1.5:
                r = 2                          # obstáculo estático cerca
            elif r == 0 and d_static < RISK_RANGE:
                r = 1
            out.append(r)
        return out, (d_f, d_l, d_r)

    def _obs(self):
        x, y, theta = self.pose
        phi = self._phi_goal() + self.rng.normal(0.0, math.radians(2))
        d_goal = float(self._noisy(self._d_goal()))
        risks, (d_f, d_l, d_r) = self._risks()
        if self.obs_mode == "doc":
            d_f, d_l, d_r = self._noisy(np.array([d_f, d_l, d_r]))
            obs = np.array([
                d_f / MAX_RANGE, d_l / MAX_RANGE, d_r / MAX_RANGE,
                phi / math.pi, min(d_goal, 10.0) / 10.0,
                risks[0] / 3.0, risks[1] / 3.0, risks[2] / 3.0,
            ], dtype=np.float32)
        else:
            rays = self.world.ray_distances(x, y, theta + EXT_RAY_ANGLES)
            rays = np.clip(self._noisy(rays), 0.0, MAX_RANGE)
            # Randomización de dominio: en producción los rayos se miden
            # sobre el MAPA pintado (celdas de eco dispersas), donde un
            # rayo puede "colarse" entre dos celdas ocupadas y reportar
            # libre un frente que no lo está. Se simula soltando algunos
            # rayos bloqueados a max_range para que la política no
            # confíe ciegamente en un hueco de un solo rayo.
            if self.sensor_noise > 0:
                drop = self.rng.random(rays.shape) < 0.08
                rays = np.where(drop & (rays < MAX_RANGE), MAX_RANGE, rays)
            hz = self.world.hazard_sector_distances(x, y, theta)
            onehot = np.zeros(4, dtype=np.float32)
            onehot[self.prev_action] = 1.0
            obs = np.concatenate([
                rays / MAX_RANGE,
                np.array(hz) / RISK_RANGE,
                np.array(risks) / 3.0,
                [math.sin(phi), math.cos(phi), min(d_goal, 10.0) / 10.0],
                onehot,
            ]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)


def make_env(obs_mode="extended", seed=None, **kw):
    def _thunk():
        return NavEnv(obs_mode=obs_mode, seed=seed, **kw)
    return _thunk
