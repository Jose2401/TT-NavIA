"""
occupancy_grid.py
Mapa de ocupación local + Ray Casting.

Responsabilidades de Ray Casting (y SOLO de Ray Casting, sin mezclarse con
el SLAM):
- proyectar rayos desde la pose estimada;
- determinar qué celdas son libres a lo largo del rayo;
- marcar como ocupada la celda donde el rayo encuentra un obstáculo;
- actualizar el mapa de ocupación (log-odds);
- mantener una CAPA DE PELIGROS de piso separada (hoyos, coladeras, agua):
  un hoyo no bloquea el rayo como una pared (visual o ultrasónicamente el
  "espacio" continúa), pero la navegación debe tratar su celda como
  intransitable. Por eso vive en una capa aparte y no en el log-odds.

El SLAM nunca modifica el mapa directamente: solo le entrega la pose
estimada y las observaciones, y opcionalmente consulta landmarks/celdas
ocupadas cercanas para la asociación de datos.

Este mismo núcleo de Ray Casting lo reutiliza map_manager.ChunkedMapManager
(mapa global por chunks con persistencia); OccupancyGrid queda como el mapa
"de un solo bloque" para simulaciones, pruebas y demos.
"""
import math
import numpy as np

# Parámetros log-odds compartidos por OccupancyGrid y por los chunks del
# ChunkedMapManager (mismo significado de celda en ambos).
# Un solo eco claro basta para marcar la celda como ocupada (sesgo
# conservador: para el usuario es más seguro sobre-detectar obstáculos
# que sub-detectarlos). Decisión validada en simulación: no bajar l_occ
# sin re-medir con simulation.py sobre varias semillas.
L_OCC = 1.2
L_FREE = -0.4
L_CLAMP = 5.0
# Umbral log-odds para considerar una celda "ocupada" al navegar/consultar.
OCC_THRESHOLD = 1.0
# Umbral de "pared sólida" (≥2 ecos de evidencia): un rayo NO despeja
# celdas más allá de una celda así de ocupada, y si el rayo quedó
# bloqueado tampoco marca su endpoint. Sin esta guarda, los rangos
# monoculares sobreestimados (o un eco emparejado con el objeto
# equivocado) perforan túneles de "libre" a través de paredes ya bien
# mapeadas, las borran y pintan marcas fantasma detrás.
L_BLOCK = 2.0
# Avistamientos necesarios para que una celda de la capa de peligros se
# considere activa (un falso positivo de segmentación de 1 frame no debe
# volverse un "hoyo" permanente).
HAZARD_MIN_HITS = 2
HAZARD_MAX_HITS = 10


def bresenham(x0, y0, x1, y1):
    """Celdas enteras atravesadas por el segmento (x0,y0)->(x1,y1).
    Solo aritmética entera: es el paso interno del Ray Casting."""
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return points


class OccupancyGrid:
    def __init__(self, size_m=8.0, resolution=0.05, origin=None):
        self.resolution = resolution
        self.n = int(size_m / resolution)
        # log-odds: 0 = desconocido, + = ocupado, - = libre
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        # capa de peligros de piso: contador de avistamientos por celda
        self.hazard_hits = np.zeros((self.n, self.n), dtype=np.uint8)
        # origen del mundo dentro de la grilla (para permitir coordenadas negativas)
        self.origin = origin if origin is not None else (size_m / 2.0, size_m / 2.0)
        self.l_occ = L_OCC
        self.l_free = L_FREE
        self.l_clamp = L_CLAMP

    # Utilidades de coordenadas
    def world_to_grid(self, x, y):
        gx = int(math.floor((x + self.origin[0]) / self.resolution))
        gy = int(math.floor((y + self.origin[1]) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution - self.origin[0]
        y = gy * self.resolution - self.origin[1]
        return x, y

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.n and 0 <= gy < self.n

    # ---------------- Ray Casting ----------------
    # Compatibilidad con código previo que usaba el método estático.
    _bresenham = staticmethod(bresenham)

    def _cells(self):
        """Interfaz de celdas que comparte el núcleo de Ray Casting con
        ChunkedMapManager: (leer, escribir log-odds, marcar hazard)."""
        return self.log_odds, self.hazard_hits

    def _add_log_odds(self, gx, gy, delta):
        if self.in_bounds(gx, gy):
            self.log_odds[gy, gx] = np.clip(
                self.log_odds[gy, gx] + delta, -self.l_clamp, self.l_clamp)

    def _add_hazard(self, gx, gy):
        if self.in_bounds(gx, gy):
            self.hazard_hits[gy, gx] = min(HAZARD_MAX_HITS,
                                           int(self.hazard_hits[gy, gx]) + 1)

    def update_from_scan(self, pose, observations, max_range=4.0):
        """Ray Casting: para cada observación (rango, bearing) relativa al
        usuario, marca las celdas libres a lo largo del rayo y la celda
        ocupada en el punto final (si el rango indica un eco real).

        - Si el punto final cae fuera del mapa, el rayo se RECORTA en el
          borde y las celdas libres dentro del mapa sí se registran (antes
          se descartaba el rayo completo y se perdía esa información).
        - La incertidumbre de rango de la observación (sigma_r, si viene)
          modula la actualización: un rango impreciso (p. ej. monocular,
          ~30% de error) solo despeja "libre" hasta donde es casi seguro
          que no hay nada (r - 2*sigma_r) y marca la celda final con menos
          peso. Sin esto, los rayos monoculares largos atraviesan y borran
          paredes que el ultrasonido ya había mapeado bien.
        - Una observación con is_hazard=True marca su celda final en la
          capa de peligros (no en el log-odds): el rayo sigue despejando
          "libre" hasta ahí porque el suelo alrededor del hoyo es
          transitable, pero la celda del hoyo queda señalada."""
        x, y, theta = pose
        gx0, gy0 = self.world_to_grid(x, y)
        if not self.in_bounds(gx0, gy0):
            return
        for obs in observations:
            r = min(obs.range_m, max_range)
            sigma = getattr(obs, "sigma_r", None)
            imprecise = sigma is not None and sigma > 0.15
            is_hazard = getattr(obs, "is_hazard", False)

            angle = theta + obs.bearing
            ex = x + r * math.cos(angle)
            ey = y + r * math.sin(angle)
            gx1, gy1 = self.world_to_grid(ex, ey)
            ray = bresenham(gx0, gy0, gx1, gy1)

            # Recorte en el borde del mapa: se conservan las celdas
            # internas y se recuerda si el final quedó dentro.
            clipped = []
            for cell in ray:
                if not self.in_bounds(*cell):
                    break
                clipped.append(cell)
            end_inside = bool(clipped) and clipped[-1] == (gx1, gy1)

            free_cells = clipped[:-1] if end_inside else clipped
            if imprecise and r > 1e-6:
                frac = max(0.0, (r - 2.0 * sigma) / r)
                free_cells = free_cells[:int(len(clipped) * frac)]

            blocked = False
            for (cx, cy) in free_cells:
                if self.log_odds[cy, cx] > L_BLOCK:
                    blocked = True     # pared sólida: no perforar
                    break
                self._add_log_odds(cx, cy, self.l_free)

            if end_inside and not blocked and obs.range_m <= max_range:
                if is_hazard:
                    # La capa de peligros tiene su propia confirmación
                    # por avistamientos (HAZARD_MIN_HITS), así que la
                    # imprecisión del rango no la contamina.
                    self._add_hazard(gx1, gy1)
                elif not imprecise:
                    # Solo los rangos PRECISOS (ultrasonido, u
                    # observaciones ancladas al landmark consolidado)
                    # pintan celda ocupada. Un rango monocular crudo
                    # (~30% de error) salpicaría marcas a lo largo del
                    # eje de profundidad (incluso tras las paredes); su
                    # objeto lo pinta el landmark consolidado vía
                    # observations_for_map(), no el rayo crudo.
                    self._add_log_odds(gx1, gy1, self.l_occ)

    # ---------------- Consultas para SLAM / navegación ----------------
    def is_occupied(self, gx, gy, threshold=OCC_THRESHOLD):
        return self.in_bounds(gx, gy) and self.log_odds[gy, gx] > threshold

    def is_hazard(self, gx, gy):
        return (self.in_bounds(gx, gy)
                and self.hazard_hits[gy, gx] >= HAZARD_MIN_HITS)

    def is_blocked(self, gx, gy):
        """Celda intransitable para la navegación: ocupada O peligro de
        piso activo."""
        return self.is_occupied(gx, gy) or self.is_hazard(gx, gy)

    def cast_distance(self, x, y, angle, max_range=4.0):
        """Ray casting de CONSULTA (no escribe el mapa): distancia desde
        (x, y) en la dirección `angle` hasta la primera celda bloqueada
        (ocupada o peligro). Devuelve max_range si el camino está libre.
        Es la fuente de d_front/d_left/d_right del estado de navegación."""
        gx0, gy0 = self.world_to_grid(x, y)
        ex = x + max_range * math.cos(angle)
        ey = y + max_range * math.sin(angle)
        gx1, gy1 = self.world_to_grid(ex, ey)
        for (cx, cy) in bresenham(gx0, gy0, gx1, gy1):
            if not self.in_bounds(cx, cy):
                break
            if (cx, cy) != (gx0, gy0) and self.is_blocked(cx, cy):
                wx, wy = self.grid_to_world(cx, cy)
                # centro de la celda
                wx += self.resolution / 2.0
                wy += self.resolution / 2.0
                return min(max_range, math.hypot(wx - x, wy - y))
        return max_range

    def occupancy_probability(self):
        """Probabilidad de ocupación [0,1] de todo el mapa (0.5 =
        desconocido)."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def local_costmap(self, pose, size_m=4.0):
        """Parche egocéntrico (2, H, W) float32 centrado en la pose, ejes
        alineados al mundo (la orientación va aparte, en la pose):
            canal 0 = probabilidad de ocupación [0,1] (0.5 = desconocido)
            canal 1 = máscara de peligros de piso {0,1}
        Es la entrada de mapa que consumirá el módulo de navegación (DRL).
        Las celdas fuera del mapa se rellenan como desconocido (0.5)."""
        x, y = pose[0], pose[1]
        half = int(round((size_m / 2.0) / self.resolution))
        side = 2 * half + 1
        out = np.zeros((2, side, side), dtype=np.float32)
        out[0].fill(0.5)

        gx, gy = self.world_to_grid(x, y)
        x0, x1 = gx - half, gx + half + 1
        y0, y1 = gy - half, gy + half + 1
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.n, x1), min(self.n, y1)
        if sx0 >= sx1 or sy0 >= sy1:
            return out
        dst_x0, dst_y0 = sx0 - x0, sy0 - y0
        occ = 1.0 / (1.0 + np.exp(-self.log_odds[sy0:sy1, sx0:sx1]))
        out[0, dst_y0:dst_y0 + (sy1 - sy0), dst_x0:dst_x0 + (sx1 - sx0)] = occ
        haz = (self.hazard_hits[sy0:sy1, sx0:sx1] >= HAZARD_MIN_HITS)
        out[1, dst_y0:dst_y0 + (sy1 - sy0),
            dst_x0:dst_x0 + (sx1 - sx0)] = haz.astype(np.float32)
        return out

    #Asociación de datos del SLAM
    def get_nearby_obstacles(self, x, y, radius=1.5, threshold=OCC_THRESHOLD):
        """Devuelve celdas ocupadas (en coordenadas del mundo) cercanas a
        (x,y). Útil si se quiere alimentar al EKF con landmarks derivados
        directamente del mapa en lugar de las observaciones crudas."""
        gx, gy = self.world_to_grid(x, y)
        rcells = int(radius / self.resolution)
        obstacles = []
        y0, y1 = max(0, gy - rcells), min(self.n, gy + rcells)
        x0, x1 = max(0, gx - rcells), min(self.n, gx + rcells)
        sub = self.log_odds[y0:y1, x0:x1]
        ys, xs = np.where(sub > threshold)
        for yy, xx in zip(ys, xs):
            wx, wy = self.grid_to_world(x0 + xx, y0 + yy)
            obstacles.append((wx, wy))
        return obstacles

    # ---------------- visualización / "croquis" ----------------
    def render_croquis(self, pose=None, trajectory=None, landmarks=None,
                       gt_trajectory=None, path="croquis_mapa.png"):
        """Genera una imagen tipo croquis del mapa de ocupación construido,
        con la posición y orientación estimadas del usuario marcadas."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        occ_prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [-self.origin[0], self.n * self.resolution - self.origin[0],
                  -self.origin[1], self.n * self.resolution - self.origin[1]]
        ax.imshow(occ_prob, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1)

        # Peligros de piso (hoyos/coladeras) en rojo
        hz = np.argwhere(self.hazard_hits >= HAZARD_MIN_HITS)
        if len(hz):
            hx = [c * self.resolution - self.origin[0] for c in hz[:, 1]]
            hy = [c * self.resolution - self.origin[1] for c in hz[:, 0]]
            ax.plot(hx, hy, "s", color="tab:red", markersize=3,
                    label="Peligros de piso")

        if gt_trajectory:
            gx = [p[0] for p in gt_trajectory]
            gy = [p[1] for p in gt_trajectory]
            ax.plot(gx, gy, "--", color="tab:green", linewidth=1.2,
                    label="Trayectoria real")

        if trajectory:
            tx = [p[0] for p in trajectory]
            ty = [p[1] for p in trajectory]
            ax.plot(tx, ty, "-", color="tab:blue", linewidth=1.5,
                    label="Trayectoria estimada")

        if landmarks:
            lx = [p[0] for p in landmarks]
            ly = [p[1] for p in landmarks]
            ax.plot(lx, ly, "x", color="tab:orange", markersize=7,
                    label="Landmarks")

        if pose is not None:
            x, y, theta = pose
            ax.plot(x, y, "o", color="tab:red", markersize=10, label="Usuario")
            ax.arrow(x, y, 0.3 * math.cos(theta), 0.3 * math.sin(theta),
                      head_width=0.12, color="tab:red")

        ax.set_title("Croquis del mapa local (SLAM + EKF)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
