"""
occupancy_grid.py
mapa de ocupacion local (log-odds) + ray casting, mas una capa aparte
de peligros de piso (hoyos, coladeras: no bloquean el rayo pero no se
pueden pisar). el slam nunca escribe el mapa directo, solo manda pose
y observaciones. map_manager reutiliza este mismo nucleo pa chunks.
"""
import math
import numpy as np

# params log-odds compartidos con los chunks del ChunkedMapManager.
# ojo: L_OCC=1.2 es a proposito, un solo eco claro marca ocupado (mas
# seguro sobre-detectar que sub-detectar). validado en simulacion,
# no bajar sin re-medir con simulation.py en varias semillas
L_OCC = 1.2
L_FREE = -0.4
L_CLAMP = 5.0
# umbral pa considerar una celda "ocupada" al consultar
OCC_THRESHOLD = 1.0
# pared solida (>=2 ecos): un rayo NO despeja celdas mas alla de una celda
# asi, ni marca su endpoint si quedo bloqueado. sin esto los rangos
# monoculares sobreestimados perforan tuneles en paredes ya mapeadas
# y pintan fantasmas detras. no mover sin re-medir
L_BLOCK = 2.0
# hits pa activar una celda de peligro (un falso positivo de 1 frame
# no debe volverse un "hoyo" permanente)
HAZARD_MIN_HITS = 2
HAZARD_MAX_HITS = 10


def bresenham(x0, y0, x1, y1):
    """celdas que cruza el segmento, pura aritmetica entera."""
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
        self.hazard_hits = np.zeros((self.n, self.n), dtype=np.uint8)
        # origen del mundo en la grilla (pa coordenadas negativas)
        self.origin = origin if origin is not None else (size_m / 2.0, size_m / 2.0)
        self.l_occ = L_OCC
        self.l_free = L_FREE
        self.l_clamp = L_CLAMP

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

    # ---------------- ray casting ----------------
    # compat con codigo viejo que usaba el metodo estatico
    _bresenham = staticmethod(bresenham)

    def _cells(self):
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
        """ray casting: marca libre a lo largo del rayo y ocupado al final.
        si el final cae fuera, el rayo se recorta en el borde (lo de adentro
        si cuenta). un rango impreciso (sigma_r alto, monocular ~30%) solo
        despeja hasta r - 2*sigma y no pinta ocupado, pq si no borra paredes
        que el ultrasonido ya mapeo bien. is_hazard marca la capa de
        peligros en vez del log-odds."""
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

            # recorte en el borde del mapa
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
                    blocked = True     # pared solida: no perforar
                    break
                self._add_log_odds(cx, cy, self.l_free)

            if end_inside and not blocked and obs.range_m <= max_range:
                if is_hazard:
                    # la capa de peligros se confirma sola con HAZARD_MIN_HITS
                    self._add_hazard(gx1, gy1)
                elif not imprecise:
                    # solo rangos precisos pintan ocupado; el monocular crudo
                    # salpicaria por todo el eje de profundidad, su objeto lo
                    # pinta el landmark consolidado via observations_for_map()
                    self._add_log_odds(gx1, gy1, self.l_occ)

    # ---------------- consultas ----------------
    def is_occupied(self, gx, gy, threshold=OCC_THRESHOLD):
        return self.in_bounds(gx, gy) and self.log_odds[gy, gx] > threshold

    def is_hazard(self, gx, gy):
        return (self.in_bounds(gx, gy)
                and self.hazard_hits[gy, gx] >= HAZARD_MIN_HITS)

    def is_blocked(self, gx, gy):
        return self.is_occupied(gx, gy) or self.is_hazard(gx, gy)

    def cast_distance(self, x, y, angle, max_range=4.0):
        """ray casting de consulta (no escribe): distancia a la primera
        celda bloqueada, max_range si esta libre. de aqui salen
        d_front/d_left/d_right de navegacion."""
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
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def local_costmap(self, pose, size_m=4.0):
        """parche egocentrico (2, H, W) float32, ejes del mundo.
        canal 0 = prob ocupacion (0.5 = desconocido), canal 1 = peligros.
        lo que cae fuera del mapa se rellena como desconocido."""
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

    def get_nearby_obstacles(self, x, y, radius=1.5, threshold=OCC_THRESHOLD):
        """celdas ocupadas (en mundo) cerca de (x,y)."""
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

    # ---------------- croquis ----------------
    def render_croquis(self, pose=None, trajectory=None, landmarks=None,
                       gt_trajectory=None, path="croquis_mapa.png"):
        """imagen tipo croquis del mapa con la pose del usuario."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        occ_prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [-self.origin[0], self.n * self.resolution - self.origin[0],
                  -self.origin[1], self.n * self.resolution - self.origin[1]]
        ax.imshow(occ_prob, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1)

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
