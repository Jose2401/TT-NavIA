"""
occupancy_grid.py
Mapa de ocupación local + Ray Casting.

Responsabilidades de Ray Casting (y SOLO de Ray Casting, sin mezclarse con
el SLAM):
- proyectar rayos desde la pose estimada;
- determinar qué celdas son libres a lo largo del rayo;
- marcar como ocupada la celda donde el rayo encuentra un obstáculo;
- actualizar el mapa de ocupación (log-odds).

El SLAM nunca modifica el mapa directamente: solo le entrega la pose
estimada y las observaciones, y opcionalmente consulta landmarks/celdas
ocupadas cercanas para la asociación de datos.
"""
import math
import numpy as np


class OccupancyGrid:
    def __init__(self, size_m=8.0, resolution=0.05, origin=None):
        self.resolution = resolution
        self.n = int(size_m / resolution)
        # log-odds: 0 = desconocido, + = ocupado, - = libre
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        # origen del mundo dentro de la grilla (para permitir coordenadas negativas)
        self.origin = origin if origin is not None else (size_m / 2.0, size_m / 2.0)
        # Un solo eco claro basta para marcar la celda como ocupada (sesgo
        # conservador: para el usuario es más seguro sobre-detectar
        # obstáculos que sub-detectarlos).
        self.l_occ = 1.2
        self.l_free = -0.4
        self.l_clamp = 5.0

    # Utilidades de coordenadas 
    def world_to_grid(self, x, y):
        gx = int((x + self.origin[0]) / self.resolution)
        gy = int((y + self.origin[1]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.resolution - self.origin[0]
        y = gy * self.resolution - self.origin[1]
        return x, y

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.n and 0 <= gy < self.n

    # ---------------- Ray Casting ----------------
    @staticmethod
    def _bresenham(x0, y0, x1, y1):
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

    def update_from_scan(self, pose, observations, max_range=4.0):
        """Ray Casting: para cada observación (rango, bearing) relativa al
        usuario, marca las celdas libres a lo largo del rayo y la celda
        ocupada en el punto final (si el rango indica un eco real).

        La incertidumbre de rango de la observación (sigma_r, si viene)
        modula la actualización: un rango impreciso (p. ej. monocular,
        ~30% de error) solo despeja "libre" hasta donde es casi seguro que
        no hay nada (r - 2*sigma_r) y marca la celda final con menos peso.
        Sin esto, los rayos monoculares largos atraviesan y borran paredes
        que el ultrasonido ya había mapeado bien."""
        x, y, theta = pose
        gx0, gy0 = self.world_to_grid(x, y)
        if not self.in_bounds(gx0, gy0):
            return
        for obs in observations:
            r = min(obs.range_m, max_range)
            sigma = getattr(obs, "sigma_r", None)
            imprecise = sigma is not None and sigma > 0.15

            angle = theta + obs.bearing
            ex = x + r * math.cos(angle)
            ey = y + r * math.sin(angle)
            gx1, gy1 = self.world_to_grid(ex, ey)
            if not self.in_bounds(gx1, gy1):
                continue
            ray = self._bresenham(gx0, gy0, gx1, gy1)

            free_cells = ray[:-1]
            if imprecise and r > 1e-6:
                frac = max(0.0, (r - 2.0 * sigma) / r)
                free_cells = ray[:int(len(ray) * frac)]
            for (cx, cy) in free_cells:
                if self.in_bounds(cx, cy):
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + self.l_free, -self.l_clamp, self.l_clamp)

            if obs.range_m <= max_range:
                l_occ = self.l_occ * (0.4 if imprecise else 1.0)
                self.log_odds[gy1, gx1] = np.clip(
                    self.log_odds[gy1, gx1] + l_occ, -self.l_clamp, self.l_clamp)

    #Asociación de datos del SLAM
    def get_nearby_obstacles(self, x, y, radius=1.5, threshold=1.0):
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
