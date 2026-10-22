import math
import numpy as np

L_OCC = 1.2
L_FREE = -0.4
L_CLAMP = 5.0
OCC_THRESHOLD = 1.0
L_BLOCK = 2.0
HAZARD_MIN_HITS = 2
HAZARD_MAX_HITS = 10


# celdas por donde pasa el segmento
def bresenham(x0, y0, x1, y1):
    puntos = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy =1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        puntos.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err+= dx
            y += sy
    return puntos


class OccupancyGrid:
    def __init__(self, sizeM=8.0, resolution=0.05, origin=None):
        self.resolution = resolution
        self.n = int(sizeM / resolution)
        self.logOdds = np.zeros((self.n, self.n), dtype=np.float32)
        self.hazardHits =np.zeros((self.n, self.n), dtype=np.uint8)
        self.origin = origin if origin is not None else (sizeM / 2.0, sizeM / 2.0)
        self.lOcc = L_OCC
        self.lFree= L_FREE
        self.lClamp = L_CLAMP

    def worldToGrid(self, x, y):
        gx = int(math.floor((x + self.origin[0]) / self.resolution))
        gy= int(math.floor((y + self.origin[1]) / self.resolution))
        return gx, gy

    def gridToWorld(self, gx, gy):
        x = gx * self.resolution - self.origin[0]
        y = gy * self.resolution - self.origin[1]
        return x, y

    def inBounds(self, gx, gy):
        return 0 <= gx < self.n and 0 <= gy < self.n

    def _addLogOdds(self, gx, gy, delta):
        if self.inBounds(gx, gy):
            self.logOdds[gy, gx] = np.clip(
                self.logOdds[gy, gx] + delta, -self.lClamp, self.lClamp)

    def _addHazard(self, gx, gy):
        if self.inBounds(gx, gy):
            self.hazardHits[gy, gx] = min(HAZARD_MAX_HITS,
                                          int(self.hazardHits[gy, gx]) + 1)

    # ray casting, libre en el rayo y ocupado al final
    def updateFromScan(self, pose, observations, maxRange=4.0):
        x, y, theta = pose
        gx0, gy0 = self.worldToGrid(x, y)
        if not self.inBounds(gx0, gy0):
            return
        for obs in observations:
            r = min(obs.range_m, maxRange)
            sigma = getattr(obs, "sigma_r", None)
            impreciso = sigma is not None and sigma > 0.15
            esHoyo= getattr(obs, "is_hazard", False)

            ang = theta + obs.bearing
            ex = x + r * math.cos(ang)
            ey =y + r * math.sin(ang)
            gx1, gy1 = self.worldToGrid(ex, ey)
            papuRay = bresenham(gx0, gy0, gx1, gy1)

            recortado = []
            for celda in papuRay:
                if not self.inBounds(*celda):
                    break
                recortado.append(celda)
            finAdentro = bool(recortado) and recortado[-1] == (gx1, gy1)

            celdasLibres = recortado[:-1] if finAdentro else recortado
            if impreciso and r > 1e-6:
                frac = max(0.0, (r - 2.0 * sigma) / r)
                celdasLibres = celdasLibres[:int(len(recortado) * frac)]

            bloqueado = False
            for (cx, cy) in celdasLibres:
                if self.logOdds[cy, cx] > L_BLOCK:
                    bloqueado = True
                    break
                self._addLogOdds(cx, cy, self.lFree)

            if finAdentro and not bloqueado and obs.range_m <= maxRange:
                if esHoyo:
                    self._addHazard(gx1, gy1)
                elif not impreciso:
                    self._addLogOdds(gx1, gy1, self.lOcc)

    def isOccupied(self, gx, gy, threshold=OCC_THRESHOLD):
        return self.inBounds(gx, gy) and self.logOdds[gy, gx] > threshold

    def isHazard(self, gx, gy):
        return (self.inBounds(gx, gy)
                and self.hazardHits[gy, gx] >= HAZARD_MIN_HITS)

    def isBlocked(self, gx, gy):
        return self.isOccupied(gx, gy) or self.isHazard(gx, gy)

    # distancia a lo primero bloqueado
    def castDistance(self, x, y, angle, maxRange=4.0):
        gx0, gy0 = self.worldToGrid(x, y)
        ex = x + maxRange * math.cos(angle)
        ey= y + maxRange * math.sin(angle)
        gx1, gy1 = self.worldToGrid(ex, ey)
        for (cx, cy) in bresenham(gx0, gy0, gx1, gy1):
            if not self.inBounds(cx, cy):
                break
            if (cx, cy) != (gx0, gy0) and self.isBlocked(cx, cy):
                wx, wy = self.gridToWorld(cx, cy)
                wx += self.resolution / 2.0
                wy +=self.resolution / 2.0
                return min(maxRange, math.hypot(wx - x, wy - y))
        return maxRange

    def occupancyProbability(self):
        return 1.0 / (1.0 + np.exp(-self.logOdds))

    def localCostmap(self, pose, sizeM=4.0):
        x, y = pose[0], pose[1]
        half = int(round((sizeM / 2.0) / self.resolution))
        lado = 2 * half + 1
        out = np.zeros((2, lado, lado), dtype=np.float32)
        out[0].fill(0.5)

        gx, gy = self.worldToGrid(x, y)
        x0, x1 = gx - half, gx + half + 1
        y0, y1 = gy - half, gy + half + 1
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(self.n, x1), min(self.n, y1)
        if sx0 >= sx1 or sy0 >= sy1:
            return out
        dstX0, dstY0 = sx0 - x0, sy0 - y0
        occ = 1.0 / (1.0 + np.exp(-self.logOdds[sy0:sy1, sx0:sx1]))
        out[0, dstY0:dstY0 + (sy1 - sy0), dstX0:dstX0 + (sx1 - sx0)] = occ
        haz =(self.hazardHits[sy0:sy1, sx0:sx1] >= HAZARD_MIN_HITS)
        out[1, dstY0:dstY0 + (sy1 - sy0),
            dstX0:dstX0 + (sx1 - sx0)] = haz.astype(np.float32)
        return out

    def getNearbyObstacles(self, x, y, radius=1.5, threshold=OCC_THRESHOLD):
        gx, gy = self.worldToGrid(x, y)
        rCeldas = int(radius / self.resolution)
        obstaculos = []
        y0, y1 = max(0, gy - rCeldas), min(self.n, gy + rCeldas)
        x0, x1 = max(0, gx - rCeldas), min(self.n, gx + rCeldas)
        sub = self.logOdds[y0:y1, x0:x1]
        ys, xs = np.where(sub > threshold)
        for yy, xx in zip(ys, xs):
            wx, wy = self.gridToWorld(x0 + xx, y0 + yy)
            obstaculos.append((wx, wy))
        return obstaculos

    def renderCroquis(self, pose=None, trajectory=None, landmarks=None,
                      gtTrajectory=None, path="croquis_mapa.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        occProb = 1.0 - 1.0 / (1.0 + np.exp(self.logOdds))
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [-self.origin[0], self.n * self.resolution - self.origin[0],
                  -self.origin[1], self.n * self.resolution - self.origin[1]]
        ax.imshow(occProb, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1)

        hz = np.argwhere(self.hazardHits >= HAZARD_MIN_HITS)
        if len(hz):
            hx = [c * self.resolution - self.origin[0] for c in hz[:, 1]]
            hy =[c * self.resolution - self.origin[1] for c in hz[:, 0]]
            ax.plot(hx, hy, "s", color="tab:red", markersize=3,
                    label="Peligros de piso")

        if gtTrajectory:
            gx = [p[0] for p in gtTrajectory]
            gy = [p[1] for p in gtTrajectory]
            ax.plot(gx, gy, "--", color="tab:green", linewidth=1.2,
                    label="Trayectoria real")

        if trajectory:
            tx= [p[0] for p in trajectory]
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
