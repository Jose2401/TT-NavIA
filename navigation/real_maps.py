"""
real_maps.py — convierte mapas reales guardados (chunks del
ChunkedMapManager) en mundos de entrenamiento/evaluación pal agente.

las celdas ocupadas quedan como hileras de discos, la capa de peligros
como círculos hazard, y el inicio/destino se muestrean en el MISMO
componente conexo de espacio libre (así el episodio siempre tiene
solución). se agregan 0-2 móviles pq el mapa guardado solo trae lo
estático. la conversión es cara, ps se cachea por carpeta.
"""
import math
import os

import numpy as np

from navigation.nav_env import NavWorld, _scatter_circles

# umbrales en log-odds (ver occupancy_grid.py)
_L_OCC = 1.0
_L_FREE = -0.6
_MIN_CLEARANCE_M = 0.30      # margen al muestrear inicio/destino
_MIN_GOAL_DIST_M = 2.0

_CACHE = {}


class RealMapWorldFactory:
    """carga un mapa por chunks y fabrica episodios sobre él."""

    def __init__(self, map_dir, cell_stride=2):
        from map_manager import ChunkedMapManager
        from occupancy_grid import HAZARD_MIN_HITS
        import cv2

        if not os.path.isdir(map_dir):
            raise FileNotFoundError(f"No existe el mapa: {map_dir}")
        mgr = ChunkedMapManager.openExisting(map_dir)
        if not mgr.knownChunkIndices():
            raise ValueError(
                f"El mapa {map_dir} está vacío: recorre el entorno con "
                "main_lap.py o video_a_mapa.py primero.")
        lo, hz, (ox, oy) = mgr.stitched(includeDisk=True)
        res = mgr.resolution
        self.map_dir = map_dir
        self.resolution = res

        h, w = lo.shape
        # celdas ocupadas submuestreadas a bloques -> círculos
        s = cell_stride
        hh, ww = (h // s) * s, (w // s) * s
        occ = (lo[:hh, :ww] > _L_OCC).reshape(hh // s, s, ww // s, s)
        occ = occ.max(axis=(1, 3))
        ys, xs = np.nonzero(occ)
        r_obs = res * s * 0.75
        cx = (ox + xs * s + s / 2.0) * res
        cy = (oy + ys * s + s / 2.0) * res
        self.statics = np.stack(
            [cx, cy, np.full(cx.shape, r_obs)], axis=1) if len(xs) \
            else np.zeros((0, 3))

        hzb = (hz[:hh, :ww] >= HAZARD_MIN_HITS).reshape(hh // s, s,
                                                        ww // s, s)
        hzb = hzb.max(axis=(1, 3))
        ys, xs = np.nonzero(hzb)
        hx = (ox + xs * s + s / 2.0) * res
        hy = (oy + ys * s + s / 2.0) * res
        self.hazards = np.stack(
            [hx, hy, np.full(hx.shape, r_obs)], axis=1) if len(xs) \
            else np.zeros((0, 3))

        # espacio libre navegable: erosionado + componente conexo más
        # grande (inicio y meta siempre caen en el mismo)
        free = (lo < _L_FREE).astype(np.uint8)
        k = max(1, int(round(_MIN_CLEARANCE_M / res)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * k + 1, 2 * k + 1))
        free_eroded = cv2.erode(free, kernel)
        n_comp, labels = cv2.connectedComponents(free_eroded)
        if n_comp <= 1:
            raise ValueError(
                f"El mapa {map_dir} no tiene espacio libre suficiente "
                f"(despeja al menos un pasillo de {2*_MIN_CLEARANCE_M:.1f} m).")
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        comp = int(np.argmax(sizes))
        ys, xs = np.nonzero(labels == comp)
        self.free_x = (ox + xs + 0.5) * res
        self.free_y = (oy + ys + 0.5) * res

        self.bounds = ((ox) * res, (oy) * res,
                       (ox + w) * res, (oy + h) * res)

    def free_area_m2(self):
        return len(self.free_x) * self.resolution ** 2

    def sample_episode(self, rng, movers=True):
        """-> (NavWorld, start, goal). la separación se adapta al
        tamaño del área libre; si no se logra, se usa el par más
        lejano encontrado — nunca metas triviales."""
        n = len(self.free_x)
        span = math.hypot(self.free_x.max() - self.free_x.min(),
                          self.free_y.max() - self.free_y.min())
        target = min(_MIN_GOAL_DIST_M, 0.6 * max(span, 0.5))
        best = (0, 1, -1.0)
        for _ in range(100):
            i, j = rng.integers(0, n, size=2)
            d = math.hypot(self.free_x[i] - self.free_x[j],
                           self.free_y[i] - self.free_y[j])
            if d > best[2]:
                best = (i, j, d)
            if d >= target:
                break
        i, j, _d = best
        start = (float(self.free_x[i]), float(self.free_y[i]),
                 float(rng.uniform(-math.pi, math.pi)))
        goal = (float(self.free_x[j]), float(self.free_y[j]))

        mv = []
        if movers:
            keep = [(start[0], start[1], 0.5), (goal[0], goal[1], 0.5)]
            for (mx, my, mr) in _scatter_circles(
                    rng, rng.integers(0, 3), self.bounds, 0.25, 0.30,
                    keep + [tuple(c) for c in self.statics[:200]]):
                sp = rng.uniform(0.05, 0.12)
                ang = rng.uniform(0, 2 * math.pi)
                mv.append((mx, my, mr, sp * math.cos(ang),
                           sp * math.sin(ang)))

        world = NavWorld(segments=np.zeros((0, 4)),
                         statics=self.statics,
                         hazards=self.hazards,
                         movers=mv, bounds=self.bounds)
        return world, start, goal


def get_factory(map_dir):
    """fábrica cacheada por carpeta (la conversión es cara)."""
    key = os.path.abspath(map_dir)
    if key not in _CACHE:
        _CACHE[key] = RealMapWorldFactory(map_dir)
    return _CACHE[key]


def sample_real_episode(map_dirs, rng):
    """elige una carpeta de mapas reales y muestrea un episodio."""
    map_dir = map_dirs[int(rng.integers(0, len(map_dirs)))]
    return get_factory(map_dir).sample_episode(rng)
