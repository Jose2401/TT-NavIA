"""
map_manager.py
Mapa de ocupación GLOBAL por chunks (cuartos/bloques) con persistencia.

Por qué chunks:
- El sistema debe funcionar en interiores Y exteriores: no existe un
  "tamaño de mapa" que sirva para ambos. Un solo OccupancyGrid grande
  desperdicia RAM (crítico en Raspberry Pi) y uno pequeño se queda corto
  en cuanto el usuario sale del cuarto.
- El plano del mundo se divide en chunks cuadrados fijos (por defecto
  8 x 8 m) anclados al sistema de coordenadas global del SLAM. Los chunks
  tejen el plano completo (índices enteros, negativos incluidos), así que
  el mapa crece SOLO hacia donde el usuario camina.
- Cada chunk se guarda/carga de disco de forma independiente: al volver a
  un lugar ya visitado (misma sesión u otra), su mapa se recupera en vez
  de reconstruirse desde cero.

Detección de "entré a un cuarto/chunk":
- Transición de CHUNK: geométrica y determinista — la pose cruzó la
  frontera del chunk actual. El evento indica si el chunk destino es
  NUEVO (nunca mapeado) o CONOCIDO (estaba en memoria o en disco).
- Transición de CUARTO (semántica, opcional): al pasar junto a un
  landmark 'door' confirmado se registra un cruce de puerta y se abre un
  cuarto nuevo (o se reconoce el cuarto guardado en el chunk destino).
  Es una heurística de apoyo: la navegación NO depende de ella.

El núcleo de Ray Casting (Bresenham + log-odds + capa de peligros) es el
mismo de occupancy_grid.py; aquí solo se resuelve a qué chunk pertenece
cada celda global. El SLAM sigue sin escribir el mapa directamente.
"""
import json
import math
import os
import time

import numpy as np

from occupancy_grid import (bresenham, L_OCC, L_FREE, L_CLAMP, L_BLOCK,
                            OCC_THRESHOLD, HAZARD_MIN_HITS, HAZARD_MAX_HITS)


class MapChunk:
    """Un bloque cuadrado del mapa global. Solo datos + (de)serialización;
    toda la lógica de rayos/coordenadas vive en ChunkedMapManager."""

    __slots__ = ("index", "cells", "log_odds", "hazard_hits", "room",
                 "visits", "created_at", "last_seen_at", "dirty")

    def __init__(self, index, cells, room=None):
        self.index = index                      # (i, j) enteros
        self.cells = cells                      # celdas por lado
        self.log_odds = np.zeros((cells, cells), dtype=np.float32)
        self.hazard_hits = np.zeros((cells, cells), dtype=np.uint8)
        self.room = room                        # etiqueta del cuarto (o None)
        self.visits = 0
        self.created_at = time.time()
        self.last_seen_at = self.created_at
        self.dirty = False

    # ---------------- persistencia ----------------
    @staticmethod
    def filename(index):
        return f"chunk_{index[0]}_{index[1]}.npz"

    def save(self, map_dir):
        np.savez_compressed(
            os.path.join(map_dir, self.filename(self.index)),
            log_odds=self.log_odds.astype(np.float16),
            hazard_hits=self.hazard_hits,
            meta=np.array([self.visits, self.created_at, self.last_seen_at]),
            room=np.array(self.room if self.room is not None else "",
                          dtype=np.str_))
        self.dirty = False

    @classmethod
    def load(cls, map_dir, index, cells):
        data = np.load(os.path.join(map_dir, cls.filename(index)),
                       allow_pickle=False)
        chunk = cls(index, cells)
        lo = data["log_odds"].astype(np.float32)
        if lo.shape != (cells, cells):
            raise ValueError(
                f"Chunk {index}: tamaño {lo.shape} != {(cells, cells)} "
                "(el mapa en disco usa otra resolución/tamaño de chunk)")
        chunk.log_odds = lo
        chunk.hazard_hits = data["hazard_hits"].astype(np.uint8)
        visits, created, last_seen = data["meta"]
        chunk.visits = int(visits)
        chunk.created_at = float(created)
        chunk.last_seen_at = float(last_seen)
        room = str(data["room"])
        chunk.room = room if room else None
        chunk.dirty = False
        return chunk


class ChunkedMapManager:
    """Mapa global por chunks con la MISMA semántica de celdas que
    OccupancyGrid (log-odds + capa de peligros) y la misma interfaz de
    actualización (update_from_scan) y consulta (cast_distance,
    local_costmap, get_nearby_obstacles), de modo que los módulos de
    SLAM/navegación no distinguen entre un grid simple y el mapa global.
    """

    def __init__(self, map_dir=None, chunk_size_m=8.0, resolution=0.05,
                 keep_loaded=25, door_radius=0.7, door_cooldown_steps=60):
        """
        map_dir: carpeta de persistencia. None = solo memoria (sin
            guardado; útil en pruebas). Si existe, los chunks/landmarks
            previos se reutilizan (mapa "conocido").
        keep_loaded: máximo de chunks en RAM. Con 8 m y 5 cm de celda,
            un chunk pesa ~128 KB en RAM; 25 chunks ≈ 3 MB. Los menos
            recientes se guardan y descargan (solo si hay map_dir).
        door_radius / door_cooldown_steps: heurística de cruce de puerta
            para la etiqueta semántica de cuarto.
        """
        self.resolution = resolution
        self.chunk_size_m = chunk_size_m
        self.cells = int(round(chunk_size_m / resolution))
        self.map_dir = map_dir
        self.keep_loaded = max(4, keep_loaded)
        self.door_radius = door_radius
        self.door_cooldown_steps = door_cooldown_steps

        self.chunks = {}            # (i, j) -> MapChunk (cargados en RAM)
        self.disk_index = set()     # chunks que existen en disco
        self.current_chunk = None   # (i, j) donde está el usuario
        self.current_room = None
        self._next_room_id = 1
        self._last_door_step = -10 ** 9
        self._step = 0
        self._last_lookup = (None, None)   # cache (index, chunk) del último acceso

        if map_dir:
            os.makedirs(map_dir, exist_ok=True)
            self._load_meta()
            for fname in os.listdir(map_dir):
                if fname.startswith("chunk_") and fname.endswith(".npz"):
                    try:
                        _, i, j = fname[:-4].split("_")
                        self.disk_index.add((int(i), int(j)))
                    except ValueError:
                        continue

        self.events = []            # eventos del último update_position()

    # ------------------------------------------------------------------
    # Coordenadas: mundo -> celda global -> (chunk, celda local)
    # ------------------------------------------------------------------
    def world_to_cell(self, x, y):
        return (int(math.floor(x / self.resolution)),
                int(math.floor(y / self.resolution)))

    def cell_to_world(self, gx, gy):
        """Centro de la celda global."""
        return ((gx + 0.5) * self.resolution, (gy + 0.5) * self.resolution)

    def cell_to_chunk(self, gx, gy):
        ci, cj = gx // self.cells, gy // self.cells
        return (ci, cj), (gx - ci * self.cells, gy - cj * self.cells)

    def chunk_of_pose(self, x, y):
        gx, gy = self.world_to_cell(x, y)
        return self.cell_to_chunk(gx, gy)[0]

    # ------------------------------------------------------------------
    # Gestión de chunks (RAM <-> disco)
    # ------------------------------------------------------------------
    def is_known_chunk(self, index):
        """¿Este chunk ya fue mapeado antes (en RAM o en disco)?"""
        return index in self.chunks or index in self.disk_index

    def get_chunk(self, index, create=True):
        chunk = self.chunks.get(index)
        if chunk is not None:
            return chunk
        if index in self.disk_index:
            chunk = MapChunk.load(self.map_dir, index, self.cells)
            self.chunks[index] = chunk
            return chunk
        if not create:
            return None
        chunk = MapChunk(index, self.cells)
        chunk.dirty = True
        self.chunks[index] = chunk
        return chunk

    def _chunk_for_cell(self, gx, gy, create=True):
        """Como los rayos recorren celdas contiguas, casi siempre caen en
        el mismo chunk que la celda anterior: se cachea el último."""
        index, local = self.cell_to_chunk(gx, gy)
        cached_index, cached_chunk = self._last_lookup
        if cached_index == index and cached_chunk is not None:
            return cached_chunk, local
        chunk = self.get_chunk(index, create=create)
        self._last_lookup = (index, chunk)
        return chunk, local

    def _evict_if_needed(self):
        if len(self.chunks) <= self.keep_loaded:
            return
        if not self.map_dir:
            return          # sin persistencia no se descarga nada (no perder datos)
        # Descarga los más lejanos al chunk actual
        cur = self.current_chunk or (0, 0)
        by_distance = sorted(
            self.chunks.keys(),
            key=lambda ij: max(abs(ij[0] - cur[0]), abs(ij[1] - cur[1])),
            reverse=True)
        for index in by_distance:
            if len(self.chunks) <= self.keep_loaded:
                break
            if index == self.current_chunk:
                continue
            chunk = self.chunks.pop(index)
            if chunk.dirty:
                chunk.save(self.map_dir)
            self.disk_index.add(index)
            if self._last_lookup[0] == index:
                self._last_lookup = (None, None)

    # ------------------------------------------------------------------
    # Transiciones de chunk / cuarto
    # ------------------------------------------------------------------
    def update_position(self, pose, landmarks=None):
        """Debe llamarse una vez por ciclo con la pose del SLAM (y
        opcionalmente los landmarks etiquetados, para la heurística de
        puertas). Devuelve la lista de eventos del ciclo:

            {"type": "chunk_change", "chunk": (i, j), "known": bool,
             "room": str|None}
            {"type": "door_crossed", "room": str}
            {"type": "room_recognized", "room": str}
        """
        self._step += 1
        self.events = []
        x, y = pose[0], pose[1]
        index = self.chunk_of_pose(x, y)

        if index != self.current_chunk:
            # "Conocido" = ya estaba en disco o ya fue VISITADO antes.
            # (El scan del ciclo puede haber creado el chunk hace un
            # instante; eso no lo vuelve "conocido" para el usuario.)
            existing = self.chunks.get(index)
            known = (index in self.disk_index
                     or (existing is not None and existing.visits > 0))
            chunk = self.get_chunk(index)
            chunk.visits += 1
            chunk.last_seen_at = time.time()
            chunk.dirty = True
            self.current_chunk = index

            if chunk.room is not None and chunk.room != self.current_room:
                # El chunk destino recuerda a qué cuarto pertenece
                self.current_room = chunk.room
                self.events.append({"type": "room_recognized",
                                    "room": chunk.room})
            elif chunk.room is None and self.current_room is not None:
                # El cuarto actual se extiende a este chunk
                chunk.room = self.current_room

            self.events.append({"type": "chunk_change", "chunk": index,
                                "known": known, "room": chunk.room})
            self._evict_if_needed()

        if landmarks:
            self._check_door_crossing(x, y, landmarks)
        return self.events

    def _check_door_crossing(self, x, y, landmarks):
        """Heurística de cuarto: pasar pegado a un landmark 'door'
        confirmado = cambio de cuarto. Con cooldown para no disparar N
        veces mientras se cruza el marco."""
        if self._step - self._last_door_step < self.door_cooldown_steps:
            return
        for lm in landmarks:
            if lm.get("label") != "door" or not lm.get("confirmed", False):
                continue
            if math.hypot(lm["x"] - x, lm["y"] - y) <= self.door_radius:
                self._last_door_step = self._step
                chunk = self.get_chunk(self.current_chunk)
                # ¿El otro lado ya tiene cuarto asignado? Se sabrá al
                # entrar al siguiente chunk; aquí solo se abre uno nuevo
                # si el actual no cambió de etiqueta.
                new_room = f"cuarto_{self._next_room_id}"
                self._next_room_id += 1
                self.current_room = new_room
                chunk.room = chunk.room or new_room
                chunk.dirty = True
                self.events.append({"type": "door_crossed",
                                    "room": new_room})
                return

    def set_room_label(self, label):
        """Etiqueta manual/por voz del cuarto actual ('cocina', 'pasillo').
        Se propaga al chunk actual y a los futuros hasta el próximo cruce."""
        self.current_room = label
        if self.current_chunk is not None:
            chunk = self.get_chunk(self.current_chunk)
            chunk.room = label
            chunk.dirty = True

    # ------------------------------------------------------------------
    # Ray Casting global (misma semántica que OccupancyGrid)
    # ------------------------------------------------------------------
    def _add_log_odds(self, gx, gy, delta):
        chunk, (lx, ly) = self._chunk_for_cell(gx, gy)
        v = chunk.log_odds[ly, lx] + delta
        chunk.log_odds[ly, lx] = min(L_CLAMP, max(-L_CLAMP, v))
        chunk.dirty = True

    def _add_hazard(self, gx, gy):
        chunk, (lx, ly) = self._chunk_for_cell(gx, gy)
        chunk.hazard_hits[ly, lx] = min(HAZARD_MAX_HITS,
                                        int(chunk.hazard_hits[ly, lx]) + 1)
        chunk.dirty = True

    def _cell_values(self, gx, gy):
        """(log_odds, hazard_hits) de la celda global; (0, 0) si el chunk
        no existe (desconocido)."""
        index, (lx, ly) = self.cell_to_chunk(gx, gy)
        chunk = self.chunks.get(index)
        if chunk is None:
            if index in self.disk_index:
                chunk = self.get_chunk(index)
            else:
                return 0.0, 0
        return float(chunk.log_odds[ly, lx]), int(chunk.hazard_hits[ly, lx])

    def update_from_scan(self, pose, observations, max_range=4.0):
        """Idéntico en semántica a OccupancyGrid.update_from_scan, pero
        sobre el plano global: los rayos cruzan fronteras de chunk sin
        recortes (el chunk destino se crea/carga al vuelo)."""
        x, y, theta = pose
        gx0, gy0 = self.world_to_cell(x, y)
        for obs in observations:
            r = min(obs.range_m, max_range)
            sigma = getattr(obs, "sigma_r", None)
            imprecise = sigma is not None and sigma > 0.15
            is_hazard = getattr(obs, "is_hazard", False)

            angle = theta + obs.bearing
            ex = x + r * math.cos(angle)
            ey = y + r * math.sin(angle)
            gx1, gy1 = self.world_to_cell(ex, ey)
            ray = bresenham(gx0, gy0, gx1, gy1)

            free_cells = ray[:-1]
            if imprecise and r > 1e-6:
                frac = max(0.0, (r - 2.0 * sigma) / r)
                free_cells = ray[:int(len(ray) * frac)]

            blocked = False
            for (cx, cy) in free_cells:
                lo, _ = self._cell_values(cx, cy)
                if lo > L_BLOCK:
                    blocked = True     # pared sólida: no perforar
                    break
                self._add_log_odds(cx, cy, L_FREE)

            if not blocked and obs.range_m <= max_range:
                if is_hazard:
                    # La capa de peligros confirma por avistamientos
                    # (HAZARD_MIN_HITS): la imprecisión no la contamina.
                    self._add_hazard(gx1, gy1)
                elif not imprecise:
                    # Solo rangos PRECISOS pintan ocupado (ver
                    # OccupancyGrid.update_from_scan): el rango
                    # monocular crudo salpica; su objeto lo pinta el
                    # landmark consolidado (observations_for_map).
                    self._add_log_odds(gx1, gy1, L_OCC)

    # ------------------------------------------------------------------
    # Consultas para navegación
    # ------------------------------------------------------------------
    def is_blocked_cell(self, gx, gy):
        lo, hz = self._cell_values(gx, gy)
        return lo > OCC_THRESHOLD or hz >= HAZARD_MIN_HITS

    def cast_distance(self, x, y, angle, max_range=4.0):
        """Distancia por ray casting hasta la primera celda bloqueada
        (ocupada o peligro de piso); max_range si el corredor está libre."""
        gx0, gy0 = self.world_to_cell(x, y)
        ex = x + max_range * math.cos(angle)
        ey = y + max_range * math.sin(angle)
        for (cx, cy) in bresenham(gx0, gy0, *self.world_to_cell(ex, ey)):
            if (cx, cy) != (gx0, gy0) and self.is_blocked_cell(cx, cy):
                wx, wy = self.cell_to_world(cx, cy)
                return min(max_range, math.hypot(wx - x, wy - y))
        return max_range

    def local_costmap(self, pose, size_m=4.0):
        """Parche egocéntrico (2, H, W) float32 centrado en la pose (ejes
        del mundo), ensamblado a través de los chunks que toque:
            canal 0 = probabilidad de ocupación (0.5 = desconocido)
            canal 1 = máscara de peligros de piso {0,1}
        Entrada de mapa para el módulo de navegación (DRL)."""
        x, y = pose[0], pose[1]
        half = int(round((size_m / 2.0) / self.resolution))
        side = 2 * half + 1
        out = np.zeros((2, side, side), dtype=np.float32)
        out[0].fill(0.5)

        gx0, gy0 = self.world_to_cell(x, y)
        lo_x, lo_y = gx0 - half, gy0 - half          # celda global de out[0,0]
        hi_x, hi_y = lo_x + side, lo_y + side        # exclusivo

        ci0, cj0 = lo_x // self.cells, lo_y // self.cells
        ci1, cj1 = (hi_x - 1) // self.cells, (hi_y - 1) // self.cells
        for cj in range(cj0, cj1 + 1):
            for ci in range(ci0, ci1 + 1):
                index = (ci, cj)
                if not self.is_known_chunk(index):
                    continue
                chunk = self.get_chunk(index)
                # intersección del chunk con el parche, en celdas globales
                cx0, cy0 = ci * self.cells, cj * self.cells
                ix0, iy0 = max(lo_x, cx0), max(lo_y, cy0)
                ix1 = min(hi_x, cx0 + self.cells)
                iy1 = min(hi_y, cy0 + self.cells)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                src = chunk.log_odds[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
                dst_y, dst_x = iy0 - lo_y, ix0 - lo_x
                out[0, dst_y:dst_y + (iy1 - iy0),
                    dst_x:dst_x + (ix1 - ix0)] = 1.0 / (1.0 + np.exp(-src))
                hz = chunk.hazard_hits[iy0 - cy0:iy1 - cy0,
                                       ix0 - cx0:ix1 - cx0]
                out[1, dst_y:dst_y + (iy1 - iy0),
                    dst_x:dst_x + (ix1 - ix0)] = \
                    (hz >= HAZARD_MIN_HITS).astype(np.float32)
        return out

    def get_nearby_obstacles(self, x, y, radius=1.5, threshold=OCC_THRESHOLD):
        """Celdas ocupadas (coordenadas de mundo) alrededor de (x, y),
        cruzando chunks. Misma interfaz que OccupancyGrid."""
        cm = self.local_costmap((x, y), size_m=2 * radius)
        half = cm.shape[1] // 2
        prob_thr = 1.0 / (1.0 + math.exp(-threshold))
        ys, xs = np.where(cm[0] > prob_thr)
        gx0, gy0 = self.world_to_cell(x, y)
        obstacles = []
        for yy, xx in zip(ys, xs):
            wx, wy = self.cell_to_world(gx0 + (xx - half), gy0 + (yy - half))
            obstacles.append((wx, wy))
        return obstacles

    # ------------------------------------------------------------------
    # Persistencia del mapa completo
    # ------------------------------------------------------------------
    def save_all(self):
        """Guarda todos los chunks sucios + metadatos. Llamar al salir y
        periódicamente (p. ej. en cada cambio de chunk)."""
        if not self.map_dir:
            return 0
        saved = 0
        for chunk in self.chunks.values():
            if chunk.dirty:
                chunk.save(self.map_dir)
                self.disk_index.add(chunk.index)
                saved += 1
        self._save_meta()
        return saved

    def _meta_path(self):
        return os.path.join(self.map_dir, "meta.json")

    def _save_meta(self):
        rooms = {}
        for chunk in self.chunks.values():
            if chunk.room:
                rooms[f"{chunk.index[0]},{chunk.index[1]}"] = chunk.room
        meta = {
            "resolution": self.resolution,
            "chunk_size_m": self.chunk_size_m,
            "next_room_id": self._next_room_id,
            "current_room": self.current_room,
            "rooms": rooms,
        }
        # conservar cuartos de chunks no cargados
        old = self._read_meta()
        if old:
            for key, room in old.get("rooms", {}).items():
                rooms.setdefault(key, room)
        with open(self._meta_path(), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)

    def _read_meta(self):
        try:
            with open(self._meta_path(), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _load_meta(self):
        meta = self._read_meta()
        if not meta:
            return
        if abs(meta.get("resolution", self.resolution) - self.resolution) > 1e-9 \
                or abs(meta.get("chunk_size_m", self.chunk_size_m)
                       - self.chunk_size_m) > 1e-9:
            raise ValueError(
                f"El mapa en {self.map_dir} fue creado con "
                f"resolution={meta.get('resolution')} y chunk_size_m="
                f"{meta.get('chunk_size_m')}; no coincide con la "
                "configuración actual. Usa otra carpeta o borra el mapa.")
        self._next_room_id = meta.get("next_room_id", 1)
        self.current_room = meta.get("current_room")

    # -------- landmarks (mapa ligero del SLAM) --------
    def save_landmarks(self, landmarks):
        """Persiste el mapa ligero de landmarks (lista de dicts de
        EKFSlam.export_landmarks()) junto al mapa de chunks."""
        if not self.map_dir:
            return
        path = os.path.join(self.map_dir, "landmarks.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(landmarks, fh, ensure_ascii=False, indent=1)

    def load_landmarks(self):
        if not self.map_dir:
            return []
        path = os.path.join(self.map_dir, "landmarks.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return []

    # ------------------------------------------------------------------
    # Visualización
    # ------------------------------------------------------------------
    def known_chunk_indices(self):
        return set(self.chunks.keys()) | set(self.disk_index)

    def stitched(self, include_disk=True):
        """Une los chunks conocidos en un solo par de arreglos
        (log_odds, hazard) + el origen del mosaico en celdas globales.
        Para visualización/croquis; no usar en el ciclo de control."""
        indices = self.known_chunk_indices() if include_disk \
            else set(self.chunks.keys())
        if not indices:
            side = self.cells
            return (np.zeros((side, side), np.float32),
                    np.zeros((side, side), np.uint8), (0, 0))
        ci0 = min(i for i, _ in indices)
        cj0 = min(j for _, j in indices)
        ci1 = max(i for i, _ in indices)
        cj1 = max(j for _, j in indices)
        w = (ci1 - ci0 + 1) * self.cells
        h = (cj1 - cj0 + 1) * self.cells
        lo = np.zeros((h, w), np.float32)
        hz = np.zeros((h, w), np.uint8)
        for index in indices:
            chunk = self.get_chunk(index)
            x0 = (index[0] - ci0) * self.cells
            y0 = (index[1] - cj0) * self.cells
            lo[y0:y0 + self.cells, x0:x0 + self.cells] = chunk.log_odds
            hz[y0:y0 + self.cells, x0:x0 + self.cells] = chunk.hazard_hits
        origin_cell = (ci0 * self.cells, cj0 * self.cells)
        return lo, hz, origin_cell

    def render_croquis(self, pose=None, trajectory=None, landmarks=None,
                       gt_trajectory=None, path="croquis_mapa.png"):
        """Croquis del mapa global (todos los chunks conocidos), con
        fronteras de chunk, cuartos, peligros y trayectoria."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lo, hz, (ox, oy) = self.stitched()
        occ_prob = 1.0 - 1.0 / (1.0 + np.exp(lo))
        h, w = lo.shape
        extent = [ox * self.resolution, (ox + w) * self.resolution,
                  oy * self.resolution, (oy + h) * self.resolution]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(occ_prob, cmap="Greys", origin="lower", extent=extent,
                  vmin=0, vmax=1)

        # fronteras de chunk + etiqueta de cuarto
        for index in self.known_chunk_indices():
            x0 = index[0] * self.chunk_size_m
            y0 = index[1] * self.chunk_size_m
            ax.add_patch(plt.Rectangle((x0, y0), self.chunk_size_m,
                                       self.chunk_size_m, fill=False,
                                       edgecolor="tab:purple",
                                       linewidth=0.6, alpha=0.5))
            chunk = self.chunks.get(index)
            room = chunk.room if chunk else None
            tag = f"({index[0]},{index[1]})" + (f" {room}" if room else "")
            ax.text(x0 + 0.1, y0 + 0.1, tag, fontsize=6,
                    color="tab:purple", alpha=0.8)

        ys, xs = np.where(hz >= HAZARD_MIN_HITS)
        if len(xs):
            ax.plot((ox + xs + 0.5) * self.resolution,
                    (oy + ys + 0.5) * self.resolution, "s",
                    color="tab:red", markersize=3, label="Peligros de piso")

        if gt_trajectory:
            ax.plot([p[0] for p in gt_trajectory],
                    [p[1] for p in gt_trajectory], "--",
                    color="tab:green", linewidth=1.2, label="Trayectoria real")
        if trajectory:
            ax.plot([p[0] for p in trajectory],
                    [p[1] for p in trajectory], "-",
                    color="tab:blue", linewidth=1.5,
                    label="Trayectoria estimada")
        if landmarks:
            ax.plot([p[0] for p in landmarks], [p[1] for p in landmarks],
                    "x", color="tab:orange", markersize=7, label="Landmarks")
        if pose is not None:
            x, y, theta = pose
            ax.plot(x, y, "o", color="tab:red", markersize=9, label="Usuario")
            ax.arrow(x, y, 0.3 * math.cos(theta), 0.3 * math.sin(theta),
                     head_width=0.12, color="tab:red")

        ax.set_title("Mapa global por chunks (SLAM + Ray Casting)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
