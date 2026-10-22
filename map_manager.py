import json
import math
import os
import time

import numpy as np

from occupancy_grid import (bresenham, L_OCC, L_FREE, L_CLAMP, L_BLOCK,
                            OCC_THRESHOLD, HAZARD_MIN_HITS, HAZARD_MAX_HITS)


class MapChunk:
    __slots__ = ("index", "cells", "logOdds", "hazardHits", "room",
                 "visits", "createdAt", "lastSeenAt", "dirty")

    def __init__(self, index, cells, room=None):
        self.index = index
        self.cells= cells
        self.logOdds = np.zeros((cells, cells), dtype=np.float32)
        self.hazardHits = np.zeros((cells, cells), dtype=np.uint8)
        self.room =room
        self.visits = 0
        self.createdAt = time.time()
        self.lastSeenAt = self.createdAt
        self.dirty = False

    @staticmethod
    def fileName(index):
        return f"chunk_{index[0]}_{index[1]}.npz"

    def save(self, mapDir):
        np.savez_compressed(
            os.path.join(mapDir, self.fileName(self.index)),
            log_odds=self.logOdds.astype(np.float16),
            hazard_hits=self.hazardHits,
            meta=np.array([self.visits, self.createdAt, self.lastSeenAt]),
            room=np.array(self.room if self.room is not None else "",
                          dtype=np.str_))
        self.dirty= False

    @classmethod
    def load(cls, mapDir, index, cells):
        data = np.load(os.path.join(mapDir, cls.fileName(index)),
                       allow_pickle=False)
        chunk = cls(index, cells)
        lo = data["log_odds"].astype(np.float32)
        if lo.shape != (cells, cells):
            raise ValueError(
                f"Chunk {index}: tamaño {lo.shape} != {(cells, cells)} "
                "(el mapa en disco usa otra resolución/tamaño de chunk)")
        chunk.logOdds = lo
        chunk.hazardHits = data["hazard_hits"].astype(np.uint8)
        visits, created, lastSeen = data["meta"]
        chunk.visits = int(visits)
        chunk.createdAt =float(created)
        chunk.lastSeenAt = float(lastSeen)
        papu = str(data["room"])
        chunk.room = papu if papu else None
        chunk.dirty = False
        return chunk


class ChunkedMapManager:
    def __init__(self, mapDir=None, chunkSizeM=8.0, resolution=0.05,
                 keepLoaded=25, doorRadius=0.7, doorCooldownSteps=60):
        self.resolution = resolution
        self.chunkSizeM = chunkSizeM
        self.cells = int(round(chunkSizeM / resolution))
        self.mapDir =mapDir
        self.keepLoaded = max(4, keepLoaded)
        self.doorRadius = doorRadius
        self.doorCooldownSteps= doorCooldownSteps

        self.chunks = {}
        self.diskIndex = set()
        self.currentChunk = None
        self.currentRoom = None
        self._nextRoomId = 1
        self._lastDoorStep = -10 ** 9
        self._step = 0
        self._lastLookup = (None, None)

        if mapDir:
            os.makedirs(mapDir, exist_ok=True)
            self._cargarMeta()
            for fname in os.listdir(mapDir):
                if fname.startswith("chunk_") and fname.endswith(".npz"):
                    try:
                        _, i, j = fname[:-4].split("_")
                        self.diskIndex.add((int(i), int(j)))
                    except ValueError:
                        continue

        self.events = []

    @classmethod
    def openExisting(cls, mapDir, **kw):
        metaPath = os.path.join(mapDir, "meta.json")
        try:
            with open(metaPath, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            kw.setdefault("resolution", meta.get("resolution", 0.05))
            kw.setdefault("chunkSizeM", meta.get("chunk_size_m", 8.0))
        except (OSError, ValueError):
            pass
        return cls(mapDir=mapDir, **kw)

    def worldToCell(self, x, y):
        return (int(math.floor(x / self.resolution)),
                int(math.floor(y / self.resolution)))

    def cellToWorld(self, gx, gy):
        return ((gx + 0.5) * self.resolution, (gy + 0.5) * self.resolution)

    def cellToChunk(self, gx, gy):
        ci, cj = gx // self.cells, gy // self.cells
        return (ci, cj), (gx - ci * self.cells, gy - cj * self.cells)

    def chunkOfPose(self, x, y):
        gx, gy = self.worldToCell(x, y)
        return self.cellToChunk(gx, gy)[0]

    def isKnownChunk(self, index):
        return index in self.chunks or index in self.diskIndex

    def getChunk(self, index, create=True):
        chunk = self.chunks.get(index)
        if chunk is not None:
            return chunk
        if index in self.diskIndex:
            chunk = MapChunk.load(self.mapDir, index, self.cells)
            self.chunks[index] = chunk
            return chunk
        if not create:
            return None
        chunk = MapChunk(index, self.cells)
        chunk.dirty =True
        self.chunks[index] = chunk
        return chunk

    def _chunkDeCelda(self, gx, gy, create=True):
        index, local = self.cellToChunk(gx, gy)
        idxCache, chunkCache = self._lastLookup
        if idxCache == index and chunkCache is not None:
            return chunkCache, local
        chunk = self.getChunk(index, create=create)
        self._lastLookup = (index, chunk)
        return chunk, local

    def _sacarSiHaceFalta(self):
        if len(self.chunks) <= self.keepLoaded:
            return
        if not self.mapDir:
            return
        cur = self.currentChunk or (0, 0)
        porDistancia = sorted(
            self.chunks.keys(),
            key=lambda ij: max(abs(ij[0] - cur[0]), abs(ij[1] - cur[1])),
            reverse=True)
        for index in porDistancia:
            if len(self.chunks) <= self.keepLoaded:
                break
            if index == self.currentChunk:
                continue
            chunk = self.chunks.pop(index)
            if chunk.dirty:
                chunk.save(self.mapDir)
            self.diskIndex.add(index)
            if self._lastLookup[0] == index:
                self._lastLookup = (None, None)

    # cambio de chunk/cuarto, regresa eventos
    def updatePosition(self, pose, landmarks=None):
        self._step += 1
        self.events = []
        x, y = pose[0], pose[1]
        index = self.chunkOfPose(x, y)

        if index != self.currentChunk:
            yaExiste = self.chunks.get(index)
            known = (index in self.diskIndex
                     or (yaExiste is not None and yaExiste.visits > 0))
            chunk = self.getChunk(index)
            chunk.visits += 1
            chunk.lastSeenAt = time.time()
            chunk.dirty = True
            self.currentChunk = index

            if chunk.room is not None and chunk.room != self.currentRoom:
                self.currentRoom = chunk.room
                self.events.append({"type": "room_recognized",
                                    "room": chunk.room})
            elif chunk.room is None and self.currentRoom is not None:
                chunk.room = self.currentRoom

            self.events.append({"type": "chunk_change", "chunk": index,
                                "known": known, "room": chunk.room})
            self._sacarSiHaceFalta()

        if landmarks:
            self._checarPuerta(x, y, landmarks)
        return self.events

    def _checarPuerta(self, x, y, landmarks):
        if self._step - self._lastDoorStep < self.doorCooldownSteps:
            return
        for lm in landmarks:
            if lm.get("label") != "door" or not lm.get("confirmed", False):
                continue
            if math.hypot(lm["x"] - x, lm["y"] - y) <= self.doorRadius:
                self._lastDoorStep = self._step
                chunk = self.getChunk(self.currentChunk)
                cuartoNuevo = f"cuarto_{self._nextRoomId}"
                self._nextRoomId += 1
                self.currentRoom = cuartoNuevo
                chunk.room = chunk.room or cuartoNuevo
                chunk.dirty = True
                self.events.append({"type": "door_crossed",
                                    "room": cuartoNuevo})
                return

    def setRoomLabel(self, label):
        self.currentRoom = label
        if self.currentChunk is not None:
            chunk = self.getChunk(self.currentChunk)
            chunk.room = label
            chunk.dirty = True

    def _addLogOdds(self, gx, gy, delta):
        chunk, (lx, ly) = self._chunkDeCelda(gx, gy)
        v = chunk.logOdds[ly, lx] + delta
        chunk.logOdds[ly, lx] = min(L_CLAMP, max(-L_CLAMP, v))
        chunk.dirty = True

    def _addHazard(self, gx, gy):
        chunk, (lx, ly) = self._chunkDeCelda(gx, gy)
        chunk.hazardHits[ly, lx] = min(HAZARD_MAX_HITS,
                                       int(chunk.hazardHits[ly, lx]) + 1)
        chunk.dirty= True

    def _cellValues(self, gx, gy):
        index, (lx, ly) = self.cellToChunk(gx, gy)
        chunk = self.chunks.get(index)
        if chunk is None:
            if index in self.diskIndex:
                chunk = self.getChunk(index)
            else:
                return 0.0, 0
        return float(chunk.logOdds[ly, lx]), int(chunk.hazardHits[ly, lx])

    # ray casting global, cruza chunks sin recortar
    def updateFromScan(self, pose, observations, maxRange=4.0):
        x, y, theta = pose
        gx0, gy0 = self.worldToCell(x, y)
        for obs in observations:
            r = min(obs.range_m, maxRange)
            sigma = getattr(obs, "sigma_r", None)
            impreciso = sigma is not None and sigma > 0.15
            esHoyo =getattr(obs, "is_hazard", False)

            ang = theta + obs.bearing
            ex = x + r * math.cos(ang)
            ey = y + r * math.sin(ang)
            gx1, gy1 = self.worldToCell(ex, ey)
            papuRay = bresenham(gx0, gy0, gx1, gy1)

            celdasLibres = papuRay[:-1]
            if impreciso and r > 1e-6:
                frac = max(0.0, (r - 2.0 * sigma) / r)
                celdasLibres = papuRay[:int(len(papuRay) * frac)]

            bloqueado = False
            for (cx, cy) in celdasLibres:
                lo, _ = self._cellValues(cx, cy)
                if lo > L_BLOCK:
                    bloqueado = True
                    break
                self._addLogOdds(cx, cy, L_FREE)

            if not bloqueado and obs.range_m <= maxRange:
                if esHoyo:
                    self._addHazard(gx1, gy1)
                elif not impreciso:
                    self._addLogOdds(gx1, gy1, L_OCC)

    def clearFootprint(self, x, y, radius=0.45):
        gx0, gy0 = self.worldToCell(x, y)
        rc = max(1, int(round(radius / self.resolution)))
        for dy in range(-rc, rc + 1):
            for dx in range(-rc, rc + 1):
                if dx * dx + dy * dy > rc * rc:
                    continue
                lo, _ = self._cellValues(gx0 + dx, gy0 + dy)
                if lo > OCC_THRESHOLD:
                    continue
                self._addLogOdds(gx0 + dx, gy0 + dy, L_FREE)

    def isBlockedCell(self, gx, gy):
        lo, hz = self._cellValues(gx, gy)
        return lo > OCC_THRESHOLD or hz >= HAZARD_MIN_HITS

    def castDistance(self, x, y, angle, maxRange=4.0):
        gx0, gy0 = self.worldToCell(x, y)
        ex = x + maxRange * math.cos(angle)
        ey =y + maxRange * math.sin(angle)
        for (cx, cy) in bresenham(gx0, gy0, *self.worldToCell(ex, ey)):
            if (cx, cy) != (gx0, gy0) and self.isBlockedCell(cx, cy):
                wx, wy = self.cellToWorld(cx, cy)
                return min(maxRange, math.hypot(wx - x, wy - y))
        return maxRange

    # parche egocentrico armado de varios chunks
    def localCostmap(self, pose, sizeM=4.0):
        x, y = pose[0], pose[1]
        half = int(round((sizeM / 2.0) / self.resolution))
        lado = 2 * half + 1
        out = np.zeros((2, lado, lado), dtype=np.float32)
        out[0].fill(0.5)

        gx0, gy0 = self.worldToCell(x, y)
        loX, loY = gx0 - half, gy0 - half
        hiX, hiY = loX + lado, loY + lado

        ci0, cj0 = loX // self.cells, loY // self.cells
        ci1, cj1 = (hiX - 1) // self.cells, (hiY - 1) // self.cells
        for cj in range(cj0, cj1 + 1):
            for ci in range(ci0, ci1 + 1):
                index = (ci, cj)
                if not self.isKnownChunk(index):
                    continue
                chunk = self.getChunk(index)
                cx0, cy0 = ci * self.cells, cj * self.cells
                ix0, iy0 = max(loX, cx0), max(loY, cy0)
                ix1 = min(hiX, cx0 + self.cells)
                iy1= min(hiY, cy0 + self.cells)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                src = chunk.logOdds[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
                dstY, dstX = iy0 - loY, ix0 - loX
                out[0, dstY:dstY + (iy1 - iy0),
                    dstX:dstX + (ix1 - ix0)] = 1.0 / (1.0 + np.exp(-src))
                hz = chunk.hazardHits[iy0 - cy0:iy1 - cy0,
                                      ix0 - cx0:ix1 - cx0]
                out[1, dstY:dstY + (iy1 - iy0),
                    dstX:dstX + (ix1 - ix0)] = \
                    (hz >= HAZARD_MIN_HITS).astype(np.float32)
        return out

    def getNearbyObstacles(self, x, y, radius=1.5, threshold=OCC_THRESHOLD):
        cm = self.localCostmap((x, y), sizeM=2 * radius)
        half = cm.shape[1] // 2
        probThr = 1.0 / (1.0 + math.exp(-threshold))
        ys, xs = np.where(cm[0] > probThr)
        gx0, gy0 = self.worldToCell(x, y)
        obstaculos = []
        for yy, xx in zip(ys, xs):
            wx, wy = self.cellToWorld(gx0 + (xx - half), gy0 + (yy - half))
            obstaculos.append((wx, wy))
        return obstaculos

    def saveAll(self):
        if not self.mapDir:
            return 0
        guardados = 0
        for chunk in self.chunks.values():
            if chunk.dirty:
                chunk.save(self.mapDir)
                self.diskIndex.add(chunk.index)
                guardados += 1
        self._guardarMeta()
        return guardados

    def _metaPath(self):
        return os.path.join(self.mapDir, "meta.json")

    def _guardarMeta(self):
        rooms = {}
        for chunk in self.chunks.values():
            if chunk.room:
                rooms[f"{chunk.index[0]},{chunk.index[1]}"] = chunk.room
        meta = {
            "resolution": self.resolution,
            "chunk_size_m": self.chunkSizeM,
            "next_room_id": self._nextRoomId,
            "current_room": self.currentRoom,
            "rooms": rooms,
        }
        viejo = self._leerMeta()
        if viejo:
            for key, room in viejo.get("rooms", {}).items():
                rooms.setdefault(key, room)
        with open(self._metaPath(), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)

    def _leerMeta(self):
        try:
            with open(self._metaPath(), "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _cargarMeta(self):
        meta = self._leerMeta()
        if not meta:
            return
        if abs(meta.get("resolution", self.resolution) - self.resolution) > 1e-9 \
                or abs(meta.get("chunk_size_m", self.chunkSizeM)
                       - self.chunkSizeM) > 1e-9:
            raise ValueError(
                f"El mapa en {self.mapDir} fue creado con "
                f"resolution={meta.get('resolution')} y chunk_size_m="
                f"{meta.get('chunk_size_m')}; no coincide con la "
                "configuración actual. Usa otra carpeta o borra el mapa.")
        self._nextRoomId = meta.get("next_room_id", 1)
        self.currentRoom= meta.get("current_room")

    def saveLandmarks(self, landmarks):
        if not self.mapDir:
            return
        path = os.path.join(self.mapDir, "landmarks.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(landmarks, fh, ensure_ascii=False, indent=1)

    def loadLandmarks(self):
        if not self.mapDir:
            return []
        path = os.path.join(self.mapDir, "landmarks.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return []

    def knownChunkIndices(self):
        return set(self.chunks.keys()) | set(self.diskIndex)

    def stitched(self, includeDisk=True):
        indices = self.knownChunkIndices() if includeDisk \
            else set(self.chunks.keys())
        if not indices:
            lado = self.cells
            return (np.zeros((lado, lado), np.float32),
                    np.zeros((lado, lado), np.uint8), (0, 0))
        ci0 = min(i for i, _ in indices)
        cj0 = min(j for _, j in indices)
        ci1 = max(i for i, _ in indices)
        cj1 = max(j for _, j in indices)
        w = (ci1 - ci0 + 1) * self.cells
        h =(cj1 - cj0 + 1) * self.cells
        lo = np.zeros((h, w), np.float32)
        hz = np.zeros((h, w), np.uint8)
        for index in indices:
            chunk = self.getChunk(index)
            x0 = (index[0] - ci0) * self.cells
            y0 = (index[1] - cj0) * self.cells
            lo[y0:y0 + self.cells, x0:x0 + self.cells] = chunk.logOdds
            hz[y0:y0 + self.cells, x0:x0 + self.cells] = chunk.hazardHits
        celdaOrigen = (ci0 * self.cells, cj0 * self.cells)
        return lo, hz, celdaOrigen

    def renderCroquis(self, pose=None, trajectory=None, landmarks=None,
                      gtTrajectory=None, path="croquis_mapa.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lo, hz, (ox, oy) = self.stitched()
        occProb = 1.0 - 1.0 / (1.0 + np.exp(lo))
        h, w = lo.shape
        extent = [ox * self.resolution, (ox + w) * self.resolution,
                  oy * self.resolution, (oy + h) * self.resolution]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(occProb, cmap="Greys", origin="lower", extent=extent,
                  vmin=0, vmax=1)

        for index in self.knownChunkIndices():
            x0 = index[0] * self.chunkSizeM
            y0 = index[1] * self.chunkSizeM
            ax.add_patch(plt.Rectangle((x0, y0), self.chunkSizeM,
                                       self.chunkSizeM, fill=False,
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

        if gtTrajectory:
            ax.plot([p[0] for p in gtTrajectory],
                    [p[1] for p in gtTrajectory], "--",
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
