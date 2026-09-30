"""
navigator.py — USA el modelo de navegación (runtime de inferencia).

Implementa, siguiendo los pseudocódigos del TT_2026_B045:
- Algoritmo 4 (inferencia): construye el vector de estado desde el
  NavigationFrame + mapa, evalúa la política y devuelve la acción.
- Algoritmo 5 (submódulo de dirección): traduce la acción a texto con
  las frases y umbrales EXACTOS del documento ("Avance N metros",
  "Gire ligeramente a la izquierda" si <=45°, "Deténgase", prefijo
  "Está cerca de su destino. " si faltan <5 m).
- Algoritmo 11 (estados de sesión): en_espera / activo / pausado, con
  los mensajes literales ("Ruta calculada. Iniciando navegación.",
  "Navegación pausada.", "Ha llegado a su destino.", etc.).
- Intenciones del PLN (Alg. 9): destino, pausa, reanudar, detener,
  ruta_alternativa, desconocido — aquí como FORMATO DE ENTRADA:
  el módulo de voz/PLN futuro solo tiene que producir un NavCommand
  (o pasar el texto a parse_command, que ya entiende español simple).

FORMATO DE SALIDA: lista de NavMessage por ciclo, cada uno con
prioridad según el diseño (0 = alerta, interrumpe el TTS; 1 =
navegación; 2 = confirmación/estado). El módulo de salida (voz) los
reproduce tal cual, uno a uno, en tiempo real.

Backends del modelo:
- SB3 (.zip) si stable_baselines3 está instalado (laptop), o
- NumpyPolicy (_policy.npz): MLP con tanh en numpy puro — la Raspberry
  no necesita SB3 ni torch para navegar.
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from navigation.nav_env import (MAX_RANGE, RISK_RANGE, EXT_RAY_ANGLES,
                                DOC_RAY_ANGLES, FRONT_HALF, SIDE_MAX,
                                FORWARD_STEP, TURN_STEP)

ARRIVAL_RADIUS = 0.6          # umbral_llegada (m)
WAYPOINT_RADIUS = 1.0         # avanzar al siguiente waypoint de la ruta
ACTION_NAMES = ("avanzar", "girar_izquierda", "girar_derecha", "detenerse")


# ----------------------------------------------------------------------
# FORMATO DE ENTRADA: comandos hacia navegación
# ----------------------------------------------------------------------
@dataclass
class NavCommand:
    """Lo que el módulo de voz/PLN (futuro) entrega a navegación.
    tipo: navegar | pausar | reanudar | detener | ruta_alternativa |
          estado | desconocido
    destino: etiqueta ('salida', 'door', 'chair', ...) o None
    punto: (x, y) explícito o None
    """
    tipo: str
    destino: Optional[str] = None
    punto: Optional[tuple] = None
    texto_original: str = ""


_DEST_WORDS = r"(?:ve|ir|vamos|llevame|llévame|navega|navegar|guiame|guíame) (?:a|al|a la|hacia|hasta)?\s*(?:el |la |los |las )?"


def parse_command(text: str) -> NavCommand:
    """Parser de español simple → NavCommand. Es un sustituto ligero
    del clasificador de intenciones del módulo de PLN (Alg. 9); cuando
    ese módulo exista, puede construir el NavCommand directamente."""
    t = text.strip().lower()
    if not t:
        return NavCommand("desconocido", texto_original=text)

    if re.search(r"\b(pausa|pausar|espera|esperate|espérate)\b", t):
        return NavCommand("pausar", texto_original=text)
    if re.search(r"\b(reanuda|reanudar|continua|continúa|sigue|seguir)\b", t):
        return NavCommand("reanudar", texto_original=text)
    if re.search(r"\b(deten|detén|detener|detente|cancela|cancelar|alto|"
                 r"terminar|termina)\b", t):
        return NavCommand("detener", texto_original=text)
    if re.search(r"(ruta alternativa|otra ruta|otro camino|recalcula)", t):
        return NavCommand("ruta_alternativa", texto_original=text)
    if re.search(r"\b(estado|donde estoy|dónde estoy|ubicacion|ubicación)\b", t):
        return NavCommand("estado", texto_original=text)

    m = re.search(r"punto\s+(-?\d+(?:\.\d+)?)[ ,]+(-?\d+(?:\.\d+)?)", t)
    if m:
        return NavCommand("navegar",
                          punto=(float(m.group(1)), float(m.group(2))),
                          texto_original=text)

    m = re.search(_DEST_WORDS + r"([a-záéíóúñ_ ]+)$", t)
    if m:
        return NavCommand("navegar", destino=m.group(1).strip(),
                          texto_original=text)
    # "destino cocina" / una sola palabra tras 'a'
    m = re.search(r"\bdestino\s+([a-záéíóúñ_ ]+)$", t)
    if m:
        return NavCommand("navegar", destino=m.group(1).strip(),
                          texto_original=text)
    return NavCommand("desconocido", texto_original=text)


# Sinónimos español -> etiquetas de landmarks del SLAM (COCO/las nuestras)
DEST_ALIASES = {
    "salida": "door", "puerta": "door", "entrada": "door",
    "silla": "chair", "mesa": "table", "sofa": "sofa", "sillon": "sofa",
    "cama": "bed", "refrigerador": "fridge", "refri": "fridge",
    "planta": "potted plant", "monitor": "monitor", "tele": "monitor",
    "television": "monitor", "escusado": "toilet", "bano": "toilet",
    "baño": "toilet", "lavabo": "sink",
}


# ----------------------------------------------------------------------
# FORMATO DE SALIDA: mensajes hacia el módulo de voz
# ----------------------------------------------------------------------
@dataclass
class NavMessage:
    """Un mensaje listo para el módulo de salida (TTS).
    priority: 0 = alerta (interrumpe la reproducción en curso),
              1 = instrucción de navegación, 2 = confirmación/estado.
    kind: alerta | instruccion | evento | respuesta
    action: índice 0-3 si el mensaje proviene de una decisión del modelo.
    """
    text: str
    priority: int = 1
    kind: str = "instruccion"
    action: Optional[int] = None


@dataclass
class NavStatus:
    """Estado consultable de la sesión (para GUI / depuración)."""
    session: str = "en_espera"     # en_espera | activo | pausado
    goal: Optional[tuple] = None
    goal_label: Optional[str] = None
    d_goal: Optional[float] = None
    phi_goal: Optional[float] = None
    last_action: Optional[int] = None
    waypoints: List[tuple] = field(default_factory=list)


# ----------------------------------------------------------------------
# Backends de la política
# ----------------------------------------------------------------------
class NumpyPolicy:
    """Inferencia del actor PPO con numpy puro (MLP + tanh + argmax).
    Es el backend para la Raspberry: solo requiere el _policy.npz."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        n = int(data["n_hidden"])
        self.layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        self.w_out = data["W_out"]
        self.b_out = data["b_out"]

    def predict(self, obs):
        h = np.asarray(obs, dtype=np.float64)
        for w, b in self.layers:
            h = np.tanh(w @ h + b)
        logits = self.w_out @ h + self.b_out
        return int(np.argmax(logits))


class SB3Policy:
    def __init__(self, zip_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(zip_path, device="cpu")

    def predict(self, obs):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=True)
        return int(action)


def load_policy(model_base):
    """model_base: prefijo (models/nav_ppo) o ruta a .zip/.npz.
    Devuelve (política, obs_mode). Prefiere SB3 si está disponible y
    existe el .zip; si no, cae al .npz con numpy (Raspberry)."""
    base = re.sub(r"(\.zip|_policy\.npz)$", "", model_base)
    meta_path = base + "_meta.json"
    obs_mode = "extended"
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            obs_mode = json.load(fh).get("obs_mode", "extended")

    zip_path, npz_path = base + ".zip", base + "_policy.npz"
    if os.path.exists(zip_path):
        try:
            return SB3Policy(zip_path), obs_mode
        except ImportError:
            pass
    if os.path.exists(npz_path):
        return NumpyPolicy(npz_path), obs_mode
    raise FileNotFoundError(
        f"No se encontró el modelo ({zip_path} ni {npz_path}). "
        "Créalo con: python -m navigation.train")


# ----------------------------------------------------------------------
# Construcción del vector de estado desde el sistema real
# ----------------------------------------------------------------------
def _hazard_sector_from_costmap(frame):
    """Distancia al peligro de piso más cercano por sector, medida sobre
    el canal 1 del costmap egocéntrico (los hoyos no aparecen en los
    rayos: no bloquean nada, igual que en el entrenamiento)."""
    cm = frame.costmap
    res = frame.costmap_resolution
    half = cm.shape[1] // 2
    ys, xs = np.nonzero(cm[1] > 0.5)
    out = [RISK_RANGE, RISK_RANGE, RISK_RANGE]
    theta = frame.slam.theta
    for yy, xx in zip(ys, xs):
        dx, dy = (xx - half) * res, (yy - half) * res
        d = math.hypot(dx, dy)
        if d > RISK_RANGE:
            continue
        bearing = _wrap(math.atan2(dy, dx) - theta)
        idx = _sector_index(bearing)
        if idx is not None:
            out[idx] = min(out[idx], d)
    return out


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sector_index(bearing):
    if abs(bearing) <= FRONT_HALF:
        return 0
    if FRONT_HALF < bearing <= SIDE_MAX:
        return 1
    if -SIDE_MAX <= bearing < -FRONT_HALF:
        return 2
    return None


def build_observation(frame, grid, goal, obs_mode, prev_action):
    """El MISMO vector que vio el modelo en entrenamiento (nav_env),
    construido desde el NavigationFrame real + el mapa (OccupancyGrid o
    ChunkedMapManager: ambos exponen cast_distance)."""
    pose = (frame.slam.x, frame.slam.y, frame.slam.theta)
    phi = _wrap(math.atan2(goal[1] - pose[1], goal[0] - pose[0]) - pose[2])
    d_goal = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
    risks = (frame.risk_front, frame.risk_left, frame.risk_right)

    if obs_mode == "doc":
        obs = np.array([
            min(frame.d_front, MAX_RANGE) / MAX_RANGE,
            min(frame.d_left, MAX_RANGE) / MAX_RANGE,
            min(frame.d_right, MAX_RANGE) / MAX_RANGE,
            phi / math.pi, min(d_goal, 10.0) / 10.0,
            risks[0] / 3.0, risks[1] / 3.0, risks[2] / 3.0,
        ], dtype=np.float32)
    else:
        # Cada rayo se mide como el MÍNIMO de un abanico de 3 sub-rayos
        # (±6°): el mapa pinta los obstáculos como celdas dispersas
        # (puntos de eco) y un rayo fino puede colarse entre dos celdas
        # y reportar libre un frente bloqueado; el abanico cierra esos
        # huecos, igual que la randomización usada en el entrenamiento.
        fan = math.radians(6)
        rays = np.array([
            min(grid.cast_distance(pose[0], pose[1], pose[2] + a + da,
                                   MAX_RANGE)
                for da in (-fan, 0.0, fan))
            for a in EXT_RAY_ANGLES])
        hz = _hazard_sector_from_costmap(frame)
        onehot = np.zeros(4, dtype=np.float32)
        onehot[prev_action] = 1.0
        obs = np.concatenate([
            rays / MAX_RANGE,
            np.array(hz) / RISK_RANGE,
            np.array(risks) / 3.0,
            [math.sin(phi), math.cos(phi), min(d_goal, 10.0) / 10.0],
            onehot,
        ]).astype(np.float32)
    return np.clip(obs, -1.0, 1.0), phi, d_goal


# ----------------------------------------------------------------------
# Submódulo de dirección (Algoritmo 5) — frases EXACTAS del documento
# ----------------------------------------------------------------------
def direction_text(action, d_goal, phi_goal, d_free=None,
                   near_prefix=True):
    """Alg. 5 del documento. d_free (opcional): espacio libre medido al
    frente; "Avance N metros" nunca promete más de lo que el mapa ve
    libre. near_prefix: el prefijo "Está cerca de su destino." se dice
    solo cuando el llamador lo decide (la primera vez por destino, para
    no repetirlo en cada frase)."""
    angulo = round(abs(math.degrees(phi_goal)))
    if action == 0:
        dist = min(d_goal, MAX_RANGE)
        if d_free is not None:
            dist = min(dist, d_free)
        metros = max(1, round(dist))
        instr = f"Avance {metros} metros" if metros > 1 else "Avance 1 metro"
    elif action == 1:
        instr = ("Gire ligeramente a la izquierda" if angulo <= 45
                 else f"Gire a la izquierda {angulo} grados")
    elif action == 2:
        instr = ("Gire ligeramente a la derecha" if angulo <= 45
                 else f"Gire a la derecha {angulo} grados")
    else:
        instr = "Deténgase"
    if near_prefix and d_goal < 5.0:
        instr = "Está cerca de su destino. " + instr
    return instr


# ----------------------------------------------------------------------
# El navegador (estados de sesión + inferencia + salida en tiempo real)
# ----------------------------------------------------------------------
class Navigator:
    """Une todo: comandos -> estados de sesión -> inferencia del modelo
    -> indicaciones una a una.

    Uso por ciclo (en el main):
        msgs = navigator.handle_command(texto)     # si hubo comando
        msgs += navigator.update(nav_frame, grid)  # cada ciclo
        para cada m en msgs: voz.reproducir(m.text, m.priority)
    """

    def __init__(self, model_base="models/nav_ppo", repeat_s=2.5,
                 decide_every_s=1.0):
        self.policy, self.obs_mode = load_policy(model_base)
        self.status = NavStatus()
        self.repeat_s = repeat_s          # no repetir la misma frase antes
        self.decide_every_s = decide_every_s
        self._prev_action = 3
        self._last_instr = None
        self._last_instr_t = 0.0
        self._last_decide_t = 0.0
        self._progress_ref = None         # (t, d_goal) para detectar bloqueo
        self._route = []                  # waypoints pendientes (futuro MAPS)
        self._near_announced = False      # "Está cerca..." una vez por destino

    # ---------------- comandos (formato de entrada) ----------------
    def handle_command(self, command, frame=None) -> List[NavMessage]:
        """command: NavCommand o texto en español. frame (opcional): el
        último NavigationFrame, para resolver destinos por landmark."""
        if isinstance(command, str):
            command = parse_command(command)
        st = self.status

        if command.tipo == "navegar":
            goal, label, err = self._resolve_destination(command, frame)
            if goal is None:
                return [NavMessage(err, priority=2, kind="respuesta")]
            st.goal, st.goal_label = goal, label
            st.waypoints = list(self._route)
            st.session = "activo"
            self._progress_ref = None
            self._near_announced = False
            return [NavMessage("Ruta calculada. Iniciando navegación.",
                               priority=2, kind="respuesta")]
        if command.tipo == "pausar":
            if st.session == "activo":
                st.session = "pausado"
                return [NavMessage("Navegación pausada.", priority=2,
                                   kind="respuesta")]
            return [NavMessage("No hay navegación activa.", priority=2,
                               kind="respuesta")]
        if command.tipo == "reanudar":
            if st.session == "pausado":
                st.session = "activo"
                return [NavMessage("Reanudando navegación.", priority=2,
                                   kind="respuesta")]
            return [NavMessage("No hay navegación pausada.", priority=2,
                               kind="respuesta")]
        if command.tipo == "detener":
            st.session = "en_espera"
            st.goal = st.goal_label = None
            st.waypoints = []
            return [NavMessage("Navegación detenida.", priority=2,
                               kind="respuesta")]
        if command.tipo == "ruta_alternativa":
            # Sin módulo de rutas globales aún: el agente local ya
            # replanifica solo. Se informa según RF-23.
            if st.session in ("activo", "pausado") and st.goal:
                self._progress_ref = None
                return [NavMessage("Calculando ruta alternativa.",
                                   priority=2, kind="respuesta")]
            return [NavMessage("No hay una ruta activa.", priority=2,
                               kind="respuesta")]
        if command.tipo == "estado":
            return [NavMessage(self._status_text(frame), priority=2,
                               kind="respuesta")]
        return [NavMessage("No entendí la instrucción. "
                           "¿Puede repetirla?", priority=2,
                           kind="respuesta")]

    def set_route(self, waypoints, destino_label=None):
        """Para el futuro módulo de rutas globales (MAPS): lista de
        waypoints (x, y); el último es el destino final."""
        self._route = list(waypoints)
        if waypoints:
            self.status.waypoints = list(waypoints)
            self.status.goal = tuple(waypoints[-1])
            self.status.goal_label = destino_label
            self.status.session = "activo"

    # ---------------- ciclo (formato de salida) ----------------
    def update(self, frame, grid) -> List[NavMessage]:
        """Llamar cada ciclo con el NavigationFrame y el mapa. Devuelve
        los mensajes NUEVOS de este ciclo (puede ser lista vacía)."""
        st = self.status
        msgs = []

        if st.session != "activo" or st.goal is None:
            st.d_goal = st.phi_goal = None
            return msgs

        pose = (frame.slam.x, frame.slam.y)
        # waypoint actual: el siguiente de la ruta, o el goal directo
        target = st.waypoints[0] if st.waypoints else st.goal
        if st.waypoints and math.hypot(target[0] - pose[0],
                                       target[1] - pose[1]) < WAYPOINT_RADIUS:
            st.waypoints.pop(0)
            target = st.waypoints[0] if st.waypoints else st.goal

        d_final = math.hypot(st.goal[0] - pose[0], st.goal[1] - pose[1])
        if d_final < ARRIVAL_RADIUS:
            st.session = "en_espera"
            st.goal = st.goal_label = None
            st.waypoints = []
            self._last_instr = None
            return [NavMessage("Ha llegado a su destino.", priority=1,
                               kind="evento")]

        now = time.time()
        if now - self._last_decide_t < self.decide_every_s:
            return msgs
        self._last_decide_t = now

        # Algoritmo 4: estado -> política -> acción
        obs, phi, d_goal = build_observation(frame, grid, target,
                                             self.obs_mode,
                                             self._prev_action)
        action = self.policy.predict(obs)

        # CAPA DE SEGURIDAD determinística (RNF-09, §6.4.3: opera por
        # encima del RL): si el modelo dice AVANZAR pero el frente está
        # bloqueado —según el mapa O la detección visual más próxima—,
        # se anula la acción y se gira hacia el lado con más espacio.
        # También rompe el estancamiento cuando el mapa aún tiene
        # huecos y el modelo insiste en avanzar contra un mueble.
        det_front = min((d.range_est for d in frame.detections
                         if d.range_est is not None
                         and abs(d.bearing) <= FRONT_HALF),
                        default=float("inf"))
        eff_front = min(frame.d_front, det_front)
        if action == 0 and eff_front < 0.6 and d_final > ARRIVAL_RADIUS:
            action = 1 if frame.d_left >= frame.d_right else 2

        self._prev_action = action
        st.last_action = action
        st.d_goal, st.phi_goal = d_final, phi

        # Algoritmo 5: acción -> instrucción, con control de repetición.
        # El prefijo "Está cerca de su destino." se anuncia UNA vez por
        # destino; "Avance N metros" se acota por el espacio libre real.
        announce_near = d_final < 5.0 and not self._near_announced
        if announce_near:
            self._near_announced = True
        instr = direction_text(action, d_final, phi,
                               d_free=frame.d_front,
                               near_prefix=announce_near)
        if instr != self._last_instr or now - self._last_instr_t > self.repeat_s:
            msgs.append(NavMessage(instr, priority=1, kind="instruccion",
                                   action=action))
            self._last_instr = instr
            self._last_instr_t = now

        # Detección de bloqueo: sin acercarse en 20 s de sesión activa
        if self._progress_ref is None or d_final < self._progress_ref[1] - 0.3:
            self._progress_ref = (now, d_final)
        elif now - self._progress_ref[0] > 20.0:
            self._progress_ref = (now, d_final)
            msgs.append(NavMessage(
                "No encuentro un paso libre hacia el destino. "
                "Puede pedir una ruta alternativa o detener la navegación.",
                priority=1, kind="evento"))
        return msgs

    # ---------------- internos ----------------
    def _resolve_destination(self, command, frame):
        """destino (texto) o punto -> (x, y). Usa los landmarks
        etiquetados del SLAM que vienen en el NavigationFrame."""
        if command.punto is not None:
            return tuple(command.punto), "punto", None
        if not command.destino:
            return None, None, "¿A dónde desea ir?"
        wanted = DEST_ALIASES.get(command.destino, command.destino)
        if frame is None or not frame.landmarks:
            return None, None, ("Aún no conozco ese lugar. "
                                "Recorra el entorno para mapearlo.")
        px, py = frame.slam.x, frame.slam.y
        candidates = [lm for lm in frame.landmarks
                      if lm.get("confirmed", True)
                      and lm.get("label") == wanted]
        if not candidates:
            return None, None, (f"No conozco ningún '{command.destino}' "
                                "en el mapa todavía.")
        best = min(candidates,
                   key=lambda lm: math.hypot(lm["x"] - px, lm["y"] - py))
        return (best["x"], best["y"]), wanted, None

    def _status_text(self, frame):
        st = self.status
        if st.session == "en_espera":
            base = "Sistema listo. Diga su destino."
        elif st.session == "pausado":
            base = "Navegación pausada."
        else:
            base = "Navegando"
            if st.goal_label:
                base += f" hacia {st.goal_label}"
            if st.d_goal is not None:
                base += f", faltan {st.d_goal:.0f} metros"
            base += "."
        if frame is not None and frame.room:
            base += f" Está en {frame.room}."
        return base
