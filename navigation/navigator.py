"""
navigator.py — el runtime que USA el modelo entrenado.

construye el estado desde el NavigationFrame + mapa, corre la política
y suelta indicaciones una por una pa el módulo de voz. ojo: las frases
de instrucciones y de sesión son las LITERALES del documento del TT
(hay tests que las comparan tal cual), no moverles ni una letra.
puede cargar el .zip de SB3 (laptop) o el _policy.npz con numpy puro
(la raspberry no necesita SB3 ni torch).
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
from navigation_feed import (CRITICAL_DISTANCE_M, effective_front_distance,
                             needs_replan)

ARRIVAL_RADIUS = 0.6
WAYPOINT_RADIUS = 1.0
ACTION_NAMES = ("avanzar", "girar_izquierda", "girar_derecha", "detenerse")
# con pose poco fiable el "Avance N metros" se acota a 1 m
LOW_CONFIDENCE = 0.35
# giros forzados seguidos en el mismo sentido (~160°) antes de re-evaluar
MAX_FORCED_STREAK = 8


@dataclass
class NavCommand:
    """lo que el módulo de voz/PLN (futuro) le entrega a navegación.
    tipo: navegar | pausar | reanudar | detener | ruta_alternativa |
    estado | desconocido; destino es etiqueta o None, punto es (x, y).
    """
    tipo: str
    destino: Optional[str] = None
    punto: Optional[tuple] = None
    texto_original: str = ""


_DEST_WORDS = r"(?:ve|ir|vamos|llevame|llévame|navega|navegar|guiame|guíame) (?:a|al|a la|hacia|hasta)?\s*(?:el |la |los |las )?"


def parse_command(text: str) -> NavCommand:
    """español simple -> NavCommand. es el sustituto ligero del
    clasificador de intenciones del PLN mientras ese módulo no existe."""
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
    m = re.search(r"\bdestino\s+([a-záéíóúñ_ ]+)$", t)
    if m:
        return NavCommand("navegar", destino=m.group(1).strip(),
                          texto_original=text)
    return NavCommand("desconocido", texto_original=text)


# sinónimos español -> etiquetas de landmarks del SLAM
DEST_ALIASES = {
    "salida": "door", "puerta": "door", "entrada": "door",
    "silla": "chair", "mesa": "table", "sofa": "sofa", "sillon": "sofa",
    "cama": "bed", "refrigerador": "fridge", "refri": "fridge",
    "planta": "potted plant", "monitor": "monitor", "tele": "monitor",
    "television": "monitor", "escusado": "toilet", "bano": "toilet",
    "baño": "toilet", "lavabo": "sink",
}


@dataclass
class NavMessage:
    """mensaje listo pal TTS. priority: 0 alerta (interrumpe),
    1 instrucción, 2 confirmación/estado."""
    text: str
    priority: int = 1
    kind: str = "instruccion"
    action: Optional[int] = None


@dataclass
class NavStatus:
    """estado consultable de la sesión (GUI / debug)."""
    session: str = "en_espera"     # en_espera | activo | pausado
    goal: Optional[tuple] = None
    goal_label: Optional[str] = None
    d_goal: Optional[float] = None
    phi_goal: Optional[float] = None
    last_action: Optional[int] = None
    waypoints: List[tuple] = field(default_factory=list)


class NumpyPolicy:
    """actor PPO con numpy puro (MLP + tanh + argmax): el backend de la
    raspberry, solo necesita el _policy.npz."""

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
    """prefiere el .zip de SB3 si se puede; si no, cae al .npz."""
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


def _hazard_sector_from_costmap(frame):
    """dist al peligro de piso por sector, leída del costmap: los hoyos
    no aparecen en los rayos pq no bloquean nada, igual que en el sim."""
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
    """el MISMO vector que vio el modelo en el sim, pero armado desde
    el NavigationFrame real + el mapa."""
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
        # cada rayo es el mínimo de un abanico de 3 sub-rayos (±6°): el
        # mapa pinta obstáculos como celdas de eco dispersas y un rayo
        # fino se cuela entre dos celdas; el abanico cierra esos huecos
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


def direction_text(action, d_goal, phi_goal, d_free=None,
                   near_prefix=True):
    """acción -> frase, con las frases exactas del documento. d_free
    acota el "Avance N metros" a lo que el mapa ve libre; near_prefix
    lo decide el llamador (una vez por destino)."""
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


class Navigator:
    """une todo: comandos -> estados de sesión -> inferencia ->
    indicaciones una a una.

    por ciclo en el main:
        msgs = navigator.handle_command(texto)     # si hubo comando
        msgs += navigator.update(nav_frame, grid, alerts)
    """

    def __init__(self, model_base="models/nav_ppo", repeat_s=2.5,
                 decide_every_s=1.0):
        self.policy, self.obs_mode = load_policy(model_base)
        self.status = NavStatus()
        self.repeat_s = repeat_s
        self.decide_every_s = decide_every_s
        self._prev_action = 3
        self._last_instr = None
        self._last_instr_t = 0.0
        self._last_action_said = None
        self._last_decide_t = 0.0
        self._forced_streak = 0
        self._progress_ref = None
        self._route = []
        self._near_announced = False

    def handle_command(self, command, frame=None) -> List[NavMessage]:
        """command: NavCommand o texto; frame opcional pa resolver
        destinos por landmark."""
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
            # sin módulo de rutas globales aún; el agente local ya
            # replanifica solo
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
        """pal futuro módulo de rutas globales: waypoints (x, y), el
        último es el destino final."""
        self._route = list(waypoints)
        if waypoints:
            self.status.waypoints = list(waypoints)
            self.status.goal = tuple(waypoints[-1])
            self.status.goal_label = destino_label
            self.status.session = "activo"

    def update(self, frame, grid, alerts=None) -> List[NavMessage]:
        """llamar cada ciclo con el frame, el mapa y las alertas de
        seguridad; devuelve solo los mensajes nuevos. una alerta
        crítica hace que el modelo decida de inmediato."""
        st = self.status
        msgs = []

        if st.session != "activo" or st.goal is None:
            st.d_goal = st.phi_goal = None
            return msgs

        pose = (frame.slam.x, frame.slam.y)
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
        replan = needs_replan(alerts or [])
        if not replan and now - self._last_decide_t < self.decide_every_s:
            return msgs
        self._last_decide_t = now

        obs, phi, d_goal = build_observation(frame, grid, target,
                                             self.obs_mode,
                                             self._prev_action)
        action = self.policy.predict(obs)

        # capa de seguridad determinística por encima del RL: si el
        # modelo dice avanzar pero el frente está bloqueado (mismo
        # umbral y distancia efectiva que las alertas), se anula y se
        # gira — con histéresis en el sentido, pq si no en una esquina
        # oirías izquierda/derecha/izquierda sin salir nunca
        eff_front = effective_front_distance(frame)
        if action == 0 and eff_front < CRITICAL_DISTANCE_M \
                and d_final > ARRIVAL_RADIUS:
            action = self._forced_turn(frame, phi)
        else:
            self._forced_streak = 0

        self._prev_action = action
        st.last_action = action
        st.d_goal, st.phi_goal = d_final, phi

        # el prefijo "Está cerca..." solo una vez por destino; con pose
        # poco fiable el avance prometido se acota a 1 m
        announce_near = d_final < 5.0 and not self._near_announced
        if announce_near:
            self._near_announced = True
        d_free = eff_front
        if frame.slam.confidence < LOW_CONFIDENCE:
            d_free = min(d_free, 1.0)
        instr = direction_text(action, d_final, phi, d_free=d_free,
                               near_prefix=announce_near)
        # "Avance N" tras "Avance M" en menos de repeat_s cuenta como la
        # misma indicación (N fluctúa con el ray casting)
        same_forward = (action == 0 and self._last_action_said == 0
                        and now - self._last_instr_t <= self.repeat_s)
        if (instr != self._last_instr and not same_forward) \
                or now - self._last_instr_t > self.repeat_s:
            msgs.append(NavMessage(instr, priority=1, kind="instruccion",
                                   action=action))
            self._last_instr = instr
            self._last_instr_t = now
            self._last_action_said = action

        # bloqueo: sin acercarse en 20 s de sesión activa
        if self._progress_ref is None or d_final < self._progress_ref[1] - 0.3:
            self._progress_ref = (now, d_final)
        elif now - self._progress_ref[0] > 20.0:
            self._progress_ref = (now, d_final)
            msgs.append(NavMessage(
                "No encuentro un paso libre hacia el destino. "
                "Puede pedir una ruta alternativa o detener la navegación.",
                priority=1, kind="evento"))
        return msgs

    def _forced_turn(self, frame, phi_goal):
        """sentido del giro forzado: seguir el giro en curso (hasta
        MAX_FORCED_STREAK, pa no dar la vuelta entera); si no, hacia el
        destino si ese lado tiene espacio, si no el más libre."""
        self._forced_streak += 1
        if self._prev_action in (1, 2) \
                and self._forced_streak <= MAX_FORCED_STREAK:
            return self._prev_action
        self._forced_streak = 1
        prefer = 1 if phi_goal >= 0 else 2
        d_pref = frame.d_left if prefer == 1 else frame.d_right
        d_other = frame.d_right if prefer == 1 else frame.d_left
        if d_pref >= CRITICAL_DISTANCE_M or d_pref >= d_other:
            return prefer
        return 3 - prefer

    def _resolve_destination(self, command, frame):
        """destino (texto) o punto -> (x, y), usando los landmarks
        etiquetados del SLAM."""
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
