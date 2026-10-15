"""
ekf_slam.py
EKF-SLAM simplificado: el estado es solo la pose [x, y, theta] (3x3 todo),
los landmarks van en una lista ligera aparte, no dentro del estado.
asi el costo no crece con los obstaculos (clave pa la raspberry).
detalles y justificacion completa en DISENO_SLAM_EKF.md.
"""

import math
import time
import numpy as np

from interfaces import MotionEstimate, Observation, SlamOutput


def wrap_angle(a):
    """normaliza a (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# etiquetas comodin: compatibles con cualquier otra al asociar/fusionar
GENERIC_LABELS = {"obstaculo", "eco_us", "libre", ""}


def labels_compatible(a, b):
    return a == b or a in GENERIC_LABELS or b in GENERIC_LABELS


class Landmark:
    """elemento del mapa ligero (fuera del estado del EKF)."""
    __slots__ = ("x", "y", "seen_count", "confirmed",
                 "last_seen_step", "label", "required_hits")

    def __init__(self, x, y, step=0, label="obstaculo", required_hits=2):
        self.x = x
        self.y = y
        self.seen_count = 1
        self.confirmed = False
        self.last_seen_step = step
        self.label = label
        self.required_hits = required_hits

    def refine(self, x_obs, y_obs):
        # promedio incremental fuera del EKF. ojo: el piso de 0.05 en alfa
        # es a proposito, sin el, el landmark se congela con el sesgo de la
        # pose temprana. probamos un filtro 1D por varianza y salio peor,
        # no cambiar sin re-medir con simulation.py en varias semillas
        alpha = max(0.05, 1.0 / self.seen_count)
        self.x = (1.0 - alpha) * self.x + alpha * x_obs
        self.y = (1.0 - alpha) * self.y + alpha * y_obs


class EKFSlam:
    def __init__(self, x0=0.0, y0=0.0, theta0=0.0,
                 initial_cov=None,
                 motion_noise=(0.02, 0.05, math.radians(2)),
                 measurement_noise=(0.05, math.radians(3)),
                 association_gate=3.0,
                 confirm_hits=2,
                 candidate_merge_radius=0.6,
                 candidate_ttl=40,
                 landmark_pos_sigma=0.4):
        """
        motion_noise: (sigma_trans_base, sigma_trans_prop, sigma_rot_base).
        measurement_noise: (sigma_r, sigma_phi) defaults; cada Observation
            puede traer los suyos.
        association_gate: umbral mahalanobis (se compara contra gate^2).
        confirm_hits: veces que un candidato debe verse antes de corregir pose.
        candidate_ttl: ciclos sin re-ver antes de podar un candidato.
        landmark_pos_sigma: ojo, afinada a 0.4 — como el landmark no esta en
            el estado, su error se mete inflando S (decrece ~1/sqrt(n)).
            sin esto el filtro trata un landmark recien visto como verdad
            absoluta y se sobre-corrige. no mover sin re-medir.
        """
        self.mu = np.array([x0, y0, theta0], dtype=np.float64)
        self.P = np.array(initial_cov, dtype=np.float64) if initial_cov is not None \
            else np.diag([0.01, 0.01, math.radians(1) ** 2])

        self.sigma_trans_base, self.sigma_trans_prop, self.sigma_rot_base = motion_noise
        self.sigma_r, self.sigma_phi = measurement_noise
        self.gate = association_gate
        self.confirm_hits = confirm_hits
        self.candidate_merge_radius = candidate_merge_radius
        self.candidate_ttl = candidate_ttl
        self.landmark_pos_sigma = landmark_pos_sigma

        self.landmarks = []          # candidatos + confirmados
        self.last_associations = []  # [(obs, landmark|None)] del ultimo correct()
        self.step_count = 0
        self.last_update_time = time.time()
        self.confidence = 1.0
        self.cycles_without_correction = 0

    # ---------- prediccion ----------
    def predict(self, motion: MotionEstimate):
        """modelo de arco: la traslacion se aplica en la direccion promedio
        del paso, mas preciso que euler simple y cuesta lo mismo."""
        x, y, theta = self.mu
        d = motion.delta_trans
        drot = motion.delta_rot

        mid_theta = theta + drot / 2.0
        x_pred = x + d * math.cos(mid_theta)
        y_pred = y + d * math.sin(mid_theta)
        theta_pred = wrap_angle(theta + drot)

        # jacobiano Gx (3x3)
        Gx = np.array([
            [1.0, 0.0, -d * math.sin(mid_theta)],
            [0.0, 1.0,  d * math.cos(mid_theta)],
            [0.0, 0.0, 1.0],
        ])

        # Q escala con la magnitud del movimiento
        sigma_trans = self.sigma_trans_base + self.sigma_trans_prop * abs(d)
        sigma_rot = self.sigma_rot_base * max(abs(drot), 0.01) + math.radians(0.3)
        Q = np.diag([sigma_trans ** 2, sigma_trans ** 2, sigma_rot ** 2])

        self.mu = np.array([x_pred, y_pred, theta_pred])
        self.P = Gx @ self.P @ Gx.T + Q
        return self.mu.copy(), self.P.copy()

    # ---------- observacion + asociacion ----------
    def _obs_noise(self, obs: Observation):
        sr = obs.sigma_r if obs.sigma_r is not None else self.sigma_r
        sp = obs.sigma_phi if obs.sigma_phi is not None else self.sigma_phi
        return np.diag([sr ** 2, sp ** 2])

    def _predict_observation(self, lm: Landmark):
        """(rango, bearing) esperados pal landmark + jacobiano H (2x3)."""
        x, y, theta = self.mu
        dx = lm.x - x
        dy = lm.y - y
        q = max(dx * dx + dy * dy, 1e-9)
        r_hat = math.sqrt(q)
        phi_hat = wrap_angle(math.atan2(dy, dx) - theta)

        H = np.array([
            [-dx / r_hat, -dy / r_hat, 0.0],
            [dy / q, -dx / q, -1.0],
        ])
        return np.array([r_hat, phi_hat]), H

    def _landmark_noise(self, lm: Landmark, r_hat):
        # error de posicion del landmark proyectado a (rango, bearing),
        # decrece ~1/sqrt(n) con re-observaciones
        sigma_lm = self.landmark_pos_sigma / math.sqrt(lm.seen_count)
        return np.diag([sigma_lm ** 2, (sigma_lm / max(r_hat, 0.3)) ** 2])

    def _associate(self, obs: Observation):
        """nearest-neighbor con gate de mahalanobis, solo contra confirmados."""
        z = np.array([obs.range_m, obs.bearing])
        R = self._obs_noise(obs)
        pool = (lm for lm in self.landmarks if lm.confirmed)

        best = None
        best_d2 = self.gate ** 2
        for lm in pool:
            if not labels_compatible(obs.label, lm.label):
                continue
            z_hat, H = self._predict_observation(lm)
            innov = z - z_hat
            innov[1] = wrap_angle(innov[1])
            S = H @ self.P @ H.T + R + self._landmark_noise(lm, z_hat[0])
            try:
                d2 = float(innov.T @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                continue
            if d2 < best_d2:
                best_d2 = d2
                best = (lm, H, S, innov)
        return best

    # ---------- correccion ----------
    def correct(self, observations):
        """correccion EKF del ciclo. lo que no se asocia alimenta candidatos.
        ojo: una obs que confirma un candidato NO corrige pose ese mismo
        ciclo, seria contar la misma evidencia doble y P se encoge de gratis."""
        matched_any = False
        self.last_associations = []
        for obs in observations:
            if not self._valid_observation(obs):
                continue

            if not obs.can_correct:
                # eco de pared sin respaldo visual: solo va al mapa,
                # se guarda pa que observations_for_map() lo entregue tal cual
                self.last_associations.append((obs, None))
                continue

            match = self._associate(obs)
            if match is None:
                lm = self._feed_candidate(obs) if obs.can_init_landmark else None
                self.last_associations.append((obs, lm))
                continue

            matched_any = True
            self._apply_update(obs, match)
            self.last_associations.append((obs, match[0]))

        if matched_any:
            self.cycles_without_correction = 0
        else:
            # sin observaciones utiles: puro dead-reckoning y baja la confianza
            self.cycles_without_correction += 1

        self._prune_candidates()
        self._dedup_confirmed()
        self._update_confidence()
        return self.mu.copy(), self.P.copy()

    def _apply_update(self, obs: Observation, match):
        """update EKF con forma de joseph (P queda simetrica y PSD aunque
        haya redondeo)."""
        lm, H, S, innov = match
        R_eff = S - H @ self.P @ H.T                 # R + ruido del landmark
        K = self.P @ H.T @ np.linalg.inv(S)          # ganancia (3x2)
        self.mu = self.mu + K @ innov
        self.mu[2] = wrap_angle(self.mu[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)           # simetria exacta
        # piso de varianza pq si P colapsa el filtro se queda sordo
        # (grave si al usuario lo mueven o tropieza)
        floor = np.array([1e-4, 1e-4, math.radians(0.5) ** 2])
        for i in range(3):
            if self.P[i, i] < floor[i]:
                self.P[i, i] = floor[i]

        # refinar landmark con la pose ya corregida
        lm.seen_count += 1
        lm.last_seen_step = self.step_count
        wx, wy = self._project(obs)
        lm.refine(wx, wy)

    @staticmethod
    def _valid_observation(obs: Observation):
        return (math.isfinite(obs.range_m) and math.isfinite(obs.bearing)
                and obs.range_m > 0.05)

    def _project(self, obs: Observation):
        x, y, theta = self.mu
        wx = x + obs.range_m * math.cos(theta + obs.bearing)
        wy = y + obs.range_m * math.sin(theta + obs.bearing)
        return wx, wy

    def _same_object(self, lm: Landmark, obs: Observation):
        """compara en espacio (rango, bearing) con elipse: estrecha en
        angulo (la camara mide bien) y larga en profundidad (el rango
        monocular fluctua ~30%). con distancia euclidiana fija la misma
        silla a 2.1 y 2.9 m creaba dos landmarks xd"""
        if not labels_compatible(obs.label, lm.label):
            return False
        x, y, theta = self.mu
        dx, dy = lm.x - x, lm.y - y
        r_lm = math.hypot(dx, dy)
        phi_lm = wrap_angle(math.atan2(dy, dx) - theta)

        sr = max(obs.sigma_r if obs.sigma_r is not None else self.sigma_r, 0.15)
        sp = max(obs.sigma_phi if obs.sigma_phi is not None else self.sigma_phi,
                 math.radians(3))
        d2 = ((r_lm - obs.range_m) / sr) ** 2 \
            + (wrap_angle(phi_lm - obs.bearing) / sp) ** 2
        if d2 < 9.0:                                   # 3 sigma
            return True
        # piso euclidiano: puntos casi encimados siempre se fusionan
        wx, wy = self._project(obs)
        return math.hypot(lm.x - wx, lm.y - wy) < 0.3

    def _feed_candidate(self, obs: Observation):
        """mete evidencia de posible landmark sin duplicar: refuerza un
        confirmado cercano, o un candidato (y lo confirma si junta hits),
        o crea candidato nuevo. min_confirm_hits puede venir de la obs
        (un eco de pared pide mas evidencia que un objeto de vision)."""
        wx, wy = self._project(obs)
        required = max(self.confirm_hits, obs.min_confirm_hits or 0)

        for lm in self.landmarks:
            if lm.confirmed and self._same_object(lm, obs):
                lm.seen_count += 1
                lm.last_seen_step = self.step_count
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label      # el eco resulto ser p.ej. una silla
                return lm

        for lm in self.landmarks:
            if not lm.confirmed and self._same_object(lm, obs):
                lm.seen_count += 1
                lm.last_seen_step = self.step_count
                lm.refine(wx, wy)
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label
                # si fuentes distintas coinciden, vale el requisito mas laxo
                lm.required_hits = min(lm.required_hits, required)
                if lm.seen_count >= lm.required_hits:
                    lm.confirmed = True
                return lm

        lm = Landmark(wx, wy, step=self.step_count, label=obs.label,
                      required_hits=required)
        if required <= 1:
            lm.confirmed = True
        self.landmarks.append(lm)
        return lm

    def _prune_candidates(self):
        self.landmarks = [
            lm for lm in self.landmarks
            if lm.confirmed or
            (self.step_count - lm.last_seen_step) <= self.candidate_ttl
        ]

    def _dedup_confirmed(self, radius=0.7):
        """fusiona confirmados duplicados (misma etiqueta no generica y
        cerca). ojo: los ecos genericos no se fusionan entre si, dos puntos
        de una misma pared son legitimamente distintos."""
        merged = True
        while merged:
            merged = False
            confirmed = [lm for lm in self.landmarks if lm.confirmed]
            for i in range(len(confirmed)):
                for j in range(i + 1, len(confirmed)):
                    a, b = confirmed[i], confirmed[j]
                    if a.label in GENERIC_LABELS or a.label != b.label:
                        continue
                    if math.hypot(a.x - b.x, a.y - b.y) >= radius:
                        continue
                    w = a.seen_count + b.seen_count
                    a.x = (a.x * a.seen_count + b.x * b.seen_count) / w
                    a.y = (a.y * a.seen_count + b.y * b.seen_count) / w
                    a.seen_count = w
                    a.last_seen_step = max(a.last_seen_step, b.last_seen_step)
                    self.landmarks.remove(b)
                    merged = True
                    break
                if merged:
                    break

    def observations_for_map(self):
        """obs "ancladas" pal mapa: las asociadas a un landmark etiquetado
        se reemplazan por el rango/bearing exacto al landmark consolidado
        (una silla = una marca, no una mancha). ojo: los ecos genericos
        (eco_us) NO se anclan — el punto de reflexion de la pared se
        desliza y su landmark deriva; anclar pintaba fantasmas y borraba
        la pared real. el eco crudo si termina sobre la superficie."""
        x, y, theta = self.mu
        out = []
        for obs, lm in self.last_associations:
            if (lm is not None and lm.confirmed
                    and lm.label not in GENERIC_LABELS):
                dx, dy = lm.x - x, lm.y - y
                out.append(Observation(
                    range_m=math.hypot(dx, dy),
                    bearing=wrap_angle(math.atan2(dy, dx) - theta),
                    label=lm.label,
                    sigma_r=0.05,       # posicion consolidada: marca firme
                    sigma_phi=obs.sigma_phi))
            else:
                out.append(obs)
        return out

    def _update_confidence(self):
        # traza de P + castigo por ciclos en dead-reckoning
        unc = float(np.trace(self.P))
        base_conf = 1.0 / (1.0 + unc)
        penalty = 0.9 ** self.cycles_without_correction
        self.confidence = max(0.0, min(1.0, base_conf * penalty))

    # ---------- ciclo completo ----------
    def step(self, motion: MotionEstimate, observations):
        """un ciclo SLAM: predict + correct."""
        self.step_count += 1
        self.predict(motion)
        self.correct(observations)
        self.last_update_time = time.time()
        return self.get_output()

    def get_output(self) -> SlamOutput:
        return SlamOutput(
            x=float(self.mu[0]),
            y=float(self.mu[1]),
            theta=float(self.mu[2]),
            covariance=self.P.tolist(),
            timestamp=self.last_update_time,
            confidence=self.confidence,
        )

    def confirmed_landmarks(self):
        return [(lm.x, lm.y) for lm in self.landmarks if lm.confirmed]

    def get_landmarks(self, only_confirmed=True):
        """mapa ligero con etiquetas, pa navegacion/rutas (ej. enrutar
        hacia un landmark 'door')."""
        return [
            {"x": float(lm.x), "y": float(lm.y), "label": lm.label,
             "seen_count": lm.seen_count, "confirmed": lm.confirmed}
            for lm in self.landmarks
            if lm.confirmed or not only_confirmed
        ]

    # ---------- persistencia del mapa ligero ----------
    def export_landmarks(self):
        """solo confirmados; los candidatos son evidencia transitoria, no mapa."""
        return [
            {"x": float(lm.x), "y": float(lm.y), "label": lm.label,
             "seen_count": int(lm.seen_count),
             "required_hits": int(lm.required_hits)}
            for lm in self.landmarks if lm.confirmed
        ]

    def import_landmarks(self, data):
        """restaura landmarks de otra sesion: entran confirmados pa poder
        re-localizar desde el primer ciclo. duplicados los fusiona
        _dedup_confirmed en el siguiente ciclo."""
        for d in data:
            lm = Landmark(float(d["x"]), float(d["y"]),
                          step=self.step_count,
                          label=d.get("label", "obstaculo"),
                          required_hits=int(d.get("required_hits", 2)))
            lm.seen_count = max(1, int(d.get("seen_count", 1)))
            lm.confirmed = True
            self.landmarks.append(lm)
        self._dedup_confirmed()
