"""
ekf_slam.py
SLAM simplificado basado en un Filtro de Kalman Extendido (EKF).

DISEÑO (ver DISENO_SLAM_EKF.md para la justificación completa):
- Estado del EKF: solo la pose 2D del usuario  mu = [x, y, theta]^T (3x1).
- Covarianza: P (3x3).
- El "mapa" de landmarks NO se guarda dentro del vector de estado (a
  diferencia del EKF-SLAM clásico con landmarks aumentados, donde el
  estado crece con cada landmark). En su lugar se mantiene una lista
  ligera de landmarks (auto-alimentada por las propias observaciones)
  que se usa para la asociación de datos y para alimentar al mapa de
  ocupación. Esto mantiene TODAS las operaciones matriciales en 3x3 de
  forma constante, sin importar cuántos obstáculos existan, lo cual es
  apropiado para hardware limitado (Raspberry Pi) y evita el crecimiento
  cuadrático de memoria/cómputo del EKF-SLAM completo.
- No se implementa SLAM visual (ORB-SLAM, LSD-SLAM, RTAB-Map, etc.).

Robustez añadida sobre el EKF de libro de texto:
- Un landmark nuevo es primero un CANDIDATO: necesita verse `confirm_hits`
  veces (en posiciones consistentes) antes de usarse para corregir la
  pose. Así una detección espuria (falso positivo de visión, eco fantasma
  del ultrasonido) no arrastra al filtro.
- La actualización de covarianza usa la forma de Joseph
  (P = (I-KH) P (I-KH)^T + K R K^T), numéricamente estable: garantiza que
  P siga siendo simétrica y semidefinida positiva aunque haya redondeo.
- Cada observación puede traer su propia incertidumbre (sigma_r,
  sigma_phi): un rango ultrasónico pesa más que un rango monocular.
- La posición de los landmarks se refina fuera del filtro con un
  promedio incremental, lo que mejora la asociación de datos sin
  agrandar el estado del EKF.
- Los candidatos que no se vuelven a ver se podan, para que la
  asociación siga siendo O(landmarks reales) y no acumule basura.
"""

import math
import time
import numpy as np

from interfaces import MotionEstimate, Observation, SlamOutput


def wrap_angle(a):
    """Normaliza un ángulo al rango (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# Etiquetas "genéricas": compatibles con cualquier otra al asociar/fusionar
# (un eco ultrasónico o una detección sin clase puede ser cualquier objeto).
GENERIC_LABELS = {"obstaculo", "eco_us", "libre", ""}


def labels_compatible(a, b):
    """Dos observaciones/landmarks pueden ser el mismo objeto físico solo
    si sus etiquetas no se contradicen (chair != table), con las etiquetas
    genéricas comodín."""
    return a == b or a in GENERIC_LABELS or b in GENERIC_LABELS


class Landmark:
    """Elemento del mapa ligero de landmarks (independiente del estado del
    EKF), usado solo para asociación de datos y como memoria de apoyo del
    mapa de ocupación."""
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
        """Promedio incremental de la posición del landmark con la nueva
        evidencia (fuera del EKF: no toca mu ni P). El peso decrece con
        seen_count pero con un PISO de 0.05: el landmark nunca se congela
        del todo y sigue co-adaptándose lentamente con la pose. Se evaluó
        sustituir esto por un filtro 1D ponderado por la varianza de cada
        observación y resultó PEOR en el banco de simulación (el landmark
        se congela con el sesgo de la pose temprana); no cambiar sin
        re-medir con simulation.py sobre varias semillas."""
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
        motion_noise: (sigma_trans_base, sigma_trans_prop, sigma_rot_base)
            desviaciones estándar del modelo de movimiento. La componente
            proporcional escala con la magnitud del desplazamiento, como es
            estándar en modelos de odometría con ruido.
        measurement_noise: (sigma_r, sigma_phi) desviaciones estándar POR
            DEFECTO de la observación fusionada; una Observation puede
            traer las suyas propias y estas se ignoran para ella.
        association_gate: umbral (en desviaciones; la distancia de
            Mahalanobis al cuadrado se compara contra gate^2) para aceptar
            la asociación observación-landmark.
        confirm_hits: número de veces que un candidato debe observarse en
            una posición consistente antes de usarse para corregir la pose.
        candidate_merge_radius: distancia (m) máxima entre la posición
            proyectada de una observación y un candidato existente para
            considerarlos el mismo objeto.
        candidate_ttl: ciclos sin re-observación tras los cuales un
            candidato NO confirmado se descarta (poda de espurios).

        landmark_pos_sigma: incertidumbre inicial (m) de la posición de un
            landmark del mapa ligero. Como el landmark NO está en el estado
            del EKF, su error de posición se contabiliza inflando S en la
            corrección (decrece con cada re-observación, ~1/sqrt(n)). Sin
            esto, el filtro trataría un landmark recién visto (posición
            derivada de UNA observación ruidosa) como verdad absoluta y se
            sobre-corregiría hacia él.
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

        self.landmarks = []          # lista de Landmark (candidatos + confirmados)
        self.last_associations = []  # [(obs, landmark|None)] del último correct()
        self.step_count = 0
        self.last_update_time = time.time()
        self.confidence = 1.0
        self.cycles_without_correction = 0

    # ------------------------------------------------------------------
    # 1) PREDICCIÓN
    # ------------------------------------------------------------------
    def predict(self, motion: MotionEstimate):
        """Predice la nueva pose a partir de la última estimación de
        movimiento disponible, usando un modelo de "arco" (la traslación
        se aplica en la dirección promedio entre la orientación inicial y
        final del paso), más preciso que Euler simple y con el mismo costo
        computacional (operaciones escalares + una matriz 3x3).
        """
        x, y, theta = self.mu
        d = motion.delta_trans
        drot = motion.delta_rot

        mid_theta = theta + drot / 2.0
        x_pred = x + d * math.cos(mid_theta)
        y_pred = y + d * math.sin(mid_theta)
        theta_pred = wrap_angle(theta + drot)

        # Jacobiano del modelo de movimiento respecto al estado (Gx), 3x3
        Gx = np.array([
            [1.0, 0.0, -d * math.sin(mid_theta)],
            [0.0, 1.0,  d * math.cos(mid_theta)],
            [0.0, 0.0, 1.0],
        ])

        # Ruido de proceso Q: escala con la magnitud del movimiento
        sigma_trans = self.sigma_trans_base + self.sigma_trans_prop * abs(d)
        sigma_rot = self.sigma_rot_base * max(abs(drot), 0.01) + math.radians(0.3)
        Q = np.diag([sigma_trans ** 2, sigma_trans ** 2, sigma_rot ** 2])

        self.mu = np.array([x_pred, y_pred, theta_pred])
        self.P = Gx @ self.P @ Gx.T + Q
        return self.mu.copy(), self.P.copy()

    # ------------------------------------------------------------------
    # 2) MODELO DE OBSERVACIÓN + ASOCIACIÓN DE DATOS
    # ------------------------------------------------------------------
    def _obs_noise(self, obs: Observation):
        """Matriz R (2x2) de la observación: usa la incertidumbre propia
        de la observación si viene definida, o los valores por defecto."""
        sr = obs.sigma_r if obs.sigma_r is not None else self.sigma_r
        sp = obs.sigma_phi if obs.sigma_phi is not None else self.sigma_phi
        return np.diag([sr ** 2, sp ** 2])

    def _predict_observation(self, lm: Landmark):
        """Observación esperada (rango, bearing) para un landmark dada la
        pose actual, más su Jacobiano H (2x3) respecto a la pose."""
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
        """Aporte (2x2) de la incertidumbre de posición del landmark a S:
        un error isotrópico de sigma_lm metros en la posición del landmark
        se proyecta como ~sigma_lm en rango y ~sigma_lm/r en bearing. La
        sigma decrece ~1/sqrt(n) con las re-observaciones."""
        sigma_lm = self.landmark_pos_sigma / math.sqrt(lm.seen_count)
        return np.diag([sigma_lm ** 2, (sigma_lm / max(r_hat, 0.3)) ** 2])

    def _associate(self, obs: Observation):
        """Asociación por vecino más cercano (nearest-neighbor) con
        "gating" por distancia de Mahalanobis, solo contra landmarks
        CONFIRMADOS. Devuelve (landmark, H, S, innovación) o None si no
        hay coincidencia suficientemente buena."""
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

    # ------------------------------------------------------------------
    # 3) CORRECCIÓN
    # ------------------------------------------------------------------
    def correct(self, observations):
        """Aplica la corrección del EKF con las observaciones disponibles
        en este ciclo. Si una observación no se asocia con ningún landmark
        confirmado, alimenta el mapa ligero de candidatos (NO el vector de
        estado del EKF, que se mantiene de tamaño constante).

        Nota: cuando una observación confirma a un candidato, NO se usa
        además para corregir la pose en ese mismo ciclo: ya se usó para
        fijar la posición del landmark, y usarla dos veces contaría la
        misma evidencia doble (P se encogería sin información real). El
        landmark corregirá a partir de la siguiente observación."""
        matched_any = False
        self.last_associations = []
        for obs in observations:
            if not self._valid_observation(obs):
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
            # Pérdida temporal de observaciones útiles (oclusión, entorno
            # sin features reconocibles, etc.): no se corrige, se continúa
            # únicamente con la predicción (dead-reckoning), y la
            # confianza reportada baja en consecuencia.
            self.cycles_without_correction += 1

        self._prune_candidates()
        self._dedup_confirmed()
        self._update_confidence()
        return self.mu.copy(), self.P.copy()

    def _apply_update(self, obs: Observation, match):
        """Actualización EKF estándar con forma de Joseph para P. El ruido
        efectivo de la observación incluye la incertidumbre de posición
        del landmark (ya venía incluida en S desde la asociación)."""
        lm, H, S, innov = match
        R_eff = S - H @ self.P @ H.T                 # R + ruido del landmark
        K = self.P @ H.T @ np.linalg.inv(S)          # ganancia de Kalman (3x2)
        self.mu = self.mu + K @ innov
        self.mu[2] = wrap_angle(self.mu[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
        self.P = 0.5 * (self.P + self.P.T)           # forzar simetría exacta
        # Piso de varianza: la pose nunca se declara "perfecta". Sin esto,
        # tras miles de correcciones P colapsa y el filtro se vuelve sordo
        # a observaciones nuevas (grave si al usuario lo mueven/tropieza).
        floor = np.array([1e-4, 1e-4, math.radians(0.5) ** 2])
        for i in range(3):
            if self.P[i, i] < floor[i]:
                self.P[i, i] = floor[i]

        # Refinar el landmark (fuera del filtro) con la pose ya corregida.
        lm.seen_count += 1
        lm.last_seen_step = self.step_count
        wx, wy = self._project(obs)
        lm.refine(wx, wy)

    @staticmethod
    def _valid_observation(obs: Observation):
        return (math.isfinite(obs.range_m) and math.isfinite(obs.bearing)
                and obs.range_m > 0.05)

    def _project(self, obs: Observation):
        """Proyecta una observación (rango, bearing) al plano del mundo
        usando la pose actual."""
        x, y, theta = self.mu
        wx = x + obs.range_m * math.cos(theta + obs.bearing)
        wy = y + obs.range_m * math.sin(theta + obs.bearing)
        return wx, wy

    def _same_object(self, lm: Landmark, obs: Observation):
        """¿La observación puede ser el mismo objeto físico que este
        landmark? Se compara en el espacio de la observación (rango,
        bearing desde la pose actual) contra la incertidumbre de la
        observación: la elipse resultante es ESTRECHA en dirección (la
        cámara mide bien el ángulo) y LARGA en profundidad (el rango
        monocular fluctúa ~30 %). Con una distancia euclidiana fija, la
        misma silla vista a 2.1 m y luego a 2.9 m creaba dos landmarks;
        con la elipse, ambos caen dentro y se fusionan."""
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
        if d2 < 9.0:                                   # dentro de 3 sigma
            return True
        # Piso euclidiano: puntos casi encimados siempre se fusionan
        wx, wy = self._project(obs)
        return math.hypot(lm.x - wx, lm.y - wy) < 0.3

    def _feed_candidate(self, obs: Observation):
        """Registra evidencia de un posible landmark nuevo SIN duplicar:
        1) si la observación es consistente (_same_object) con un landmark
           CONFIRMADO cercano (la asociación EKF pudo fallar por el gate),
           solo lo refuerza — no se crea nada;
        2) si es consistente con un candidato, lo refuerza (y se confirma
           al acumular los avistamientos requeridos);
        3) si no, se crea un candidato nuevo. El número de avistamientos
           requerido puede venir de la observación (min_confirm_hits): un
           eco de pared necesita más evidencia que un objeto de visión."""
        wx, wy = self._project(obs)
        required = max(self.confirm_hits, obs.min_confirm_hits or 0)

        for lm in self.landmarks:
            if lm.confirmed and self._same_object(lm, obs):
                lm.seen_count += 1
                lm.last_seen_step = self.step_count
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label      # el eco resultó ser p.ej. una silla
                return lm

        for lm in self.landmarks:
            if not lm.confirmed and self._same_object(lm, obs):
                lm.seen_count += 1
                lm.last_seen_step = self.step_count
                lm.refine(wx, wy)
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label
                # Si fuentes distintas coinciden en el mismo punto, vale el
                # requisito más laxo (la evidencia visual valida al eco).
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
        """Elimina candidatos no confirmados que llevan demasiados ciclos
        sin re-observarse (detecciones espurias)."""
        self.landmarks = [
            lm for lm in self.landmarks
            if lm.confirmed or
            (self.step_count - lm.last_seen_step) <= self.candidate_ttl
        ]

    def _dedup_confirmed(self, radius=0.7):
        """Fusiona landmarks CONFIRMADOS duplicados: mismo objeto (misma
        etiqueta no genérica) a menos de `radius` metros. Los duplicados
        aparecen cuando la pose derivó entre la primera y la segunda vez
        que se vio el objeto. Los ecos genéricos NO se fusionan entre sí:
        dos puntos de una misma pared son legítimamente distintos."""
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
        """Observaciones "ancladas" para el mapa de ocupación: las que se
        asociaron a un landmark se reemplazan por el rango/bearing EXACTO
        hacia la posición consolidada de ese landmark. Así el Ray Casting
        pinta el mismo objeto siempre en la misma celda (una silla = una
        marca), en vez de una mancha dispersa por el ruido del rango
        monocular. Las no asociadas se entregan tal cual (con su peso
        bajo, ver OccupancyGrid.update_from_scan)."""
        x, y, theta = self.mu
        out = []
        for obs, lm in self.last_associations:
            if lm is not None and lm.confirmed:
                dx, dy = lm.x - x, lm.y - y
                out.append(Observation(
                    range_m=math.hypot(dx, dy),
                    bearing=wrap_angle(math.atan2(dy, dx) - theta),
                    label=lm.label,
                    sigma_r=0.05,       # posición consolidada: marca firme
                    sigma_phi=obs.sigma_phi))
            else:
                out.append(obs)
        return out

    def _update_confidence(self):
        """Nivel de confianza en [0,1]: combina la incertidumbre de la
        pose (traza de P) con una penalización por ciclos consecutivos sin
        corrección (dead-reckoning puro)."""
        unc = float(np.trace(self.P))
        base_conf = 1.0 / (1.0 + unc)
        penalty = 0.9 ** self.cycles_without_correction
        self.confidence = max(0.0, min(1.0, base_conf * penalty))

    # ------------------------------------------------------------------
    # 4) CICLO INCREMENTAL COMPLETO
    # ------------------------------------------------------------------
    def step(self, motion: MotionEstimate, observations):
        """Ejecuta un ciclo completo del SLAM: predicción + corrección.
        Ver DISENO_SLAM_EKF.md, sección "Ciclo incremental", para el
        pseudocódigo equivalente."""
        self.step_count += 1
        self.predict(motion)
        self.correct(observations)
        self.last_update_time = time.time()
        return self.get_output()

    def get_output(self) -> SlamOutput:
        """Estructura de salida que el resto del sistema (mapa de
        ocupación, navegación local) debe consumir."""
        return SlamOutput(
            x=float(self.mu[0]),
            y=float(self.mu[1]),
            theta=float(self.mu[2]),
            covariance=self.P.tolist(),
            timestamp=self.last_update_time,
            confidence=self.confidence,
        )

    def confirmed_landmarks(self):
        """Landmarks confirmados (x, y) — útiles para depuración y para
        dibujarlos sobre el croquis."""
        return [(lm.x, lm.y) for lm in self.landmarks if lm.confirmed]

    def get_landmarks(self, only_confirmed=True):
        """Mapa ligero completo, con etiquetas. Pensado para los módulos
        de navegación/rutas: p. ej., si visión etiquetó un landmark como
        'door'/'salida', el planificador puede enrutar al usuario hacia
        su (x, y) dentro del mismo sistema de coordenadas de la pose."""
        return [
            {"x": float(lm.x), "y": float(lm.y), "label": lm.label,
             "seen_count": lm.seen_count, "confirmed": lm.confirmed}
            for lm in self.landmarks
            if lm.confirmed or not only_confirmed
        ]
