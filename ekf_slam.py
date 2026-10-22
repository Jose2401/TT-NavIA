import math
import time
import numpy as np

from interfaces import MotionEstimate, Observation, SlamOutput


def wrapAngle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


GENERIC_LABELS = {"obstaculo", "eco_us", "libre", ""}


def labelsCompatibles(a, b):
    return a == b or a in GENERIC_LABELS or b in GENERIC_LABELS


class Landmark:
    __slots__ = ("x", "y", "seenCount", "confirmed",
                 "lastSeenStep", "label", "requiredHits")

    def __init__(self, x, y, step=0, label="obstaculo", requiredHits=2):
        self.x = x
        self.y= y
        self.seenCount = 1
        self.confirmed =False
        self.lastSeenStep = step
        self.label = label
        self.requiredHits= requiredHits

    def refine(self, xObs, yObs):
        alfa = max(0.05, 1.0 / self.seenCount)
        self.x = (1.0 - alfa) * self.x + alfa * xObs
        self.y =(1.0 - alfa) * self.y + alfa * yObs


class EKFSlam:
    def __init__(self, x0=0.0, y0=0.0, theta0=0.0,
                 initialCov=None,
                 motionNoise=(0.02, 0.05, math.radians(2)),
                 measurementNoise=(0.05, math.radians(3)),
                 associationGate=3.0,
                 confirmHits=2,
                 candidateMergeRadius=0.6,
                 candidateTtl=40,
                 landmarkPosSigma=0.4):
        self.mu = np.array([x0, y0, theta0], dtype=np.float64)
        if initialCov is not None:
            self.P = np.array(initialCov, dtype=np.float64)
        else:
            self.P=np.diag([0.01, 0.01, math.radians(1) ** 2])

        self.sigmaTransBase, self.sigmaTransProp, self.sigmaRotBase = motionNoise
        self.sigmaR, self.sigmaPhi = measurementNoise
        self.gate = associationGate
        self.confirmHits= confirmHits
        self.candidateMergeRadius = candidateMergeRadius
        self.candidateTtl =candidateTtl
        self.landmarkPosSigma = landmarkPosSigma

        self.landmarks = []
        self.lastAssociations=[]
        self.stepCount = 0
        self.lastUpdateTime = time.time()
        self.confidence = 1.0
        self.cyclesWithoutCorrection = 0

    # prediccion con modelo de arco
    def predict(self, motion: MotionEstimate):
        x, y, theta = self.mu
        d = motion.delta_trans
        dRot= motion.delta_rot

        thetaMid = theta + dRot / 2.0
        xPred = x + d * math.cos(thetaMid)
        yPred =y + d * math.sin(thetaMid)
        thetaPred = wrapAngle(theta + dRot)

        Gx = np.array([
            [1.0, 0.0, -d * math.sin(thetaMid)],
            [0.0, 1.0,  d * math.cos(thetaMid)],
            [0.0, 0.0, 1.0],
        ])

        sigTrans = self.sigmaTransBase + self.sigmaTransProp * abs(d)
        sigRot= self.sigmaRotBase * max(abs(dRot), 0.01) + math.radians(0.3)
        Q = np.diag([sigTrans ** 2, sigTrans ** 2, sigRot ** 2])

        self.mu = np.array([xPred, yPred, thetaPred])
        self.P = Gx @ self.P @ Gx.T + Q
        return self.mu.copy(), self.P.copy()

    def _ruidoObs(self, obs: Observation):
        sr = obs.sigma_r if obs.sigma_r is not None else self.sigmaR
        sp=obs.sigma_phi if obs.sigma_phi is not None else self.sigmaPhi
        return np.diag([sr ** 2, sp ** 2])

    def _predecirObs(self, lm: Landmark):
        x, y, theta = self.mu
        dx = lm.x - x
        dy= lm.y - y
        q = max(dx * dx + dy * dy, 1e-9)
        rHat = math.sqrt(q)
        phiHat = wrapAngle(math.atan2(dy, dx) - theta)

        H = np.array([
            [-dx / rHat, -dy / rHat, 0.0],
            [dy / q, -dx / q, -1.0],
        ])
        return np.array([rHat, phiHat]), H

    def _ruidoLm(self, lm: Landmark, rHat):
        sigmaLm = self.landmarkPosSigma / math.sqrt(lm.seenCount)
        return np.diag([sigmaLm ** 2, (sigmaLm / max(rHat, 0.3)) ** 2])

    def _asociar(self, obs: Observation):
        z = np.array([obs.range_m, obs.bearing])
        R = self._ruidoObs(obs)
        pool = (lm for lm in self.landmarks if lm.confirmed)

        mejor = None
        mejorD2 =self.gate ** 2
        for papuLm in pool:
            if not labelsCompatibles(obs.label, papuLm.label):
                continue
            zHat, H = self._predecirObs(papuLm)
            innov = z - zHat
            innov[1] = wrapAngle(innov[1])
            S = H @ self.P @ H.T + R + self._ruidoLm(papuLm, zHat[0])
            try:
                d2 = float(innov.T @ np.linalg.solve(S, innov))
            except np.linalg.LinAlgError:
                continue
            if d2 < mejorD2:
                mejorD2 = d2
                mejor= (papuLm, H, S, innov)
        return mejor

    # correccion, lo que no asocia se va a candidatos
    def correct(self, observations):
        huboMatch = False
        self.lastAssociations = []
        for obs in observations:
            if not self._obsValida(obs):
                continue

            if not obs.can_correct:
                self.lastAssociations.append((obs, None))
                continue

            match = self._asociar(obs)
            if match is None:
                lm = self._feedCandidato(obs) if obs.can_init_landmark else None
                self.lastAssociations.append((obs, lm))
                continue

            huboMatch =True
            self._aplicarUpdate(obs, match)
            self.lastAssociations.append((obs, match[0]))

        if huboMatch:
            self.cyclesWithoutCorrection = 0
        else:
            self.cyclesWithoutCorrection += 1

        self._podarCandidatos()
        self._dedupConfirmados()
        self._updateConfianza()
        return self.mu.copy(), self.P.copy()

    def _aplicarUpdate(self, obs: Observation, match):
        lm, H, S, innov = match
        rEff = S - H @ self.P @ H.T
        K = self.P @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ innov
        self.mu[2] = wrapAngle(self.mu[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ rEff @ K.T
        self.P= 0.5 * (self.P + self.P.T)
        piso = np.array([1e-4, 1e-4, math.radians(0.5) ** 2])
        for i in range(3):
            if self.P[i, i] < piso[i]:
                self.P[i, i] = piso[i]

        lm.seenCount += 1
        lm.lastSeenStep = self.stepCount
        wx, wy = self._proyectar(obs)
        lm.refine(wx, wy)

    @staticmethod
    def _obsValida(obs: Observation):
        return (math.isfinite(obs.range_m) and math.isfinite(obs.bearing)
                and obs.range_m > 0.05)

    def _proyectar(self, obs: Observation):
        x, y, theta = self.mu
        wx = x + obs.range_m * math.cos(theta + obs.bearing)
        wy =y + obs.range_m * math.sin(theta + obs.bearing)
        return wx, wy

    def _mismoObjeto(self, lm: Landmark, obs: Observation):
        if not labelsCompatibles(obs.label, lm.label):
            return False
        x, y, theta = self.mu
        dx, dy = lm.x - x, lm.y - y
        rLm = math.hypot(dx, dy)
        phiLm = wrapAngle(math.atan2(dy, dx) - theta)

        sr = max(obs.sigma_r if obs.sigma_r is not None else self.sigmaR, 0.15)
        sp= max(obs.sigma_phi if obs.sigma_phi is not None else self.sigmaPhi,
                 math.radians(3))
        d2 = ((rLm - obs.range_m) / sr) ** 2 \
            + (wrapAngle(phiLm - obs.bearing) / sp) ** 2
        if d2 < 9.0:
            return True
        wx, wy = self._proyectar(obs)
        return math.hypot(lm.x - wx, lm.y - wy) < 0.3

    def _feedCandidato(self, obs: Observation):
        wx, wy = self._proyectar(obs)
        requeridos = max(self.confirmHits, obs.min_confirm_hits or 0)

        for lm in self.landmarks:
            if lm.confirmed and self._mismoObjeto(lm, obs):
                lm.seenCount += 1
                lm.lastSeenStep = self.stepCount
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label
                return lm

        for lm in self.landmarks:
            if not lm.confirmed and self._mismoObjeto(lm, obs):
                lm.seenCount+= 1
                lm.lastSeenStep = self.stepCount
                lm.refine(wx, wy)
                if lm.label in GENERIC_LABELS and obs.label not in GENERIC_LABELS:
                    lm.label = obs.label
                lm.requiredHits = min(lm.requiredHits, requeridos)
                if lm.seenCount >= lm.requiredHits:
                    lm.confirmed = True
                return lm

        nuevo = Landmark(wx, wy, step=self.stepCount, label=obs.label,
                         requiredHits=requeridos)
        if requeridos <= 1:
            nuevo.confirmed=True
        self.landmarks.append(nuevo)
        return nuevo

    def _podarCandidatos(self):
        self.landmarks = [
            lm for lm in self.landmarks
            if lm.confirmed or
            (self.stepCount - lm.lastSeenStep) <= self.candidateTtl
        ]

    def _dedupConfirmados(self, radio=0.7):
        merged = True
        while merged:
            merged = False
            confirmados = [lm for lm in self.landmarks if lm.confirmed]
            for i in range(len(confirmados)):
                for j in range(i + 1, len(confirmados)):
                    a, b = confirmados[i], confirmados[j]
                    if a.label in GENERIC_LABELS or a.label != b.label:
                        continue
                    if math.hypot(a.x - b.x, a.y - b.y) >= radio:
                        continue
                    w =a.seenCount + b.seenCount
                    a.x = (a.x * a.seenCount + b.x * b.seenCount) / w
                    a.y = (a.y * a.seenCount + b.y * b.seenCount) / w
                    a.seenCount = w
                    a.lastSeenStep = max(a.lastSeenStep, b.lastSeenStep)
                    self.landmarks.remove(b)
                    merged = True
                    break
                if merged:
                    break

    def observationsForMap(self):
        x, y, theta = self.mu
        out = []
        for obs, lm in self.lastAssociations:
            if (lm is not None and lm.confirmed
                    and lm.label not in GENERIC_LABELS):
                dx, dy = lm.x - x, lm.y - y
                out.append(Observation(
                    range_m=math.hypot(dx, dy),
                    bearing=wrapAngle(math.atan2(dy, dx) - theta),
                    label=lm.label,
                    sigma_r=0.05,
                    sigma_phi=obs.sigma_phi))
            else:
                out.append(obs)
        return out

    def _updateConfianza(self):
        incert = float(np.trace(self.P))
        confBase = 1.0 / (1.0 + incert)
        castigo= 0.9 ** self.cyclesWithoutCorrection
        self.confidence = max(0.0, min(1.0, confBase * castigo))

    def step(self, motion: MotionEstimate, observations):
        self.stepCount += 1
        self.predict(motion)
        self.correct(observations)
        self.lastUpdateTime = time.time()
        return self.getOutput()

    def getOutput(self) -> SlamOutput:
        return SlamOutput(
            x=float(self.mu[0]),
            y=float(self.mu[1]),
            theta=float(self.mu[2]),
            covariance=self.P.tolist(),
            timestamp=self.lastUpdateTime,
            confidence=self.confidence,
        )

    def confirmedLandmarks(self):
        return [(lm.x, lm.y) for lm in self.landmarks if lm.confirmed]

    def getLandmarks(self, onlyConfirmed=True):
        return [
            {"x": float(lm.x), "y": float(lm.y), "label": lm.label,
             "seen_count": lm.seenCount, "confirmed": lm.confirmed}
            for lm in self.landmarks
            if lm.confirmed or not onlyConfirmed
        ]

    def exportLandmarks(self):
        return [
            {"x": float(lm.x), "y": float(lm.y), "label": lm.label,
             "seen_count": int(lm.seenCount),
             "required_hits": int(lm.requiredHits)}
            for lm in self.landmarks if lm.confirmed
        ]

    def importLandmarks(self, data):
        for papu in data:
            lm = Landmark(float(papu["x"]), float(papu["y"]),
                          step=self.stepCount,
                          label=papu.get("label", "obstaculo"),
                          requiredHits=int(papu.get("required_hits", 2)))
            lm.seenCount = max(1, int(papu.get("seen_count", 1)))
            lm.confirmed= True
            self.landmarks.append(lm)
        self._dedupConfirmados()
