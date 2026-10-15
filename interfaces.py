"""
interfaces.py
los dataclasses que comparten slam, vision, mapa y navegacion.
basicamente los "contratos" pa que los modulos que faltan se
conecten sin mover la logica del EKF.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotionEstimate:
    """movimiento entre dos iteraciones.
    TODO: cambiar SimulatedMotion (sim_world.py) por odometria real (visual o IMU)."""
    delta_trans: float      # desplazamiento (m)
    delta_rot: float        # cambio de orientacion (rad)
    dt: float = 0.1


@dataclass
class VisionDetection:
    """deteccion del modulo de vision (YOLOv8-seg via vision_bridge).
    ojo con la convencion: bearing en rad, 0 = al frente, + = izquierda.
    range_est es monocular y gruesa; si hay ultrasonido en esa direccion, gana.
    moving=True -> no usar como landmark."""
    bearing: float
    label: str = "obstaculo"
    pixel_width: int = 0
    range_est: Optional[float] = None
    moving: bool = False
    confidence: float = 1.0
    # peligro a nivel de piso (hoyo, coladera): el ultrasonido no lo ve
    # pq no sobresale, va a la capa de peligros aunque el rayo "pase"
    hazard: bool = False


@dataclass
class UltrasonicReading:
    """lectura de un sensor ultrasonico.
    TODO: hardware real (HC-SR04 en la raspberry); por ahora sim_world la simula."""
    range_m: float
    sensor_bearing: float = 0.0   # direccion fija del sensor
    max_range: float = 4.0
    valid: bool = True


@dataclass
class Observation:
    """observacion fusionada (rango + bearing) lista pal EKF,
    la arma vision_bridge.fuse_observations().
    sigma_r/sigma_phi: incertidumbre propia; None = defaults del EKF
    (asi el ultrasonido pesa mas que el rango monocular)."""
    range_m: float
    bearing: float
    label: str = "obstaculo"
    sigma_r: Optional[float] = None
    sigma_phi: Optional[float] = None
    # False = puede corregir contra landmarks conocidos pero no crear nuevos
    # (ecos de pared: el punto de eco se desliza al caminar, no es punto fijo)
    can_init_landmark: bool = True
    # hits minimos pa confirmar landmark (None = default del EKF)
    min_confirm_hits: Optional[int] = None
    # True -> la celda final va a la capa de peligros (hoyo/agua),
    # no bloquea el rayo pero la navegacion no debe pisarla
    is_hazard: bool = False
    # ojo: False = no toca el EKF, solo alimenta el mapa. es pal eco de
    # pared sin respaldo visual: su punto de reflexion se desliza al caminar
    # y si lo dejas corregir arruina la pose. medido en sim: ATE 0.53 -> 0.27
    can_correct: bool = True


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class SlamOutput:
    """salida estandar del SLAM pal resto del sistema."""
    x: float
    y: float
    theta: float
    covariance: List[List[float]]
    timestamp: float
    confidence: float

    def as_dict(self):
        return {
            "pose": {"x": self.x, "y": self.y, "theta": self.theta},
            "covarianza": self.covariance,
            "timestamp": self.timestamp,
            "confianza": self.confidence,
        }


@dataclass
class NavigationFrame:
    """paquete de entrada pal modulo de navegacion (DRL, aun no existe).
    costmap: (2, H, W) float32 centrado en el usuario, ejes del mundo;
    canal 0 = prob ocupacion (0.5 = desconocido), canal 1 = peligros de piso.
    d_*/risk_* son el vector de estado del agente DRL (TT_2026_B045 §7.3.1):
    distancias por ray casting y riesgo max por sector (0-3).
    phi_goal/d_goal los pondra el modulo de rutas (no existe aun)."""
    slam: SlamOutput
    costmap: object                  # np.ndarray (2, H, W) float32
    costmap_resolution: float
    costmap_size_m: float
    detections: List[VisionDetection]
    landmarks: List[dict]
    d_front: float = float("inf")
    d_left: float = float("inf")
    d_right: float = float("inf")
    risk_front: int = 0
    risk_left: int = 0
    risk_right: int = 0
    chunk: Optional[tuple] = None    # (i, j) del chunk actual
    room: Optional[str] = None
    ultrasonic: Optional[List[UltrasonicReading]] = None
