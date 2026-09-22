"""
interfaces.py
Interfaces / estructuras de datos compartidas entre el módulo SLAM y el
resto del sistema (percepción, mapa de ocupación, navegación).

Estas clases actúan como "contratos" para que los módulos que aún no están
implementados (odometría / estimación de movimiento real, sensores
ultrasónicos físicos, mapa de ocupación real) puedan conectarse sin cambiar
la lógica interna del SLAM. Donde el proyecto todavía no tiene un módulo
real, se deja un dato o clase simulada claramente marcada con TODO.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotionEstimate:
    """Estimación de movimiento entre dos iteraciones.
    TODO: reemplazar el generador simulado (SimulatedMotion en sim_world.py)
    por el módulo real de estimación de movimiento (por ejemplo, odometría
    visual o IMU) cuando esté disponible.
    """
    delta_trans: float      # desplazamiento lineal (m) entre t-1 y t
    delta_rot: float        # cambio de orientación (rad) entre t-1 y t
    dt: float = 0.1         # intervalo de tiempo (s) de la medición


@dataclass
class VisionDetection:
    """Detección de obstáculo obtenida por el módulo de visión real
    (vision/TT-NavIA, YOLOv8-seg) a través de vision_bridge.py.

    bearing: ángulo relativo al usuario (rad). 0 = al frente,
             + = hacia la izquierda (convención matemática estándar).
    range_est: distancia monocular estimada a partir del tamaño del bbox
             (m). Es una estimación gruesa: si hay un sensor ultrasónico
             apuntando en esa dirección, su rango tiene prioridad.
    moving: True si el módulo de visión determinó que el objeto se mueve
             (los objetos en movimiento NO deben usarse como landmarks).
    """
    bearing: float
    label: str = "obstaculo"
    pixel_width: int = 0
    range_est: Optional[float] = None
    moving: bool = False
    confidence: float = 1.0


@dataclass
class UltrasonicReading:
    """Lectura de un sensor ultrasónico.
    TODO: sustituir por lectura real de hardware (por ejemplo HC-SR04 sobre
    GPIO en Raspberry Pi). Mientras tanto, sim_world.SimulatedUltrasonicArray
    genera lecturas coherentes con un cuarto simulado.
    """
    range_m: float
    sensor_bearing: float = 0.0   # dirección fija hacia la que apunta el sensor
    max_range: float = 4.0
    valid: bool = True


@dataclass
class Observation:
    """Observación fusionada (rango + orientación) lista para el EKF.
    Se construye en vision_bridge.fuse_observations() a partir de una
    VisionDetection emparejada con una UltrasonicReading (o, en su defecto,
    con la distancia monocular estimada por visión).

    sigma_r / sigma_phi: incertidumbre (desv. estándar) propia de ESTA
    observación. Si se dejan en None, el EKF usa sus valores por defecto.
    Esto permite que un rango ultrasónico (preciso) pese más que un rango
    monocular (impreciso) sin cambiar la lógica del filtro.

    can_init_landmark: si False, la observación puede CORREGIR la pose
    (asociándose a un landmark ya conocido) pero nunca crear un landmark
    nuevo. Se usa para los ecos ultrasónicos sin detección visual: una
    pared no es un punto fijo (el punto de eco se desliza al caminar) y
    convertirla en landmark puntual sesga el filtro.
    """
    range_m: float
    bearing: float
    label: str = "obstaculo"
    sigma_r: Optional[float] = None
    sigma_phi: Optional[float] = None
    can_init_landmark: bool = True
    # Avistamientos mínimos para confirmar un landmark creado por esta
    # observación (None = el valor por defecto del EKF). Los ecos de pared
    # piden más evidencia porque su punto de reflexión se desliza.
    min_confirm_hits: Optional[int] = None


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class SlamOutput:
    """Estructura de salida estándar del módulo SLAM, tal como la requiere
    el resto del sistema (mapa de ocupación / navegación local)."""
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
