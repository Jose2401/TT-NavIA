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
    (paquete vision/, YOLOv8-seg) a través de vision_bridge.py.

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
    # True si es un peligro A NIVEL DE PISO (hoyo, coladera, agua, escalón
    # descendente): no sobresale del suelo, así que el ultrasonido no lo ve
    # y el mapa debe marcarlo como celda peligrosa aunque el rayo "pase".
    hazard: bool = False


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
    # True si la celda final del rayo debe marcarse en la CAPA DE PELIGROS
    # del mapa (hoyo/coladera/agua): no bloquea el rayo como una pared,
    # pero la navegación debe tratarla como intransitable.
    is_hazard: bool = False
    # Si False, la observación NO participa en la corrección del EKF (ni
    # crea landmarks): solo llega al mapa de ocupación. Es el caso de los
    # ecos de pared sin respaldo visual: su punto de reflexión se DESLIZA
    # a lo largo de la pared al caminar (no es un punto fijo del mundo) y
    # tratarlo como landmark puntual corrompe la pose. Medido en
    # simulación con geometría correcta: ATE 0.53 -> 0.27 al excluirlos.
    can_correct: bool = True


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


@dataclass
class NavigationFrame:
    """Paquete de entrada para el módulo de NAVEGACIÓN (aprendizaje por
    refuerzo profundo, aún no implementado). Es el contrato de salida
    conjunto de SLAM + mapeo + visión: todo lo que la política necesita
    por ciclo, ya en el mismo sistema de coordenadas.

    costmap: arreglo (2, H, W) float32 centrado en el usuario, ejes
        alineados al mundo (theta de la pose indica hacia dónde mira
        dentro del parche):
          canal 0 = probabilidad de ocupación [0,1] (0.5 = desconocido)
          canal 1 = máscara de peligros de piso [0,1] (hoyos, coladeras)
    costmap_resolution / costmap_size_m: geometría del parche.
    detections: lista de VisionDetection del ciclo (obstáculos con
        dirección, distancia estimada, si se mueven, si son hazard).
    landmarks: mapa ligero etiquetado (dicts de EKFSlam.get_landmarks()),
        útil para metas semánticas ("ir a la puerta").
    chunk / room: dónde está el usuario dentro del mapa por chunks.
    ultrasonic: lecturas del ciclo (lista vacía hasta que exista el
        hardware; la interfaz ya queda fija).

    d_front / d_left / d_right y risk_front / risk_left / risk_right son
    el vector de estado que la documentación (TT_2026_B045, §7.3.1)
    define para el agente DRL:
        s_t = [d_front, d_left, d_right, phi_goal, d_goal,
               r_front, r_left, r_right]
    Las distancias (m) salen de ray casting sobre el mapa de ocupación
    desde la pose actual; los riesgos son el máximo por sector
    (0 libre, 1 bajo, 2 medio, 3 alto). phi_goal/d_goal los aportará el
    módulo de rutas (no existen aún).
    """
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
    room: Optional[str] = None       # etiqueta del cuarto actual, si se conoce
    ultrasonic: Optional[List[UltrasonicReading]] = None
