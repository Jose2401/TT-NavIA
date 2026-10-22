from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotionEstimate:
    delta_trans: float
    delta_rot: float
    dt: float = 0.1


@dataclass
class VisionDetection:
    bearing: float
    label: str = "obstaculo"
    pixel_width: int = 0
    range_est: Optional[float] = None
    moving: bool = False
    confidence: float = 1.0
    hazard: bool = False


@dataclass
class UltrasonicReading:
    range_m: float
    sensor_bearing: float = 0.0
    max_range: float = 4.0
    valid: bool = True


@dataclass
class Observation:
    range_m: float
    bearing: float
    label: str = "obstaculo"
    sigma_r: Optional[float] = None
    sigma_phi: Optional[float] = None
    can_init_landmark: bool = True
    min_confirm_hits: Optional[int] = None
    is_hazard: bool = False
    can_correct: bool = True


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class SlamOutput:
    x: float
    y: float
    theta: float
    covariance: List[List[float]]
    timestamp: float
    confidence: float

    # pa mandarlo como json
    def asDict(self):
        papu = {"x": self.x, "y": self.y, "theta": self.theta}
        return {
            "pose": papu,
            "covarianza": self.covariance,
            "timestamp": self.timestamp,
            "confianza": self.confidence,
        }


@dataclass
class NavigationFrame:
    slam: SlamOutput
    costmap: object
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
    chunk: Optional[tuple] = None
    room: Optional[str] = None
    ultrasonic: Optional[List[UltrasonicReading]] = None
