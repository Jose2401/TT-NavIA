import math
import random

from interfaces import MotionEstimate, UltrasonicReading, VisionDetection
from ekf_slam import wrapAngle


class SimulatedWorld:
    def __init__(self, walls, obstacles):
        self.walls = walls
        self.obstacles= obstacles

    @staticmethod
    def defaultRoom():
        paredes = [
            (0.0, 0.0, 6.0, 0.0),
            (6.0, 0.0, 6.0, 4.0),
            (6.0, 4.0, 0.0, 4.0),
            (0.0, 4.0, 0.0, 0.0),
        ]
        cosas = [
            (2.0, 1.0, 0.40, "table"),
            (4.6, 1.1, 0.25, "chair"),
            (1.0, 3.1, 0.45, "sofa"),
            (5.3, 3.3, 0.20, "potted plant"),
            (3.3, 2.9, 0.30, "chair"),
        ]
        return SimulatedWorld(paredes, cosas)

    # primer choque del rayo con pared u obstaculo
    def rayDistance(self, x, y, angle, maxRange=4.0):
        c, s = math.cos(angle), math.sin(angle)
        mejor = float("inf")

        for (ax, ay, bx, by) in self.walls:
            ex, ey = bx - ax, by - ay
            denom = c * ey - s * ex
            if abs(denom) < 1e-12:
                continue
            t = ((ax - x) * ey - (ay - y) * ex) / denom
            u =((x - ax) * s - (y - ay) * c) / -denom
            if t > 0.0 and 0.0 <= u <= 1.0:
                mejor = min(mejor, t)

        for (cx, cy, r, _label) in self.obstacles:
            fx, fy = x - cx, y - cy
            b = fx * c + fy * s
            disc = b * b - (fx * fx + fy * fy - r * r)
            if disc < 0.0:
                continue
            t= -b - math.sqrt(disc)
            if t > 0.0:
                mejor = min(mejor, t)

        return mejor if mejor <= maxRange else float("inf")

    def occupiedCells(self, grid, samplesPerM=40):
        celdas = set()
        for (ax, ay, bx, by) in self.walls:
            n = max(2, int(math.hypot(bx - ax, by - ay) * samplesPerM))
            for i in range(n + 1):
                t = i / n
                gx, gy = grid.worldToGrid(ax + t * (bx - ax), ay + t * (by - ay))
                if grid.inBounds(gx, gy):
                    celdas.add((gy, gx))
        for (cx, cy, r, _label) in self.obstacles:
            n = max(8, int(2 * math.pi * r * samplesPerM))
            for i in range(n):
                a = 2 * math.pi * i / n
                gx, gy = grid.worldToGrid(cx + r * math.cos(a), cy + r * math.sin(a))
                if grid.inBounds(gx, gy):
                    celdas.add((gy, gx))
        return celdas


class SimulatedUltrasonicArray:
    def __init__(self, world: SimulatedWorld,
                 bearings=(0.0, math.radians(40), math.radians(-40)),
                 maxRange=4.0, sigmaR=0.02):
        self.world = world
        self.bearings = list(bearings)
        self.maxRange =maxRange
        self.sigmaR = sigmaR

    def read(self, truePose):
        x, y, theta = truePose
        lecturas = []
        for b in self.bearings:
            d = self.world.rayDistance(x, y, theta + b, self.maxRange)
            if math.isinf(d):
                lecturas.append(UltrasonicReading(
                    range_m=self.maxRange, sensor_bearing=b,
                    max_range=self.maxRange, valid=False))
            else:
                lecturas.append(UltrasonicReading(
                    range_m=max(0.05, d + random.gauss(0.0, self.sigmaR)),
                    sensor_bearing=b, max_range=self.maxRange, valid=True))
        return lecturas


# detecciones fake tipo camara
def simulatedVision(world: SimulatedWorld, truePose,
                    fov=math.radians(70), maxRange=4.5,
                    sigmaBearing=math.radians(2), sigmaRangePct=0.25,
                    dropProb=0.0):
    x, y, theta = truePose
    detecciones = []
    for (cx, cy, r, label) in world.obstacles:
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        if dist > maxRange or dist < 1e-6:
            continue
        bearing = wrapAngle(math.atan2(dy, dx) - theta)
        if abs(bearing) > fov / 2.0:
            continue
        hit = world.rayDistance(x, y, theta + bearing, maxRange=dist + r)
        if hit < dist - r - 1e-6:
            continue
        if random.random() < dropProb:
            continue
        surf = max(0.1, dist - r)
        detecciones.append(VisionDetection(
            bearing=bearing + random.gauss(0.0, sigmaBearing),
            label=label,
            range_est=max(0.1, surf * (1.0 + random.gauss(0.0, sigmaRangePct))),
            moving=False,
        ))
    return detecciones


class KeyboardUser:
    ODOM_SIGMA_TRANS = 0.006
    ODOM_SIGMA_ROT = math.radians(0.5)

    def __init__(self, world, start=(0.7, 0.7, 0.0)):
        self.world = world
        self.truePose = list(start)

    def command(self, forward=0.0, rot=0.0, dt=1 / 20.0):
        x, y, theta = self.truePose

        if forward > 0:
            libre = self.world.rayDistance(x, y, theta, maxRange=2.0)
            if libre < forward + 0.15:
                forward = 0.0
        elif forward < 0:
            libre = self.world.rayDistance(x, y, theta + math.pi,
                                           maxRange=2.0)
            if libre < -forward + 0.15:
                forward = 0.0

        mid = theta + rot / 2.0
        self.truePose[0] = x + forward * math.cos(mid)
        self.truePose[1] =y + forward * math.sin(mid)
        self.truePose[2] = wrapAngle(theta + rot)

        ruidoT = random.gauss(0.0, self.ODOM_SIGMA_TRANS) if forward else 0.0
        ruidoR = random.gauss(0.0, self.ODOM_SIGMA_ROT) \
            if (forward or rot) else 0.0
        return MotionEstimate(delta_trans=forward + ruidoT,
                              delta_rot=rot + ruidoR, dt=dt)


class SimulatedMotion:
    def __init__(self, start=(0.7, 0.7, 0.0), waypoints=None,
                 stepDist=0.06, maxRot=math.radians(9),
                 sigmaTrans=0.006, sigmaRot=math.radians(0.5),
                 dt=1 / 15.0, loop=True):
        self.truePose = list(start)
        self.waypoints = list(waypoints) if waypoints else [
            (4.0, 0.7), (5.3, 2.0), (4.4, 3.3), (2.4, 2.2),
            (0.8, 2.4), (0.7, 0.7),
        ]
        self.wpIndex = 0
        self.stepDist = stepDist
        self.maxRot= maxRot
        self.sigmaTrans = sigmaTrans
        self.sigmaRot = sigmaRot
        self.dt = dt
        self.loop =loop
        self.finished = False

    # un ciclo hacia el waypoint, regresa odom con ruido
    def get(self):
        x, y, theta = self.truePose

        if self.finished:
            return MotionEstimate(0.0, 0.0, self.dt)

        wx, wy = self.waypoints[self.wpIndex]
        if math.hypot(wx - x, wy - y) < 0.25:
            self.wpIndex += 1
            if self.wpIndex >= len(self.waypoints):
                if self.loop:
                    self.wpIndex = 0
                else:
                    self.finished = True
                    return MotionEstimate(0.0, 0.0, self.dt)
            wx, wy = self.waypoints[self.wpIndex]

        deseado = math.atan2(wy - y, wx - x)
        err = wrapAngle(deseado - theta)
        dRot = max(-self.maxRot, min(self.maxRot, err))
        d = self.stepDist if abs(err) < math.radians(25) else 0.0

        mid = theta + dRot / 2.0
        self.truePose[0] = x + d * math.cos(mid)
        self.truePose[1] = y + d * math.sin(mid)
        self.truePose[2] =wrapAngle(theta + dRot)

        return MotionEstimate(
            delta_trans=d + random.gauss(0.0, self.sigmaTrans),
            delta_rot=dRot + random.gauss(0.0, self.sigmaRot),
            dt=self.dt,
        )
