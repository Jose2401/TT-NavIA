"""
figuras pa la doc del TT2 (SLAM, ray casting y mapeo), todo sale del
codigo real, nada dibujado a mano. correr:
    python -m tests.figuras_doc --out docs/TT_2026_B045/Capturas_TT2
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle, Wedge

from ekf_slam import EKFSlam
from occupancy_grid import (OccupancyGrid, L_OCC, L_FREE, L_CLAMP,
                            OCC_THRESHOLD, L_BLOCK, bresenham)
from map_manager import ChunkedMapManager
from interfaces import Observation, MotionEstimate
from sim_world import (SimulatedWorld, SimulatedMotion,
                       SimulatedUltrasonicArray, simulatedVision)
from vision_bridge import fuseObservations, freeSpaceRays
from navigation_feed import (buildNavigationFrame, FRONT_HALF_ANGLE,
                             SIDE_MAX_ANGLE)
from tests.metrics import (compute_ate, compute_orientation_error,
                           map_occupancy_accuracy)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def correrSim(seed=0, steps=300, oclusion=None, snapshots=(), dropProb=0.0,
              escala=1.0, sesgoRot=0.0, usBearings=None):
    random.seed(seed)
    np.random.seed(seed)
    world = SimulatedWorld.defaultRoom()
    motionSim = SimulatedMotion(start=(0.7, 0.7, 0.0))
    usArray = SimulatedUltrasonicArray(world) if usBearings is None \
        else SimulatedUltrasonicArray(world, bearings=usBearings)
    slam = EKFSlam(x0=0.7, y0=0.7, theta0=0.0)
    # mismo ruido de odometria pero sin corregir nunca = dead reckoning
    soloOdo = EKFSlam(x0=0.7, y0=0.7, theta0=0.0)
    grid = OccupancyGrid(sizeM=8.0, resolution=0.05, origin=(1.0, 1.0))

    res = {"gt": [], "est": [], "odo": [], "P": [], "conf": [],
           "snaps": {}, "t": []}
    for paso in range(steps):
        motion = motionSim.get()
        # odometria con sesgo (paso mal calibrado / giroscopio con bias)
        motion = MotionEstimate(motion.delta_trans * escala,
                                motion.delta_rot + sesgoRot, motion.dt)
        truePose = tuple(motionSim.truePose)
        us = usArray.read(truePose)
        dets = simulatedVision(world, truePose, dropProb=dropProb)
        if oclusion and oclusion[0] <= paso < oclusion[1]:
            dets = []
        obs = fuseObservations(dets, us)

        t0 = time.perf_counter()
        out = slam.step(motion, obs)
        t1 = time.perf_counter()
        grid.updateFromScan((out.x, out.y, out.theta),
                            slam.observationsForMap() + freeSpaceRays(us))
        t2 = time.perf_counter()
        res["t"].append((t1 - t0, t2 - t1))

        o2 = soloOdo.step(motion, [])
        res["gt"].append(truePose)
        res["est"].append((out.x, out.y, out.theta))
        res["odo"].append((o2.x, o2.y, o2.theta))
        res["P"].append(np.array(out.covariance))
        res["conf"].append(out.confidence)
        if paso + 1 in snapshots:
            res["snaps"][paso + 1] = (grid.logOdds.copy(),
                                      list(res["est"]))
    res.update(world=world, slam=slam, grid=grid)
    return res


def dibujarCuarto(ax, world, alpha=1.0):
    for (ax0, ay0, bx, by) in world.walls:
        ax.plot([ax0, bx], [ay0, by], "-", color="k", lw=2, alpha=alpha)
    for (cx, cy, r, label) in world.obstacles:
        ax.add_patch(Circle((cx, cy), r, fill=False, ec="tab:brown",
                            lw=1.2, ls="--", alpha=alpha))
        ax.text(cx, cy - r - 0.18, label, ha="center", fontsize=7,
                color="tab:brown")


def imshowGrid(ax, grid, logOdds=None):
    lo = grid.logOdds if logOdds is None else logOdds
    occ = 1.0 / (1.0 + np.exp(-lo))
    ext = [-grid.origin[0], grid.n * grid.resolution - grid.origin[0],
           -grid.origin[1], grid.n * grid.resolution - grid.origin[1]]
    ax.imshow(occ, cmap="Greys", origin="lower", extent=ext, vmin=0, vmax=1)


# ---------------- SLAM ----------------

def figTrayectorias(r, out):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    dibujarCuarto(ax, r["world"])
    gt, est, odo = (np.array(r[k]) for k in ("gt", "est", "odo"))
    ax.plot(gt[:, 0], gt[:, 1], "--", color="tab:green", lw=1.6,
            label="Trayectoria real")
    ax.plot(odo[:, 0], odo[:, 1], ":", color="tab:gray", lw=1.8,
            label="Solo odometría (sin corrección)")
    ax.plot(est[:, 0], est[:, 1], "-", color="tab:blue", lw=1.6,
            label="Estimada por el EKF")
    for i in range(0, len(est), 30):
        P = r["P"][i][:2, :2]
        vals, vecs = np.linalg.eigh(P)
        ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(Ellipse(est[i, :2], 2 * 2 * math.sqrt(vals[1]),
                             2 * 2 * math.sqrt(vals[0]), angle=ang,
                             fill=False, ec="tab:red", lw=0.9))
    lms = r["slam"].getLandmarks()
    ax.plot([l["x"] for l in lms], [l["y"] for l in lms], "x",
            color="tab:orange", ms=8, mew=2, label="Landmarks confirmados")
    ax.plot([], [], color="tab:red", lw=0.9, label="Elipse 2σ de la pose")
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(-0.4, 4.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8)
    ax.set_title("Trayectoria estimada vs. real (cuarto simulado, semilla 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slam_trayectorias.png"), dpi=170)
    plt.close(fig)


def figIncertidumbre(rNorm, rOcl, oclusion, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for r, nombre, col in ((rNorm, "Operación normal", "tab:blue"),
                           (rOcl, "Con oclusión visual", "tab:red")):
        tr = [np.trace(P) for P in r["P"]]
        a1.plot(tr, color=col, lw=1.3, label=nombre)
        a2.plot(r["conf"], color=col, lw=1.3, label=nombre)
    for a in (a1, a2):
        a.axvspan(*oclusion, color="tab:red", alpha=0.08)
        a.grid(alpha=0.3)
    a1.set_ylabel("traza(P)")
    a1.set_yscale("log")
    a1.legend(fontsize=8, loc="upper left")
    a1.set_title("Evolución de la incertidumbre de la pose")
    a2.set_ylabel("confianza")
    a2.set_xlabel("ciclo")
    a2.text(np.mean(oclusion), 0.05, "sin detecciones\nvisuales",
            ha="center", fontsize=8, color="tab:red")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slam_incertidumbre.png"), dpi=170)
    plt.close(fig)


def tablaSemillas(out):
    filas = []
    for seed in range(6):
        for drop in (0.0, 0.6):
            r = correrSim(seed=seed, dropProb=drop)
            ate = compute_ate(r["est"], r["gt"])
            ateOdo = compute_ate(r["odo"], r["gt"])
            ori = math.degrees(compute_orientation_error(r["est"], r["gt"]))
            acc = map_occupancy_accuracy(r["grid"],
                                         r["world"].occupiedCells(r["grid"]))
            tSlam = 1000 * np.mean([t[0] for t in r["t"]])
            tRay = 1000 * np.mean([t[1] for t in r["t"]])
            filas.append(dict(seed=seed, drop=drop, ate=ate, ate_odo=ateOdo,
                              ori=ori, prec=acc["precision"],
                              rec=acc["recall"],
                              lms=len(r["slam"].confirmedLandmarks()),
                              t_slam_ms=tSlam, t_ray_ms=tRay))
    with open(os.path.join(out, "metricas_slam.json"), "w") as fh:
        json.dump(filas, fh, indent=1)
    return filas


def experimentoDeriva(out):
    casos = (("300 ciclos, sin sesgo", 300, 1.0, 0.0),
             ("1200 ciclos, sin sesgo", 1200, 1.0, 0.0),
             ("1200 ciclos, paso +10 %", 1200, 1.10, 0.0),
             ("1200 ciclos, giro +0.1°/ciclo", 1200, 1.0, math.radians(0.1)),
             ("1200 ciclos, ambos sesgos", 1200, 1.10, math.radians(0.1)))
    filas = []
    for nombre, steps, esc, sr in casos:
        ekf, odo = [], []
        for seed in range(6):
            r = correrSim(seed=seed, steps=steps, escala=esc, sesgoRot=sr)
            ekf.append(compute_ate(r["est"], r["gt"]))
            odo.append(compute_ate(r["odo"], r["gt"]))
        filas.append(dict(caso=nombre, ekf=float(np.mean(ekf)),
                          ekf_max=float(np.max(ekf)),
                          odo=float(np.mean(odo)),
                          odo_max=float(np.max(odo))))
    with open(os.path.join(out, "metricas_deriva.json"), "w") as fh:
        json.dump(filas, fh, indent=1, ensure_ascii=False)
    return filas


# banco usa 3 US al frente, el prototipo 2 MB1232 laterales
def experimentoUltrasonido(out):
    casos = (("3 virtuales (0°, ±40°)", None),
             ("2 laterales (±90°), como el prototipo",
              (math.radians(90), math.radians(-90))),
             ("sin ultrasonido", ()))
    filas = []
    for nombre, b in casos:
        a, pr, rc = [], [], []
        for seed in range(6):
            r = correrSim(seed=seed, usBearings=b)
            a.append(compute_ate(r["est"], r["gt"]))
            acc = map_occupancy_accuracy(r["grid"],
                                         r["world"].occupiedCells(r["grid"]))
            pr.append(acc["precision"])
            rc.append(acc["recall"])
        filas.append(dict(caso=nombre, ate=float(np.mean(a)),
                          prec=float(np.mean(pr)), rec=float(np.mean(rc))))
    with open(os.path.join(out, "metricas_ultrasonido.json"), "w") as fh:
        json.dump(filas, fh, indent=1, ensure_ascii=False)
    return filas


def figDeriva(out):
    r = correrSim(seed=0, steps=1200, sesgoRot=math.radians(0.1))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    dibujarCuarto(ax, r["world"], alpha=0.7)
    gt, est, odo = (np.array(r[k]) for k in ("gt", "est", "odo"))
    ax.plot(gt[:, 0], gt[:, 1], "--", color="tab:green", lw=1.2,
            label="Trayectoria real")
    ax.plot(odo[:, 0], odo[:, 1], ":", color="tab:gray", lw=1.4,
            label="Solo odometría")
    ax.plot(est[:, 0], est[:, 1], "-", color="tab:blue", lw=1.0,
            label="EKF")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8)
    ax.set_title("1200 ciclos con giroscopio sesgado (+0.1°/ciclo)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "slam_deriva.png"), dpi=170)
    plt.close(fig)


# ---------------- RAY CASTING ----------------

# quita reglas del ray casting una por una y mide el mapa
def experimentoAblacion(out):
    import occupancy_grid as og

    def correr(anclar=True, crudoOcupa=False):
        pr, rc, fal, tot = [], [], [], []
        for seed in range(6):
            random.seed(seed)
            np.random.seed(seed)
            world = SimulatedWorld.defaultRoom()
            motionSim = SimulatedMotion(start=(0.7, 0.7, 0.0))
            usArray = SimulatedUltrasonicArray(world)
            slam = EKFSlam(0.7, 0.7, 0.0)
            grid = og.OccupancyGrid(8.0, 0.05, (1.0, 1.0))
            for _ in range(300):
                motion = motionSim.get()
                tp = tuple(motionSim.truePose)
                us = usArray.read(tp)
                obs = fuseObservations(simulatedVision(world, tp), us)
                o = slam.step(motion, obs)
                mapa = slam.observationsForMap() if anclar else obs
                if crudoOcupa:
                    mapa = [Observation(**{**m.__dict__, "sigma_r": min(
                        m.sigma_r or 0.05, 0.15)}) for m in mapa]
                grid.updateFromScan((o.x, o.y, o.theta),
                                    mapa + freeSpaceRays(us))
            gt = world.occupiedCells(grid)
            acc = map_occupancy_accuracy(grid, gt)
            occ = set(map(tuple, np.argwhere(grid.logOdds > OCC_THRESHOLD)))
            falsas = sum(1 for (cy, cx) in occ if not any(
                (cy + dy, cx + dx) in gt
                for dy in range(-2, 3) for dx in range(-2, 3)))
            pr.append(acc["precision"])
            rc.append(acc["recall"])
            fal.append(falsas)
            tot.append(len(occ))
        return dict(prec=float(np.mean(pr)), rec=float(np.mean(rc)),
                    ocupadas=float(np.mean(tot)), falsas=float(np.mean(fal)))

    filas = []
    filas.append(dict(caso="vigente", **correr()))
    og.L_BLOCK = 1e9
    filas.append(dict(caso="sin L_block", **correr()))
    og.L_BLOCK = L_BLOCK
    filas.append(dict(caso="rango monocular crudo marca ocupado",
                      **correr(crudoOcupa=True)))
    filas.append(dict(caso="sin anclar a landmarks", **correr(anclar=False)))
    og.L_BLOCK = 1e9
    filas.append(dict(caso="sin las tres reglas",
                      **correr(anclar=False, crudoOcupa=True)))
    og.L_BLOCK = L_BLOCK
    with open(os.path.join(out, "metricas_ablacion_ray.json"), "w") as fh:
        json.dump(filas, fh, indent=1, ensure_ascii=False)
    return filas

def _pintarCeldas(ax, grid, ray=None, titulo=""):
    lo = grid.logOdds
    n = grid.n
    cmap = plt.get_cmap("RdBu_r")
    for gy in range(n):
        for gx in range(n):
            v = lo[gy, gx]
            if v == 0:
                color = "#eeeeee"
            else:
                color = cmap(0.5 + 0.5 * max(-1.0, min(1.0, v / 2.5)))
            ax.add_patch(Rectangle((gx, gy), 1, 1, fc=color, ec="#bbbbbb",
                                   lw=0.4))
            if v != 0:
                ax.text(gx + 0.5, gy + 0.5, f"{v:+.1f}", ha="center",
                        va="center", fontsize=5.5)
    if ray is not None:
        (x0, y0), (x1, y1) = ray
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
        ax.plot(x0, y0, "o", mfc="none", mec="tab:green", mew=2, ms=9)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titulo, fontsize=9)


def figRayCasting(out):
    res = 0.1
    origen = (0.05, 0.05)

    def nuevoGrid():
        return OccupancyGrid(sizeM=1.6, resolution=res, origin=origen)

    pose = (0.0, 0.0, 0.0)
    ang = math.radians(28)
    r = 1.2

    def aCelda(px, py):
        return ((px + origen[0]) / res, (py + origen[1]) / res)

    fin = aCelda(r * math.cos(ang), r * math.sin(ang))
    ini = aCelda(0, 0)

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
    g1 = nuevoGrid()
    g1.updateFromScan(pose, [Observation(r, ang, "chair", sigma_r=0.05)])
    _pintarCeldas(axs[0], g1, (ini, fin),
                  "(a) Rango preciso (ultrasonido\no landmark anclado)")

    g2 = nuevoGrid()
    g2.updateFromScan(pose, [Observation(r, ang, "chair", sigma_r=0.36)])
    _pintarCeldas(axs[1], g2, (ini, fin),
                  "(b) Rango monocular crudo\n(σ = 0.36 m, sin ocupado)")

    g3 = nuevoGrid()
    for _ in range(3):
        g3.updateFromScan(pose, [Observation(0.6, ang, "wall",
                                             sigma_r=0.05)])
    g3.updateFromScan(pose, [Observation(r, ang, "chair", sigma_r=0.05)])
    _pintarCeldas(axs[2], g3, (ini, fin),
                  f"(c) Pared consolidada (l > {L_BLOCK:.0f})\ndetiene el rayo")
    fig.suptitle("Actualización log-odds de un rayo (celdas de 10 cm, "
                 "código real de occupancy_grid.py)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "raycasting_casos.png"), dpi=170)
    plt.close(fig)


def figBresenham(out):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x0, y0, x1, y1 = 1, 1, 13, 6
    celdas = bresenham(x0, y0, x1, y1)
    for gx in range(15):
        for gy in range(8):
            fc = "#ffffff"
            if (gx, gy) in celdas[:-1]:
                fc = "#9ecae1"
            if (gx, gy) == celdas[-1]:
                fc = "#de2d26"
            ax.add_patch(Rectangle((gx, gy), 1, 1, fc=fc, ec="#999999",
                                   lw=0.5))
    ax.plot([x0 + 0.5, x1 + 0.5], [y0 + 0.5, y1 + 0.5], "k-", lw=1)
    ax.plot(x0 + 0.5, y0 + 0.5, "o", color="tab:green", ms=8)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc="#9ecae1"),
                       Rectangle((0, 0), 1, 1, fc="#de2d26")],
              labels=[r"libre ($l_{free}$)", r"impacto ($l_{occ}$)"],
              loc="upper left", fontsize=8)
    ax.set_title("Celdas recorridas por Bresenham", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "raycasting_bresenham.png"), dpi=170)
    plt.close(fig)


def figLogOdds(out):
    # celda que primero se ve libre y luego le aparece un obstaculo
    seq = ["libre"] * 14 + ["ocupado"] * 10 + ["libre"] * 6
    l, hist = 0.0, []
    for s in seq:
        # igual que updateFromScan: el rayo libre no pasa por celdas > L_BLOCK
        if s == "libre" and l <= L_BLOCK:
            l += L_FREE
        elif s == "ocupado":
            l += L_OCC
        l = max(-L_CLAMP, min(L_CLAMP, l))
        hist.append(l)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.step(range(1, len(hist) + 1), hist, where="post", color="tab:blue")
    ax.axhline(OCC_THRESHOLD, ls="--", color="tab:red", lw=1,
               label=f"umbral ocupado ({OCC_THRESHOLD})")
    ax.axhline(L_BLOCK, ls=":", color="tab:purple", lw=1,
               label=f"$L_{{block}}$ ({L_BLOCK})")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-L_CLAMP, -L_CLAMP + 0.05, color="k")
    ax.axvspan(14.5, 24.5, color="tab:orange", alpha=0.1)
    ax.text(19.5, -4.2, "aparece un\nobstáculo", ha="center", fontsize=8)
    ax.text(27.5, -4.2, "se retira:\nlos rayos se\ndetienen en la celda", ha="center",
            fontsize=8)
    ax.set_xlabel("observación")
    ax.set_ylabel("log-odds $l$")
    ax.set_ylim(-L_CLAMP - 0.3, L_CLAMP + 0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("Evolución de una celda (mismas reglas que updateFromScan)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "raycasting_logodds.png"), dpi=170)
    plt.close(fig)


def figSectores(r, out, paso=200):
    grid = r["grid"]
    x, y, th = r["est"][paso - 1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    imshowGrid(ax, grid)
    colores = {"front": "tab:red", "left": "tab:green", "right": "tab:blue"}
    sectores = (("front", -FRONT_HALF_ANGLE, FRONT_HALF_ANGLE),
                ("left", FRONT_HALF_ANGLE, SIDE_MAX_ANGLE),
                ("right", -SIDE_MAX_ANGLE, -FRONT_HALF_ANGLE))
    dist = {}
    for nombre, a0, a1 in sectores:
        mejor = 4.0
        for k in range(5):
            a = th + a0 + (a1 - a0) * k / 4
            d = grid.castDistance(x, y, a, 4.0)
            mejor = min(mejor, d)
            ax.plot([x, x + d * math.cos(a)], [y, y + d * math.sin(a)],
                    "-", color=colores[nombre], lw=1)
            ax.plot(x + d * math.cos(a), y + d * math.sin(a), ".",
                    color=colores[nombre], ms=5)
        dist[nombre] = mejor
        ax.add_patch(Wedge((x, y), 0.5, math.degrees(th + a0),
                           math.degrees(th + a1), fc=colores[nombre],
                           alpha=0.25))
    ax.plot(x, y, "o", color="k", ms=6)
    ax.arrow(x, y, 0.35 * math.cos(th), 0.35 * math.sin(th),
             head_width=0.1, color="k")
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Rayos de consulta: d_front={dist['front']:.2f} m, "
                 f"d_left={dist['left']:.2f} m, d_right={dist['right']:.2f} m",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "raycasting_sectores.png"), dpi=170)
    plt.close(fig)
    return dist


# ---------------- MAPEO ----------------

def figEvolucionMapa(r, out):
    pasos = sorted(r["snaps"])
    fig, axs = plt.subplots(1, len(pasos), figsize=(3.6 * len(pasos), 3.3))
    for ax, p in zip(axs, pasos):
        lo, tray = r["snaps"][p]
        imshowGrid(ax, r["grid"], lo)
        t = np.array(tray)
        ax.plot(t[:, 0], t[:, 1], "-", color="tab:blue", lw=1)
        ax.plot(t[-1, 0], t[-1, 1], "o", color="tab:red", ms=5)
        ax.set_xlim(-0.6, 6.6)
        ax.set_ylim(-0.6, 4.6)
        ax.set_title(f"ciclo {p}", fontsize=9)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([0, 2, 4])
    fig.suptitle("Construcción incremental del mapa de ocupación", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mapeo_evolucion.png"), dpi=170)
    plt.close(fig)


def figCostmap(r, out, paso=200):
    x, y, th = r["est"][paso - 1]
    grid = r["grid"]
    # le meto un hoyo pa que se vea el 2do canal
    hx, hy = x + 1.0 * math.cos(th + 0.3), y + 1.0 * math.sin(th + 0.3)
    gx, gy = grid.worldToGrid(hx, hy)
    grid.hazardHits[gy - 2:gy + 3, gx - 2:gx + 3] = 3
    cm = grid.localCostmap((x, y), sizeM=4.0)
    grid.hazardHits[gy - 2:gy + 3, gx - 2:gx + 3] = 0
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ext = [-2, 2, -2, 2]
    im = axs[0].imshow(cm[0], cmap="Greys", origin="lower", extent=ext,
                       vmin=0, vmax=1)
    axs[0].set_title("Canal 0: P(ocupado)", fontsize=9)
    fig.colorbar(im, ax=axs[0], fraction=0.046)
    axs[1].imshow(cm[1], cmap="Reds", origin="lower", extent=ext,
                  vmin=0, vmax=1)
    axs[1].set_title("Canal 1: peligros de piso", fontsize=9)
    for ax in axs:
        ax.plot(0, 0, "o", color="tab:blue", ms=5)
        ax.arrow(0, 0, 0.4 * math.cos(th), 0.4 * math.sin(th),
                 head_width=0.1, color="tab:blue")
        ax.set_xlabel("x relativo (m)")
    axs[0].set_ylabel("y relativo (m)")
    fig.suptitle("Costmap local de 4×4 m centrado en el usuario", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mapeo_costmap.png"), dpi=170)
    plt.close(fig)


COLORES_LM = {"door": "tab:green", "hoyo": "tab:red", "escalera": "tab:red",
              "chair": "tab:orange", "monitor": "tab:purple"}


def figMapaReal(mapDir, out, nombre, titulo, zoom=None):
    mgr = ChunkedMapManager.openExisting(mapDir)
    lo, hz, (ox, oy) = mgr.stitched()
    occ = 1.0 / (1.0 + np.exp(-lo))
    h, w = lo.shape
    res = mgr.resolution
    ext = [ox * res, (ox + w) * res, oy * res, (oy + h) * res]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.imshow(occ, cmap="Greys", origin="lower", extent=ext, vmin=0, vmax=1)
    ys, xs = np.where(hz >= 2)
    if len(xs):
        ax.plot((ox + xs + 0.5) * res, (oy + ys + 0.5) * res, "s",
                color="tab:red", ms=2)
    for (ci, cj) in mgr.knownChunkIndices():
        ax.add_patch(Rectangle((ci * mgr.chunkSizeM, cj * mgr.chunkSizeM),
                               mgr.chunkSizeM, mgr.chunkSizeM, fill=False,
                               ec="tab:purple", lw=0.8, alpha=0.6))
        ax.text(ci * mgr.chunkSizeM + 0.2, cj * mgr.chunkSizeM + 0.2,
                f"({ci},{cj})", fontsize=7, color="tab:purple",
                clip_on=True)
    lms = mgr.loadLandmarks()
    vistos = set()
    for lm in lms:
        lab = lm["label"]
        col = COLORES_LM.get(lab, "tab:blue")
        etiqueta = lab if lab not in vistos else None
        vistos.add(lab)
        ax.plot(lm["x"], lm["y"], "x", color=col, ms=6, mew=1.6,
                label=etiqueta if lab in COLORES_LM else None)
    ax.plot([], [], "x", color="tab:blue", label="otras clases")
    if zoom is None:
        ys, xs = np.where(np.abs(lo) > 0.05)
        px = list((ox + xs) * res) + [l["x"] for l in lms]
        py = list((oy + ys) * res) + [l["y"] for l in lms]
        zoom = ((min(px) - 1.5, max(px) + 1.5), (min(py) - 1.5, max(py) + 1.5))
    ax.set_xlim(*zoom[0])
    ax.set_ylim(*zoom[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre), dpi=170)
    plt.close(fig)
    return len(mgr.knownChunkIndices()), len(lms)


# corre el mismo pipeline de main_lap sobre un video y guarda el panel
def capturasVideo(video, out, prefijo, guardarEn, stride=2, hfov=70.0,
                  hazards=True):
    import cv2
    from vision_bridge import (VisionBridge, VisualYawEstimator,
                               GroundHazardDetector)
    from video_a_mapa import WalkDetector
    from main_lap import (annotateCamera, drawChunkMapPanel, drawNavBar,
                          _hstackPad)
    from navigation_feed import safetyAlerts

    bridge = VisionBridge(hfovDeg=hfov)
    yawEst = VisualYawEstimator(hfovDeg=hfov)
    caminata = WalkDetector(hfov)
    detPel = GroundHazardDetector(hfovDeg=hfov) if hazards else None
    slam = EKFSlam(0.0, 0.0, 0.0)
    mgr = ChunkedMapManager(mapDir=None)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = stride / fps
    n = usados = 0
    peligros, tiempos, tray = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % stride:
            continue
        usados += 1
        t0 = time.perf_counter()
        dRot, okYaw = yawEst.update(frame)
        dTrans = 1.1 * dt if caminata.update(frame, dRot) else 0.0
        t1 = time.perf_counter()
        dets = bridge.process(frame, egoYaw=dRot)
        t2 = time.perf_counter()
        if detPel is not None and usados % 4 == 0:
            peligros = detPel.process(frame, dets)
        dets = dets + peligros
        t3 = time.perf_counter()
        outp = slam.step(MotionEstimate(dTrans, dRot, dt),
                         fuseObservations(dets, []))
        t4 = time.perf_counter()
        pose = (outp.x, outp.y, outp.theta)
        mgr.updateFromScan(pose, slam.observationsForMap())
        mgr.clearFootprint(outp.x, outp.y)
        mgr.updatePosition(pose, slam.getLandmarks())
        t5 = time.perf_counter()
        nav = buildNavigationFrame(slam, mgr, dets, chunk=mgr.currentChunk)
        alertas = safetyAlerts(nav)
        t6 = time.perf_counter()
        tiempos.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4,
                        t6 - t5))
        tray.append((outp.x, outp.y))
        if n in guardarEn or (guardarEn and n >= max(guardarEn)
                              and not os.path.exists(os.path.join(
                                  out, f"{prefijo}_f{max(guardarEn)}.jpg"))):
            cam = annotateCamera(cv2.resize(frame, (960, 540)), dets,
                                 bridge.hfov, okYaw, fps / stride)
            mapa = drawChunkMapPanel(mgr, slam, outp, [], size=540)
            arriba = _hstackPad(cam, mapa)
            img = np.vstack([arriba, drawNavBar(nav, alertas,
                                                arriba.shape[1])])
            cv2.imwrite(os.path.join(out, f"{prefijo}_f{n}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    t = 1000 * np.array(tiempos)
    resumen = {"video": os.path.basename(video), "frames": usados,
               "landmarks": slam.exportLandmarks(),
               "chunks": len(mgr.knownChunkIndices()),
               "ms_prom": dict(zip(("odometria_visual", "yolo",
                                    "segformer_amortizado", "ekf",
                                    "raycasting_mapa", "nav_frame"),
                                   [float(v) for v in t.mean(axis=0)]))}
    with open(os.path.join(out, f"{prefijo}_resumen.json"), "w") as fh:
        json.dump(resumen, fh, indent=1, ensure_ascii=False)
    mgr.renderCroquis(trajectory=tray, landmarks=slam.confirmedLandmarks(),
                      path=os.path.join(out, f"{prefijo}_croquis.png"))
    return resumen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/TT_2026_B045/Capturas_TT2")
    p.add_argument("--video", default=None,
                   help="solo capturas del pipeline real sobre este video")
    p.add_argument("--prefijo", default="captura")
    p.add_argument("--frames", default="300,600,900")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.video:
        r = capturasVideo(args.video, args.out, args.prefijo,
                          {int(f) for f in args.frames.split(",")})
        print(r["frames"], r["chunks"], len(r["landmarks"]), r["ms_prom"])
        return

    oclusion = (120, 200)
    rNorm = correrSim(seed=0, snapshots=(60, 150, 300))
    rOcl = correrSim(seed=0, oclusion=oclusion)
    figTrayectorias(rNorm, args.out)
    figIncertidumbre(rNorm, rOcl, oclusion, args.out)
    figEvolucionMapa(rNorm, args.out)
    dist = figSectores(rNorm, args.out)
    figCostmap(rNorm, args.out)
    figRayCasting(args.out)
    figBresenham(args.out)
    figLogOdds(args.out)
    print("sectores:", dist)
    print("ATE oclusion:", compute_ate(rOcl["est"], rOcl["gt"]))

    for carpeta, nombre, titulo in (
            ("maps/real/v04_pasillo", "mapeo_real_pasillo.png",
             "Mapa por chunks generado de un video real (pasillo)"),
            ("maps/real/v10_oficina4", "mapeo_real_oficina.png",
             "Mapa por chunks generado de un video real (oficina)")):
        if os.path.isdir(carpeta):
            print(carpeta, figMapaReal(carpeta, args.out, nombre, titulo))

    for f in experimentoAblacion(args.out):
        print(f)
    for f in experimentoUltrasonido(args.out):
        print(f)
    figDeriva(args.out)
    for f in experimentoDeriva(args.out):
        print(f)
    filas = tablaSemillas(args.out)
    for f in filas:
        print({k: round(v, 3) if isinstance(v, float) else v
               for k, v in f.items()})


if __name__ == "__main__":
    main()
