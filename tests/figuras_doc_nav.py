"""
figuras de navegacion y alertas pa la doc del TT2, todo con el codigo
real. correr:
    python -m tests.figuras_doc_nav --out docs/TT_2026_B045/Capturas_TT2/nav
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
from matplotlib.patches import Circle

from ekf_slam import EKFSlam
from map_manager import ChunkedMapManager
from sim_world import (SimulatedWorld, SimulatedMotion,
                       SimulatedUltrasonicArray, simulatedVision,
                       KeyboardUser)
from vision_bridge import (fuseObservations, freeSpaceRays,
                           validarConUltrasonido)
from navigation_feed import buildNavigationFrame
from navigation.nav_env import (NavEnv, gen_room, gen_two_rooms,
                                gen_outdoor, FORWARD_STEP, TURN_STEP)
from navigation.navigator import Navigator, load_policy, NumpyPolicy
from navigation.route_planner import planear
from alertas import ModuloAlertas

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


# ---------------- episodios en mundos procedurales ----------------

def _dibujarMundo(ax, world):
    for (x0, y0, x1, y1) in world.segments:
        ax.plot([x0, x1], [y0, y1], "k-", lw=2)
    for (cx, cy, r) in world.statics:
        ax.add_patch(Circle((cx, cy), r, fc="#999999", ec="k", lw=0.6))
    for (cx, cy, r) in world.hazards:
        ax.add_patch(Circle((cx, cy), r, fc="#f4a3a3", ec="tab:red",
                            lw=0.8, hatch="///"))


def figEpisodios(policy, obsMode, out, semillas=(3, 11, 7)):
    gens = ((gen_room, "Cuarto"), (gen_two_rooms, "Dos cuartos con puerta"),
            (gen_outdoor, "Exterior con coladeras"))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    resumen = []
    for ax, (gen, nombre), seed in zip(axs, gens, semillas):
        env = NavEnv(obs_mode=obsMode, seed=seed, sensor_noise=0.05)
        env.reset()
        world, start, goal = gen(env.rng)
        env.world, env.pose, env.goal = world, list(start), goal
        env.prev_d_goal = env._d_goal()
        env.steps = 0
        movers0 = world.movers.copy()
        obs = env._obs()
        tray, movTray = [tuple(env.pose)], []
        done, info = False, {}
        while not done:
            obs, _r, term, trunc, info = env.step(policy.predict(obs))
            tray.append(tuple(env.pose))
            movTray.append(world.movers[:, :2].copy())
            done = term or trunc
        _dibujarMundo(ax, world)
        t = np.array(tray)
        ax.plot(t[:, 0], t[:, 1], "-", color="tab:blue", lw=1.6)
        ax.plot(t[0, 0], t[0, 1], "o", color="tab:green", ms=8)
        ax.plot(goal[0], goal[1], "*", color="tab:purple", ms=15)
        if len(movers0):
            for k in range(len(movers0)):
                mt = np.array([m[k] for m in movTray])
                ax.plot(mt[:, 0], mt[:, 1], ":", color="tab:orange", lw=1)
                ax.add_patch(Circle(mt[-1], movers0[k, 2], fc="tab:orange",
                                    alpha=0.5))
        res = ("llega" if info.get("success") else
               "choque" if info.get("collision") else
               "caída" if info.get("fell") else "tiempo")
        resumen.append((nombre, res, env.steps))
        ax.set_title(f"{nombre}: {res} en {env.steps} pasos", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nav_episodios.png"), dpi=170)
    plt.close(fig)
    return resumen


def figCompObs(out):
    nombres = {"gen_room": "Cuarto", "gen_two_rooms": "Dos cuartos",
               "gen_outdoor": "Exterior"}
    metas = {"Observación del documento (8)": "models/nav_ppo_doc_cmp_meta.json",
             "Observación extendida (24)": "models/nav_ppo_extended_cmp_meta.json"}
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    w = 0.38
    for k, (lab, path) in enumerate(metas.items()):
        ev = json.load(open(path))["eval"]
        vals = [100 * ev["by_world"][g]["success_rate"] for g in nombres]
        vals.append(100 * ev["success_rate"])
        xs = np.arange(len(vals)) + (k - 0.5) * w
        bars = ax.bar(xs, vals, w, label=lab,
                      color=("#BAB0AC", "#4C78A8")[k])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{b.get_height():.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(nombres.values()) + ["Global"])
    ax.set_ylabel("episodios exitosos (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Comparación de observaciones (300 000 pasos de "
                 "entrenamiento cada una)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nav_comparacion_obs.png"), dpi=170)
    plt.close(fig)


def figEvalMundos(evalJson, out):
    data = json.load(open(evalJson))["res"]["by_world"]
    nombres = {"gen_room": "Cuarto (proc.)",
               "gen_two_rooms": "Dos cuartos (proc.)",
               "gen_outdoor": "Exterior (proc.)"}
    etiquetas, exito, choque = [], [], []
    for k, s in data.items():
        lab = nombres.get(k, k.replace("real:maps/real/", ""))
        etiquetas.append(lab)
        exito.append(100 * s["success_rate"])
        choque.append(100 * s["collision_rate"])
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    xs = np.arange(len(etiquetas))
    ax.bar(xs, exito, color=["#72B7B2"] * 3 + ["#4C78A8"] * (len(xs) - 3),
           label="llega")
    ax.bar(xs, choque, bottom=exito, color="#E45756", label="choque")
    ax.set_xticks(xs)
    ax.set_xticklabels(etiquetas, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% de episodios")
    ax.axhline(90, ls="--", color="k", lw=0.8)
    ax.text(len(xs) - 0.6, 91, "90 % (objetivo 3)", fontsize=8, ha="right")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Evaluación del modelo final por tipo de mundo "
                 "(40 episodios por mundo)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nav_eval_mundos.png"), dpi=170)
    plt.close(fig)


# ---------------- sesion completa en simulacion ----------------

def _cicloSensores(world, user, usArray, slam, mgr, motion, persona=None):
    truePose = tuple(user.truePose)
    if persona is not None:
        world.obstacles.append((persona["x"], persona["y"], 0.28, "person"))
    dets = simulatedVision(world, truePose)
    if persona is not None:
        for d in dets:
            if d.label == "person":
                d.moving = True
    # igual que main_nav_lap: los frontales validan a la camara
    us = validarConUltrasonido(dets, usArray.read(truePose))
    if persona is not None:
        world.obstacles.pop()
    out = slam.step(motion, fuseObservations(dets, us))
    pose = (out.x, out.y, out.theta)
    mgr.updateFromScan(pose, slam.observationsForMap() + freeSpaceRays(us))
    mgr.updatePosition(pose, slam.getLandmarks())
    frame = buildNavigationFrame(slam, mgr, dets, ultrasonic=us)
    return out, frame


def _explorar(seed, pasos=320):
    random.seed(seed)
    np.random.seed(seed)
    world = SimulatedWorld.defaultRoom()
    user = KeyboardUser(world, start=(0.7, 0.7, 0.0))
    piloto = SimulatedMotion(start=(0.7, 0.7, 0.0))
    usArray = SimulatedUltrasonicArray(world)
    slam = EKFSlam(0.7, 0.7, 0.0)
    mgr = ChunkedMapManager(mapDir=None)
    for _ in range(pasos):
        piloto.truePose = list(user.truePose)
        motion = piloto.get()
        user.truePose = list(piloto.truePose)
        out, frame = _cicloSensores(world, user, usArray, slam, mgr, motion)
    return world, user, usArray, slam, mgr, out, frame


def sesionGuiada(navigator, inter, seed=0, destino="llévame al sillón",
                 maxPasos=700, cajaEn=None, capturas=(), out=None):
    """el usuario simulado hace exactamente lo que dice el navegador
    (como main_nav_lap --sim). regresa el log y la trayectoria."""
    import cv2
    from main_nav_lap import drawRutaViva
    world, user, usArray, slam, mgr, outp, frame = _explorar(seed)
    log = []
    for m in inter.manejarTexto(destino, frame):
        log.append((0, m.priority, m.kind, m.text))
    tray, trayReal = [], []
    ruta = None
    rutas = {}
    bloqueos = 0
    llego = False
    for paso in range(1, maxPasos + 1):
        st = navigator.status
        forward = rot = 0.0
        if st.session == "activo" and st.last_action is not None:
            a = st.last_action
            if a == 0:
                forward = FORWARD_STEP / 2
            elif a == 1:
                rot = TURN_STEP / 2
            elif a == 2:
                rot = -TURN_STEP / 2
        antes = tuple(user.truePose)
        motion = user.command(forward, rot)
        if forward and tuple(user.truePose)[:2] == antes[:2]:
            bloqueos += 1           # KeyboardUser no deja atravesar
        if cajaEn is not None and paso == cajaEn:
            ux, uy, uth = user.truePose
            world.obstacles.append((ux + 1.2 * math.cos(uth),
                                    uy + 1.2 * math.sin(uth), 0.25, "box"))
            log.append((paso, 9, "demo", "[caja nueva 1.2 m adelante]"))
            rutas["antes"] = ruta
        outp, frame = _cicloSensores(world, user, usArray, slam, mgr, motion)
        tray.append((outp.x, outp.y))
        trayReal.append(tuple(user.truePose[:2]))
        if st.session == "activo" and st.goal is not None and paso % 6 == 0:
            ruta = planear(navigator.policy, mgr,
                           (outp.x, outp.y, outp.theta), st.goal)
            if cajaEn is not None and paso >= cajaEn + 6 \
                    and "despues" not in rutas:
                rutas["despues"] = ruta
        for m in inter.ciclo(frame, mgr, []):
            log.append((paso, m.priority, m.kind, m.text))
            if m.kind == "evento" or "llegamos" in m.text.lower() \
                    or "llegado" in m.text.lower() \
                    or "lograste" in m.text.lower():
                llego = llego or navigator.status.session == "en_espera"
        if out and paso in capturas:
            img = drawRutaViva(mgr, slam, navigator, ruta, outp, frame,
                               sizePx=640, ventanaM=8.0)
            cv2.imwrite(os.path.join(out, f"nav_vivo_p{paso}.png"), img)
        if navigator.status.session == "en_espera" and paso > 2:
            llego = True
            break
    # distancia REAL (superficie) al objeto pedido al terminar
    ux, uy = user.truePose[0], user.truePose[1]
    dReal = None
    if r_label := _etiquetaPedida(destino):
        ds = [math.hypot(cx - ux, cy - uy) - rr
              for (cx, cy, rr, lab) in world.obstacles if lab == r_label]
        dReal = min(ds) if ds else None
    return dict(log=log, tray=tray, trayReal=trayReal, pasos=paso, dReal=dReal,
                llego=llego, bloqueos=bloqueos, mgr=mgr, slam=slam,
                world=world, rutas=rutas, ruta=ruta)


def _etiquetaPedida(destino):
    from navigation.navigator import DEST_ALIASES
    d = destino.translate(str.maketrans("áéíóú", "aeiou"))
    for pal, lab in DEST_ALIASES.items():
        if d.endswith(" " + pal):
            return lab
    return None


def figSesion(r, out, nombre="nav_sesion.png", titulo=""):
    mgr = r["mgr"]
    lo, hz, (ox, oy) = mgr.stitched()
    occ = 1.0 / (1.0 + np.exp(-lo))
    h, w = lo.shape
    res = mgr.resolution
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1,
              extent=[ox * res, (ox + w) * res, oy * res, (oy + h) * res])
    for (cx, cy, rr, lab) in r["world"].obstacles:
        ax.add_patch(Circle((cx, cy), rr, fill=False, ec="tab:brown",
                            ls="--", lw=1))
        ax.text(cx, cy - rr - 0.2, lab, fontsize=7, ha="center",
                color="tab:brown")
    for nom, col in (("antes", "tab:cyan"), ("despues", "tab:olive")):
        ru = r["rutas"].get(nom)
        if ru:
            p = np.array(ru["points"])
            ax.plot(p[:, 0], p[:, 1], "--", color=col, lw=2,
                    label=f"ruta planeada {'antes' if nom == 'antes' else 'después'}"
                          " de la caja")
    t = np.array(r["trayReal"])
    ax.plot(t[:, 0], t[:, 1], "-", color="tab:blue", lw=1.8,
            label="recorrido del usuario simulado")
    ax.plot(t[0, 0], t[0, 1], "o", color="tab:green", ms=8)
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(-0.4, 4.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=2)
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, nombre), dpi=170)
    plt.close(fig)


def lazoCerrado(navigator, inter, out, semillas=range(5),
                destinos=("llévame al sillón", "llévame a la mesa",
                          "llévame a la planta", "ve a la silla")):
    filas = []
    for seed in semillas:
        for dest in destinos:
            navigator.handle_command("detener")
            r = sesionGuiada(navigator, inter, seed=seed, destino=dest,
                             maxPasos=600)
            instr = sum(1 for x in r["log"] if x[2] == "instruccion")
            filas.append(dict(seed=seed, destino=dest, llego=r["llego"],
                              pasos=r["pasos"], instrucciones=instr,
                              bloqueos=r["bloqueos"], dReal=r["dReal"]))
            print(filas[-1])
    json.dump(filas, open(os.path.join(out, "lazo_cerrado.json"), "w"),
              indent=1, ensure_ascii=False)
    return filas


def figRutaAlternativa(out, seed=0, destino="llévame al sillón"):
    """ruta planeada antes y despues de pedir 'busca otro camino'"""
    from tests.eval_interaccion import _armarPln
    from interaccion import Interaccion
    world, user, usArray, slam, mgr, outp, frame = _explorar(seed)
    nav = Navigator("models/nav_ppo", decide_every_s=0.0, repeat_s=2.5)
    inter = Interaccion(nav, nlp=_armarPln())
    inter.manejarTexto(destino, frame)
    inter.ciclo(frame, mgr, [])          # el ejecutor guarda mapa y pose
    pose = (outp.x, outp.y, outp.theta)
    antes = planear(nav.policy, mgr, pose, nav.status.goal)
    msgs = inter.manejarTexto("busca otro camino", frame)
    bloqueos = list(mgr.bloqueos)
    despues = planear(nav.policy, mgr, pose, nav.status.goal)
    lo, hz, (ox, oy) = mgr.stitched()
    occ = 1.0 / (1.0 + np.exp(-lo))
    h, w = lo.shape
    res = mgr.resolution
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1,
              extent=[ox * res, (ox + w) * res, oy * res, (oy + h) * res])
    for (bx, by, br, _t) in bloqueos:
        ax.add_patch(Circle((bx, by), br, fc="tab:red", alpha=0.25, ec="none"))
    for ru, col, nom in ((antes, "tab:cyan", "ruta original"),
                         (despues, "tab:olive", "ruta alternativa")):
        p = np.array(ru["points"])
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2.2,
                label=f"{nom} ({ru['pasos']} pasos)")
    ax.plot(pose[0], pose[1], "o", color="tab:green", ms=8)
    ax.plot(*nav.status.goal, "*", color="tab:purple", ms=15)
    ax.plot([], [], "o", color="tab:red", alpha=0.3, ms=10,
            label="tramo bloqueado por 3 min")
    ax.set_xlim(-0.4, 6.4)
    ax.set_ylim(-0.4, 4.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3)
    ax.set_title("Ruta antes y después de pedir «busca otro camino»",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "nav_ruta_alternativa.png"), dpi=170)
    plt.close(fig)
    return dict(respuesta=[m.text for m in msgs], antes=antes["estado"],
                despues=despues["estado"], pasos=(antes["pasos"],
                                                   despues["pasos"]),
                bloqueos=len(bloqueos))


# ---------------- alertas ----------------

def escenarioAlertas(out):
    """camina derecho hacia la pared del fondo mientras una persona le
    cruza enfrente; registra distancias y alertas del modulo real."""
    random.seed(4)
    np.random.seed(4)
    world, user, usArray, slam, mgr, outp, frame = _explorar(4, pasos=320)
    # se coloca mirando a la pared derecha (x=6) desde el centro
    user.truePose = [1.0, 2.0, 0.0]
    slam.mu = np.array([1.0, 2.0, 0.0])
    world.obstacles = [o for o in world.obstacles if o[3] != "chair"
                       or o[0] < 4.0]
    import types
    import alertas as modAl
    reloj = [0.0]
    relojReal = modAl.time
    # el cooldown usa time.time(): aca el tiempo es el simulado (0.1 s/ciclo)
    modAl.time = types.SimpleNamespace(time=lambda: reloj[0],
                                       monotonic=relojReal.monotonic,
                                       sleep=relojReal.sleep)
    mod = ModuloAlertas(conPip=False)
    persona = {"x": 4.8, "y": 3.4, "vx": 0.0, "vy": -0.03}
    filas = []
    for paso in range(1, 175):
        reloj[0] = 1000.0 + paso * 0.1
        forward = 0.03 if user.truePose[0] < 5.45 else 0.0
        motion = user.command(forward, 0.0)
        p = persona if 15 <= paso <= 120 else None
        if p is not None:
            persona["y"] += persona["vy"]
        outp, frame = _cicloSensores(world, user, usArray, slam, mgr,
                                     motion, persona=p)
        t0 = time.perf_counter()
        alertas, recalc = mod.evaluar(frame)
        dt = time.perf_counter() - t0
        detMov = [d for d in frame.detections if d.moving]
        filas.append(dict(paso=paso, t=paso * 0.1, d_front=frame.d_front,
                          d_movil=(min(d.range_est for d in detMov)
                                   if detMov else None),
                          alertas=alertas, recalcular=recalc,
                          ms=1000 * dt))
    modAl.time = relojReal
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    t = [f["t"] for f in filas]
    ax.plot(t, [min(f["d_front"], 4.0) for f in filas], "-",
            color="tab:blue", label="distancia libre al frente (mapa)")
    ax.plot(t, [f["d_movil"] if f["d_movil"] else np.nan for f in filas],
            "-", color="tab:orange", label="persona en movimiento (visión)")
    ax.axhline(0.8, ls="--", color="tab:red", lw=0.9)
    ax.axhline(1.5, ls="--", color="tab:olive", lw=0.9)
    ax.text(t[-1], 0.83, "crítica 0.80 m", fontsize=7, ha="right",
            color="tab:red")
    ax.text(t[-1], 1.53, "precaución 1.50 m", fontsize=7, ha="right",
            color="tab:olive")
    puestos = False
    for f in filas:
        for prio, txt in f["alertas"]:
            col = "tab:red" if prio == 0 else "tab:olive"
            ax.plot(f["t"], 3.7 if prio == 0 else 3.4, "v", color=col, ms=7)
    ax.plot([], [], "v", color="tab:red", label="alerta crítica emitida")
    ax.plot([], [], "v", color="tab:olive", label="precaución emitida")
    ax.set_ylim(0, 4.0)
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("distancia (m)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.set_title("Alertas emitidas al caminar hacia una pared mientras "
                 "cruza una persona", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "alertas_escenario.png"), dpi=170)
    plt.close(fig)
    salida = [dict(t=f["t"], alertas=f["alertas"]) for f in filas
              if f["alertas"]]
    json.dump({"emitidas": salida,
               "ms_prom": float(np.mean([f["ms"] for f in filas])),
               "ms_max": float(np.max([f["ms"] for f in filas]))},
              open(os.path.join(out, "alertas_escenario.json"), "w"),
              indent=1, ensure_ascii=False)
    return salida


def figPitidos(out):
    import sys
    sys.path.insert(0, "VozYPLN")
    import importlib
    pm = importlib.import_module("VozYPLN.alertas")
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 4.6), sharex="col")
    for col, (nivel, nombre) in enumerate(((pm.CRITICAL, "crítica"),
                                           (pm.WARNING, "precaución"))):
        partes = []
        izq, der = pm.PAN["izquierda"]
        for f, on, off in pm.PATTERNS[nivel]:
            partes.append(np.frombuffer(pm._tone(f, on, izq, der), np.int16))
            if off:
                partes.append(np.frombuffer(pm._silence(off), np.int16))
        s = np.concatenate(partes).reshape(-1, 2) / 32768.0
        t = np.arange(len(s)) / pm.SAMPLE_RATE * 1000
        axs[0, col].plot(t, s[:, 0], lw=0.3, color="tab:blue")
        axs[1, col].plot(t, s[:, 1], lw=0.3, color="tab:orange")
        axs[0, col].set_title(f"Alerta {nombre}, obstáculo a la izquierda",
                              fontsize=9)
        axs[1, col].set_xlabel("tiempo (ms)")
        for a in axs[:, col]:
            a.set_ylim(-1, 1)
    axs[0, 0].set_ylabel("canal izq.")
    axs[1, 0].set_ylabel("canal der.")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "alertas_pitidos.png"), dpi=170)
    plt.close(fig)


# ---------------- tiempos ----------------

def tiempos(out):
    from navigation.navigator import build_observation
    world, user, usArray, slam, mgr, outp, frame = _explorar(0, pasos=200)
    sb3, mode = load_policy("models/nav_ppo")
    npy = NumpyPolicy("models/nav_ppo_policy.npz")
    goal = (1.0, 3.1)
    res = {}
    t0 = time.perf_counter()
    for _ in range(200):
        obs, _p, _d = build_observation(frame, mgr, goal, mode, 3)
    res["observacion_ms"] = 1000 * (time.perf_counter() - t0) / 200
    for nom, pol in (("politica_sb3_ms", sb3), ("politica_numpy_ms", npy)):
        t0 = time.perf_counter()
        for _ in range(500):
            pol.predict(obs)
        res[nom] = 1000 * (time.perf_counter() - t0) / 500
    iguales = 0
    rng = np.random.default_rng(0)
    for _ in range(500):
        o = np.clip(rng.uniform(-1, 1, size=obs.shape), -1, 1)
        iguales += sb3.predict(o) == npy.predict(o)
    res["acuerdo_sb3_numpy"] = iguales / 500
    t0 = time.perf_counter()
    n = 0
    for _ in range(10):
        planear(sb3, mgr, (outp.x, outp.y, outp.theta), goal)
        n += 1
    res["planear_ms"] = 1000 * (time.perf_counter() - t0) / n
    json.dump(res, open(os.path.join(out, "tiempos_nav.json"), "w"),
              indent=1)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/TT_2026_B045/Capturas_TT2/nav")
    p.add_argument("--rapido", action="store_true",
                   help="sin la tabla de lazo cerrado")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    policy, mode = load_policy("models/nav_ppo")
    print("episodios:", figEpisodios(policy, mode, args.out))
    figCompObs(args.out)
    ev = os.path.join(args.out, "eval_nav_ppo.json")
    if os.path.exists(ev):
        figEvalMundos(ev, args.out)
    figPitidos(args.out)
    print("alertas:", escenarioAlertas(args.out))
    print("tiempos:", tiempos(args.out))

    from tests.eval_interaccion import _armarPln
    from interaccion import Interaccion
    navigator = Navigator("models/nav_ppo", decide_every_s=0.0,
                          repeat_s=2.5)
    inter = Interaccion(navigator, nlp=_armarPln(), amigable=True)
    r = sesionGuiada(navigator, inter, seed=0, cajaEn=40,
                     capturas=(5, 40, 75), out=args.out)
    figSesion(r, args.out, titulo="Sesión guiada: «llévame al sillón» "
              "con una caja que aparece en el camino")
    with open(os.path.join(args.out, "sesion_log.json"), "w") as fh:
        json.dump({"log": r["log"], "pasos": r["pasos"],
                   "llego": r["llego"], "bloqueos": r["bloqueos"]}, fh,
                  indent=1, ensure_ascii=False)
    for x in r["log"]:
        print(x)
    if not args.rapido:
        lazoCerrado(navigator, inter, args.out)


if __name__ == "__main__":
    main()
