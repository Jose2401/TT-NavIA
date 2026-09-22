"""
simulation.py
Simulación end-to-end del módulo SLAM sin hardware, para validar el
funcionamiento según los requisitos del proyecto:
  - el usuario inicia en una posición conocida;
  - avanza y cambia de orientación (recorrido por waypoints en un cuarto);
  - encuentra obstáculos (visión simulada) y paredes (ultrasonido simulado
    por ray casting contra el cuarto real);
  - el mapa de ocupación recibe nuevas observaciones;
  - la pose se actualiza correctamente;
  - la incertidumbre cambia conforme se reciben (o se pierden) mediciones.

El flujo de datos es EXACTAMENTE el mismo que en demo_realtime.py con la
cámara real: sensores -> fuse_observations() -> EKFSlam.step() ->
OccupancyGrid.update_from_scan(). Solo cambia el origen de los datos
(sim_world en lugar de vision_bridge + hardware), así que pasar a real no
toca este pipeline.
"""
import math
import random
import numpy as np

from ekf_slam import EKFSlam
from occupancy_grid import OccupancyGrid
from sim_world import (SimulatedWorld, SimulatedMotion,
                       SimulatedUltrasonicArray, simulated_vision)
from vision_bridge import fuse_observations, free_space_rays
from metrics import (compute_ate, compute_orientation_error,
                     uncertainty_series, map_occupancy_accuracy)


def run_simulation(steps=300, seed=0, drop_prob=0.0, verbose=True,
                   croquis_path="croquis_simulacion.png"):
    random.seed(seed)
    np.random.seed(seed)

    world = SimulatedWorld.default_room()
    motion_sim = SimulatedMotion(start=(0.7, 0.7, 0.0))
    us_array = SimulatedUltrasonicArray(world)

    slam = EKFSlam(x0=0.7, y0=0.7, theta0=0.0)   # posición inicial conocida
    grid = OccupancyGrid(size_m=8.0, resolution=0.05, origin=(1.0, 1.0))

    gt_traj, est_traj, cov_hist, conf_hist = [], [], [], []

    for step in range(steps):
        # 1) Movimiento (odometría simulada con ruido) + pose real
        motion = motion_sim.get()
        true_pose = tuple(motion_sim.true_pose)

        # 2) Sensores simulados coherentes con el cuarto
        us_readings = us_array.read(true_pose)
        vision_dets = simulated_vision(world, true_pose, drop_prob=drop_prob)

        # 3) Fusión -> observaciones para el EKF
        observations = fuse_observations(vision_dets, us_readings)

        # 4) Ciclo SLAM (predicción + corrección)
        output = slam.step(motion, observations)

        # 5) Mapa de ocupación (Ray Casting) con la pose ya corregida y
        #    las observaciones ancladas a los landmarks consolidados;
        #    también recibe los rayos "sin eco" (solo espacio libre)
        grid.update_from_scan((output.x, output.y, output.theta),
                              slam.observations_for_map()
                              + free_space_rays(us_readings))

        gt_traj.append(true_pose)
        est_traj.append((output.x, output.y, output.theta))
        cov_hist.append(output.covariance)
        conf_hist.append(output.confidence)

        if verbose and step % 30 == 0:
            print(f"[t={step:03d}] real=({true_pose[0]:.2f},{true_pose[1]:.2f},"
                  f"{math.degrees(true_pose[2]):6.1f}°)  "
                  f"est=({output.x:.2f},{output.y:.2f},"
                  f"{math.degrees(output.theta):6.1f}°)  "
                  f"conf={output.confidence:.2f}  "
                  f"landmarks={len(slam.confirmed_landmarks())}")

    ate = compute_ate(est_traj, gt_traj)
    ori_err = compute_orientation_error(est_traj, gt_traj)
    map_acc = map_occupancy_accuracy(grid, world.occupied_cells(grid))
    unc = uncertainty_series(cov_hist)

    if verbose:
        print(f"\nError medio de trayectoria (ATE): {ate:.4f} m")
        print(f"Error medio de orientación: {math.degrees(ori_err):.2f}°")
        print(f"Confianza final: {conf_hist[-1]:.2f}  "
              f"(traza P: {unc[0]:.4f} -> {unc[-1]:.4f})")
        if map_acc:
            print(f"Mapa vs cuarto real: precisión={map_acc['precision']:.2f} "
                  f"recall={map_acc['recall']:.2f}")
        print(f"Landmarks confirmados: {len(slam.confirmed_landmarks())} "
              f"(candidatos vivos: {len(slam.landmarks)})")

    path = grid.render_croquis(
        pose=est_traj[-1],
        trajectory=[(p[0], p[1]) for p in est_traj],
        gt_trajectory=[(p[0], p[1]) for p in gt_traj],
        landmarks=slam.confirmed_landmarks(),
        path=croquis_path)
    if verbose:
        print(f"Croquis de la simulación guardado en: {path}")

    return {
        "gt_traj": gt_traj, "est_traj": est_traj, "cov_hist": cov_hist,
        "ate": ate, "orientation_error": ori_err, "map_accuracy": map_acc,
        "slam": slam, "grid": grid, "croquis_path": path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--drop", type=float, default=0.0,
                        help="probabilidad de perder cada detección visual "
                             "(prueba de oclusión / dead-reckoning)")
    args = parser.parse_args()
    run_simulation(steps=args.steps, seed=args.seed, drop_prob=args.drop)
