import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _ensure(path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    This function creates a directory at the specified path if it does not already exist.
    If the directory already exists, it does nothing and returns the path unchanged.
    
    Args:
        path (str): The directory path to ensure exists.
    
    Returns:
        str: The input path, unchanged.
    """
    os.makedirs(path, exist_ok=True)
    return path


def _stack_positions_phase1(robots):
    """
    Stack position histories from multiple robots into a single array.
    
    This function organizes the position history data from Phase 1 for all robots
    into a structured numpy array, aligning them by time step. This is useful for
    collective analysis or visualization of robot trajectories during Phase 1.
    
    Args:
        robots (list): A list of robot objects, each containing a `pos_history_phase1`
                      attribute with position data (sequence of (x, y) coordinates).
    
    Returns:
        np.ndarray: A 3D array of shape (T, N, 2) where:
                    - T is the maximum length of position history across all robots
                    - N is the number of robots
                    - 2 represents the (x, y) coordinates
                    - Rows are padded with NaN for robots with shorter histories.
    """
    T = max(len(r.pos_history_phase1) for r in robots)
    N = len(robots)
    arr = np.full((T, N, 2), np.nan)
    for j, r in enumerate(robots):
        for k, p in enumerate(r.pos_history_phase1):
            arr[k, j] = p
    return arr


def _stack_positions_phase2(robots):
    """
    Stack the position histories of all robots from phase 2 into a single array.
    
    This function aggregates the position history data from multiple robots during
    phase 2 of the simulation into a 3D numpy array for efficient batch processing.
    
    Args:
        robots (list): List of robot objects, each containing a pos_history_phase2
                      attribute with position data.
    
    Returns:
        np.ndarray: A 3D array of shape (T, N, 2) where:
                   - T is the maximum length of position history across all robots
                   - N is the number of robots
                   - 2 represents the x, y coordinates
                   - NaN values fill positions for robots with shorter histories
    """
    T = max(len(r.pos_history_phase2) for r in robots)
    N = len(robots)
    arr = np.full((T, N, 2), np.nan)
    for j, r in enumerate(robots):
        for k, p in enumerate(r.pos_history_phase2):
            arr[k, j] = p
    return arr



def run_phase1_analysis(world_cfg, victims, robots, phase1_result):
    """
    Generate comprehensive analysis visualizations and metrics for Phase 1 of the multi-robot disaster response simulation.
    
    This function processes robot trajectories, measurements, and consensus data from Phase 1 to produce
    multiple diagnostic plots and numerical summaries. It generates coverage heatmaps, trajectory plots,
    consensus error analysis, detection delays, belief variance, false positive confusions, measurement noise
    scatter plots, and per-victim measurement errors.
    
    Args:
        world_cfg: Configuration object containing world parameters (width, height, dt_phase1).
        victims: List of victim objects with position attributes (pos, x, y).
        robots: List of robot objects containing Phase 1 history data:
            - pos_history_phase1: Position history throughout Phase 1
            - belief_history_phase1: Belief state history
            - belief_var_history_phase1: Belief variance history
            - detection_time_phase1: Dictionary mapping victim IDs to detection times
            - fp_confusion_events_phase1: List of false positive confusion events
            - meas_noise_scatter_phase1: Tuples of (measurement, true_position)
        phase1_result: Result object from Phase 1 execution (currently unused).
    
    Returns:
        None. Outputs are saved as PNG visualization files to "simulation_plots/phase1/" directory
        and binary NumPy files (.npy) containing structured data for detection delays and measurement errors.
    
    Generated outputs:
        - phase1_coverage.png: 2D histogram heatmap of robot coverage
        - phase1_trajectory.png: Plot of all robot trajectories
        - phase1_consensus_error.png: Consensus error over time steps
        - phase1_detection_delay.png: Bar chart of victim detection delays
        - phase1_belief_variance.png: Belief variance evolution per robot
        - phase1_fp_confusion.png: False positive confusion count per robot
        - phase1_meas_noise.png: Scatter plot of noisy measurements vs true victim positions
        - phase1_mean_measurement_error.png: Mean measurement error per victim
        - detection_delay.npy: Saved detection delay values
        - mean_errors.npy: Saved mean measurement error values
    """
    out = _ensure("simulation_plots/phase1")

    # ---- 1. Coverage Heatmap ----
    pos = _stack_positions_phase1(robots)
    xs = pos[..., 0].ravel()
    ys = pos[..., 1].ravel()
    xs = xs[~np.isnan(xs)]
    ys = ys[~np.isnan(ys)]

    H, xedges, yedges = np.histogram2d(
        xs, ys,
        bins=40,
        range=[[0, world_cfg.width], [0, world_cfg.height]]
    )
    plt.figure()
    plt.imshow(H.T, origin="lower",
               extent=[0, world_cfg.width, 0, world_cfg.height],
               aspect='equal')
    plt.title("Phase 1 Coverage Heatmap")
    plt.colorbar()
    plt.savefig(f"{out}/phase1_coverage.png", dpi=200)
    plt.close()

    # ---- 2. Trajectory Plot ----
    plt.figure()
    for r in robots:
        ph = np.array(r.pos_history_phase1)
        plt.plot(ph[:, 0], ph[:, 1], lw=0.7)
    plt.xlim(0, world_cfg.width)
    plt.ylim(0, world_cfg.height)
    plt.gca().set_aspect('equal')
    plt.title("Phase 1 Robot Trajectories")
    plt.savefig(f"{out}/phase1_trajectory.png", dpi=200)
    plt.close()

    # ---- 3. Consensus Error vs Time ----
    if len(victims) > 0:
        true = victims[0].pos
        err_list = []
        positions = _stack_positions_phase1(robots)
        T = positions.shape[0]
        for k in range(T):
            beliefs = []
            for r in robots:
                if len(r.belief_history_phase1) > k:
                    beliefs.append(r.belief_history_phase1[k][0])
            if len(beliefs) > 0:
                bmean = np.mean(beliefs, axis=0)
                err_list.append(np.linalg.norm(bmean - true))
            else:
                err_list.append(np.nan)

        plt.figure()
        plt.plot(err_list)
        plt.title("Phase 1 Consensus Error vs Time")
        plt.xlabel("step")
        plt.ylabel("error")
        plt.savefig(f"{out}/phase1_consensus_error.png", dpi=200)
        plt.close()

    # ---- Phase 1 Detection Delay ----
    det_times = []
    labels = []
    dt1 = getattr(world_cfg, "dt_phase1", 0.1)   # default = 1 second per step

    for r in robots:
        for vid, t in r.detection_time_phase1.items():
            det_times.append(t * dt1)
            labels.append(f"Victim {vid+1}")


    plt.figure(figsize=(12, 5))
    # --- convert step index to actual simulated time ---
    dt1 = getattr(world_cfg, "dt_phase1", 0.1)   # default 1.0 if missing
    det_times_sec = [t * dt1 for t in det_times]
    plt.bar(labels, det_times_sec)
    plt.xticks(rotation=65)
    plt.ylabel("detection time (sec)")
    plt.title("Phase 1 Detection Delay (real time)")
    plt.tight_layout()
    plt.savefig(f"{out}/phase1_detection_delay.png")
    plt.close()
    phase1_det = dict(zip(labels, det_times_sec))
    np.save("simulation_plots/phase1/detection_delay.npy", phase1_det)

    # ---- Phase 1 Belief Variance ----
    plt.figure(figsize=(8, 5))
    for r in robots:
        if len(r.belief_var_history_phase1) > 0:
            vars_ = [np.linalg.norm(v) for v in r.belief_var_history_phase1]
            plt.plot(vars_, alpha=0.3)

    plt.title("Phase 1 Belief Variance Over Time")
    plt.xlabel("step")
    plt.ylabel("variance")
    plt.grid(True)
    plt.savefig(f"{out}/phase1_belief_variance.png")
    plt.close()

    # ---- Phase 1 FP Confusion Events ----
    fp_counts = [len(r.fp_confusion_events_phase1) for r in robots]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(fp_counts)), fp_counts)
    plt.title("Phase 1 FP Confusion Count per Robot")
    plt.xlabel("robot id")
    plt.ylabel("# FP confusions")
    plt.savefig(f"{out}/phase1_fp_confusion.png")
    plt.close()

    # ---- Phase 1 Measurement Noise Scatter ----
    plt.figure(figsize=(7, 7))
    for r in robots:
        for z, true_pos in r.meas_noise_scatter_phase1:
            plt.scatter(z[0], z[1], s=5, c="blue", alpha=0.25)

    plt.scatter([v.x for v in victims],
                [v.y for v in victims],
                c="red", s=80, label="true victim")

    plt.title("Phase 1 Noisy Measurement Scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(f"{out}/phase1_meas_noise.png")
    plt.close()

    # ---- Phase-1 Measurement Error (per victim) ----
    errors_by_victim = {i: [] for i in range(len(victims))}
    for r in robots:
        for meas, true_pos in getattr(r, "meas_noise_scatter_phase1", []):
            for vi, v in enumerate(victims):
                if np.allclose(true_pos, v.pos):
                    err = np.linalg.norm(meas - v.pos)
                    errors_by_victim[vi].append(err)
                    break

    mean_errors = []
    labels = []
    for vi in sorted(errors_by_victim.keys()):
        vals = errors_by_victim[vi]
        if len(vals) > 0:
            mean_errors.append(np.mean(vals))
        else:
            mean_errors.append(0.0)
        labels.append(f"Victim {vi+1}")

    plt.figure(figsize=(7, 5))
    plt.bar(labels, mean_errors)
    plt.title("Phase 1 Mean Measurement Error (per victim)")
    plt.ylabel("error")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out}/phase1_mean_measurement_error.png")
    plt.close()
    phase1_error_dict = dict(zip(labels, mean_errors))
    np.save("simulation_plots/phase1/mean_errors.npy", phase1_error_dict)




def run_phase2_analysis(world_cfg, victims, false_sites, robots, phase2_cfg, phase2_result, log_csv="phase2_log.csv"):
    """
    Generate comprehensive visualization and analysis plots for Phase 2 of the multi-robot disaster response simulation.
    This function creates a series of diagnostic plots to evaluate robot performance during Phase 2, including
    environment mapping, potential field visualization, trajectory analysis, coverage heatmaps, and performance
    metrics such as belief accuracy, signal convergence, and detection delays.
    Args:
        world_cfg: Configuration object containing world dimensions (width, height).
        victims (list): List of victim objects with attributes x, y, pos.
        false_sites (list): List of false positive site objects with attributes x, y, pos.
        robots (list): List of robot objects containing position history, beliefs, and detection data.
        phase2_cfg: Configuration object for Phase 2 containing parameters like influence_radius, safe_radius,
                   victim_signal_mean, and dt (time step).
        phase2_result: Result object from Phase 2 simulation (currently unused in function body).
        log_csv (str, optional): CSV log filename. Defaults to "phase2_log.csv" (currently unused).
    Returns:
        None. Generates and saves PNG plots to "simulation_plots/phase2/" directory and NPY files for
        exact numerical values (belief_errors.npy, detection_delay.npy).
    Outputs:
        - environment_map.png: Scatter plot of victims and false positives
        - potential_field.png: 3D surface plot of combined potential field
        - phase2_trajectory.png: Robot trajectories during Phase 2
        - phase2_coverage.png: 2D heatmap of spatial coverage
        - belief_mean_error.png: Localization error per victim (C1 metric)
        - signal_convergence.png: Signal strength convergence per victim (S1 metric)
        - detection_delay.png: Time to detect each victim
        - belief_errors.npy: Dictionary of belief mean errors
        - detection_delay.npy: Dictionary of detection times in seconds
    """

    out = _ensure("simulation_plots/phase2")

    # ---- stack positions ----
    pos = _stack_positions_phase2(robots)

    # ---- 1. Environment Map ----
    plt.figure()
    plt.scatter([v.x for v in victims],
                [v.y for v in victims],
                c='red', label='victims')
    plt.scatter([f.x for f in false_sites],
                [f.y for f in false_sites],
                c='blue', label='FPs')
    plt.xlim(0, world_cfg.width)
    plt.ylim(0, world_cfg.height)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("Environment Map")
    plt.savefig(f"{out}/environment_map.png", dpi=200)
    plt.close()

    # ---- 2. Combined Potential Field ----
    xs = np.linspace(0, world_cfg.width, 80)
    ys = np.linspace(0, world_cfg.height, 80)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros_like(X)

    def V_victim(p, c):
        diff = p - c
        r = np.linalg.norm(diff)
        so = 1.2 * phase2_cfg.influence_radius
        si = 0.5 * phase2_cfg.safe_radius
        ko = 1.5
        ki = 2.5
        Vo = ko * np.exp(-(r * r) / (2 * so * so))
        Vi = ki * np.exp(-(r * r) / (2 * si * si))
        return Vo - Vi + ki

    def V_fp(p, c):
        diff = p - c
        r = np.linalg.norm(diff)
        so = 1.0 * phase2_cfg.influence_radius
        si = 0.4 * phase2_cfg.safe_radius
        ko = 0.8
        ki = 1.6
        Vo = ko * np.exp(-(r * r) / (2 * so * so))
        Vi = ki * np.exp(-(r * r) / (2 * si * si))
        return Vo - Vi + ki

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], Y[i, j]])
            val = 0
            for v in victims:
                val += V_victim(p, v.pos)
            for f in false_sites:
                val += V_fp(p, f.pos)
            V[i, j] = val

    V = (V - np.min(V)) / (np.max(V) - np.min(V))
    V = 25 + 2 * V

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, V, cmap='viridis',
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.92)

    ax.scatter([v.x for v in victims],
               [v.y for v in victims],
               zs=0, zdir='z',
               c='blue', marker='*', s=80, label='victims')

    ax.scatter([f.x for f in false_sites],
               [f.y for f in false_sites],
               zs=0, zdir='z',
               c='orange', marker='x', s=50, label='false pos')

    ax.set_title("Combined potential field (victims = tall donut hills, FP = shorter hills)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("V(x,y)")
    ax.view_init(elev=32, azim=230)

    plt.legend()
    fig.savefig(f"{out}/potential_field.png", dpi=200)
    plt.close(fig)

    # ---- 3. Phase 2 Trajectory Plot ----
    plt.figure()
    for r in robots:
        ph = np.array(r.pos_history_phase2)
        plt.plot(ph[:, 0], ph[:, 1], lw=0.7)
    plt.xlim(0, world_cfg.width)
    plt.ylim(0, world_cfg.height)
    plt.gca().set_aspect('equal')
    plt.title("Phase 2 Trajectories")
    plt.savefig(f"{out}/phase2_trajectory.png", dpi=200)
    plt.close()

    # ---- Phase 2 Coverage Heatmap ----
    pos2 = _stack_positions_phase2(robots)
    xs2 = pos2[..., 0].ravel()
    ys2 = pos2[..., 1].ravel()

    xs2 = xs2[~np.isnan(xs2)]
    ys2 = ys2[~np.isnan(ys2)]

    H2, xedges2, yedges2 = np.histogram2d(
        xs2, ys2,
        bins=40,
        range=[[0, world_cfg.width], [0, world_cfg.height]]
    )

    plt.figure()
    plt.imshow(
        H2.T,
        origin="lower",
        extent=[0, world_cfg.width, 0, world_cfg.height],
        aspect='equal'
    )
    plt.colorbar()
    plt.title("Phase 2 Coverage Heatmap")
    plt.savefig(f"{out}/phase2_coverage.png", dpi=200)
    plt.close()


    # ---- 4. Belief Mean Error ----
    errors = []
    labels = []
    for i, v in enumerate(victims):
        team = [r.belief_victims[i] for r in robots
                if getattr(r, "assigned_victim", None) == i]
        if len(team) > 0:
            meanb = np.mean(np.stack(team), axis=0)
            errors.append(np.linalg.norm(meanb - v.pos))
            labels.append(f"Victim {i+1}")

    if errors:
        plt.figure()
        plt.bar(range(len(errors)), errors)
        plt.xticks(range(len(errors)), labels, rotation=30)
        plt.ylabel("error")
        plt.title("Phase 2 Belief Mean Error (C1)")
        plt.savefig(f"{out}/belief_mean_error.png", dpi=200)
        plt.close()
        phase2_error_dict = dict(zip(labels, errors))
        np.save("simulation_plots/phase2/belief_errors.npy", phase2_error_dict)


    # ---- 5. Signal Convergence  ----
    means = []
    labels = []
    for i, v in enumerate(victims):
        sigs = []
        for r in robots:
            if getattr(r, "assigned_victim", None) == i:
                if hasattr(r, "victim_signal") and len(r.victim_signal) > i:
                    sigs.append(r.victim_signal[i])
        if len(sigs) > 0:
            means.append(np.mean(sigs))
            labels.append(f"Victim {i+1}")

    if means:
        plt.figure()
        plt.bar(range(len(means)), means)
        plt.axhline(phase2_cfg.victim_signal_mean, color='r', linestyle='--')
        plt.xticks(range(len(means)), labels, rotation=30)
        plt.ylabel("signal")
        plt.title("Phase 2 Signal Convergence (S1)")
        plt.savefig(f"{out}/signal_convergence.png", dpi=200)
        plt.close()

    # ---- 6. Detection Delay  ----
    det_times = {}
    dt2 = getattr(phase2_cfg, "dt", 0.1)

    for r in robots:
        for vid, t in getattr(r, "detection_time_phase2", {}).items():
            if vid not in det_times:
                det_times[vid] = t
            else:
                det_times[vid] = min(det_times[vid], t)

    # convert to sorted dict { "Victim 1" : time_sec, ... }
    det_sorted = {
        f"Victim {vid+1}": det_times[vid] * dt2
        for vid in sorted(det_times.keys())
    }

    if det_sorted:
        labels = list(det_sorted.keys())
        vals = list(det_sorted.values())

        plt.figure()
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=30)
        plt.ylabel("Detection Time (sec)")
        plt.title("Detection Delay (Phase 2, Corrected)")
        plt.savefig(f"{out}/detection_delay.png", dpi=200)
        plt.close()
        np.save("simulation_plots/phase2/detection_delay.npy", det_sorted)


def make_combined_plots():
    """
    Generate combined comparison plots for Phase 1 and Phase 2 simulation results.
    
    This function creates visualization comparing detection delays and belief mean errors
    between Phase 1 and Phase 2 of a multi-robot disaster response simulation. It loads
    pre-computed metrics from numpy files, generates side-by-side bar charts, and saves
    the plots as PNG images.
    
    Inputs:
        - Requires pre-computed numpy files in 'simulation_plots/' directory:
          * phase1/mean_errors.npy: Phase 1 mean error metrics
          * phase2/belief_errors.npy: Phase 2 belief error metrics
          * phase1/detection_delay.npy: Phase 1 detection delay metrics
          * phase2/detection_delay.npy: Phase 2 detection delay metrics
    
    Outputs:
        - Generates two PNG plot files in 'simulation_plots/combined/' directory:
          * combined_detection_delay.png: Bar chart comparing detection times
          * combined_mean_error.png: Bar chart comparing belief mean errors
    
    Returns:
        None
    """

    out = _ensure("simulation_plots/combined")
    p1_err = np.load("simulation_plots/phase1/mean_errors.npy", allow_pickle=True).item()
    p2_err = np.load("simulation_plots/phase2/belief_errors.npy", allow_pickle=True).item()

    p1_det = np.load("simulation_plots/phase1/detection_delay.npy", allow_pickle=True).item()
    p2_det = np.load("simulation_plots/phase2/detection_delay.npy", allow_pickle=True).item()

    victims = sorted(p1_err.keys(), key=lambda x: int(x.split()[1]))
    x = np.arange(len(victims))
    width = 0.35

    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, [p1_det[v] for v in victims], width,
            label="Phase 1", color="royalblue")
    plt.bar(x + width/2, [p2_det[v] for v in victims], width,
            label="Phase 2", color="crimson")

    plt.xticks(x, victims, rotation=30)
    plt.ylabel("Detection Time (sec)")
    plt.title("Phase 1 vs Phase 2 Victim Detection Delay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/combined_detection_delay.png", dpi=300)
    plt.close()


    plt.figure(figsize=(10,5))
    plt.bar(x - width/2, [p1_err[v] for v in victims], width,
            label="Phase 1 Error", color="royalblue")
    plt.bar(x + width/2, [p2_err[v] for v in victims], width,
            label="Phase 2 Error", color="crimson")

    plt.xticks(x, victims, rotation=30)
    plt.ylabel("Error")
    plt.title("Phase 1 vs Phase 2 Belied Mean Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out}/combined_mean_error.png", dpi=300)
    plt.close()
