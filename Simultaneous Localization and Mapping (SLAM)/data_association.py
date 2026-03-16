import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pandas as pd

# ── Load Track from CSV ───────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_HERE, "small_track.csv"))

BLUE_CONES   = df[df["tag"] == "blue"      ][["x", "y"]].values.astype(float)
YELLOW_CONES = df[df["tag"] == "yellow"    ][["x", "y"]].values.astype(float)
BIG_ORANGE   = df[df["tag"] == "big_orange"][["x", "y"]].values.astype(float)

_cs               = df[df["tag"] == "car_start"].iloc[0]
CAR_START_POS     = np.array([float(_cs["x"]), float(_cs["y"])])
CAR_START_HEADING = float(_cs["direction"])   # radians (0 = east)

MAP_CONES = np.vstack([BLUE_CONES, YELLOW_CONES])


# ── Build Approximate Centerline ──────────────────────────────────────────────
def _build_centerline():
    """
    Pair each blue cone with its nearest yellow cone, take the midpoint,
    then sort CLOCKWISE around the track centroid so pure-pursuit drives CW.
    """
    center = np.mean(MAP_CONES, axis=0)
    D      = distance.cdist(BLUE_CONES, YELLOW_CONES)
    mids   = np.array(
        [(BLUE_CONES[i] + YELLOW_CONES[np.argmin(D[i])]) / 2.0
         for i in range(len(BLUE_CONES))]
    )
    angles = np.arctan2(mids[:, 1] - center[1], mids[:, 0] - center[0])
    return mids[np.argsort(angles)[::-1]]   # descending angle = clockwise


CENTERLINE = _build_centerline()


# ── Simulation Parameters ─────────────────────────────────────────────────────
SENSOR_RANGE = 12.0   # metres – sensor visibility radius
NOISE_STD    = 0.20   # metres – measurement noise std-dev
WHEELBASE    = 3.0    # metres – bicycle model wheelbase
DT           = 0.1    # seconds – time step
SPEED        = 7.0    # m/s
LOOKAHEAD    = 5.5    # pure-pursuit lookahead distance (m)
N_FRAMES     = 130    # ≈ one full lap


# ── Utility Functions ─────────────────────────────────────────────────────────
def angle_wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def pure_pursuit(pos: np.ndarray, heading: float, path: np.ndarray) -> float:
    """Compute steering angle (rad) to follow *path* via pure-pursuit."""
    dists   = np.linalg.norm(path - pos, axis=1)
    nearest = int(np.argmin(dists))
    n       = len(path)
    target  = path[(nearest + 5) % n]       # fallback lookahead
    for k in range(nearest, nearest + n):
        pt = path[k % n]
        if np.linalg.norm(pt - pos) >= LOOKAHEAD:
            target = pt
            break
    alpha = angle_wrap(
        np.arctan2(target[1] - pos[1], target[0] - pos[0]) - heading
    )
    steer = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), LOOKAHEAD)
    return float(np.clip(steer, -0.6, 0.6))


def local_to_global(local_pts: np.ndarray,
                    pos: np.ndarray, heading: float) -> np.ndarray:
    """Rotate + translate points from the car's local frame to world frame."""
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, -s], [s, c]])       # local → world rotation
    return (R @ local_pts.T).T + pos


def get_measurements(pos: np.ndarray, heading: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left),
    along with their true indices in MAP_CONES for evaluation.
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible_idx = np.where(dists < SENSOR_RANGE)[0]
    visible = MAP_CONES[visible_idx]
    if len(visible) == 0:
        return np.zeros((0, 2)), np.array([], dtype=int)
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    noisy_local = local + np.random.normal(0, NOISE_STD, local.shape)
    return noisy_local, visible_idx


def step_kinematic(pos: np.ndarray, heading: float,
                   velocity: float, steering: float):
    """One bicycle-model step; returns (new_pos, new_heading)."""
    new_pos = pos.copy()
    new_pos[0] += velocity * np.cos(heading) * DT
    new_pos[1] += velocity * np.sin(heading) * DT
    new_heading = angle_wrap(
        heading + (velocity / WHEELBASE) * np.tan(steering) * DT
    )
    return new_pos, new_heading


def draw_track(ax, alpha_b: float = 0.4, alpha_y: float = 0.4) -> None:
    ax.scatter(BLUE_CONES[:, 0],   BLUE_CONES[:, 1],
               c="royalblue", marker="^", s=65,  alpha=alpha_b,
               zorder=2, label="Blue cones")
    ax.scatter(YELLOW_CONES[:, 0], YELLOW_CONES[:, 1],
               c="gold",      marker="^", s=65,  alpha=alpha_y,
               zorder=2, label="Yellow cones")
    ax.scatter(BIG_ORANGE[:, 0],   BIG_ORANGE[:, 1],
               c="darkorange", marker="s", s=100, alpha=0.7,
               zorder=2, label="Start gate")


def draw_car(ax, pos: np.ndarray, heading: float) -> None:
    ax.scatter(pos[0], pos[1], c="red", s=160, zorder=7, label="Car")
    ax.arrow(pos[0], pos[1],
             2.2 * np.cos(heading), 2.2 * np.sin(heading),
             head_width=0.8, fc="red", ec="red", zorder=8)


def setup_ax(ax, subtitle: str = "") -> None:
    ax.set_xlim(-28, 28)
    ax.set_ylim(-22, 22)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    if subtitle:
        ax.set_title(subtitle, fontsize=10)


# ── Abstract Base ─────────────────────────────────────────────────────────────
class Bot:
    def __init__(self):
        self.pos     = CAR_START_POS.copy()   # (2,) float64
        self.heading = CAR_START_HEADING      # radians

    def data_association(self, measurements, current_map):
        raise NotImplementedError

    def localization(self, velocity, steering):
        raise NotImplementedError

    def mapping(self, measurements):
        raise NotImplementedError


# ──  Solution ──────────────────────────────────────────────────────────
class Solution(Bot):
    def __init__(self):
        super().__init__()
        self.learned_map  = []                    # list of np.ndarray (2,)
        # Internal state exposed for visualisation
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)

    # ------------------------------------------------------------------
    def data_association(self, measurements, current_map):
        """
        Global Assignment with Spatial Gating for data association.
        Optimized with KD-Tree subset extraction to maintain O(N log M) running time.
        Steps:
          1. Transform local measurements → world frame using current pose.
          2. Query KD-Tree of the map for candidates near the measurements.
          3. Compute pairwise distance matrix ONLY on the subset.
          4. Use Hungarian algorithm (linear_sum_assignment) on the reduced dense matrix.
        Returns an array mapping each measurement to a map index (-1 if unassociated).
        """
        if len(measurements) == 0:
            self._global_meas = np.zeros((0, 2))
            self._assoc       = np.array([], dtype=int)
            return self._assoc

        gm = local_to_global(measurements, self.pos, self.heading)
        self._global_meas = gm

        if len(current_map) == 0:
            self._assoc = np.full(len(measurements), -1, dtype=int)
            return self._assoc

        from scipy.optimize import linear_sum_assignment
        from scipy.spatial import cKDTree
        
        GATE_THRESHOLD = 3.0
        
        # 1. Identify candidate map landmarks using KD-Tree to avoid dense N x M distance matrix
        map_tree = cKDTree(current_map)
        
        # query_ball_point returns list of lists.
        # Find all map indices that are within GATE_THRESHOLD of ANY measurement
        candidate_indices_lists = map_tree.query_ball_point(gm, r=GATE_THRESHOLD)
        
        # Flatten and get unique map indices
        unique_candidate_idx = np.unique([idx for sublist in candidate_indices_lists for idx in sublist])
        
        self._assoc = np.full(len(measurements), -1, dtype=int)
        
        if len(unique_candidate_idx) == 0:
            return self._assoc
            
        # 2. Extract the subset of the map and compute small dense D
        map_subset = current_map[unique_candidate_idx]
        
        D_subset = distance.cdist(gm, map_subset)
        D_subset[D_subset > GATE_THRESHOLD] = 1e6
        
        row_ind, col_ind_subset = linear_sum_assignment(D_subset)
        
        # 3. Map the subset indices back to original map indices
        for r, c_sub in zip(row_ind, col_ind_subset):
            if D_subset[r, c_sub] < GATE_THRESHOLD:
                self._assoc[r] = unique_candidate_idx[c_sub]
                
        return self._assoc

# ── Problem 1 – Data Association ──────────────────────────────────────────────
def make_problem1():
    """
    Visualise nearest-neighbour association: cyan dots = sensor measurements
    transformed to world frame; green dashed lines connect each measurement
    to its matched map cone.
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 1 – Data Association  (Nearest Neighbour)",
                 fontsize=13, fontweight="bold")

    global_correct_assocs = 0
    global_total_assocs = 0
    lap_count = 1

    def update(frame):
        nonlocal global_correct_assocs, global_total_assocs, lap_count
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas, true_indices = get_measurements(sol.pos, sol.heading)
        assocs = sol.data_association(meas, MAP_CONES)
        
        # Evaluate associations
        if len(true_indices) > 0:
            correct = np.sum(assocs == true_indices)
            global_correct_assocs += correct
            global_total_assocs += len(true_indices)
            
        sol.pos, sol.heading = step_kinematic(sol.pos, sol.heading, SPEED, steer)

        draw_track(ax)

        if len(sol._global_meas) > 0:
            for idx, gm in zip(sol._assoc, sol._global_meas):
                if idx != -1:
                    mc = MAP_CONES[idx]
                    ax.plot([gm[0], mc[0]], [gm[1], mc[1]],
                            "g--", lw=1.0, alpha=0.65, zorder=3)
            ax.scatter(sol._global_meas[:, 0], sol._global_meas[:, 1],
                       c="cyan", s=45, zorder=5,
                       label=f"Measurements ({len(sol._global_meas)})")

        draw_car(ax, sol.pos, sol.heading)
        
        acc_str = f"Assoc Acc: {100.0 * global_correct_assocs / max(1, global_total_assocs):.1f}%" if global_total_assocs > 0 else ""
        setup_ax(ax, f"Frame {frame+1}/{N_FRAMES}  –  "
                     f"green lines = NN association  |  {acc_str}")
                     
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        if frame == N_FRAMES - 1:
            final_acc = 100.0 * global_correct_assocs / max(1, global_total_assocs)
            print(f"\n--- Lap {lap_count} ---")
            print(f"[Metrics] Cumulative Association Accuracy: {final_acc:.2f}% ({global_correct_assocs}/{global_total_assocs})")
            lap_count += 1

    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=100, repeat=True)
    return fig, ani

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Driverless Car Hackathon – SLAM Visualisation ===")
    print(f"  Blue cones   : {len(BLUE_CONES)}")
    print(f"  Yellow cones : {len(YELLOW_CONES)}")
    print(f"  Big orange   : {len(BIG_ORANGE)}")
    print(f"  Car start    : {CAR_START_POS}  "
          f"heading={np.degrees(CAR_START_HEADING):.1f}°")
    print(f"  Centerline   : {len(CENTERLINE)} waypoints (clockwise)")
    print("\nOpening 1 animation window …")

    # Keep references to prevent garbage collection of FuncAnimation objects.
    fig1, ani1 = make_problem1()

    plt.show()