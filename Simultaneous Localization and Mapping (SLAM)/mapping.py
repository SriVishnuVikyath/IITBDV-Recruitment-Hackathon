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
        
        # Track raw structures: {'pos': np.ndarray(2,), 'hits': int, 'misses': int}
        self._trackers = []
        
        # Internal state exposed for visualisation
        self._global_meas = np.zeros((0, 2))
        self._assoc       = np.array([], dtype=int)

    # ------------------------------------------------------------------
    def mapping(self, measurements):
        """
        Landmark Lifecycle Mapping with Linear Kalman Filter.
        """
        if len(measurements) == 0:
            gm = np.zeros((0, 2))
        else:
            gm = local_to_global(measurements, self.pos, self.heading)
            
        matched_trackers = set()
        
        # Measurement noise covariance
        R_cov = np.diag([NOISE_STD**2, NOISE_STD**2])
        
        for p in gm:
            if not self._trackers:
                P_init = np.diag([1.0, 1.0])
                self._trackers.append({'pos': p.copy(), 'P': P_init, 'hits': 1, 'misses': 0})
                matched_trackers.add(0)
            else:
                dists = [np.linalg.norm(p - t['pos']) for t in self._trackers]
                min_idx = np.argmin(dists)
                if dists[min_idx] < 2.0:
                    # Match found, update via Kalman Filter
                    mu = self._trackers[min_idx]['pos']
                    P = self._trackers[min_idx]['P']
                    
                    # Innovation
                    y = p - mu
                    # Innovation covariance (H = I)
                    S = P + R_cov
                    # Kalman Gain
                    K = P @ np.linalg.inv(S)
                    
                    # Update state
                    self._trackers[min_idx]['pos'] = mu + K @ y
                    self._trackers[min_idx]['P'] = (np.eye(2) - K) @ P
                    self._trackers[min_idx]['hits'] += 1
                    matched_trackers.add(min_idx)
                else:
                    P_init = np.diag([1.0, 1.0])
                    self._trackers.append({'pos': p.copy(), 'P': P_init, 'hits': 1, 'misses': 0})
                    matched_trackers.add(len(self._trackers) - 1)
                    
        # Update misses for trackers in FOV that were NOT matched
        for i, t in enumerate(self._trackers):
            if i not in matched_trackers:
                # If within sensor range but not seen, it's a miss
                if np.linalg.norm(t['pos'] - self.pos) < SENSOR_RANGE:
                    t['misses'] += 1
                    
        # Pruning and filtering building learned_map
        kept_trackers = []
        self.learned_map = []
        for t in self._trackers:
            if t['misses'] > 5 and t['hits'] < 3:
                continue # Prune ghost cone
            kept_trackers.append(t)
            
            # Confirm if hits >= 2
            if t['hits'] >= 2:
                self.learned_map.append(t['pos'].copy())
                
        self._trackers = kept_trackers

# ── Problem 3 – Mapping ───────────────────────────────────────────────────────
def make_problem3():
    """
    Visualise incremental mapping: green × marks show the car's accumulated
    global cone map built from local sensor measurements.  Ground-truth cones
    are faded so the learned map stands out.
    """
    sol = Solution()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 3 – Mapping  (Local → Global Transform + Deduplication)",
                 fontsize=13, fontweight="bold")
                 
    seen_true_cones = set()
    lap_count = 1

    def update(frame):
        nonlocal lap_count
        ax.clear()
        steer = pure_pursuit(sol.pos, sol.heading, CENTERLINE)
        meas, true_indices = get_measurements(sol.pos, sol.heading)
        
        for idx in true_indices:
            seen_true_cones.add(int(idx))
            
        sol.pos, sol.heading = step_kinematic(sol.pos, sol.heading, SPEED, steer)
        sol.mapping(meas)

        draw_track(ax, alpha_b=0.15, alpha_y=0.15)

        if sol.learned_map:
            lm = np.array(sol.learned_map)
            ax.scatter(lm[:, 0], lm[:, 1],
                       c="limegreen", marker="x", s=90, linewidths=2.0,
                       zorder=5, label=f"Mapped cones ({len(lm)})")

        draw_car(ax, sol.pos, sol.heading)
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"map size: {len(sol.learned_map)} / {len(MAP_CONES)} cones")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        if frame == N_FRAMES - 1:
            true_count = len(seen_true_cones)
            pred_count = len(sol.learned_map)
            
            # Map RMSE: nearest ground-truth cone for every estimated cone
            total_sq_err = 0.0
            if pred_count > 0:
                for est_pt in sol.learned_map:
                    dists = np.linalg.norm(MAP_CONES - est_pt, axis=1)
                    total_sq_err += np.min(dists)**2
                map_rmse = np.sqrt(total_sq_err / pred_count)
            else:
                map_rmse = float('inf')
                
            print(f"\n--- Lap {lap_count} ---")
            print(f"[Metrics] Cumulative Visible True Cones: {true_count}")
            print(f"[Metrics] Estimated Map Size: {pred_count}")
            print(f"[Metrics] Map Size Error: {abs(pred_count - true_count)} cones")
            print(f"[Metrics] Map Positional RMSE: {map_rmse:.3f}m")
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
    fig3, ani3 = make_problem3()

    plt.show()