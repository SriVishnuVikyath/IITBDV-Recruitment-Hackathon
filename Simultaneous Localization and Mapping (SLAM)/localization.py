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


def get_measurements(pos: np.ndarray, heading: float) -> np.ndarray:
    """
    Simulate a 2-D lidar: return visible cone positions as noisy
    measurements in the car's LOCAL frame (x = forward, y = left).
    """
    dists   = np.linalg.norm(MAP_CONES - pos, axis=1)
    visible = MAP_CONES[dists < SENSOR_RANGE]
    if len(visible) == 0:
        return np.zeros((0, 2))
    c, s = np.cos(heading), np.sin(heading)
    R    = np.array([[c, s], [-s, c]])       # world → local (transpose of above)
    local = (R @ (visible - pos).T).T
    return local + np.random.normal(0, NOISE_STD, local.shape)


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
        
        # Localization uncertainty covariance P (x, y, theta)
        self.P = np.zeros((3, 3))
        
        # Process noise Q (variance of odometry steps)
        self.Q = np.diag([0.05**2, 0.05**2, np.deg2rad(1.0)**2])

    # ------------------------------------------------------------------
    def localization(self, velocity, steering):
        """
        Bicycle kinematic model with Exact Analytical Circular-Arc Integration:
            If steer != 0, follows a circular arc.
            If steer ≈ 0, follows linear path.
        """
        # Save old values
        theta = self.heading

        if abs(steering) > 1e-4:
            # Exact circular arc
            R = WHEELBASE / np.tan(steering)
            d_theta = (velocity / R) * DT
            
            self.pos[0] += R * (np.sin(theta + d_theta) - np.sin(theta))
            self.pos[1] += R * (-np.cos(theta + d_theta) + np.cos(theta))
            self.heading = angle_wrap(theta + d_theta)
            
            # Jacobian F of the motion model w.r.t state (x, y, theta)
            F = np.eye(3)
            F[0, 2] = R * (np.cos(theta + d_theta) - np.cos(theta))
            F[1, 2] = R * (np.sin(theta + d_theta) - np.sin(theta))
        else:
            # Linear approximation for near-zero steering
            self.pos[0] += velocity * np.cos(theta) * DT
            self.pos[1] += velocity * np.sin(theta) * DT
            self.heading = angle_wrap(theta + (velocity / WHEELBASE) * np.tan(steering) * DT)
            
            # Jacobian F
            F = np.eye(3)
            F[0, 2] = -velocity * np.sin(theta) * DT
            F[1, 2] = velocity * np.cos(theta) * DT
            
        # Propagate covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q


# ── Problem 2 – Localization ───────────────────────────────────────────────────
def make_problem2():
    """
    Visualise dead-reckoning: the magenta trail is the car's estimated
    trajectory built purely from the kinematic model and steering commands.
    """
    sol     = Solution()
    true_pos = CAR_START_POS.copy()
    true_heading = CAR_START_HEADING
    
    path_x  = [float(sol.pos[0])]
    path_y  = [float(sol.pos[1])]
    
    true_path_x = [float(true_pos[0])]
    true_path_y = [float(true_pos[1])]
    
    squared_errors = []
    lap_count = 1
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle("Problem 2 – Localization  (Dead Reckoning / Kinematic Model)",
                 fontsize=13, fontweight="bold")

    def update(frame):
        nonlocal true_pos, true_heading, lap_count
        ax.clear()
        
        # Ground Truth Update
        steer_true = pure_pursuit(true_pos, true_heading, CENTERLINE)
        true_pos, true_heading = step_kinematic(true_pos, true_heading, SPEED, steer_true)
        true_path_x.append(float(true_pos[0]))
        true_path_y.append(float(true_pos[1]))
        
        # Estimated Update (with synthetic noise on odometry inputs)
        noisy_speed = SPEED + np.random.normal(0, 0.05)
        noisy_steer = steer_true + np.random.normal(0, np.deg2rad(1.0))
        sol.localization(noisy_speed, noisy_steer)
        path_x.append(float(sol.pos[0]))
        path_y.append(float(sol.pos[1]))
        
        # Metrics Tracking
        err = np.linalg.norm(sol.pos - true_pos)
        squared_errors.append(err**2)

        draw_track(ax)
        
        from matplotlib.patches import Ellipse
        
        # Draw uncertainty ellipse (3-sigma confidence)
        cov_xy = sol.P[0:2, 0:2]
        eigenvalues, eigenvectors = np.linalg.eig(cov_xy)
        # Using the dominant eigenvector for angle
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * 3 * np.sqrt(max(0, eigenvalues[0]))
        height = 2 * 3 * np.sqrt(max(0, eigenvalues[1]))
        
        ellipse = Ellipse(xy=sol.pos, width=width, height=height, angle=angle,
                          edgecolor='blue', facecolor='blue', alpha=0.15, zorder=3,
                          label="3σ Uncertainty")
        ax.add_patch(ellipse)
        
        ax.plot(true_path_x, true_path_y, color="black", lw=1.5, linestyle="--",
                alpha=0.6, zorder=3, label="True path")
        ax.plot(path_x, path_y, color="magenta", lw=2.0,
                alpha=0.85, zorder=4, label="Estimated path")
        draw_car(ax, sol.pos, sol.heading)
        draw_car(ax, true_pos, true_heading) # Draw true car behind
        ax.scatter(true_pos[0], true_pos[1], c="black", s=160, zorder=6, alpha=0.3)
        
        current_rmse = np.sqrt(np.mean(squared_errors))
        setup_ax(ax,
            f"Frame {frame+1}/{N_FRAMES}  –  "
            f"RMSE: {current_rmse:.2f}m")
        
        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.8)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        if frame == N_FRAMES - 1:
            final_rmse = np.sqrt(np.mean(squared_errors))
            final_drift = np.linalg.norm(sol.pos - true_pos)
            print(f"\n--- Lap {lap_count} ---")
            print(f"[Metrics] Cumulative Trajectory RMSE: {final_rmse:.3f}m")
            print(f"[Metrics] Current Position Drift: {final_drift:.3f}m")
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
    fig2, ani2 = make_problem2()

    plt.show()