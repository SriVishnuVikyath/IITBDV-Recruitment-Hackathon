# IITB Driverless Vehicle Recruitment Hackathon

**Author:** Sri Vishnu Vikyath S

This hackathon covers three core areas of a Formula Student Driverless car: **Perception**, **Path Planning & Control (PPC)**, and **SLAM**. Each module is self-contained but together they represent a complete autonomous driving pipeline — see, plan, drive.

---

## 1. Perception — Cone Distance Estimation

**Folder:** `Perception/`  
**File:** `main.py`

### What it does
Given a single camera image, detect all traffic cones on the track and report how far away each one is in metres.

### How it works

**Step 1 – Detection:**  
A pre-trained **YOLOv11s-Carmaker** model runs inference on the input image (`testing/image.webp`) at a confidence threshold of **0.5**. It outputs bounding boxes for each detected cone with its class (`Blue`, `Yellow`, `Small Orange`).

**Step 2 – Distance calculation:**  
The pixel height `h` of each bounding box is used with the pinhole camera formula:

```
distance (m) = (H × f) / (h × 100)
```

Where:
- `H` = 30 cm (real-world cone height)
- `f` = 100 cm (camera focal length)
- `h` = pixel height of the bounding box (extracted as `y2 − y1`)

**Step 3 – Output:**  
Each detected cone gets a labelled bounding box drawn on the image (colour-coded by type) with its distance annotated. The result is saved to `annotated_output.jpg` and printed to console.

### Why it works
Perspective geometry: the bigger a cone appears in the image (larger pixel height), the closer it is. YOLO gives us the bounding box heights automatically, so no depth sensor or stereo camera is needed — just one image and simple maths.

### Assumptions
- Cones are upright on flat ground.
- The bounding box fully captures the cone's vertical extent.

---

## 2. PPC — Path Planning & Vehicle Control

**Folder:** `PPC/`  
**Files:** `participant/planner.py`, `participant/controller.py`

### What it does
Given the positions of all cones around the track, plan a centre-line path and control the car to follow it smoothly.

### Part A — Planner (`planner.py`)

Cones come with a `side` (`"left"` / `"right"`) and an `index` (matching number for left-right pairs).

**Algorithm:**
1. Separate cones into left (blue) and right (yellow) lists, sorted by index.
2. For each matched pair `(left_cone, right_cone)`, compute the midpoint:
   ```
   mid_x = (lc.x + rc.x) / 2
   mid_y = (lc.y + rc.y) / 2
   ```
3. The ordered list of midpoints becomes the waypoint path.

This produces a clean, collision-safe **centre-line** through the entire track in a single pass — O(N) with no curve fitting needed.

### Part B — Controller (`controller.py`)

Runs every **50 ms** during simulation. Receives the car's position, heading, and velocity; outputs throttle, steer, and brake.

**Steering — Stanley Controller:**

```
steer = heading_error + arctan(k × cross_track_error / (vx + 0.1))
```

- Finds the closest waypoint and the segment after it.
- `heading_error (ψ)` — angle between the car's heading and the path direction.
- `cross_track_error (e)` — signed perpendicular distance from the car to the path segment.
- Gain `k = 1.0`. Clipped to **[−0.5, 0.5] rad**.
- The `(vx + 0.1)` denominator ensures the cross-track correction shrinks at high speed — naturally smoother steering.

**Speed — PI Controller:**

```
output = 1.5 × error + 0.1 × integral
```

- Target speed: **5 m/s**.
- `error = target − current_vx`.
- Integral is clamped to **[−5, 5]** to prevent windup.
- Positive output → throttle; negative output → brake. Both clipped to **[0, 1]**.

### Results

| Metric | Value |
|---|---|
| Lap completed | ✅ Yes |
| Cone hits | 0 |
| Penalty time | +0.0 s |
| Lap time | ~23–60 s |

---

## 3. SLAM — Simultaneous Localisation and Mapping

**Folder:** `Simultaneous Localization and Mapping (SLAM)/`  
**Files:** `data_association.py`, `localization.py`, `mapping.py`

### Simulation environment
- **Sensor:** 2-D lidar, 12 m range, noise σ = 0.2 m
- **Vehicle:** Bicycle model, wheelbase = 3 m, speed = 7 m/s, dt = 0.1 s
- **Track data:** `small_track.csv` (blue, yellow, and big-orange cones + car start pose)
- **Lap length:** 130 frames ≈ one full lap

### Part A — Data Association (`data_association.py`)

**Problem with the baseline:** Greedy nearest-neighbour — when two measurements are near the same cone, one "steals" it, forcing the other to match a wrong cone far away.

**Solution: KD-Tree spatial gating + Hungarian algorithm**

1. Build a KD-Tree of all known map cones.
2. For each batch of measurements (in world frame), query the tree for all candidates within **3.0 m** (spatial gating). This limits the Hungarian assignment to a small subset.
3. Run `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) on the small distance matrix — finds the globally optimal one-to-one matching.
4. Distances above the gate threshold are set to `1e6` (effectively forbidden).

Result: **O(N log M)** runtime (KD-Tree query) + small exact global assignment — best of both worlds.

**Metric tracked:** Association Accuracy (%) — fraction of measurements correctly matched to their true map cone.

---

### Part B — Localisation (`localization.py`)

**Problem with the baseline:** Euler (straight-line) integration — each step assumes the car moved in a straight line, which drifts badly on curves.

**Solution: Exact circular-arc integration + covariance propagation**

When steering angle `|δ| > 1e-4`:
```
R       = wheelbase / tan(δ)          # turning radius
dθ      = (v / R) × dt
new_x  += R × (sin(θ + dθ) − sin(θ))
new_y  += R × (−cos(θ + dθ) + cos(θ))
```

When steering ≈ 0 → standard linear update.

**Uncertainty tracking:**  
A 3×3 covariance matrix `P` is propagated each step via the motion Jacobian `F`:
```
P = F × P × Fᵀ + Q
```
where process noise `Q = diag([0.05², 0.05², (1°)²])`.

This is visualised as a **blue 3-sigma ellipse** around the car — you can see uncertainty grow on straights and shrink with corrections.

**Metrics tracked:**
- Trajectory RMSE (m) — average distance between estimated and true path.
- Final drift (m) — endpoint position error after one full lap.

---

### Part C — Mapping (`mapping.py`)

**Problem with the baseline:** Every sensor reading is appended to a list and deduplicated only by distance. No noise averaging, no ghost rejection.

**Solution: Kalman Filter per landmark + hit/miss lifecycle**

Each detected cone is tracked as a hypothesis `{pos, P, hits, misses}`:

- **On first detection:** a new tracker is created with position `p` and initial covariance `P = I`.
- **On re-detection (distance < 2 m):** Kalman update:
  ```
  y = p − μ           # innovation (residual)
  S = P + R           # innovation covariance (H = I, R = diag(σ²))
  K = P × S⁻¹         # Kalman gain
  μ ← μ + K × y       # updated position
  P ← (I − K) × P    # updated covariance
  hits += 1
  ```
- **If within sensor range but not detected:** `misses += 1`.
- **Confirmed** (added to final map): `hits ≥ 2`.
- **Pruned** (ghost/noise): `misses > 5` and `hits < 3`.

**Metrics tracked:**
- Estimated vs. true visible cone count.
- Map size error (|predicted − true| cones).
- Map positional RMSE (m) — average distance from each estimated cone to its nearest ground-truth cone.

---

## Summary

| Module | Core algorithm | Key strength |
|---|---|---|
| Perception | YOLOv11s + pinhole formula | No depth sensor needed |
| PPC Planner | Midpoint interpolation | Simple, zero cone hits |
| PPC Controller | Stanley + PI | Stable at speed, no tuning instability |
| SLAM – Data Association | KD-Tree gating + Hungarian | Globally optimal, real-time speed |
| SLAM – Localisation | Circular-arc kinematics + Jacobian covariance | Follows curves exactly, tracks uncertainty |
| SLAM – Mapping | Kalman filter + lifecycle pruning | Noise-robust, ghost-free map |

---

## How to Run

```bash
# Perception
cd Perception
python main.py
# Output: annotated_output.jpg + console cone list

# PPC
cd PPC
python run.py
# Requires Python 3.11, numpy

# SLAM (run each module independently)
cd "Simultaneous Localization and Mapping (SLAM)"
python data_association.py   # visualises association accuracy
python localization.py       # visualises trajectory + uncertainty ellipse
python mapping.py            # visualises incremental cone map

# SLAM metrics only (no animation)
python verify_metrics.py
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `ultralytics`, `opencv-python`
