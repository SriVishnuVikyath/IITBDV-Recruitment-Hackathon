# PPC Hackathon – Solution README

**Author:** Samruddhi  
**Task:** Plan a path through a cone-marked track and control a Formula Student Driverless car to follow it as fast as possible.

---

## How It Works

The solution is split into two parts: a **Planner** and a **Controller**.

---

## Part 1 – Planner (`participant/planner.py`)

### What does it do?
The planner looks at all the cones on the track and decides **where the car should drive** — specifically, down the middle of the track.

### How?
1. The cones come in pairs — **blue cones on the left** and **yellow cones on the right** of the track.
2. Each pair of cones has a matching **index number** so we know which left cone belongs to which right cone.
3. For every matched pair, we calculate the **midpoint** (average of their positions).
4. These midpoints become our **waypoints** — the list of target positions the car should drive through, from start to finish.

```
Left cone  ●               ● Right cone
                  ★  ← midpoint (waypoint)
```

This gives us a clean **centre line** path through the entire track.

---

## Part 2 – Controller (`participant/controller.py`)

### What does it do?
The controller runs every 50ms during the simulation. It reads the car's current position, heading, and speed, and decides how much to **steer**, **throttle**, and **brake**.

### Steering – Stanley Controller
We use a classic algorithm called the **Stanley Controller**:

1. **Find the closest waypoint** on the path to the car's current position.
2. Calculate two errors:
   - **Heading error (ψ):** How much is the car pointing away from the path direction?
   - **Cross-track error (e):** How far sideways is the car from the path?
3. Combine both into a single steering angle:

   ```
   steer = heading_error + arctan(gain × cross_track_error / speed)
   ```

4. Clip the result to the allowed range **[-0.5, 0.5] radians**.

> The faster the car goes, the less the cross-track error matters (the arctan term gets smaller), which naturally gives smoother steering at high speed.

### Throttle – PI Controller
We use a simple **PI (Proportional + Integral) controller** to maintain a target speed of **5 m/s**:

1. Calculate `error = target_speed − current_speed`
2. Accumulate the error over time (integral term) to correct persistent speed offsets
3. If the output is positive → apply **throttle**
4. If the output is negative → apply **brake**

---

## Result

| Metric | Value |
|---|---|
| Lap Completed | ✅ Yes |
| Cone Hits | 0 |
| Penalty | +0.0 s |
| Lap Time | ~23–60 s (varies by run) |

The car successfully navigates the full track without hitting any cones.

---

## Files

| File | Purpose |
|---|---|
| `participant/planner.py` | Generates the centre-line path from cone positions |
| `participant/controller.py` | Steers and throttles the car along the path |
| `run.py` | Simulation runner (do not modify) |
| `tracks/` | Track cone layout data |

---

## How to Run

```bash
cd PPC
python run.py
```

Make sure you are using **Python 3.11** and have **numpy** installed.
