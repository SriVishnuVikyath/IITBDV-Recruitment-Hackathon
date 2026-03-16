# SLAM Project Improvements & Metrics

This README provides a simple and clear explanation of the improvements made to the SLAM system in `data_association.py`, `localization.py`, and `mapping.py`, as well as a breakdown of the specific metrics we are tracking to evaluate performance.

---

## What We Have Done

We upgraded the three core components of the SLAM pipeline from basic baseline approaches to more robust, advanced algorithms modeled after real-world autonomous driving systems.

### 1. Data Association (`data_association.py`)
**Problem with the baseline:** The original approach just matched each measurement to the closest cone (Greedy Nearest-Neighbor). This easily makes mistakes when cones are clustered closely together.
**Our Solution:** **Global Assignment with Spatial Gating**
*   **Spatial Gating (KD-Tree):** Instead of checking the distance to every single cone on the entire track, we use a KD-Tree to quickly gather a small list of candidate cones that are strictly within a 3.0-meter radius of our measurements. This saves massive amounts of computation time.
*   **Global Assignment (Hungarian Algorithm):** Once we have our candidates, we use the Hungarian algorithm (`linear_sum_assignment`). Instead of greedily matching pairs one by one, it evaluates all possible match combinations simultaneously to find the setup that minimizes the total overall distance error. This mathematically guarantees the best overall arrangement and stops "stolen" matches.

**Why this solution?**
Nearest-neighbor is fast but greedy. If two measurements are close to the same map cone, one might "steal" it, forcing the other to match with a completely wrong cone further away, causing cascading errors. Global assignment solves this by evaluating all distance pairs at once, ensuring the lowest total error layout. However, running the Hungarian algorithm against every cone on the entire track is computationally expensive. We combined it with Spatial Gating via KD-Trees to narrow the search space to only nearby cones, giving us the accuracy of global assignment without sacrificing real-time performance.

### 2. Localization (`localization.py`)
**Problem with the baseline:** The baseline used a simple stepwise line projection (Euler integration) to guess where the car moved. Because cars drive in curves, drawing straight lines causes the estimated position to quickly drift further and further out of the curve.
**Our Solution:** **Exact Analytical Circular-Arc Integration & Uncertainty Tracking**
*   **Circular-Arc Kinematics:** We replaced the straight-line math with accurate circular-arc calculations. When the steering wheel is turned, we calculate the exact curve the car is driving along, which heavily reduces drift over time.
*   **Covariance Tracking:** We added uncertainty tracking using Jacobians (motion model derivatives). Since sensors and movements are noisy, we track how "unsure" we are of our position over time. This is represented visually by a blue 3-sigma ellipse around the car, letting us know the bounds of where the car actually could be at any moment.

**Why this solution?**
The baseline Euler integration simply draws straight lines at each step. Because a car physically moves in a continuous curve rather than sharp straight segments when steering, this piecewise linear approximation diverges from the true path very quickly. Exact circular-arc kinematics mathematically define the continuous curves the car actually travels, preventing this structural drift. Furthermore, simple dead reckoning blindly assumes perfect sensors, which never happens in the real world. By formally tracking covariance (uncertainty), we model exactly how errors in speed and steering angle compound over time, giving higher-level mapping algorithms a stable, probabilistically sound foundation of trust.

### 3. Mapping (`mapping.py`)
**Problem with the baseline:** The baseline essentially just dumped every single sensor reading into a list and checked if it was close to an existing one. It had no way of updating cone positions accurately over time or handling sensor "ghosts".
**Our Solution:** **Landmark Lifecycle Management and Linear Kalman Filters**
*   **Kalman Filtering:** When we see a cone, we don't just blindly place it. We mathematically track it. Every subsequent time our sensor sees that same cone, we use a Kalman Filter to merge the new measurement with our previous estimate, averaging out the sensor noise to find its true, highly precise location.
*   **Lifecycle Tracking (Hits & Misses):** We treat every cone as a hypothesis. If we "see" it multiple times (Hits ≥ 2), we formally confirm it and add it to our map. If the cone should be directly in our sensor range but our sensor detects nothing (a Miss), we add to its Miss counter. If a cone gets too many misses and not enough hits, we classify it as an error/ghost and delete it.

**Why this solution?**
A simple distance-threshold list cannot handle real-world sensor issues. Lidars are inherently noisy and occasionally report false positives (ghost cones) or jumpy positions. By using Linear Kalman Filters, we recursively average out multiple noisy detections of the same physical cone, allowing the math to smoothly converge to a much tighter, more accurate, true position estimate. Moreover, Lifecycle Management (tracking hits and misses) ensures we don't permanently pollute our map with random glitches or false positives; only consistently confirmed signatures are kept.

---

## Metrics We Are Tracking

We built a standalone evaluation script (`verify_metrics.py`) to simulate laps without visualization overhead and calculate our performance metrics. They are saved continuously into `slam_metrics_history.csv` to track progress.

Here is what each metric means:

### Data Association Metrics
*   **Association Accuracy Percent (%)**: The percentage of times our algorithm correctly paired a sensor measurement to its exact ground-truth map cone. (e.g. 98% means out of 100 measurements, 98 correctly locked onto the right cone).

### Localization Metrics
*   **Localization Trajectory RMSE (meters)**: Root Mean Square Error. We measure the distance between our estimated trajectory and the car's *true* trajectory at every frame. We square them, average them, and take the square root. A lower number means our estimation hugged the true path very tightly.
*   **Localization Final Drift (meters)**: The absolute straight-line distance between where our system thinks the car ended up vs where it *actually* is at the final frame of the simulation.

### Mapping Metrics
*   **Mapping Estimated vs. Total Visible Cones**: We compare exactly how many cones our map successfully verified against the number of unique ground-truth cones the car physically drove past.
*   **Mapping Size Error (Cones)**: The absolute difference between our estimated cone count and the true visible cone count. An error of 0 means we found every cone perfectly without adding duplicates.
*   **Mapping Positional RMSE (meters)**: For every cone we formally placed into our final map, we find the closest real-world track cone and measure the distance between them. This averages out that error to tell us how sharply and precisely we placed the cones.
