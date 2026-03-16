
'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Path: list of waypoints [{"x": float, "y": float}, ...]
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"} 
# CmdFeedback: {"throttle", "steer"}         

# ─── CONTROLLER ───────────────────────────────────────────────────────────────
import numpy as np


integral = 0.0

def steering(path: list[dict], state: dict):

    length_of_car = 2.6
    # Calculate steering angle based on path and vehicle state
    if not path:
        steer = 0.0
        return np.clip(steer, -0.5, 0.5)

    car_x, car_y, car_yaw = state["x"], state["y"], state["yaw"]

    # Find closest waypoint
    closest_idx = min(range(len(path)), key=lambda i: (path[i]["x"] - car_x)**2 + (path[i]["y"] - car_y)**2)
    if closest_idx >= len(path) - 1:
        steer = 0.0
        return np.clip(steer, -0.5, 0.5)

    # Stanley controller: cross-track error + heading error
    p1 = path[closest_idx]
    p2 = path[closest_idx + 1]
    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]
    ex = car_x - p1["x"]
    ey = car_y - p1["y"]
    e = (ex * dy - ey * dx) / np.hypot(dx, dy)   # signed cross-track error
    path_yaw = np.arctan2(dy, dx)
    psi = np.arctan2(np.sin(path_yaw - car_yaw), np.cos(path_yaw - car_yaw))  # heading error
    k = 1.0
    steer = psi + np.arctan(k * e / (state["vx"] + 0.1))

    # 0.5 is the max steering angle in radians (about 28.6 degrees)
    return np.clip(steer, -0.5, 0.5)


def throttle_algorithm(target_speed, current_speed, dt):
    global integral

    error = target_speed - current_speed
    integral = float(np.clip(integral + error * dt, -5.0, 5.0))

    # PI control output
    output = 1.5 * error + 0.1 * integral

    # generate the output for throttle command
    throttle = output if output > 0 else 0.0
    brake = (-output) if output < 0 else 0.0
    # clip throttle and brake to [0, 1]
    return np.clip(throttle, 0.0, 1.0), np.clip(brake, 0.0, 1.0)

def control(
    path: list[dict],
    state: dict,
    cmd_feedback: dict,
    step: int,
) -> tuple[float, float, float]:
    """
    Generate throttle, steer, brake for the current timestep.
    Called every 50ms during simulation.

    Args:
        path:         Your planned path (waypoints)
        state:        Noisy vehicle state observation
                        x, y        : position (m)
                        yaw         : heading (rad)
                        vx, vy      : velocity in body frame (m/s)
                        yaw_rate    : (rad/s)
        cmd_feedback: Last applied command with noise
                        throttle, steer, brake
        step:         Current simulation timestep index

    Returns:
        throttle  : float in [0.0, 1.0]   — 0=none, 1=full
        steer     : float in [-0.5, 0.5]  — rad, neg=left
        brake     : float in [0.0, 1.0]   — 0=none, 1=full
    
    Note: throttle and brake cannot both be > 0 simultaneously.
    """
    global integral
    throttle = 0.0
    steer    = 0.0
    brake    = 0.0

    # TODO: implement your controller here
    if step == 0:
        integral = 0.0

    steer = steering(path, state)
    target_speed = 5.0  # m/s, adjust as needed
    throttle, brake = throttle_algorithm(target_speed, state["vx"], 0.05)

    return float(throttle), float(steer), float(brake)
