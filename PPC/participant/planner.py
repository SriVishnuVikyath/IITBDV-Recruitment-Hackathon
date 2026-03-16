'''
PPC Hackathon — Participant Boilerplate
You must implement two functions: plan() and control()
'''

# ─── TYPES (for reference) ────────────────────────────────────────────────────

# Cone: {"x": float, "y": float, "side": "left" | "right", "index": int}
# State: {"x", "y", "yaw", "vx", "vy", "yaw_rate"}  
# CmdFeedback: {"throttle", "steer"}        

# ─── PLANNER ──────────────────────────────────────────────────────────────────
import numpy as np

def plan(cones: list[dict]) -> list[dict]:
    """
    Generate a path from the cone layout.
    Called ONCE before the simulation starts.

    Args:
        cones: List of cone dicts with keys x, y, side ("left"/"right"), index

    Returns:
        path: List of waypoints [{"x": float, "y": float}, ...]
              Ordered from start to finish.
    
    Tip: Try midline interpolation between matched left/right cones.
         You can also compute a curvature-optimised racing line.
    """
    path = []
    # TODO: implement your path planning here
    blue   = np.array([[cone["x"], cone["y"]] for cone in cones if cone["side"] == "left"])
    yellow = np.array([[cone["x"], cone["y"]] for cone in cones if cone["side"] == "right"])

    # implement a planning algorithm to generate a path from the blue and yellow cones
    # Sort cones by index so waypoints are ordered along the track
    left_cones  = sorted([c for c in cones if c["side"] == "left"],  key=lambda c: c["index"])
    right_cones = sorted([c for c in cones if c["side"] == "right"], key=lambda c: c["index"])

    # Midline interpolation: average of matched left/right cones at each index
    for lc, rc in zip(left_cones, right_cones):
        mid_x = (lc["x"] + rc["x"]) / 2.0
        mid_y = (lc["y"] + rc["y"]) / 2.0
        path.append({"x": float(mid_x), "y": float(mid_y)})

    return path
