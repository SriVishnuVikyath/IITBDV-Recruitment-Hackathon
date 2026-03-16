import matplotlib
matplotlib.use('Agg')
import data_association
import localization
import mapping
import numpy as np
import csv
import os
from datetime import datetime

# Let's write a clean script to calculate all metrics without depending on matplotlib FuncAnimation state

def calculate_all_metrics():
    # 1. Data Association
    sol_da = data_association.Solution()
    da_correct = 0
    da_total = 0
    for frame in range(data_association.N_FRAMES):
        steer = data_association.pure_pursuit(sol_da.pos, sol_da.heading, data_association.CENTERLINE)
        meas, true_indices = data_association.get_measurements(sol_da.pos, sol_da.heading)
        assocs = sol_da.data_association(meas, data_association.MAP_CONES)
        
        if len(true_indices) > 0:
            correct = np.sum(assocs == true_indices)
            da_correct += correct
            da_total += len(true_indices)
            
        sol_da.pos, sol_da.heading = data_association.step_kinematic(sol_da.pos, sol_da.heading, data_association.SPEED, steer)
    
    final_assoc_acc = 100.0 * da_correct / max(1, da_total)

    # 2. Localization
    sol_loc = localization.Solution()
    true_pos = localization.CAR_START_POS.copy()
    true_heading = localization.CAR_START_HEADING
    squared_errors = []
    
    for frame in range(localization.N_FRAMES):
        steer_true = localization.pure_pursuit(true_pos, true_heading, localization.CENTERLINE)
        true_pos, true_heading = localization.step_kinematic(true_pos, true_heading, localization.SPEED, steer_true)
        
        noisy_speed = localization.SPEED + np.random.normal(0, 0.05)
        noisy_steer = steer_true + np.random.normal(0, np.deg2rad(1.0))
        sol_loc.localization(noisy_speed, noisy_steer)
        
        err = np.linalg.norm(sol_loc.pos - true_pos)
        squared_errors.append(err**2)
        
    final_loc_rmse = np.sqrt(np.mean(squared_errors))
    final_drift = np.linalg.norm(sol_loc.pos - true_pos)

    # 3. Mapping
    sol_map = mapping.Solution()
    seen_true_cones = set()
    
    for frame in range(mapping.N_FRAMES):
        steer = mapping.pure_pursuit(sol_map.pos, sol_map.heading, mapping.CENTERLINE)
        meas, true_indices = mapping.get_measurements(sol_map.pos, sol_map.heading)
        for idx in true_indices:
            seen_true_cones.add(int(idx))
            
        sol_map.pos, sol_map.heading = mapping.step_kinematic(sol_map.pos, sol_map.heading, mapping.SPEED, steer)
        sol_map.mapping(meas)
        
    true_count = len(seen_true_cones)
    pred_count = len(sol_map.learned_map)
    total_sq_err = 0.0
    if pred_count > 0:
        for est_pt in sol_map.learned_map:
            dists = np.linalg.norm(mapping.MAP_CONES - est_pt, axis=1)
            total_sq_err += np.min(dists)**2
        map_rmse = np.sqrt(total_sq_err / pred_count)
    else:
        map_rmse = float('inf')

    map_size_error = abs(pred_count - true_count)

    # Compile Results
    results = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Association_Accuracy_Percent': round(final_assoc_acc, 2),
        'Localization_Trajectory_RMSE_m': round(final_loc_rmse, 3),
        'Localization_Final_Drift_m': round(final_drift, 3),
        'Mapping_Total_Visible_Cones': true_count,
        'Mapping_Estimated_Cones': pred_count,
        'Mapping_Size_Error_Cones': map_size_error,
        'Mapping_Positional_RMSE_m': round(map_rmse, 3)
    }
    
    return results

if __name__ == "__main__":
    print("Calculating final metric loop...")
    metrics = calculate_all_metrics()
    
    csv_file = "slam_metrics_history.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
        
    print(f"\n[Success] Metrics successfully appended to {csv_file}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
