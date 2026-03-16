# Perception Assignment: Distance Estimation

## Overview
This folder contains the solution for the "Distance Estimation using YOLO and Monocular Vision" task. The goal is to detect traffic cones using a pre-trained YOLO model and estimate their depth (Z-distance) from the camera using Pin-hole Camera Geometry.

## How the Code Solves the Problem
The implementation in `main.py` successfully fulfills the assignment requirements:

1. **Object Detection**: It loads the provided YOLO model (`YOLOv11s-Carmaker.pt`) using the `ultralytics` library and performs inference on the static input image (`testing/image.webp`).
2. **Depth Calculation**: It calculates the distance to each detected cone using the Similarity of Triangles (Lens Formula): $d = \frac{H \cdot f}{h}$.
   - **Parameters Used:**
     - `H` (Real-world height of the cone) = 30 cm.
     - `f` (Camera Focal Length) = 1000 mm (used as 100 cm in the code).
   - The pixel height `h` of each cone is extracted dynamically from the YOLO bounding box coordinates (`y2 - y1`).
   - The `dist` function intelligently computes `(H * f / h) / 100` returning the physical distance converted into meters.
3. **Visualization & Output**: 
   - Draws bounding boxes around detected cones.
   - Annotates each box with the cone's class name and its computed distance in meters.
   - Saves the annotated result as `annotated_output.jpg` and prints a list of detected cones with their depths directly to the console.

## Assumptions
- The cones are standing upright on a flat ground.
- Bounding box height accurately bounds the full vertical extent of the cone.
