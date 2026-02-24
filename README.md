




https://github.com/user-attachments/assets/3919724e-4943-4c0c-a566-a9443c88e818





Road Edge Detection under Icy and Low-Contrast Conditions
Overview

This project investigates classical computer vision methods for detecting road edges and estimating drivable boundaries in icy, snow-covered, and low-contrast winter conditions.

Winter road environments introduce several perception challenges:

Reduced texture and low contrast between road and surroundings

Snow cover obscuring lane markings

Glare and reflections from ice

Weak or fragmented edge structures

The goal of this project is not to achieve perfect lane detection, but to study the robustness and limitations of classical edge-based methods under visually degraded conditions. The work emphasizes structured experimentation, parameter tuning, and stability analysis.

Problem Motivation

Reliable perception in winter conditions is critical for:

Autonomous ground vehicles

Mobile robots operating outdoors

Driver assistance systems in Nordic environments

Unlike ideal highway datasets, icy roads present weak gradients and inconsistent visual cues. This project explores how far a classical OpenCV-based pipeline can operate before more advanced learning-based methods become necessary.

Key Objectives

Detect road boundaries in low-contrast icy environments

Maintain detection stability under noise and surface variability

Evaluate sensitivity to parameter tuning

Analyze failure cases and robustness limitations

Explore perception pipeline design relevant to autonomous systems

System Pipeline

The implementation is based on modular classical image processing techniques using OpenCV.

1. Preprocessing

Grayscale conversion

Gaussian blurring for noise reduction

Contrast normalization (where applicable)

Purpose: Improve edge continuity while suppressing high-frequency noise caused by snow texture.

2. Edge Detection

Canny edge detection with tuned thresholds

Threshold ranges experimentally adjusted to retain weak gradients

Observation: Lower thresholds preserve faint edges but increase noise sensitivity.

3. Region of Interest (ROI) Masking

Polygonal mask restricting processing to likely road region

Reduces false positives from sky, trees, and roadside objects

This improves stability and computational efficiency.

4. Line Detection

Probabilistic Hough Transform

Extraction of candidate line segments

Detected segments are filtered based on slope and spatial constraints.

5. Lane Estimation & Smoothing

Separation of left and right boundary candidates

Least-squares line fitting

Temporal smoothing across frames to reduce flickering

Temporal smoothing improves visual stability but introduces slight latency.

6. Centerline Estimation

Virtual centerline computed between left and right boundaries

Provides approximate drivable direction estimate

This demonstrates how perception outputs can be transformed into control-relevant information.

Experimental Observations

Detection stability is highly sensitive to Canny thresholds

Strong glare reduces edge continuity

Snow-covered regions produce fragmented edges

ROI masking significantly reduces false detections

Temporal smoothing improves robustness but may hide sudden detection failures

These observations highlight the limitations of purely gradient-based perception methods in winter conditions.

Limitations

Performance strongly depends on camera viewpoint and lighting

Fixed thresholds do not generalize across environments

No adaptive exposure or contrast enhancement

No learning-based semantic understanding

Not optimized for embedded real-time deployment

Future Improvements

Adaptive thresholding techniques

Histogram equalization for low-contrast enhancement

Comparison with Sobel and Laplacian gradients

Integration of semantic segmentation models

Sensor fusion with depth or stereo input

Deployment benchmarking on embedded platforms

Integration into ROS2 perception pipeline

Engineering Reflection

This project demonstrates that classical edge-based methods can provide baseline functionality in degraded winter conditions, but robustness degrades rapidly under glare and heavy snow coverage.

For industrial perception systems, a hybrid approach combining classical preprocessing with learning-based segmentation would likely provide better generalization and reliability.

The experiment reinforces the importance of:

Structured parameter tuning

Explicit failure case analysis

Modular perception pipeline design

Stability evaluation over time
https://www.dropbox.com/scl/fi/6vftd88o2ij74ts5s8bwu/lane_detection_output.avi?rlkey=ahkncmidrlubr4u9a0s15lwlw&st=70ir6mi2&dl=0



