




https://github.com/user-attachments/assets/3919724e-4943-4c0c-a566-a9443c88e818





**Road Edge Detection on Icy Roads (Computer Vision)**


**Overview**

This project implements a classical computer vision pipeline for detecting road edges and drivable boundaries in icy, snow-covered, and low-contrast winter road conditions. The work is motivated by challenges in autonomous driving and robotic navigation in Nordic environments, where visual cues such as lane markings are often partially or fully obscured.

The focus of the project is robustness under visually noisy conditions rather than ideal road scenarios.

**Key Objectives**

Detect road edges and drivable boundaries in icy and low-contrast environments

Handle challenging conditions such as snow cover, glare, and poor illumination

Evaluate the stability of detections across varying surface textures

Explore perception techniques relevant to autonomous systems and robotics

**Methodology**

The pipeline is based on classical image processing techniques using OpenCV:

Preprocessing

  Grayscale conversion

  Gaussian blurring for noise reduction

Edge Detection

  Canny edge detector with relaxed thresholds to retain weak edges
  
Region of Interest (ROI) Masking

  Polygonal ROI focused on the road area to suppress irrelevant features

Line Detection

  Probabilistic Hough Transform to extract candidate edge segments

Lane Estimation & Smoothing

  Separation of left and right road boundaries based on slope and position
  
  Line fitting using least-squares methods
  
  Temporal smoothing across frames for stability

Centerline Estimation

  Computation of a virtual centerline between detected road edges

**Limitations**

Performance depends on lighting conditions and camera viewpoint

Purely classical approach (no learning-based segmentation)

Does not explicitly model road geometry or elevation

**Future Improvements**

Integration of learning-based segmentation models

Use of depth or stereo data for improved robustness

Adaptation for real-time deployment on robotic platforms

Evaluation on larger winter-road datasets


https://www.dropbox.com/scl/fi/6vftd88o2ij74ts5s8bwu/lane_detection_output.avi?rlkey=ahkncmidrlubr4u9a0s15lwlw&st=70ir6mi2&dl=0



