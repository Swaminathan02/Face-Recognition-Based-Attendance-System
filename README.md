# Face Recognition–Based Attendance System  
### Digital Image Processing (DIP) Project

## Overview
This project implements an **Automated Attendance System** using **Digital Image Processing (DIP)** and **Machine Learning** techniques.  
The system detects faces from images, extracts meaningful features, classifies individuals, and automatically records attendance.

Unlike end-result–only systems, this project emphasizes **intermediate results** at every stage of the DIP pipeline, making the processing flow transparent and interpretable.

##  Objectives
- Apply core **Digital Image Processing concepts** in a real-world application  
- Visualize **intermediate outputs** at each processing stage  
- Perform **face recognition** using classical feature extraction and ML  
- Automatically generate an **attendance log**

## Methodology (DIP Pipeline)

### Pre-processing
- Image resizing
- RGB → Grayscale conversion
- Histogram Equalization
- Gaussian Blur (noise reduction)

**Intermediate Outputs:**
- Original image  
- Grayscale image  
- Contrast-enhanced image  
- Noise-reduced image  

---

### Feature Visualization
- **Canny Edge Detection**
- **Histogram of Oriented Gradients (HOG)**

**Intermediate Outputs:**
- Edge-detected image  
- HOG gradient visualization  

---

### Segmentation
- Face detection using **Haar Cascade Classifier**
- Bounding boxes drawn on detected faces

**Intermediate Outputs:**
- Detected face regions on test image  

---

### Feature Extraction
- HOG feature vectors extracted from detected faces
- Fixed-size (64×64) face normalization

**Intermediate Outputs:**
- Face crops  
- HOG feature images  
- Feature vector dimensions  

---

### ML-Based Modelling
- **Support Vector Machine (SVM)** classifier
- StandardScaler for feature normalization
- Probability-based face recognition

**Intermediate Outputs:**
- Training class distribution  
- Training accuracy  
- Confidence heatmap of predictions  

---

### Attendance Logging
- Recognized faces saved with timestamps
- Attendance stored in CSV format

**Final Outputs:**
- Annotated image with labels  
- `attendance.csv` file  
- Attendance summary visualization  

---
