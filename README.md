# 🎯 Real-Time Face Detection & Age Estimation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A lightweight, real-time computer vision pipeline that performs fast face detection using **Google MediaPipe** and estimates exact age using an **OpenCV DNN (Caffe)** model. Built for high-FPS execution on standard CPUs with interactive privacy controls.

---

## ✨ Key Features

* **Two-Stage Detection Pipeline:** Combines MediaPipe's BlazeFace detector for rapid bounding-box extraction with a Deep Neural Network for age inference.
* **Weighted Age Estimation:** Computes a smooth, exact age estimate using midpoint probability weighting rather than jumping between coarse age brackets.
* **Smart ROI Padding:** Automatically expands face crops by 25% to capture hair and forehead context, significantly boosting age estimation accuracy.
* **Live Privacy Mode:** Toggle real-time Gaussian face blurring on the fly.
* **Auto-Asset Management:** Automatically fetches required neural network weights (`~44 MB`) on first execution.
* **Real-Time Telemetry:** Displays live FPS counter and active face count overlay.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
```

### 2. Set Up the Environment
Using `uv` (recommended) or standard `venv` with **Python 3.11**:
```bash
# Using uv
uv venv --python 3.11
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```
*(Note: On first launch, the script will automatically download `age_deploy.prototxt` and `age_net.caffemodel` into the project directory).*

---

## ⌨️ Interactive Controls

While the webcam window is active, use the following keyboard shortcuts:

| Key | Action |
| :---: | :--- |
| **`b`** | Toggle **Privacy Blur Mode** (Gaussian blur on detected faces) |
| **`s`** | Save a **Snapshot** (`snapshot_<timestamp>.jpg`) to the project folder |
| **`q`** | Quit the application and release the webcam |

---

## 🛠️ Tech Stack & Architecture

1. **Face Detection:** `MediaPipe Face Detection` (BlazeFace short-range model).
2. **Age Estimation:** Caffe-based CNN (`GilLevi/AgeGenderDeepLearning`) running via `cv2.dnn`.
3. **Post-Processing:** Boundary clamping, 25% spatial padding, and dot-product expectation over age midpoints `[1, 5, 10, 18, 28, 40, 50, 75]`.