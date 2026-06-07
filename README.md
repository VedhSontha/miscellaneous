# Miscellaneous Code Vault

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/notebooks-Jupyter-orange?logo=jupyter&logoColor=white" alt="Jupyter Notebooks">
  <img src="https://img.shields.io/badge/domain-ML%20%7C%20Computer%20Vision%20%7C%20NLP-brightgreen" alt="Machine Learning & Vision">
</p>

A consolidated repository of miscellaneous Python scripts, helper modules, and Jupyter Notebooks. These files cover machine learning algorithms, deep neural network training scripts (CNN, RNN, Attention), 3D face/image reconstruction routines, RAG trials, RealSense camera integrations, and weekly coursework assignments.

---

## 📂 Catalog of Files

### 1. 🧠 Deep Learning & NLP Exploration
Notebooks covering fundamental neural architectures, transformer mechanisms, and large language model concepts:
* **`Multi_head_attention.ipynb`** — Custom implementation of Multi-Head Attention mechanisms from scratch.
* **`PRE_LLM_concepts.ipynb`** — Foundational concepts leading up to LLM architectures.
* **`RAG_implementaion(trial).ipynb`** — A trial workflow for Retrieval-Augmented Generation (RAG).
* **`CNN.3.ipynb`** — Convolutional Neural Networks (CNN) modeling and experiments.
* **`RNN.2.ipynb`** — Recurrent Neural Networks (RNN) and sequence modeling.
* **`linear_regression.ipynb`** — Basic linear regression implementation.

### 2. 📷 Computer Vision & 3D Reconstruction
Scripts for 3D visual reconstruction, real-time cameras, and segmentation:
* **`2d_to_3d.ipynb` / `3d_2d.ipynb`** — Transformations between 2D coordinates and 3D space projections.
* **`face_reconstruct.ipynb`** — Facial mesh and 3D face shape reconstruction routines.
* **`realsense_live_pointcloud.py`** — Python script interfacing with an Intel RealSense camera to capture and project live 3D point clouds.
* **`auto_reconstruct_safe.py`** — Automation script for visual reconstructions.
* **`dehazing-gamma-cnn.ipynb`** — Deep learning notebook for image dehazing using custom CNNs and Gamma adjustments.
* **`Sign_lang.ipynb`** — Sign language detection and recognition model training.
* **`CAM.py`** — Class Activation Mapping (Grad-CAM) visualization routines for CNN model explainability.
* **`controls.py` / `gallery.py` / `best.py`** — Interactive helpers, custom UI controls, and visualization utilities.

### 3. 🗓️ Weekly Assignments & Tasks
A collection of notebooks tracking progress across weekly coursework and assignments:
* **`rugvedweek3.ipynb`** through **`rugvedweek7.ipynb`** — Weekly data science, modeling, and deep learning assignments.
* **`Week10.ipynb`** — Final-stage coursework tasks and evaluations.
* **`Hack.ipynb`** — Quick hackathon experiments, prototyping, and draft calculations.

---

## 🛠️ Requirements & Setup

To run these notebooks and scripts, set up a python environment with standard data science and computer vision dependencies:

```bash
# Set up environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install core packages
pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision opencv-python jupyter ipykernel
```

For RealSense cameras:
```bash
pip install pyrealsense2
```

---

## 📄 License
This repository is configured for personal portfolio and reference use. Distributed under the MIT License.

---

## 📊 Repository Insights

| Metric | Details |
| :--- | :--- |
| **Total Consolidated Files** | 29 Files |
| **Jupyter Notebooks (`.ipynb`)** | 22 Notebooks |
| **Python Scripts (`.py`)** | 7 Scripts |
| **Last Maintenance Check** | June 8, 2026 |
| **Status** | Fully Structured & Indexed |
