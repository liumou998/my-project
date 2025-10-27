# A Progressive Fusion Architecture for Validating Distance Estimation in Aftermarket ADAS Installations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation for the paper **"A Progressive Fusion Architecture for Validating Distance Estimation in Aftermarket ADAS Installations"**. 

This project addresses the critical challenge of validating the performance of retrofitted Advanced Driver-Assistance Systems (ADAS), particularly for their distance estimation capabilities. We introduce a progressive fusion architecture that intelligently integrates outputs from multiple existing ADAS sources to generate a reliable distance estimate, which serves as a validation benchmark (a "ground truth proxy") without requiring expensive, professional testing equipment.

## Table of Contents
- [1. Model Architecture & Methodology](#1-model-architecture--methodology)
- [2. Repository Structure](#2-repository-structure)
- [3. Dataset Information](#3-dataset-information)
- [4. Setup and Installation](#4-setup-and-installation)
- [5. Usage Instructions](#5-usage-instructions)
- [6. License](#6-license)

## 1. Model Architecture & Methodology

Our approach consists of a data preprocessing pipeline and an end-to-end multi-source, multi-stage fusion network.

### Data Preprocessing
Raw CAN bus messages from ADAS devices are transformed into standardized, time-consistent vectors through a three-stage process:
1.  **Message Parsing**: Decodes raw data packets to extract timestamps and obstacle information.
2.  **Frame Synchronization**: Aligns data streams from different sources based on timestamps.
3.  **Data Standardization**: Normalizes and formats the data into fixed-dimension vectors suitable for the neural network.

![Data Preprocessing Pipeline](images/data.png)
*Fig. 1: The data preprocessing workflow.*

### Multi-source Multi-stage Fusion Network
The core of our method is a deep learning network designed to fuse features from two ADAS sources and predict a reliable distance vector.

-   **Feature Extraction Module**: Two parallel encoders (with unshared weights) map the heterogeneous ADAS input vectors into a unified, compact representation space.
-   **Multi-stage Fusion Module**: A three-stage progressive fusion process integrates the feature representations. Each stage refines the features through modality-specific transformations, cross-modal information exchange, and residual connections, ensuring a comprehensive fusion.

![Model Architecture](images/model.png)
*Fig. 2: The architecture of the proposed fusion network.*

## 2. Repository Structure

```
my-project/
│
├── images/                     # Contains images for the README
│   ├── data.png
│   └── model.png
│
├── data/                       # Directory for input data
│   └── sample/
│       ├── adas1_sample_data.txt
│       └── adas2_sample_data.txt
│
├── e2e_self_supervised_results/ # Default output directory for fusion results
│
├── some_py/                    # Scripts for performance evaluation and analysis
│   └── evaluate_performance.py # Example script to calculate metrics
│
├── train_e2e.py                # Main script for training and fusion
├── config.py                   # Configuration file for data paths and parameters
├── requirements.txt            # List of dependencies for pip
└── README.md                   # This file
```

## 3. Dataset Information

The dataset used in this study was specifically collected to evaluate ADAS performance in a controlled environment.

-   **Data Source**: The dataset comprises **24 high-quality street scene videos** (60-90 seconds each), featuring common obstacles such as pedestrians, vehicles, and non-motorized vehicles.
-   **Ground Truth**: Ground truth for obstacle distances was established through direct physical measurements within the controlled experimental setting.
-   **Data Format**: The raw data consists of CAN messages captured from various commercial ADAS devices (e.g., MAXEYE, JMS3, MINIEYE, MOTOVIS). This repository works with the **post-processed numerical data**, which has been parsed and synchronized into `.txt` or `.npy` files.
-   **Availability**:
    -   **Sample Data**: To ensure reproducibility, sample processed data files are provided in the `data/sample/` directory.
    -   **Full Dataset**: The full raw dataset and the hardware-specific parsing scripts are not publicly distributed due to their proprietary nature. However, they can be made available for academic purposes upon reasonable request to the corresponding author.

## 4. Setup and Installation

This project is built using Python 3.8 and PyTorch. We recommend using a `conda` virtual environment.

**1. Clone the repository:**
```bash
git clone https://github.com/liumou998/my-project.git
cd my-project
```

**2. Create and activate a conda environment:**
```bash
conda create -n adas-fusion python=3.8
conda activate adas-fusion
```

**3. Install dependencies:**
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. (Optional) GPU Support:**
For GPU acceleration, please install a version of PyTorch that matches your system's CUDA toolkit. You can find the appropriate command on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## 5. Usage Instructions

### Step 1: End-to-End Fusion

This step trains the end-to-end model and generates the fused benchmark output in a single run.

1.  **Configure Data Paths**: Open the `config.py` file. In the `end_to_end` section, set the `data_path` for `adas1` and `adas2` to point to the desired input files (e.g., the provided sample data).

2.  **Run the Fusion Script**: Execute the following command from the project's root directory.
    ```bash
    python train_e2e.py --mode fusion
    ```

3.  **Check the Output**: After the script finishes, the fused result (`fused_feature.npy`) and other relevant files will be saved in the `e2e_self_supervised_results/` directory.

### Step 2: Performance Evaluation

This step uses the fused benchmark generated in Step 1 to evaluate the performance of an under-test ADAS device.

1.  **Prerequisites**: Ensure you have successfully run Step 1.

2.  **Run Evaluation Script**: The scripts in the `some_py/` directory are used for analysis. For example, to compute performance metrics (MSE, MAE, etc.) against the fused benchmark, you can run:
    ```bash
    python some_py/evaluate_performance.py
    ```
    *(Note: You may need to adjust the file paths inside the evaluation script to load the correct files.)*

3.  **Expected Output**: The script will print a quantitative report comparing the raw ADAS signals to the fused benchmark, allowing for an objective performance assessment.

## 6. License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
