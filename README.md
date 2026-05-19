# SOL-Nav: Structured Observation Language for Efficient and Generalizable Vision-Language Navigation


[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.27577-b31b1b)](https://arxiv.org/abs/2603.27577)
[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)](https://www.python.org/)

Official implementation of **SOL-Nav**, a novel Vision-Language Navigation (VLN) framework that converts egocentric RGB-D visual observations into compact structured language descriptions, enabling efficient and generalizable navigation via pure pre-trained language models (PLMs). 
<!-- This work is a submission to ECCV 2026 (Paper ID: 11964). -->

## 📺 Video Demo

A detailed video demonstration of SOL-Nav's performance on simulated benchmarks and real-world robotic deployments is available in the repository (same directory as this README.md):

<!-- [ECCV_2026_SOL-Nav_Video.mp4](ECCV_2026_SOL-Nav_Video.mp4) -->

## 📜 Paper Abstract

Vision-Language Navigation (VLN) requires embodied agents to navigate complex environments by following natural language instructions, demanding tight fusion of visual and language modalities. Existing methods convert raw images into visual tokens/implicit features, relying on large-scale visual pre-training and suffering from poor generalization under environmental variations (e.g., lighting, texture).

To address these issues, we propose **SOL-Nav**, which translates egocentric RGB-D observations into **structured textual descriptions** (semantic, color, depth information in an N×N grid) and concatenates this with language instructions as **pure language input** to a PLM. Experimental results on standard VLN benchmarks (R2R-CE, RxR-CE) and real-world robotic deployments show that SOL-Nav:

- Significantly reduces model size and training data dependency

- Fully leverages PLMs' reasoning and representation capabilities

- Achieves strong generalization to unseen environments

- Matches or outperforms SOTA methods with a tiny model (0.6B parameters)

## 🚀 Core Advantages

1. **Reduced Training Cost**: Eliminates scratch training of visual encoders, only requires small-scale navigation data for PLM fine-tuning (LoRA).

2. **Strong Generalization**: Structured text avoids environmental noise (lighting/texture) and enables robust adaptation to unseen scenes.

3. **Simplified Pipeline**: No complex multimodal encoders/fusion modules—pure language model for navigation decision-making, lower computational cost.

4. **Real-World Deployable**: Small model size (0.6B params) and low inference latency (0.8s on edge devices) for physical robot integration.

## 🏗️ Engineering Framework

The SOL-Nav codebase is modularly designed for easy extension, reproduction, and real-world deployment. The framework is divided into **6 core modules**, with clear data flow and minimal dependencies between modules.

### Overall Architecture

```bash
SOL-Nav/
├── configs/                # Configuration files (model, data, training, deployment)
├── data/                   # Data processing pipeline
│   ├── dataset/            # VLN dataset loaders (R2R-CE, RxR-CE)
│   ├── preprocess/         # RGB-D to structured text conversion
│   └── augmentation/       # Instruction and observation augmentation
├── models/                 # Model implementation
│   ├── backbone/           # PLM backbone (Qwen3-Embedding-0.6B)
│   ├── heads/              # Multi-step action prediction heads
│   └── lora/               # LoRA fine-tuning implementation
├── navigation/             # Navigation execution
│   ├── agent/              # Embodied agent controller
│   ├── action/             # Action space definition and execution
│   └── metrics/            # VLN evaluation metrics (NE, SR, OS, SPL)
├── deployment/             # Real-world robot deployment
│   ├── sensor/             # RGB-D sensor driver (Intel RealSense D435i)
│   ├── robot/              # Robot control (Unitree Go2)
│   └── edge/               # Edge computing optimization (NVIDIA Jetson Orin)
├── utils/                  # Utility functions (logging, visualization, tools)
├── train.py                # Training script (fine-tune PLM for action prediction)
├── eval.py                 # Evaluation script (simulated benchmarks)
├── deploy.py               # Real-world deployment script
└── demo.py                 # Quick demo script (sim/real)
```

### Module Details

#### 1. configs/

Unified configuration management with YAML files for all hyperparameters:

- `model.yaml`: PLM backbone, LoRA rank, action head settings

- `data.yaml`: Dataset paths, grid resolution (2×2/4×4/6×6), preprocessing params

- `train.yaml`: Batch size, epochs, loss function, optimizer (AdamW)

- `deploy.yaml`: Sensor resolution, robot speed, inference latency threshold

#### 2. data/preprocess/ (Core Module)

Converts raw RGB-D images to **structured textual observations**—the core innovation of SOL-Nav:

- **Grid Division**: Split RGB-D/semantic segmentation maps into N×N grids (6×6 for current, 4×4 short-term, 2×2 long-term)

- **Feature Extraction**:
        

    - Depth: Average depth value of each grid cell (string format)

    - Semantic: Dominant semantic category (via pre-trained SegFormer/Grounded SAM)

    - Color: HSV-to-standard color name mapping (predefined lookup table)

- **Text Structuring**: Format each grid cell as `[i,j]: depth=d, semantic=s, color=c` and concatenate with time-step info

#### 3. models/

Lightweight PLM-based model for action block prediction:

- **Backbone**: Qwen3-Embedding-0.6B (extended context window for long structured observations)

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning on VLN data (no full PLM retraining)

- **Multi-Step Classification Heads**: 4 linear heads (for 4-action block prediction) with balanced class weights (address action imbalance)

- **Loss Function**: Weighted cross-entropy loss (average over 4 action steps)

#### 4. navigation/

VLN agent and evaluation pipeline for simulated environments (Habitat):

- **Agent**: Embodied agent with historical observation memory (short/long-term)

- **Action Space**: Discrete actions (stop=0, turn left 15°=1, turn right 15°=2, move forward 25cm=3)

- **Metrics**: Standard VLN evaluation (NE, SR, OS, SPL) with automatic result logging/visualization

#### 5. deployment/

Real-world robotic deployment pipeline (Unitree Go2 + NVIDIA Jetson Orin + Intel RealSense D435i):

- **Sensor Driver**: RGB-D image acquisition (640×480) and real-time preprocessing

- **Robot Control**: Low-level robot action execution (compatible with ROS/ROS2)

- **Edge Optimization**: TensorRT quantization for PLM inference (0.8s latency on Jetson Orin)

- **Semantic Segmentation**: Fine-tuned SegFormer on real-world + VLN dataset data (1000 manually annotated real images)

#### 6. utils/

Helper functions for the entire pipeline:

- **Logging**: Structured logging (training/eval/deployment) with WandB integration

- **Visualization**: Grid observation visualization, trajectory plotting, metric curves

- **Tools**: Text prompt construction, model checkpoint saving/loading, data conversion

## 📊 Key Experimental Results

SOL-Nav achieves state-of-the-art or comparable performance on **R2R-CE** and **RxR-CE** (val-unseen splits) with a **0.6B parameter model** (10× smaller than SOTA multimodal models), without additional training data or waypoint predictors.

### R2R-CE Val-Unseen (No Extra Data / No Waypoint Predictor)

|Metric|NE (↓)|OS (↑)|SR (↑)|SPL (↑)|
|---|---|---|---|---|
|SOL-Nav|5.11|72.9|53.6|49.2|
|NaVILA|5.37|57.6|49.7|45.5|
|UniNaVid|5.58|53.3|47.0|42.7|
### RxR-CE Val-Unseen (No Extra Data)

|Metric|NE (↓)|OS (↑)|SR (↑)|SPL (↑)|
|---|---|---|---|---|
|SOL-Nav|6.87|60.5|48.6|42.3|
|UniNaVid|6.24|55.5|48.7|40.9|
### Ablation Study (R2R-CE Val-Unseen)

All core components are critical for performance—**6×6 grid resolution, historical observations, and depth information** are indispensable:

|Ablation|NE (↓)|OS (↑)|SR (↑)|SPL (↑)|
|---|---|---|---|---|
|Full Model (SOL-Nav)|5.11|72.9|53.6|49.2|
|Lower Res (4×4)|6.84|43.4|34.5|29.8|
|No Historical Obs|7.81|39.4|26.5|21.9|
|No Depth Info|7.98|31.2|21.6|17.8|
## 🛠️ Installation & Dependencies

### Prerequisites

- Python 3.8+

- PyTorch 2.0+

- CUDA 11.7+ (for training) / JetPack 5.1+ (for Jetson Orin deployment)

- Habitat-Sim 0.2.4 (for VLN simulation)

- Hugging Face Transformers (for PLM backbone)

- LoRA: peft 0.6.0

- RGB-D Sensor: pyrealsense2 2.54.1

- Robot Control: unitree_api (for Unitree Go2)

### Installation

```bash
# Clone the repository
git clone https://github.com/DaojiePENG/sol-nav.git
cd sol-nav

# Create a conda environment
conda create -n solnav python=3.8
conda activate solnav

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install core dependencies
pip install -r requirements.txt

# Install Habitat-Sim (for simulation)
conda install -c conda-forge habitat-sim=0.2.4 headless=True

# Install edge deployment dependencies (Jetson Orin)
# pip install torch2trt tensorrt
```

### Dataset Preparation

Note: The official **R2R-CE** and **RxR-CE** datasets cannot be directly downloaded and used. You need to run them in the Habitat simulator first, extract RGB-D images from the simulation process, save them, and then perform structured text conversion on the extracted RGB-D images. Organize the processed dataset in `data/dataset/` following the structure in `configs/data.yaml`:

Download the official **R2R-CE** and **RxR-CE** datasets and organize them in `data/dataset/` following the structure in `configs/data.yaml`:

```bash
data/dataset/
├── R2R-CE/
│   ├── train/
│   ├── val/
│   └── val-unseen/
└── RxR-CE/
    ├── train/
    ├── val/
    └── val-unseen/
```

The dataset will be automatically preprocessed into structured textual observations on first run (configurable grid resolution).

## 🚦 Quick Start

### 1. Train SOL-Nav

Fine-tune the Qwen3-Embedding-0.6B model with LoRA on R2R-CE/RxR-CE (4 RTX 4090 GPUs, ~2 weeks for 10 epochs):

```bash
python train.py --config configs/train.yaml --dataset R2R-CE --gpu 0,1,2,3
```

### 2. Evaluate on Simulated Benchmarks

Evaluate SOL-Nav on R2R-CE/RxR-CE val-unseen split and generate metric reports:

```bash
python eval.py --config configs/train.yaml --dataset R2R-CE --checkpoint runs/best_model.pth --gpu 0
```

### 3. Real-World Deployment

Run SOL-Nav on Unitree Go2 robot (NVIDIA Jetson Orin + Intel RealSense D435i):

```bash
python deploy.py --config configs/deploy.yaml --checkpoint runs/best_model.pth
```

### 4. Run Demo

Quick demo on simulated environment with pre-trained checkpoint:

```bash
python demo.py --config configs/demo.yaml --checkpoint runs/best_model.pth --instruction "Go to the dining table and stop"
```

## 📈 Visualization

- **Structured Observation Visualization**: The `utils/visualization.py` script plots the N×N grid observations with depth/semantic/color info.

- **Trajectory Plotting**: Evaluation results include trajectory plots of the agent in Habitat simulator (top-down view).

- **Metric Curves**: Training/evaluation metrics (loss, NE, SR, SPL) are logged to WandB and saved as PDF in `runs/`.

## 🎯 Future Work

The SOL-Nav framework is designed for easy extension—key future directions supported by the codebase:

1. Enrich Structured Observations: Add fine-grained features (shape, texture) to grid cells.

2. Adaptive Grid Resolution: Dynamic N×N grid based on scene complexity.

3. Multi-Sensor Fusion: Integrate IMU/LiDAR data into structured textual observations.

4. Complex Scenarios: Extend to outdoor navigation, dynamic environments (moving objects).

5. Model Compression: Further optimize PLM for edge devices (quantization, pruning).

6. Embodied Manipulation: Extend SOL-Nav to vision-language manipulation tasks.

## 📝 Citation

If you find SOL-Nav useful for your research, please cite our paper:

```bibtex
@article{peng2026structured,
  title={Structured Observation Language for Efficient and Generalizable Vision-Language Navigation},
  author={Peng, Daojie and Ma, Fulong and Ma, Jun},
  journal={arXiv preprint arXiv:2603.27577},
  year={2026}
}
```

## 📧 Contact

For questions, issues, or collaboration, please open an issue in the repository or contact the authors at Daojie.PENG@qq.com.

## 📄 License

This project is licensed under the **MIT License**—see the [LICENSE](LICENSE) file for details.