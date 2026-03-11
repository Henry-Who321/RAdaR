# RAdaR: RL-Native Adaptive Reasoning with Capability-Aware Data Curation for VLMs
## 📝 Abstract
Vision-Language Models (VLMs) excel in complex reasoning tasks but are often constrained by the issue of overthinking and underthinking, limiting their applicability in real-world scenarios. Existing adaptive reasoning approaches face critical challenges, including data scarcity, catastrophic forgetting, and sensitivity to prompts. To address these limitations, we propose AdaR, an RL-native framework for adaptive reasoning with a two-stage training process: In Stage 1, the model is trained to produce outputs that follow the prescribed formats of the two reasoning modes, Thinking and Instant, using a curriculum-style formatting strategy. In Stage 2, we model adaptive reasoning as a reasoning mode selection problem and train the model to dynamically choose appropriate strategies for each input by predicting the corresponding mode-control start token. In addition, we introduce a capability-aware data construction pipeline that provides highly discriminative supervision for adaptive reasoning in VLMs. Experimental results demonstrate that RAdaR achieves a significant reduction in reasoning overhead while improving accuracy by up to 7.6% over the base model and 14.7% over SOTA methods, respectively. To ensure reproducibility and promote further research, we will release the code, datasets, and model weights.

![Three-stage pipeline](figures/framework.png)

## 🚀 Getting Started

## 🔧 0. Before You Start
### System Requirements

| Component | Version |
|-----------|---------|
| OS | Linux (Ubuntu recommended) |
| Python | 3.12 |
| CUDA Toolkit | 12.x (nvcc required for compiling flash-attn, etc.) |
| CUDA Driver | ≥ 550.x |
| GPU | NVIDIA A800/A100/H100 (80GB recommended) |
| PyTorch | 2.8.0+cu128 |

### Installation
1. Clone the repository:
```bash
git clone git@github.com:henryhu12321-beef/RAdaR.git
cd RAdaR
```

2. Create a conda environment and install dependencies:
```bash
# 1. Create conda environment
conda create -n RAdaR python=3.12 -y
conda activate RAdaR

# 2. Install PyTorch with CUDA 12.8 support (MUST install first)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Set CUDA_HOME and install packages that require CUDA compilation
export CUDA_HOME=/usr/local/cuda  # adjust to your CUDA 12.x path
pip install -r requirements-special.txt --no-build-isolation

# 5. (Optional) Install dev tools
pip install -r requirements-dev.txt

# 6. (Optional) If you need latex2sympy2 for evaluation:
pip install -e evaluation/latex2sympy
```

## 🚀 1. Stage 1.1
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage1_1_train.py --config examples/vlm/radar_gspo_stage1_1_bs32_rollout32.yaml
```

## 🚀 2. Stage 1.2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage1_2_train.py --config examples/vlm/radar_gspo_stage1_2_bs32_rollout32.yaml
```

## 🚀 3. Stage 2
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m areal.launcher.local examples/vlm/radar_stage2_train.py --config examples/vlm/radar_gspo_stage2_bs32_rollout32.yaml
```

## 📖 Resources

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)
- [CLI Configurations](https://inclusionai.github.io/AReaL/cli_reference.html)
- [Asynchronous RL Explained](https://inclusionai.github.io/AReaL/algorithms/async.html)
- [Fine-Tuning Large MoE](https://inclusionai.github.io/AReaL/tutorial/megatron.html)
- [Agentic RL](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html)
- [Debugging Best Practices](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)

## 📄 Citation

```bibtex
```
