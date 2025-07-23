<div align="center">
  
# **TARS: Training Adaptive Reasoners for Safety**

[**📄 Paper**](https://arxiv.org/abs/2507.00971) | [**🤗 TARS Model**](https://huggingface.co/CMU-AIRe/TARS-1.5B) | [**🤗 Lightweight SFT Model**](https://huggingface.co/danielkty22/TARS-SFT-1.5B) | [**📝 Blog Post**](https://training-adaptive-reasoners-safety.github.io) | 

*Training repository for "Reasoning as an Adaptive Defense for Safety"*

</div>

---

## 🎯 Overview

This repository contains the training code and datasets for **TARS (Training Adaptive Reasoners for Safety)**, an online RL training approach that uses reasoning as an adaptive defense for LLM safety. The training code uses a modified version of [verl](https://github.com/volcengine/verl), which is adapted from a previous version of [rLLM](https://github.com/agentica-project/rllm).

---

## Table of Contents

- [Getting Started](#getting-started)
- [Training](#training)

---

## Getting Started

This repository includes:
- **Datasets:** train_lambda_0.1/0.3/0.5/0.7/0.9.parquet
- **Training Script:** Online RL safety training for reasoning using GRPO

First, install the Python packages.
```bash
conda env create --file environment.yml
```
Second, install the modified version of verl and additional packages.
```bash
pip install -e ./verl
pip install git+https://github.com/dsbowen/strong_reject.git@main
pip install flash-attn
```
  
### Training

Train through online RL starting from the base lightweight [SFT model](https://huggingface.co/danielkty22/TARS-SFT-1.5B) used for TARS.
```bash
bash scripts/train/run_train.sh 
```

### Citation

If you find this work useful, please cite our paper:

```bibtex
@article{kim2025reasoning,
  title={Reasoning as an Adaptive Defense for Safety},
  author={Kim, Taeyoun and Tajwar, Fahim and Raghunathan, Aditi and Kumar, Aviral},
  journal={arXiv preprint arXiv:2507.00971},
  year={2025}
}
