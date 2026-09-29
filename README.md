<div align="center">
  
<h1>Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning</h1>

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.26+-yellow.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📑 Overview

This repository provides the implementation of **"Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning"**. DSP is a diffusion-based framework for high-level subgoal generation in offline goal-conditioned reinforcement learning. Instead of explicitly selecting subgoals through high-level value estimates, DSP uses classifier-free guidance over learned conditional and unconditional subgoal flows to generate data-supported, goal-directed subgoals in long-horizon tasks.

## 🔧 Requirements

Ensure your system meets the following specifications:
- Python 3.9+
- JAX 0.4.26+
- CUDA support (recommended)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🏗️ Repository Structure

```bash
├── agents/                 
│   ├── dsp.py          	# Core DSP implementation
│   ├── hiql_w_o.py         # HIQL without subgoal representation implementation  
│   └── __init__.py
├── utils/
│   ├── datasets.py         # Dataset processing, metrics
│   ├── encoders.py         # Encoder implementations
│   ├── env_utils.py		# Environment utils
│   ├── evaluation.py		# Evaluation function
│   ├── flax_utils.py		# Flax utils
│   ├── log_utils.py		# Log utils
│   ├── networks.py			# Network implementations
│   └── __init__.py     
├── d4rl/               # DSP for D4RL antmaze environment
│   ├── antmaze_aux/        
│   ├── d4rl_ext/
│   ├── jaxrl_m/        # Utils
│   ├── src             # Implementations
│   └── main.py
├── main.py					# main
├── requirements.txt        # Dependencies
├── pyproject.toml
└── README.md
```

## ⚙️  Training and Evaluation

### OGBench

Please note that we **do not provide the entire baseline methods** here.

Run the following command to reproduce our method (OGBench):

```bash
python main.py --agent=agents/dsp.py --env_name=antmaze-medium-navigate-v0 --agent.expectile=0.7 --agent.low_alpha=3.0 --agent.cfg=3.0 --agent.actor_p_curgoal=0.0 --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0
```

If you want reproduce our method in visual datasets, please run following command:

```bash
python main.py --agent=agents/dsp.py --env_name=visual-antmaze-large-navigate-v0 --agent.batch_size=256 --agent.expectile=0.7 --agent.low_alpha=3.0 --agent.cfg=3.0 --agent.actor_p_curgoal=0.0 --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0 --agent.low_actor_rep_grad=True --agent.p_aug=0
```

### D4RL-Antmaze

Run the following command to reproduce our method (D4RL):

```bash
python main.py --run_group EXP --seed 0 --env_name antmaze-large-diverse-v2 --algo_name dsp --use_waypoints 1 --way_steps 25 --pretrain_steps 1000000 --log_interval 100000 --eval_interval 100000 --save_interval 100000 --eval_episodes 100
```

## 📝 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{zhang2026dsp,
  title={Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning},
  author={Zhang, Hengrui and Cheng, Yuhu and Chen, C. L. Philip and Wang, Xuesong},
  journal={arXiv preprint arXiv:2609.34575},
  year={2026}
}
```

## 🙏 Acknowledgements

This repository builds upon the excellent works of [OGBench](https://github.com/seohongpark/ogbench) and [HIQL](https://github.com/seohongpark/HIQL). 
We thank the authors for providing open-source implementations and benchmarks that made this research possible.

## 📜 License

This project is licensed under the MIT License
