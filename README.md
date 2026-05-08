<div align="center">
  
  # Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.26+-yellow.svg)](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## 📑 Overview

This repository provides the implementation of **"Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning"**.

DSP is a diffusion-based hierarchical framework for offline goal-conditioned reinforcement learning. It replaces explicit value-based guidance in high-level subgoal generation with guided generative sampling, while retaining a value-trained low-level executor. By learning both conditional and unconditional subgoal flows, DSP uses classifier-free guidance at inference time to generate data-supported, goal-directed subgoals for long-horizon tasks.

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

Please note that we **do not provide the entire baseline methods** here.

Run the following command to reproduce our method (OGBench):

```bash
python main.py --agent=agents/dsp.py --env_name=antmaze-medium-navigate-v0 --agent.expectile=0.7 --agent.low_alpha=3.0 --agent.cfg=3.0 --agent.actor_p_curgoal=0.0 --agent.actor_p_trajgoal=1.0 --agent.actor_p_randomgoal=0.0
```

Run the following command to reproduce our method (D4RL):

```bash
python main.py --run_group EXP --seed 0 --env_name antmaze-large-diverse-v2 --algo_name dsp --use_waypoints 1 --way_steps 25 --pretrain_steps 1000000 --log_interval 100000 --eval_interval 100000 --save_interval 100000 --eval_episodes 100
```

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{neurips2026submission,
  title={Diffusion Subgoal Planning for Long-Horizon Offline Goal-Conditioned Reinforcement Learning},
  author={Anonymous Authors},
  journal={},
  year={2026}
}
```

## 🙏 Acknowledgements

This repository builds upon the excellent works of [OGBench](https://github.com/seohongpark/ogbench) and [HIQL](https://github.com/seohongpark/HIQL). 
We thank the authors for providing open-source implementations and benchmarks that made this research possible.

## 📜 License

This project is licensed under the MIT License

## 📞 Contact

For any questions or issues, please contact: anon.email@domain.com
