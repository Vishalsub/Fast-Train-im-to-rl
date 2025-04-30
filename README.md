# LOKI-G: Fast Policy Learning via Imitation + Reinforcement

## Description
This module implements **LOKI** (Locally Optimal search after K-step Imitation), a hybrid IL+RL algorithm for fast policy learning. It uses expert demonstrations for warm-starting and transitions to policy gradient reinforcement learning for long-term performance.

> 📊 **LOKI** = K-step Imitation + Mirror Descent RL

## Highlights
- Mirrors the LOKI algorithm from "Fast Policy Learning through Imitation and Reinforcement".
- Supports demonstration dataset loading from `.npz` format.
- Configurable switching strategy for IL to RL transition.
- Compatible with PPO-style actor-critic models.

## Directory Structure (Suggested)
```
loki_rl/
├── loki_main.py           # Main entrypoint to train using LOKI
├── policy.py              # Actor-critic network and PPO agent logic
├── loki_utils.py          # Data loading, switching scheduler, advantage estimates
├── configs/
│   └── loki_config.yaml  # Contains all training hyperparameters
├── data/
    └── f_high_quality_demonstrations_1.npz
```

## Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run LOKI Training
```bash
python loki_main.py --config configs/loki_config.yaml
```

### Configuration
Modify `configs/loki_config.yaml` to control:
- Number of IL steps `K`
- Learning rates for IL/RL phases
- PPO clipping, batch size, discount factor

## To-do
- [ ] Add evaluation script per training checkpoint
- [ ] Add support for AGGREVATE or THOR-style IL baselines
- [ ] Benchmark LOKI against GAIL and DAgger
- [ ] Visualize learning curves and IL-RL phase separation

## License
MIT License - see the [LICENSE](/LICENSE) file.

## References
- [Fast Policy Reinforcement Learning (LOKI)](https://arxiv.org/abs/1805.10413)
- Cheng, C.-A., Yan, X., Wagener, N., & Boots, B. (2018). Fast Policy Learning through Imitation and Reinforcement. *arXiv preprint arXiv:1805.10413*.
- [PPO - Schulman et al.](https://arxiv.org/abs/1707.06347)

## Issues & Discussions
To report a bug or start a discussion, please open an [Issue](https://github.com/your-username/loki_g_rl/issues).

---

> ✉️ For advanced usage or theoretical background, check `Fast Policy Reinforcement Learning.pdf` included in this repo.


