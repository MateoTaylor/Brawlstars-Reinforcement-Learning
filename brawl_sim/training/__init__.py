"""Training layer: PPO plumbing, shaped reward, and the bot-difficulty curriculum.

Kept apart from the simulator: the reward, policy architecture and hyperparameters live here,
and this package touches the sim only through its public seams (`BrawlVecEnv`'s `reward_fn` and
`params_hook`, `BrawlSB3VecEnv`'s `infos`). Nothing under `core/` imports from here.

`builder.py`, `callbacks.py` and `evaluation.py` need the `[sb3]` optional dependency group;
`config.py`, `reward.py`, `curriculum.py` and `schedules.py` are torch/yaml-only.
"""
