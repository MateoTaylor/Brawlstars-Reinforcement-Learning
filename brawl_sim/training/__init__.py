"""Training layer: PPO plumbing, shaped reward, and the bot-difficulty curriculum.

Deliberately a separate package from the simulator. `BRAWL_SIM_BUILD_PLAN.md` puts "the reward
function itself, policy architecture, training runs, hyperparameters" explicitly out of scope
for everything under `core/`/`bots/`/`env.py` -- this package is where those live, and it only
ever touches the simulator through its public seams (`BrawlVecEnv`'s `reward_fn` and
`params_hook`, `BrawlSB3VecEnv`'s `infos`). Nothing under `core/` imports from here.

Requires the `[sb3]` optional dependency group (`callbacks.py` and `ppo.py` import
stable_baselines3 / sb3_contrib); `config.py`, `reward.py`, `curriculum.py`, and `schedules.py`
are torch/yaml-only and import cleanly without it.
"""
