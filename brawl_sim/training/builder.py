"""Assembles a TrainConfig into a ready-to-`learn()` (venv, model) pair.

Kept out of `scripts/train.py` so `tests/test_training.py` can build the exact same objects the
CLI builds, at `n_envs=2`, without shelling out. `scripts/train.py` is a thin argument parser
over `build_run`.

**The one sharp edge this module exists to handle.** `BrawlVecEnv`'s own docstring warns that a
`load_config(..., overrides=...)` override touching a SimParams-only field (`base_hp`,
`zone.dps`, `entities.enemy_hp_mult`, ...) is invisible to `EnvConfig` and is silently lost
unless the matching merged dict is passed separately as `spec=`. `run.env_overrides` in
configs/train.yaml is exactly such an override channel, so `build_env` always merges it into
BOTH the `EnvConfig` and the `spec` dict. Without that, `env_overrides: {boxes: {hp: 1}}` would
appear to do nothing at all.
"""
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from ..config import load_config
from ..core import obs_select
from ..env import BrawlVecEnv
from ..wrappers.sb3_features import default_policy_kwargs, float_group_names
from ..wrappers.sb3_vecenv import BrawlSB3VecEnv
from .config import TrainConfig
from .curriculum import CurriculumManager
from .reward import ShapedReward
from .schedules import make_schedule

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = REPO_ROOT / "configs"


def _resolve(path) -> Path:
    """Config paths in train.yaml are repo-relative by convention; absolute paths pass through."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_spec(tcfg: TrainConfig) -> dict:
    """The merged `default.yaml + brawlers.yaml + run.env_overrides` dict `BrawlVecEnv(spec=...)`
    wants. Mirrors `env._load_base_spec`'s shallow top-level merge (the two files have disjoint
    top-level keys), then deep-merges the overrides so a partial section (`{zone: {dps: 500}}`)
    patches rather than replaces."""
    base = {
        **yaml.safe_load(_resolve(tcfg.run.env_config).read_text()),
        **yaml.safe_load((_CONFIGS_DIR / "brawlers.yaml").read_text()),
    }
    return _deep_merge(base, tcfg.run.env_overrides or {})


def sees_pickups(spec: obs_select.AgentObsSpec) -> bool:
    """Whether any group shows the agent a cube lying on the ground: a grid's `pickup` channel, or
    any `pickups.*` field. `hero.cubes` does not count -- it says how many the hero HOLDS, which is
    no help in finding the next one."""
    for g in spec.groups:
        if g.view_channels is not None and "pickup" in g.view_channels:
            return True
        if any(f.startswith("pickups.") for f in g.fields):
            return True
    return False


def check_reward_is_observable(reward, spec: obs_select.AgentObsSpec, spec_path) -> None:
    """Refuses a run that pays `reward.cube_pickup` under a spec that cannot see pickups.

    Such a run trains an approach behaviour toward something the policy has no input for -- it
    learns to loiter where cubes tend to be -- and nothing in the run's own curves says so. The
    reward lives in the shared configs/train.yaml and the spec is chosen per run, so this is the
    first point where the two are both known; a static test over the files cannot tell a spec that
    sees cubes from one that does not. One direction only: a spec that sees cubes with the term at
    0 is a legitimate unshaped run.
    """
    if reward.cube_pickup != 0.0 and not sees_pickups(spec):
        raise ValueError(
            f"reward.cube_pickup is {reward.cube_pickup}, but {spec_path} has no grid `pickup` "
            "channel and no `pickups.*` field, so the agent would be paid for collecting something "
            "it cannot see. Pass `--set reward.cube_pickup=0` for this spec (the deploy and deploy2 "
            "specs need it), or train on one that sees cubes (agent_obs_deploy3.yaml)."
        )


def build_env(tcfg: TrainConfig, n_envs: int | None = None, verbose: bool | None = None):
    """Returns `(venv, parts)` where `venv` is the fully wrapped SB3 `VecEnv` and `parts` is a
    dict of the pieces the caller still needs handles on: `sim` (the native `BrawlVecEnv`),
    `curriculum` (a `CurriculumManager` or None), `reward_fn`, `agent_spec`, `env_cfg`."""
    n_envs = tcfg.run.n_envs if n_envs is None else n_envs
    verbose = (tcfg.run.verbose > 0) if verbose is None else verbose

    env_cfg = load_config(_resolve(tcfg.run.env_config), overrides=tcfg.run.env_overrides or None)
    agent_spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)
    check_reward_is_observable(tcfg.reward, agent_spec, tcfg.run.agent_obs)
    reward_fn = ShapedReward(tcfg.reward)

    sim = BrawlVecEnv(
        env_cfg, n_envs=n_envs, device=tcfg.run.device, seed=tcfg.run.seed,
        reward_fn=reward_fn,
        randomization=str(_resolve(tcfg.run.randomization)) if tcfg.run.randomization else None,
        spec=build_spec(tcfg),
        verbose=verbose,
    )

    curriculum = None
    if tcfg.curriculum.enabled:
        # Shares the env's own Generator on purpose: one seed reproduces the whole run, tier
        # draws included. Installed as the params_hook AFTER construction because the manager
        # needs `sim.gen`, which doesn't exist until BrawlVecEnv.__init__ returns.
        curriculum = CurriculumManager(tcfg.curriculum, device=sim.device, gen=sim.gen)
        sim.params_hook = curriculum

    venv = VecMonitor(BrawlSB3VecEnv(sim, agent_spec, reward_fn, info_mode=tcfg.run.info_mode))

    if tcfg.normalize.enabled:
        venv = VecNormalize(
            venv, training=True,
            norm_obs=tcfg.normalize.obs, norm_reward=tcfg.normalize.reward,
            clip_obs=tcfg.normalize.clip_obs, clip_reward=tcfg.normalize.clip_reward,
            gamma=tcfg.ppo.gamma,
            # Excludes the uint8 grid group: VecNormalize would otherwise apply a running
            # mean/std to occupancy COUNTS, which is meaningless and corrupts the CNN's input.
            # Only passed when norm_obs is on -- SB3 ignores it otherwise.
            **({"norm_obs_keys": float_group_names(agent_spec)} if tcfg.normalize.obs else {}),
        )

    return venv, {
        "sim": sim, "curriculum": curriculum, "reward_fn": reward_fn,
        "agent_spec": agent_spec, "env_cfg": env_cfg,
    }


def build_policy_kwargs(tcfg: TrainConfig, agent_spec, env_cfg) -> dict:
    """`default_policy_kwargs` (features extractor + the required `normalize_images=False`) plus
    this config's extractor widths and head architecture."""
    kwargs = default_policy_kwargs(agent_spec, env_cfg)
    kwargs["features_extractor_kwargs"] = {
        "cnn_output_dim": tcfg.policy.cnn_output_dim,
        "mlp_hidden_dims": tuple(tcfg.policy.mlp_hidden_dims),
    }
    if tcfg.policy.net_arch:
        kwargs["net_arch"] = dict(tcfg.policy.net_arch)
    return kwargs


def tensorboard_available() -> bool:
    """Whether SB3 can actually write TensorBoard events. SB3 imports `SummaryWriter` from
    `torch.utils.tensorboard` and sets it to None if the import fails -- and that import needs
    the standalone `tensorboard` PACKAGE, which torch does not pull in. When it's missing,
    `configure_logger` raises `ImportError` from inside `learn()`, i.e. AFTER env construction
    and model setup have already run. Checked up front instead so a missing optional dependency
    degrades to "no TB, everything else still logs" with a clear message, rather than blowing up
    a run that has already paid its startup cost."""
    from stable_baselines3.common.logger import SummaryWriter
    return SummaryWriter is not None


def make_logger(tcfg: TrainConfig, log_dir):
    """An explicit SB3 `Logger` written to `log_dir`, replacing the one `learn()` would
    otherwise configure for itself (`model.set_logger` marks it custom, so `learn()` leaves it
    alone).

    Done by hand because SB3's own default is **stdout only** -- with `tensorboard_log` set it
    becomes stdout + tensorboard, and in neither case is there a machine-readable file on disk.
    That's a poor fit for "monitor a run": a scrolling console table can't be replotted, diffed
    between runs, or read after the terminal is gone. This always writes:

      progress.csv   every logged scalar, one row per dump -- pandas/matplotlib read it directly
      log.txt        the same human-readable tables the console prints, kept for the record
      events.*       TensorBoard, when the optional `tensorboard` package is installed

    All three land in one directory, so `tensorboard --logdir runs` still discovers the events
    (it recurses) without SB3's extra `<Algo>_1` nesting level.
    """
    from stable_baselines3.common.logger import configure

    formats = ["csv", "log"]
    if tcfg.run.verbose:
        formats.insert(0, "stdout")
    if tcfg.run.tensorboard:
        if tensorboard_available():
            formats.append("tensorboard")
        else:
            print("[logging] run.tensorboard is true but the `tensorboard` package is not "
                  "installed, so no event files will be written (progress.csv and log.txt "
                  "still are). Install it with:  pip install tensorboard")
    return configure(str(log_dir), formats)


def build_model(tcfg: TrainConfig, venv, agent_spec, env_cfg):
    """`MaskablePPO` (default) or plain `PPO`, wired with both schedules.

    `MaskablePPO` is the right default here: `core/hero.action_mask` already knows exactly when
    firing is illegal (dead / out of ammo / on cooldown / mid-dash) and `BrawlSB3VecEnv` already
    exposes it, so masking those actions out costs nothing and stops the policy from spending
    early training learning that "fire" is a no-op two thirds of the time."""
    algo_cls = PPO
    if tcfg.run.algo == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        algo_cls = MaskablePPO

    # PPO builds a fresh torch.distributions.Categorical every action-selection call, and by
    # default each one re-validates its own (already policy-guaranteed-valid) logits -- a real
    # training run profiled this at ~12% of total wall-clock (Distribution.__init__), pure
    # overhead since a softmax/masked-logits tensor can never fail that check in normal use.
    # Global and process-wide by design (torch's own default_validate_args is a single flag),
    # so this only needs to run once per process; calling it again here is a harmless no-op.
    torch.distributions.Distribution.set_default_validate_args(False)

    return algo_cls(
        "MultiInputPolicy", venv,
        policy_kwargs=build_policy_kwargs(tcfg, agent_spec, env_cfg),
        learning_rate=make_schedule(tcfg.learning_rate),
        clip_range=make_schedule(tcfg.clip_range),
        n_steps=tcfg.ppo.n_steps,
        batch_size=tcfg.ppo.batch_size,
        n_epochs=tcfg.ppo.n_epochs,
        gamma=tcfg.ppo.gamma,
        gae_lambda=tcfg.ppo.gae_lambda,
        ent_coef=tcfg.ppo.ent_coef,
        vf_coef=tcfg.ppo.vf_coef,
        max_grad_norm=tcfg.ppo.max_grad_norm,
        target_kl=tcfg.ppo.target_kl,
        normalize_advantage=tcfg.ppo.normalize_advantage,
        # No `tensorboard_log=`: logging is configured explicitly via `make_logger` +
        # `model.set_logger` (see make_logger for why SB3's own default isn't enough).
        seed=tcfg.run.seed,
        device=tcfg.run.device,
        verbose=tcfg.run.verbose,
    )


def build_run(tcfg: TrainConfig, log_dir=None, n_envs: int | None = None):
    """(model, venv, parts) -- everything `scripts/train.py` needs to call `.learn()`. Passing
    `log_dir` also installs the CSV/log.txt/TensorBoard logger described in `make_logger`."""
    venv, parts = build_env(tcfg, n_envs=n_envs)
    model = build_model(tcfg, venv, parts["agent_spec"], parts["env_cfg"])
    if log_dir is not None:
        model.set_logger(make_logger(tcfg, log_dir))
    return model, venv, parts
