"""TrainConfig: the typed, validated form of configs/train.yaml.

Mirrors `brawl_sim/config.py`'s split in spirit -- frozen dataclasses, everything validated once
at load, nothing hardcoded downstream -- but stays a separate module because it describes the
RUN, not the world, and must not become a dependency of anything under `core/`.

Validation is deliberately eager and specific: a run that dies 40 minutes in because
`batch_size` didn't divide `n_steps * n_envs`, or because a curriculum stage referenced a tier
that isn't defined, is far more expensive than one that refuses to start.
"""
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

# Multiplier names a `tiers:` entry may set, and the value each defaults to when omitted.
# 1.0 everywhere means "leave configs/brawlers.yaml's own number alone", so a tier only has to
# name the knobs it actually moves.
TIER_FIELDS: dict[str, float] = {
    "aim_noise": 1.0,
    "reaction_delay": 1.0,
    "lead_target": 1.0,
    "decision_period": 1.0,
    "move_speed": 1.0,
    "hp": 1.0,
    "damage": 1.0,
}

SCHEDULE_KINDS = ("constant", "linear", "cosine", "exponential")
ALGOS = ("maskable_ppo", "ppo")
INFO_MODES = ("minimal", "episode", "full")


@dataclass(frozen=True)
class ScheduleConfig:
    schedule: str = "linear"
    initial: float = 3.0e-4
    final: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULE_KINDS:
            raise ValueError(f"schedule must be one of {SCHEDULE_KINDS}, got {self.schedule!r}")
        if self.initial <= 0:
            raise ValueError(f"schedule initial must be > 0, got {self.initial}")
        if self.final < 0:
            raise ValueError(f"schedule final must be >= 0, got {self.final}")
        if self.schedule == "exponential" and self.final <= 0:
            raise ValueError("exponential schedule needs final > 0 (it decays by a ratio, "
                             "so it can never reach exactly 0)")


@dataclass(frozen=True)
class RunConfig:
    name: str = "mortis_ppo"
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "runs"
    total_timesteps: int = 20_000_000
    n_envs: int = 512
    algo: str = "maskable_ppo"
    env_config: str = "configs/default.yaml"
    agent_obs: str = "configs/agent_obs.yaml"
    randomization: str | None = None
    env_overrides: dict = field(default_factory=dict)
    info_mode: str = "episode"
    checkpoint_every_steps: int = 1_000_000
    tensorboard: bool = True
    log_interval: int = 1
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.algo not in ALGOS:
            raise ValueError(f"run.algo must be one of {ALGOS}, got {self.algo!r}")
        if self.info_mode not in INFO_MODES:
            raise ValueError(f"run.info_mode must be one of {INFO_MODES}, got {self.info_mode!r}")
        if self.n_envs < 1:
            raise ValueError(f"run.n_envs must be >= 1, got {self.n_envs}")
        if self.total_timesteps < 1:
            raise ValueError(f"run.total_timesteps must be >= 1, got {self.total_timesteps}")


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int = 128
    batch_size: int = 8192
    n_epochs: int = 4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    normalize_advantage: bool = True


@dataclass(frozen=True)
class PolicyConfig:
    cnn_output_dim: int = 128
    mlp_hidden_dims: tuple[int, ...] = (256,)   # BrawlFeaturesExtractor's vector-group MLP, one width per layer
    net_arch: dict = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


@dataclass(frozen=True)
class NormalizeConfig:
    obs: bool = False
    reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    @property
    def enabled(self) -> bool:
        return self.obs or self.reward


@dataclass(frozen=True)
class RewardConfig:
    win_bonus: float = 10.0
    death_penalty: float = -5.0
    rank_bonus: float = 1.0
    damage_dealt: float = 1.0e-3
    damage_taken: float = -1.0e-3
    hp_healed: float = 1.0e-3      # deliberately +damage_taken: HP won back returns the reward
    kill: float = 3.0
    cube_pickup: float = 0.25
    survive_per_step: float = 0.002
    in_zone_per_step: float = -0.05
    scale: float = 1.0


@dataclass(frozen=True)
class DifficultyTier:
    """A named set of multipliers on configs/brawlers.yaml's own per-archetype numbers. Every
    field defaults to 1.0 (see TIER_FIELDS), so `hard: {}` is a perfectly valid way to spell
    "the bots exactly as authored"."""
    name: str
    aim_noise: float = 1.0
    reaction_delay: float = 1.0
    lead_target: float = 1.0
    decision_period: float = 1.0
    move_speed: float = 1.0
    hp: float = 1.0
    damage: float = 1.0

    def __post_init__(self) -> None:
        for key in TIER_FIELDS:
            value = getattr(self, key)
            if value < 0:
                raise ValueError(f"tier {self.name!r}: {key} must be >= 0, got {value}")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    tier_weights: dict          # {tier_name: weight}; normalized at use, need not sum to 1
    advance_win_rate: float | None = None   # None => terminal stage, never advances

    def __post_init__(self) -> None:
        if not self.tier_weights:
            raise ValueError(f"stage {self.name!r} has no tier_weights")
        for tier_name, weight in self.tier_weights.items():
            if weight < 0:
                raise ValueError(f"stage {self.name!r}: tier_weights[{tier_name!r}] must be >= 0")
        if sum(self.tier_weights.values()) <= 0:
            raise ValueError(f"stage {self.name!r}: tier_weights sum to 0")
        if self.advance_win_rate is not None and not 0.0 <= self.advance_win_rate <= 1.0:
            raise ValueError(f"stage {self.name!r}: advance_win_rate must be in [0, 1], "
                             f"got {self.advance_win_rate}")


@dataclass(frozen=True)
class CurriculumConfig:
    enabled: bool = True
    window_episodes: int = 500
    min_episodes_at_stage: int = 500
    demote_win_rate: float | None = None
    max_timesteps_at_stage: int | None = None
    tiers: dict = field(default_factory=dict)       # {name: DifficultyTier}
    stages: tuple = ()                              # (CurriculumStage, ...)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if not self.stages:
            raise ValueError("curriculum.enabled is true but no stages are defined")
        if self.window_episodes < 1:
            raise ValueError(f"curriculum.window_episodes must be >= 1, got {self.window_episodes}")
        if self.min_episodes_at_stage > self.window_episodes:
            # CurriculumCallback also requires the window to hold min_episodes_at_stage outcomes,
            # which a smaller window never can: no stage would ever advance on its win rate.
            raise ValueError(
                f"curriculum.min_episodes_at_stage ({self.min_episodes_at_stage}) must not exceed "
                f"curriculum.window_episodes ({self.window_episodes}), or no stage can ever advance"
            )
        for stage in self.stages:
            for tier_name in stage.tier_weights:
                if tier_name not in self.tiers:
                    raise ValueError(
                        f"stage {stage.name!r} references undefined tier {tier_name!r}; "
                        f"defined tiers are {sorted(self.tiers)}"
                    )
        if self.stages[-1].advance_win_rate is not None:
            raise ValueError(
                f"the last stage ({self.stages[-1].name!r}) must have advance_win_rate: null -- "
                "it is terminal and has nowhere to advance to"
            )


@dataclass(frozen=True)
class EvalConfig:
    """Periodic stationary evaluation -- fixed difficulty, unaffected by the curriculum.

    This is the only signal in the whole run that is comparable ACROSS TIME. `rollout/
    ep_rew_mean` is measured against the training distribution, which the curriculum
    deliberately makes harder, so it cannot distinguish "policy improved" from "curriculum got
    easier". Every eval here replays the same seeded scenarios against the same pinned bots.
    """
    enabled: bool = True
    every_timesteps: int = 500_000
    episodes_per_tier: int = 32
    tiers: tuple = ()            # empty => every tier defined under `curriculum.tiers`
    deterministic: bool = True
    seed: int = 999_983          # deliberately unrelated to run.seed; see TierEvaluator
    at_start: bool = True        # one eval before any training, as the baseline row

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.episodes_per_tier < 1:
            raise ValueError(f"eval.episodes_per_tier must be >= 1, got {self.episodes_per_tier}")
        if self.every_timesteps < 1:
            raise ValueError(f"eval.every_timesteps must be >= 1, got {self.every_timesteps}")


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    learning_rate: ScheduleConfig = field(default_factory=ScheduleConfig)
    clip_range: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(schedule="linear", initial=0.2, final=0.05)
    )
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    raw: dict = field(default_factory=dict, repr=False)   # the merged YAML, for verbatim archiving

    @property
    def rollout_transitions(self) -> int:
        return self.ppo.n_steps * self.run.n_envs


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def deep_merge(base: dict, patch: dict) -> dict:
    """Recursive dict merge; `patch` wins at the leaves. Public because scripts/train.py's
    `--smoke` needs to layer two override dicts before either reaches `load_train_config`."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_deep_merge = deep_merge  # internal alias, kept so the call sites below read as before


def _coerce_literal(text: str):
    """Parses a `--set key=value` value with YAML's own scalar rules, so `true`/`3`/`null`/
    `[1, 2]` all mean what they mean everywhere else in this file, and anything that doesn't
    parse stays a plain string.

    With one deliberate widening: **YAML 1.1's float resolver requires a decimal point**, so
    `yaml.safe_load("1e-4")` returns the STRING `"1e-4"`, not `0.0001`. Inside a YAML file that's
    a non-issue (this repo's configs all write `1.0e-4`), but on a command line `--set
    learning_rate.initial=1e-4` is exactly what anyone would type, and the string would sail
    through into a `float` dataclass field and only blow up later comparing `str <= int`. Any
    scalar YAML leaves as a string is retried as an int, then a float, before being accepted as
    a genuine string."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        value = text
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def parse_overrides(assignments) -> dict:
    """`["ppo.n_steps=256", "curriculum.enabled=false"]` -> a nested dict ready for _deep_merge."""
    out: dict = {}
    for item in assignments or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        dotted, _, raw_value = item.partition("=")
        parts = dotted.strip().split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {dotted!r} conflicts with an earlier scalar override")
        node[parts[-1]] = _coerce_literal(raw_value.strip())
    return out


def _build(cls, data: dict, **extra):
    """Constructs a dataclass from `data`, rejecting unknown keys by name rather than letting a
    typo silently fall back to the default (the single most common way a config edit does
    nothing at all)."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known - set(extra)
    if unknown:
        raise ValueError(f"unknown key(s) in {cls.__name__}: {sorted(unknown)}; "
                         f"valid keys are {sorted(known - set(extra))}")
    return cls(**{k: v for k, v in data.items() if k in known}, **extra)


def _build_curriculum(data: dict) -> CurriculumConfig:
    # No `curriculum:` section at all means "no curriculum" -- NOT "the enabled-by-default
    # curriculum, with zero stages", which would raise a confusing "enabled but no stages"
    # error for anyone writing a minimal config. Writing the section and leaving out `stages`
    # still errors, because that IS a mistake.
    if not data:
        return CurriculumConfig(enabled=False)

    body = {k: v for k, v in data.items() if k not in ("tiers", "stages")}

    tiers = {}
    for name, raw in (data.get("tiers") or {}).items():
        raw = raw or {}
        unknown = set(raw) - set(TIER_FIELDS)
        if unknown:
            raise ValueError(f"tier {name!r} has unknown key(s) {sorted(unknown)}; "
                             f"valid keys are {sorted(TIER_FIELDS)}")
        tiers[name] = DifficultyTier(name=name, **{k: float(v) for k, v in raw.items()})

    stages = []
    for i, raw in enumerate(data.get("stages") or ()):
        unknown = set(raw) - {"name", "tier_weights", "advance_win_rate"}
        if unknown:
            raise ValueError(f"stage {i} has unknown key(s) {sorted(unknown)}")
        advance = raw.get("advance_win_rate")
        stages.append(CurriculumStage(
            name=str(raw.get("name", f"stage_{i}")),
            tier_weights={str(k): float(v) for k, v in raw["tier_weights"].items()},
            advance_win_rate=None if advance is None else float(advance),
        ))

    return _build(CurriculumConfig, body, tiers=tiers, stages=tuple(stages))


def _build_eval(data: dict) -> EvalConfig:
    data = dict(data)
    if "tiers" in data and data["tiers"] is not None:
        data["tiers"] = tuple(str(t) for t in data["tiers"])
    return _build(EvalConfig, data)


def load_train_config(path, overrides: dict | None = None) -> TrainConfig:
    """Loads configs/train.yaml (or any file shaped like it), deep-merges `overrides` (from
    `parse_overrides`), and validates the whole thing. Every section is optional -- an empty
    file yields the dataclass defaults above."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    unknown = set(raw) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown top-level section(s) in {path}: {sorted(unknown)}")

    cfg = TrainConfig(
        run=_build(RunConfig, raw.get("run") or {}),
        ppo=_build(PPOConfig, raw.get("ppo") or {}),
        learning_rate=_build(ScheduleConfig, raw.get("learning_rate") or {}),
        clip_range=_build(ScheduleConfig, raw.get("clip_range")
                          or {"schedule": "linear", "initial": 0.2, "final": 0.05}),
        policy=_build(PolicyConfig, raw.get("policy") or {}),
        normalize=_build(NormalizeConfig, raw.get("normalize") or {}),
        reward=_build(RewardConfig, raw.get("reward") or {}),
        curriculum=_build_curriculum(raw.get("curriculum") or {}),
        eval=_build_eval(raw.get("eval") or {}),
        raw=raw,
    )
    validate_train_config(cfg)
    return cfg


def validate_train_config(cfg: TrainConfig) -> None:
    """Cross-section checks -- the ones no single dataclass can make on its own."""
    transitions = cfg.rollout_transitions
    if cfg.ppo.batch_size > transitions:
        raise ValueError(
            f"ppo.batch_size ({cfg.ppo.batch_size}) exceeds one rollout's transitions "
            f"(n_steps {cfg.ppo.n_steps} * n_envs {cfg.run.n_envs} = {transitions})"
        )
    if transitions % cfg.ppo.batch_size != 0:
        raise ValueError(
            f"ppo.batch_size ({cfg.ppo.batch_size}) must divide n_steps * n_envs "
            f"({cfg.ppo.n_steps} * {cfg.run.n_envs} = {transitions}); SB3 silently drops the "
            "remainder minibatch otherwise"
        )
    if cfg.curriculum.enabled and cfg.run.info_mode == "minimal":
        raise ValueError(
            "curriculum.enabled requires run.info_mode 'episode' or 'full' -- info[\"outcome\"] "
            "(the per-episode win signal the curriculum advances on) is only emitted by "
            "BrawlSB3VecEnv at those modes"
        )
    if cfg.curriculum.enabled and cfg.curriculum.min_episodes_at_stage > cfg.curriculum.window_episodes:
        raise ValueError(
            f"curriculum.min_episodes_at_stage ({cfg.curriculum.min_episodes_at_stage}) exceeds "
            f"window_episodes ({cfg.curriculum.window_episodes}); the win rate is only ever "
            "measured over the window, so the floor could never be satisfied"
        )
    if cfg.eval.enabled:
        if not cfg.curriculum.tiers:
            raise ValueError(
                "eval.enabled requires `curriculum.tiers` to be defined -- evaluation pins bots "
                "to those same named tiers, so that an eval score describes the same bots the "
                "curriculum trains against. Define tiers (curriculum.enabled may stay false), "
                "or set eval.enabled: false."
            )
        unknown = set(cfg.eval.tiers) - set(cfg.curriculum.tiers)
        if unknown:
            raise ValueError(
                f"eval.tiers names undefined tier(s) {sorted(unknown)}; defined tiers are "
                f"{sorted(cfg.curriculum.tiers)}"
            )


def with_overrides(cfg: TrainConfig, **sections) -> TrainConfig:
    """`with_overrides(cfg, run=replace(cfg.run, n_envs=8))` -- used by the tests and the
    `--smoke` path to shrink a real config without re-parsing YAML."""
    out = replace(cfg, **sections)
    validate_train_config(out)
    return out
