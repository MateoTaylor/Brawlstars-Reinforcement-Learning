"""Tests for brawl_sim/training/ -- config, schedules, shaped reward, and the curriculum.

Follows CONVENTIONS.md's testing rule: CPU by default, `n_envs = 8` with
`presets/debug_tiny.yaml` for anything that needs a real simulator. The curriculum-sampling
tests deliberately break the n_envs=8 rule and use a few thousand envs instead -- they assert a
*distribution* matches its configured weights, which needs a sample size, and the tiny-map sim
is cheap enough on CPU that it costs a fraction of a second.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.env import BrawlVecEnv
from brawl_sim.training import schedules
from brawl_sim.training.config import (
    CurriculumConfig, CurriculumStage, DifficultyTier, RewardConfig, RunConfig, ScheduleConfig,
    load_train_config, parse_overrides,
)
from brawl_sim.training.curriculum import CurriculumManager
from brawl_sim.training.reward import ShapedReward

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
DEBUG_TINY = REPO_ROOT / "configs" / "presets" / "debug_tiny.yaml"


@pytest.fixture
def tcfg():
    return load_train_config(TRAIN_CONFIG)


def _tiny_env(n_envs=8, seed=0, params_hook=None):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      overrides=yaml.safe_load(DEBUG_TINY.read_text()))
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed, verbose=False,
                      params_hook=params_hook)
    return cfg, env


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_shipped_train_config_loads_and_validates(tcfg):
    assert tcfg.run.algo == "maskable_ppo"
    assert tcfg.rollout_transitions % tcfg.ppo.batch_size == 0
    assert len(tcfg.curriculum.stages) >= 2
    assert tcfg.curriculum.stages[-1].advance_win_rate is None


def test_shipped_curriculum_advance_thresholds_are_well_formed(tcfg):
    """Every non-terminal stage has an advance threshold in (0, 1); the terminal one has none.

    This used to assert `== 0.5` for every stage, and had been failing against the shipped
    config for some time -- `configs/train.yaml` moved to 0.2/0.3 thresholds and the test was
    never updated. The exact threshold is a TUNING KNOB (it is retuned whenever bot difficulty
    changes, which the bot overhaul does on purpose); what is actually invariant is that a
    non-terminal stage is escapable and a terminal stage is not. Thresholds must also be
    non-decreasing: a later stage should never be easier to leave than an earlier one.
    """
    thresholds = [s.advance_win_rate for s in tcfg.curriculum.stages[:-1]]
    for stage, rate in zip(tcfg.curriculum.stages, thresholds):
        assert rate is not None and 0.0 < rate < 1.0, (
            f"stage {stage.name!r} has a non-terminal advance_win_rate of {rate!r}"
        )
    assert thresholds == sorted(thresholds), (
        f"advance thresholds must be non-decreasing across the walk, got {thresholds}"
    )
    assert tcfg.curriculum.stages[-1].advance_win_rate is None   # terminal


def test_shipped_curriculum_walks_easy_to_hard(tcfg):
    """The user-facing shape of the default curriculum: it starts weighted toward the weakest
    tier and shifts monotonically toward the strongest. If someone reorders the stages, this
    fails.

    Deliberately does NOT require the terminal stage to be a single pure tier. It used to assert
    `"medium" not in last.tier_weights`, which the shipped config broke when the terminal stage
    became a MIXTURE (`{medium: 0.3, hard: 0.5, elite: 0.2}`) -- keeping some weaker opponents in
    the final stage is an anti-overfitting choice, not a regression. What must hold is that the
    weakest tier drains away and the strongest tier only ever grows.
    """
    stages = tcfg.curriculum.stages
    first, last = stages[0], stages[-1]
    share = lambda s, tier: s.tier_weights.get(tier, 0.0) / sum(s.tier_weights.values())  # noqa: E731

    assert first.tier_weights["easy"] == max(first.tier_weights.values())
    assert "easy" not in last.tier_weights, "the terminal stage must contain no `easy` bots"

    easy_share = [share(s, "easy") for s in stages]
    assert easy_share == sorted(easy_share, reverse=True), (
        f"`easy` share must be non-increasing across the walk, got {easy_share}"
    )
    # The mirror condition, which is what actually makes it a walk toward "hard" rather than
    # merely away from "easy": the two strongest tiers' combined share only ever grows.
    top_share = [share(s, "hard") + share(s, "elite") for s in stages]
    assert top_share == sorted(top_share), (
        f"combined `hard`+`elite` share must be non-decreasing across the walk, got {top_share}"
    )


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("ppo: {n_stpes: 128}\n")   # typo
    with pytest.raises(ValueError, match="n_stpes"):
        load_train_config(path)


def test_batch_size_must_divide_rollout(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("run: {n_envs: 8}\nppo: {n_steps: 64, batch_size: 100}\n")
    with pytest.raises(ValueError, match="must divide"):
        load_train_config(path)


def test_stage_referencing_undefined_tier_is_rejected():
    with pytest.raises(ValueError, match="undefined tier"):
        CurriculumConfig(
            tiers={"easy": DifficultyTier("easy")},
            stages=(CurriculumStage("s", {"nope": 1.0}),),
        )


def test_last_stage_must_be_terminal():
    with pytest.raises(ValueError, match="terminal"):
        CurriculumConfig(
            tiers={"easy": DifficultyTier("easy")},
            stages=(CurriculumStage("s", {"easy": 1.0}, advance_win_rate=0.5),),
        )


def test_curriculum_requires_outcome_carrying_info_mode(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("run: {info_mode: minimal}\n"
                    "curriculum:\n"
                    "  tiers: {hard: {}}\n"
                    "  stages: [{name: only, tier_weights: {hard: 1.0}}]\n")
    with pytest.raises(ValueError, match="info_mode"):
        load_train_config(path)


@pytest.mark.parametrize("text,expected", [
    ("1e-4", 1e-4),          # YAML 1.1 would leave this a string; we coerce it
    ("3.0e-4", 3.0e-4),
    ("256", 256),
    ("false", False),
    ("null", None),
    ("linear", "linear"),
])
def test_set_override_coercion(text, expected):
    assert parse_overrides([f"k={text}"])["k"] == expected


def test_set_override_reaches_the_dataclass():
    cfg = load_train_config(TRAIN_CONFIG, overrides=parse_overrides(["learning_rate.initial=1e-4"]))
    assert cfg.learning_rate.initial == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "cosine", "exponential"])
def test_schedule_endpoints_and_direction(kind):
    """SB3 passes progress_REMAINING (1.0 -> 0.0). Every schedule must therefore start at
    `initial` when it's 1.0 and end at `final` when it's 0.0 -- getting this backwards silently
    trains with a RISING learning rate."""
    fn = schedules.make_schedule(ScheduleConfig(schedule=kind, initial=1.0, final=0.01))
    assert fn(1.0) == pytest.approx(1.0)
    assert fn(0.0) == pytest.approx(0.01)
    values = [fn(1.0 - i / 20) for i in range(21)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:])), "must decrease monotonically"


def test_schedule_clamps_timestep_overshoot():
    """SB3's num_timesteps advances n_envs at a time and can overshoot total_timesteps, making
    progress_remaining slightly negative on the final update."""
    fn = schedules.make_schedule(ScheduleConfig(schedule="exponential", initial=1.0, final=0.01))
    assert fn(-0.05) == pytest.approx(0.01)
    assert fn(1.5) == pytest.approx(1.0)


def test_constant_schedule_ignores_final():
    fn = schedules.make_schedule(ScheduleConfig(schedule="constant", initial=7.0, final=0.1))
    assert fn(1.0) == fn(0.5) == fn(0.0) == 7.0


def test_exponential_requires_positive_final():
    with pytest.raises(ValueError, match="final > 0"):
        ScheduleConfig(schedule="exponential", initial=1.0, final=0.0)


# ---------------------------------------------------------------------------
# shaped reward
# ---------------------------------------------------------------------------

def _reward_inputs(n=4, device="cpu"):
    """Mirrors what core/events.compute_info actually returns for ONE decision at
    action_repeat=1: every per-tick delta zeroed, the hero alive for the single tick, and none of
    it spent outside the safe rect. `alive_ticks`/`in_zone_ticks`/`hero_alive` are what
    ShapedReward reads for the rate and death terms -- obs is no longer consulted for them (see
    training/reward.py), so raising alive_ticks to k is how a test says "action_repeat = k"."""
    zeros_ne = torch.zeros((n, 3), dtype=torch.float32, device=device)
    obs = {"hero": {
        "alive": torch.ones(n, dtype=torch.bool, device=device),
        "in_zone": torch.zeros(n, dtype=torch.bool, device=device),
    }}
    info = {
        "damage_dealt_tick": zeros_ne.clone(),
        "damage_taken_tick": zeros_ne.clone(),
        "hp_healed_tick": zeros_ne.clone(),
        "kills_tick": torch.zeros((n, 3), dtype=torch.int32, device=device),
        "cubes_gained_tick": torch.zeros((n, 3), dtype=torch.int64, device=device),
        "hero_rank": torch.zeros(n, dtype=torch.int64, device=device),
        "terminated": torch.zeros(n, dtype=torch.bool, device=device),
        "truncated": torch.zeros(n, dtype=torch.bool, device=device),
        "hero_alive": torch.ones(n, dtype=torch.bool, device=device),
        "alive_ticks": torch.ones(n, dtype=torch.int32, device=device),
        "in_zone_ticks": torch.zeros(n, dtype=torch.int32, device=device),
        "n_ticks": torch.ones(n, dtype=torch.int32, device=device),
    }
    return obs, info


def test_reward_terminal_win_and_death():
    cfg = RewardConfig(win_bonus=10.0, death_penalty=-5.0, rank_bonus=0.0, damage_dealt=0.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0, survive_per_step=0.0,
                       in_zone_per_step=0.0)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=4)
    env_cfg = SimpleNamespace(n_entities=3)

    # env 0: won (terminated, alive, rank 0). env 1: died. env 2: timed out alive at rank 1.
    # env 3: still running.
    info["terminated"][0] = info["terminated"][1] = True
    info["truncated"][2] = True
    info["hero_alive"][1] = False
    info["hero_rank"][1] = 2
    info["hero_rank"][2] = 1

    r = fn(obs, info, env_cfg)
    assert r.tolist() == [10.0, -5.0, 0.0, 0.0]


def test_reward_rank_bonus_scales_with_placement():
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=1.0, damage_dealt=0.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0, survive_per_step=0.0,
                       in_zone_per_step=0.0)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=3)
    info["terminated"][:] = True
    info["hero_rank"] = torch.tensor([0, 3, 6])          # 1st, 4th, last of 7
    r = fn(obs, info, SimpleNamespace(n_entities=7))
    assert r.tolist() == [6.0, 3.0, 0.0]


def test_reward_per_tick_terms_and_scale():
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0,
                       damage_dealt=1e-3, damage_taken=-1e-3, hp_healed=1e-3, kill=3.0,
                       cube_pickup=0.25, survive_per_step=0.002, in_zone_per_step=-0.05, scale=2.0)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=1)
    info["damage_dealt_tick"][0, 0] = 1000.0
    info["damage_taken_tick"][0, 0] = 400.0
    info["hp_healed_tick"][0, 0] = 150.0
    info["kills_tick"][0, 0] = 1
    info["cubes_gained_tick"][0, 0] = 2
    info["in_zone_ticks"][0] = 1
    r = fn(obs, info, SimpleNamespace(n_entities=3))
    expected = (1.0 - 0.4 + 0.15 + 3.0 + 0.5 + 0.002 - 0.05) * 2.0
    assert r.item() == pytest.approx(expected)


def test_reward_healing_refunds_exactly_what_the_damage_cost():
    """The point of hp_healed being the mirror of damage_taken: a wound taken and then fully
    healed back nets to zero, so chip damage the hero out-regens is not a permanent debt."""
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                       kill=0.0, cube_pickup=0.0, survive_per_step=0.0, in_zone_per_step=0.0,
                       damage_taken=-1e-4, hp_healed=1e-4)
    fn = ShapedReward(cfg, track_terms=False)
    env_cfg = SimpleNamespace(n_entities=3)

    obs, info = _reward_inputs(n=1)
    info["damage_taken_tick"][0, 0] = 900.0
    hurt = fn(obs, info, env_cfg).item()

    obs, info = _reward_inputs(n=1)
    info["hp_healed_tick"][0, 0] = 900.0
    healed = fn(obs, info, env_cfg).item()

    assert hurt == pytest.approx(-0.09)
    assert hurt + healed == pytest.approx(0.0)


def test_reward_ignores_healing_by_entities_other_than_the_hero():
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0, survive_per_step=0.0,
                       in_zone_per_step=0.0, hp_healed=1e-3)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=1)
    info["hp_healed_tick"][0, 1] = 999.0        # a bot regenerating, not the hero
    assert fn(obs, info, SimpleNamespace(n_entities=3)).item() == 0.0


def test_reward_rate_terms_scale_with_ticks_not_calls():
    """The property that makes sim.action_repeat safe to change: the two rate terms are priced
    per SIM TICK and integrate over the decision, so one decision covering k ticks pays exactly
    what k separate one-tick decisions would have. Everything else in the reward is already a
    delta env.py sums over the same window, so total episode return is invariant to the
    decision rate -- no reward weight has to be retuned when action_repeat changes."""
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0,
                       survive_per_step=0.002, in_zone_per_step=-0.05)
    fn = ShapedReward(cfg, track_terms=False)
    env_cfg = SimpleNamespace(n_entities=3)

    obs, info = _reward_inputs(n=1)
    info["in_zone_ticks"][0] = 1    # one decision at action_repeat=1: alive, in the zone, 1 tick
    one_tick = fn(obs, info, env_cfg).item()

    obs, info = _reward_inputs(n=1)
    info["alive_ticks"][0] = 5      # one decision, action_repeat=5, alive and in the zone throughout
    info["in_zone_ticks"][0] = 5
    info["n_ticks"][0] = 5
    assert fn(obs, info, env_cfg).item() == pytest.approx(5 * one_tick)


def test_reward_death_penalty_reads_latched_info_not_live_obs():
    """An env that WON mid-decision and was then finished off by the zone on a later sub-tick
    still shows a dead hero in the final obs. Charging the death penalty off that obs would turn
    a win into a loss; ShapedReward reads info["hero_alive"], which env.py latched at the
    sub-tick the episode actually ended. See core/events.advance_decision_tally."""
    cfg = RewardConfig(win_bonus=10.0, death_penalty=-5.0, rank_bonus=0.0, damage_dealt=0.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0, survive_per_step=0.0,
                       in_zone_per_step=0.0)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=1)
    info["terminated"][0] = True
    info["hero_rank"][0] = 0            # won: last one standing
    info["hero_alive"][0] = True        # latched at the winning sub-tick
    obs["hero"]["alive"][0] = False     # but dead by the end of the decision

    assert fn(obs, info, SimpleNamespace(n_entities=3)).item() == pytest.approx(10.0)


def test_reward_reads_hero_row_only():
    """info[...] tensors are (N, E) over ALL entities -- a reward that summed them instead of
    indexing entity 0 would pay the agent for what the BOTS did."""
    cfg = RewardConfig(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=1.0,
                       damage_taken=0.0, kill=0.0, cube_pickup=0.0, survive_per_step=0.0,
                       in_zone_per_step=0.0)
    fn = ShapedReward(cfg, track_terms=False)
    obs, info = _reward_inputs(n=1)
    info["damage_dealt_tick"][0, 1] = 999.0     # a bot's damage, not the hero's
    assert fn(obs, info, SimpleNamespace(n_entities=3)).item() == 0.0


def test_reward_term_means_reports_and_resets():
    fn = ShapedReward(RewardConfig(), track_terms=True)
    obs, info = _reward_inputs(n=2)
    for _ in range(4):
        fn(obs, info, SimpleNamespace(n_entities=3))
    means = fn.term_means()
    assert means["survive_per_step"] == pytest.approx(0.002)   # alive every tick
    assert fn.term_means() == {}                               # accumulators were reset


def test_reward_satisfies_the_simulator_protocol():
    from brawl_sim.core.reward import RewardFn
    assert isinstance(ShapedReward(RewardConfig()), RewardFn)


def test_reward_runs_inside_a_real_env():
    fn = ShapedReward(RewardConfig())
    cfg, env = _tiny_env()
    env.reward_fn = fn
    env.reset()
    action = torch.zeros((8, 2), dtype=torch.int64)
    for _ in range(20):
        _, reward, _, _, _ = env.step(action)
    assert reward.shape == (8,) and reward.dtype == torch.float32
    assert torch.isfinite(reward).all()


# ---------------------------------------------------------------------------
# curriculum manager
# ---------------------------------------------------------------------------

def _manager(stage_weights, tiers=None, device="cpu", seed=0):
    tiers = tiers or {
        "easy": DifficultyTier("easy", aim_noise=3.0, reaction_delay=2.0, lead_target=0.25,
                               decision_period=2.0, move_speed=0.9, hp=0.5, damage=0.5),
        "hard": DifficultyTier("hard"),
    }
    ccfg = CurriculumConfig(
        tiers=tiers,
        stages=(CurriculumStage("a", stage_weights, advance_win_rate=0.5),
                CurriculumStage("b", {"hard": 1.0})),
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return CurriculumManager(ccfg, device=device, gen=gen), ccfg


def test_manager_never_touches_the_hero_column():
    """Per-kind SimParams tensors are (N, K) with column 0 = the hero. A curriculum that leaked
    into column 0 would nerf (or buff) the agent itself along with the bots."""
    mgr, _ = _manager({"easy": 1.0})
    cfg, env = _tiny_env(n_envs=64)
    before = {a: getattr(env.params, a)[:, 0].clone()
              for a in ("aim_noise_std_rad", "base_hp", "base_damage", "move_speed",
                        "reaction_delay", "lead_target_fraction", "decision_period")}
    env.params_hook = mgr
    env.reset()
    for attr, col in before.items():
        assert torch.equal(getattr(env.params, attr)[:, 0], col), f"hero's {attr} was modified"


def test_manager_applies_tier_multipliers_to_bot_columns():
    mgr, _ = _manager({"easy": 1.0})
    cfg, env = _tiny_env(n_envs=64)
    base_hp = env.params.base_hp[:, 1:].clone()
    base_aim = env.params.aim_noise_std_rad[:, 1:].clone()
    env.params_hook = mgr
    env.reset()
    assert torch.allclose(env.params.base_hp[:, 1:], base_hp * 0.5)
    assert torch.allclose(env.params.aim_noise_std_rad[:, 1:], base_aim * 3.0)


def test_manager_mixture_matches_configured_weights():
    mgr, _ = _manager({"easy": 0.7, "hard": 0.3}, seed=7)
    cfg, env = _tiny_env(n_envs=4096)
    env.params_hook = mgr
    env.reset()
    fractions = mgr.tier_spawn_fractions()
    assert fractions["easy"] == pytest.approx(0.7, abs=0.02)
    assert fractions["hard"] == pytest.approx(0.3, abs=0.02)


def test_manager_varies_difficulty_within_one_env():
    """Tiers are drawn per (env, archetype), so a single episode can mix difficulties -- the
    'difficulty varies' half of the requirement. With a 50/50 mixture across every bot archetype,
    all-same across every one of 512 envs is astronomically unlikely."""
    mgr, _ = _manager({"easy": 0.5, "hard": 0.5}, seed=3)
    cfg, env = _tiny_env(n_envs=512)
    # Baseline captured BEFORE the hook is attached, then divided out elementwise, so what remains
    # is each (env, archetype) cell's tier MULTIPLIER with the per-kind HP differences normalized
    # away. Was a hardcoded 4-vector of the shipped bot HP values, which silently encoded both the
    # roster size (it broke the day a fifth bot archetype landed) and the assumption that base_hp
    # is not per-env randomized. Reading it off params costs one clone and assumes neither.
    baseline = env.params.base_hp[:, 1:].clone()
    env.params_hook = mgr
    env.reset()
    per_env_distinct = env.params.base_hp[:, 1:].div(baseline).round(decimals=3)
    n_mixed = int((per_env_distinct.min(dim=1).values != per_env_distinct.max(dim=1).values).sum())
    assert n_mixed > 400


def test_manager_clamps_lead_target_and_floors_decision_period():
    tiers = {"silly": DifficultyTier("silly", lead_target=99.0, decision_period=0.0)}
    ccfg = CurriculumConfig(tiers=tiers, stages=(CurriculumStage("only", {"silly": 1.0}),))
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    mgr = CurriculumManager(ccfg, device="cpu", gen=gen)
    cfg, env = _tiny_env(n_envs=16, params_hook=mgr)
    env.reset()
    assert bool((env.params.lead_target_fraction <= 1.0).all())
    # Bot columns only: the hero's own decision_period is unset in brawlers.yaml (it's 0, and
    # never read -- the hero is agent-controlled) and the curriculum must leave it exactly there
    # rather than floor it to 1 along with the bots. See test_manager_never_touches_the_hero_column.
    assert int(env.params.decision_period[:, 1:].min()) >= 1
    assert int(env.params.decision_period[:, 0].max()) == 0


def test_manager_only_touches_reset_rows():
    mgr, _ = _manager({"easy": 1.0})
    cfg, env = _tiny_env(n_envs=8, params_hook=mgr)
    env.reset()
    snapshot = env.params.base_hp.clone()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[3] = True
    env.reset(reset_mask=mask)
    untouched = [i for i in range(8) if i != 3]
    assert torch.equal(env.params.base_hp[untouched], snapshot[untouched])


def test_manager_stage_control_and_state_roundtrip():
    mgr, _ = _manager({"easy": 1.0})
    assert mgr.stage.name == "a" and not mgr.is_final_stage
    assert mgr.advance() is True and mgr.stage.name == "b"
    assert mgr.is_final_stage and mgr.advance() is False    # terminal
    assert mgr.demote() is True and mgr.stage.name == "a"
    assert mgr.demote() is False                            # already at stage 0

    mgr.advance()
    state = mgr.state_dict()
    other, _ = _manager({"easy": 1.0})
    other.load_state_dict(state)
    assert other.stage_index == mgr.stage_index


def test_manager_rejects_a_checkpoint_naming_a_removed_stage():
    mgr, _ = _manager({"easy": 1.0})
    with pytest.raises(ValueError, match="no longer exists"):
        mgr.load_state_dict({"stage_index": 0, "stage_name": "renamed_away"})


def test_curriculum_composes_with_randomization_ranges():
    """A {low, high} range and a tier multiplier are independent mechanisms that must stack:
    the range jitters the base value per env, the tier then scales it."""
    mgr, _ = _manager({"easy": 1.0})   # hp multiplier 0.5
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      overrides=yaml.safe_load(DEBUG_TINY.read_text()))
    env = BrawlVecEnv(cfg, n_envs=256, device="cpu", seed=0, verbose=False, params_hook=mgr,
                      randomization={"bot_sniper.base_hp": {"low": 1000.0, "high": 2000.0}})
    env.reset()
    sniper_hp = env.params.base_hp[:, 1]
    assert sniper_hp.min() >= 500.0 and sniper_hp.max() <= 1000.0   # (1000..2000) * 0.5
    assert sniper_hp.std() > 1.0                                    # the range still varies


# ---------------------------------------------------------------------------
# training monitor (continuous, curriculum-independent train/* metrics)
# ---------------------------------------------------------------------------

pytest.importorskip("stable_baselines3")
from brawl_sim.training.callbacks import CurriculumCallback, TrainingMonitorCallback  # noqa: E402


def _monitor(window_episodes=10, **kw):
    cb = TrainingMonitorCallback(window_episodes=window_episodes, verbose=0, **kw)
    recorded = {}
    cb.model = SimpleNamespace(logger=SimpleNamespace(record=lambda k, v, **_: recorded.__setitem__(k, v)))
    cb.num_timesteps = 0
    return cb, recorded


def _outcome_info(won: bool, rank: int, **stats) -> dict:
    info = {"outcome": {"won": won, "rank": rank}}
    if stats:
        info["episode_stats"] = {
            "kills": 0, "damage_dealt": 0.0, "damage_taken": 0.0, "cubes": 0, "shots_fired": 0,
            **stats,
        }
    return info


def test_training_monitor_reports_empty_defaults_before_any_episode():
    cb, recorded = _monitor()
    cb.locals = {"infos": [{}]}
    cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/win_rate"] == 0.0
    import math
    assert math.isnan(recorded["train/mean_rank"])
    assert recorded["train/episodes_total"] == 0
    assert recorded["train/episodes_window"] == 0


def test_training_monitor_computes_win_rate_and_episode_stats():
    cb, recorded = _monitor()
    cb.locals = {"infos": [
        _outcome_info(True, 0, kills=2, damage_dealt=1000.0, damage_taken=200.0, cubes=3, shots_fired=5),
        _outcome_info(False, 3, kills=0, damage_dealt=100.0, damage_taken=800.0, cubes=1, shots_fired=2),
    ]}
    cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/win_rate"] == pytest.approx(0.5)
    assert recorded["train/mean_rank"] == pytest.approx(1.5)
    assert recorded["train/episodes_total"] == 2
    assert recorded["train/episodes_window"] == 2
    assert recorded["train/kills_mean"] == pytest.approx(1.0)
    assert recorded["train/damage_dealt_mean"] == pytest.approx(550.0)
    assert recorded["train/damage_taken_mean"] == pytest.approx(500.0)
    assert recorded["train/cubes_mean"] == pytest.approx(2.0)
    assert recorded["train/shots_fired_mean"] == pytest.approx(3.5)


def test_training_monitor_never_resets_unlike_the_curriculum_window():
    """The whole reason this is a SEPARATE callback: CurriculumCallback's own win_rate is
    deliberately cleared on every stage transition (see its module docstring), but the
    training-visibility signal must survive that -- it answers a different question."""
    cb, recorded = _monitor(window_episodes=100)
    for _ in range(5):
        cb.locals = {"infos": [_outcome_info(True, 0)]}
        cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/episodes_total"] == 5
    assert recorded["train/episodes_window"] == 5

    # Simulate what a curriculum stage transition does to ITS OWN window -- nothing here should
    # touch this callback's state, since nothing ever calls a reset on it.
    for _ in range(3):
        cb.locals = {"infos": [_outcome_info(False, 2)]}
        cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/episodes_total"] == 8   # cumulative, never zeroed
    assert recorded["train/episodes_window"] == 8


def test_training_monitor_window_is_rolling_not_unbounded():
    cb, recorded = _monitor(window_episodes=4)
    for won in (True, True, True, True, False, False):
        cb.locals = {"infos": [_outcome_info(won, 0 if won else 3)]}
        cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/episodes_window"] == 4          # capped at the window size
    assert recorded["train/episodes_total"] == 6            # but the cumulative count is not
    assert recorded["train/win_rate"] == pytest.approx(0.5)  # only the last 4: T,T,F,F


def test_training_monitor_ignores_infos_without_an_outcome():
    cb, recorded = _monitor()
    cb.locals = {"infos": [{}, {"TimeLimit.truncated": False}, _outcome_info(True, 0)]}
    cb._on_step()
    cb._on_rollout_end()
    assert recorded["train/episodes_total"] == 1


def test_training_monitor_tolerates_missing_episode_stats():
    """`episode_stats` is only ever present alongside `outcome` in real infos, but this callback
    must not crash if some other caller constructs an outcome without it."""
    cb, recorded = _monitor()
    cb.locals = {"infos": [{"outcome": {"won": True, "rank": 0}}]}
    cb._on_step()
    cb._on_rollout_end()
    import math
    assert recorded["train/episodes_total"] == 1
    assert math.isnan(recorded["train/kills_mean"])


# ---------------------------------------------------------------------------
# curriculum callback (advancement policy)
# ---------------------------------------------------------------------------


def _callback(ccfg, mgr, **kw):
    """`BaseCallback.logger` is a read-only property proxying `self.model.logger`, and
    `num_timesteps` is a plain attribute the training loop refreshes -- so a stub model is
    enough to drive `_on_step` directly, without standing up a real algorithm."""
    cb = CurriculumCallback(mgr, ccfg, verbose=0, **kw)
    cb.model = SimpleNamespace(logger=SimpleNamespace(record=lambda *a, **k: None))
    cb.num_timesteps = 0
    return cb


def _feed(cb, wins, losses, rank_for_loss=2):
    cb.locals = {"infos": (
        [{"outcome": {"won": True, "rank": 0}}] * wins
        + [{"outcome": {"won": False, "rank": rank_for_loss}}] * losses
    )}
    cb._on_step()


def _cc(**kw):
    defaults = dict(window_episodes=10, min_episodes_at_stage=10,
                    tiers={"easy": DifficultyTier("easy"), "hard": DifficultyTier("hard")},
                    stages=(CurriculumStage("a", {"easy": 1.0}, advance_win_rate=0.5),
                            CurriculumStage("b", {"hard": 1.0}, advance_win_rate=0.5),
                            CurriculumStage("c", {"hard": 1.0})))
    defaults.update(kw)
    return CurriculumConfig(**defaults)


def _mgr_for(ccfg):
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    return CurriculumManager(ccfg, device="cpu", gen=gen)


def test_callback_advances_once_the_win_rate_clears_the_bar():
    ccfg = _cc(); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    _feed(cb, wins=6, losses=4)
    assert mgr.stage.name == "b"
    assert cb.history[0]["how"] == "advanced"
    assert cb.episodes_at_stage == 0 and len(cb._window) == 0   # evidence reset on transition


def test_callback_waits_for_min_episodes():
    ccfg = _cc(min_episodes_at_stage=10); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    _feed(cb, wins=9, losses=0)      # 90% win rate but only 9 episodes
    assert mgr.stage.name == "a"
    _feed(cb, wins=1, losses=0)      # 10th episode
    assert mgr.stage.name == "b"


def test_callback_does_not_advance_below_the_bar():
    ccfg = _cc(); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    _feed(cb, wins=4, losses=6)
    assert mgr.stage.name == "a" and cb.win_rate == pytest.approx(0.4)


def test_callback_never_advances_past_the_terminal_stage():
    ccfg = _cc(); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    for _ in range(5):
        _feed(cb, wins=10, losses=0)
    assert mgr.stage.name == "c" and mgr.is_final_stage
    assert len([h for h in cb.history if h["how"] == "advanced"]) == 2


def test_callback_demotes_on_collapse():
    ccfg = _cc(demote_win_rate=0.1); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    _feed(cb, wins=6, losses=4)
    assert mgr.stage.name == "b"
    _feed(cb, wins=0, losses=10)
    assert mgr.stage.name == "a" and cb.history[-1]["how"] == "demoted"


def test_callback_force_advances_a_stalled_stage():
    ccfg = _cc(max_timesteps_at_stage=1000); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    _feed(cb, wins=0, losses=10)
    assert mgr.stage.name == "a"
    cb.num_timesteps = 1000
    _feed(cb, wins=0, losses=1)
    assert mgr.stage.name == "b" and "force-advanced" in cb.history[-1]["how"]


def test_callback_writes_a_resumable_state_file(tmp_path):
    ccfg = _cc(); mgr = _mgr_for(ccfg)
    cb = _callback(ccfg, mgr, state_path=tmp_path / "curriculum.json")
    _feed(cb, wins=6, losses=4)
    state = json.loads((tmp_path / "curriculum.json").read_text())
    assert state["stage_name"] == "b" and len(state["history"]) == 1

    restored = _mgr_for(ccfg)
    restored.load_state_dict(state)
    assert restored.stage.name == "b"


def test_callback_ignores_infos_without_an_outcome():
    """Only envs that FINISHED this tick carry info["outcome"]; every other env's dict must not
    be miscounted as an episode."""
    ccfg = _cc(); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    cb.locals = {"infos": [{}, {"TimeLimit.truncated": False}, {"outcome": {"won": True, "rank": 0}}]}
    cb._on_step()
    assert cb.total_episodes == 1


# ---------------------------------------------------------------------------
# SB3 seam
# ---------------------------------------------------------------------------

def test_sb3_vecenv_emits_the_outcome_the_curriculum_needs():
    from brawl_sim.core import obs_select
    from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv
    import numpy as np

    cfg, env = _tiny_env(n_envs=8)
    spec = obs_select.load_agent_spec(REPO_ROOT / "configs" / "agent_obs.yaml", cfg)
    venv = BrawlSB3VecEnv(env, spec, ShapedReward(RewardConfig()), info_mode="episode")
    venv.reset()

    seen, seen_stats = [], []
    for _ in range(cfg.max_episode_steps + 1):
        _, _, dones, infos = venv.step(np.zeros((8, 2), dtype=np.int64))
        seen += [infos[i]["outcome"] for i in range(8) if dones[i]]
        seen_stats += [infos[i]["episode_stats"] for i in range(8) if dones[i]]
        if seen:
            break
    assert seen, "no episode finished within the step cap"
    for outcome in seen:
        assert set(outcome) == {"rank", "won"}
        assert 0 <= outcome["rank"] < cfg.n_entities
        assert outcome["won"] == (outcome["rank"] == 0)
    for stats in seen_stats:
        assert set(stats) == {"kills", "damage_dealt", "damage_taken", "cubes", "shots_fired"}
        for v in stats.values():
            assert v >= 0


def test_sb3_vecenv_seed_reseeds_the_shared_generator():
    """SB3's `set_random_seed` calls `env.seed(seed)` whenever a model is built with `seed=`;
    this used to raise and made a seeded MaskablePPO impossible to construct."""
    from brawl_sim.core import obs_select
    from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

    cfg, env = _tiny_env(n_envs=8)
    spec = obs_select.load_agent_spec(REPO_ROOT / "configs" / "agent_obs.yaml", cfg)
    venv = BrawlSB3VecEnv(env, spec, ShapedReward(RewardConfig()), info_mode="episode")
    assert venv.seed(123) == [123] * 8
    a = torch.rand(4, generator=env.gen)
    venv.seed(123)
    assert torch.equal(a, torch.rand(4, generator=env.gen))


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# stationary evaluation
# ---------------------------------------------------------------------------

from brawl_sim.training.config import EvalConfig  # noqa: E402
from brawl_sim.training.curriculum import FixedTierHook  # noqa: E402
from brawl_sim.training.evaluation import METRICS, TierEvaluator  # noqa: E402


def test_fixed_tier_hook_pins_every_env_to_one_tier():
    tiers = {"easy": DifficultyTier("easy", hp=0.5), "hard": DifficultyTier("hard")}
    cfg, env = _tiny_env(n_envs=32)
    base = env.params.base_hp[:, 1:].clone()
    env.params_hook = FixedTierHook.uniform(tiers, "cpu", 32, "easy")
    env.reset()
    assert torch.allclose(env.params.base_hp[:, 1:], base * 0.5)


def test_fixed_tier_hook_block_assignment_scores_several_tiers_at_once():
    """The evaluator's core trick: one batched env, contiguous blocks pinned to different
    tiers, so every difficulty is scored in a single rollout."""
    tiers = {"easy": DifficultyTier("easy", hp=0.5), "hard": DifficultyTier("hard")}
    cfg, env = _tiny_env(n_envs=8)
    base = env.params.base_hp[:, 1:].clone()
    assignment = torch.arange(8) // 4          # envs 0-3 easy, 4-7 hard
    env.params_hook = FixedTierHook(tiers, "cpu", assignment)
    env.reset()
    assert torch.allclose(env.params.base_hp[:4, 1:], base[:4] * 0.5)
    assert torch.allclose(env.params.base_hp[4:, 1:], base[4:])


def test_fixed_tier_hook_is_deterministic_across_resets():
    """No RNG at all -- two identically seeded envs must produce identical bot stats, which is
    what makes an eval number comparable between two checkpoints."""
    tiers = {"hard": DifficultyTier("hard", aim_noise=0.5)}
    out = []
    for _ in range(2):
        cfg, env = _tiny_env(n_envs=16, seed=3)
        env.params_hook = FixedTierHook.uniform(tiers, "cpu", 16, "hard")
        env.reset()
        out.append(env.params.aim_noise_std_rad.clone())
    assert torch.equal(out[0], out[1])


def test_fixed_tier_hook_rejects_unknown_tier():
    with pytest.raises(ValueError):
        FixedTierHook.uniform({"hard": DifficultyTier("hard")}, "cpu", 4, "nope")


def test_eval_config_rejects_undefined_tier(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "eval: {tiers: [nope]}\n"
        "curriculum:\n"
        "  tiers: {hard: {}}\n"
        "  stages: [{name: only, tier_weights: {hard: 1.0}}]\n"
    )
    with pytest.raises(ValueError, match="undefined tier"):
        load_train_config(path)


def test_eval_requires_tiers_to_exist(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("eval: {enabled: true}\ncurriculum: {enabled: false}\n")
    with pytest.raises(ValueError, match="curriculum.tiers"):
        load_train_config(path)


def _eval_tcfg(episodes_per_tier=2, tiers=("easy", "hard")):
    return load_train_config(TRAIN_CONFIG, overrides={
        "run": {"device": "cpu", "env_overrides": yaml.safe_load(DEBUG_TINY.read_text())},
        "eval": {"episodes_per_tier": episodes_per_tier, "tiers": list(tiers)},
    })


class _StubModel:
    """Returns idle actions. Enough to drive the evaluator's plumbing without training
    anything -- what's under test here is the accounting, not the policy."""
    def __init__(self):
        self.saved = []
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True, action_masks=None):
        n = next(iter(obs.values())).shape[0]
        return np.zeros((n, 2), dtype=np.int64), None

    def get_vec_normalize_env(self):
        return None

    def save(self, path):
        self.saved.append(str(path))


def test_evaluator_reports_every_tier_with_the_requested_episode_count():
    tcfg = _eval_tcfg(episodes_per_tier=3, tiers=("easy", "hard"))
    ev = TierEvaluator(tcfg)
    assert ev.n_envs == 6
    results = ev.evaluate(_StubModel())
    ev.close()

    assert set(results) == {"easy", "hard"}
    for name, r in results.items():
        assert r["episodes"] == 3, f"{name} scored {r['episodes']} episodes, expected 3"
        for metric in METRICS:
            assert np.isfinite(r[metric]), f"{name}.{metric} is not finite"
        assert 0.0 <= r["win_rate"] <= 1.0


def test_evaluator_is_reproducible_across_calls():
    """Same seed, same scenarios: two evaluations of the SAME policy must agree exactly, or the
    metric is measuring env noise rather than the policy."""
    tcfg = _eval_tcfg()
    ev = TierEvaluator(tcfg)
    model = _StubModel()
    first, second = ev.evaluate(model), ev.evaluate(model)
    ev.close()
    assert first == second


def test_evaluator_is_unaffected_by_the_training_curriculum():
    """The whole point of "stationary": advancing the training curriculum must not move the
    eval env's bots, because the two use separate hooks on separate envs."""
    tcfg = _eval_tcfg()
    ev = TierEvaluator(tcfg)
    model = _StubModel()
    before = ev.evaluate(model)

    # Walk a TRAINING curriculum all the way to its hardest stage. It shares the tier
    # definitions but not the env, so it must not move these numbers by one bit.
    gen = torch.Generator(device="cpu"); gen.manual_seed(0)
    training_mgr = CurriculumManager(tcfg.curriculum, device="cpu", gen=gen)
    while training_mgr.advance():
        pass
    assert training_mgr.is_final_stage

    after = ev.evaluate(model)
    ev.close()

    assert isinstance(ev.sim.params_hook, FixedTierHook)
    assert ev.sim.params_hook is not training_mgr
    assert before == after, "advancing the training curriculum changed the eval numbers"


def test_eval_callback_logs_per_tier_scalars_and_tier_event_dirs(tmp_path):
    from brawl_sim.training.callbacks import TierEvalCallback

    tcfg = _eval_tcfg()
    ev = TierEvaluator(tcfg)
    recorded = {}
    cb = TierEvalCallback(ev, every_timesteps=1000, log_dir=tmp_path,
                          at_start=True, best_model_path=tmp_path / "best_model.zip", verbose=0)
    model = _StubModel()
    model.logger = SimpleNamespace(record=lambda k, v, **kw: recorded.__setitem__(k, v))
    cb.model = model
    cb.num_timesteps = 0
    cb.locals = {}

    cb._on_training_start()
    cb._on_step()
    cb._on_training_end()
    ev.close()

    # (1) named scalars -> separate TB charts and progress.csv columns
    for tier in ("easy", "hard"):
        for metric in METRICS:
            assert f"eval/{metric}_{tier}" in recorded
    assert "eval/win_rate_mean" in recorded

    # (2) one event-file directory per tier -> a single overlaid eval/win_rate chart
    for tier in ("easy", "hard"):
        d = tmp_path / f"eval_{tier}"
        assert d.is_dir() and list(d.glob("events.out.tfevents.*")), f"no event file in {d}"

    assert model.saved, "best_model was never saved"


def test_eval_callback_fires_once_per_interval_not_once_per_overshoot(tmp_path):
    """num_timesteps advances n_envs at a time, so a rollout can jump past several intervals at
    once. A fixed grid would then fire repeatedly to catch up, burning minutes of eval on
    identical weights."""
    from brawl_sim.training.callbacks import TierEvalCallback

    tcfg = _eval_tcfg()
    ev = TierEvaluator(tcfg)
    calls = []
    cb = TierEvalCallback(ev, every_timesteps=100, log_dir=tmp_path, at_start=False, verbose=0)
    cb.model = SimpleNamespace(logger=SimpleNamespace(record=lambda *a, **k: None))
    cb._evaluate = lambda: calls.append(cb.num_timesteps)

    cb.num_timesteps = 0
    cb._on_training_start()
    for t in (50, 1000, 1050, 1100, 1101):
        cb.num_timesteps = t
        cb._on_step()
    ev.close()
    assert calls == [1000, 1100], f"expected two evals, got {calls}"


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_script_smoke_then_watch_the_checkpoint(tmp_path):
    """The whole pipeline in one pass, since training is the expensive part:
    config -> env + curriculum hook -> MaskablePPO -> learn (with stationary eval) -> save,
    then load the saved checkpoint back and play a watchable match against a pinned tier."""
    pytest.importorskip("sb3_contrib")
    from scripts.train import main as train_main

    assert train_main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    run_dir = tmp_path / "smoke"
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "train.yaml").exists()
    assert (run_dir / "best_model.zip").exists(), "stationary eval never saved a best model"
    assert json.loads((run_dir / "curriculum.json").read_text())["n_stages"] == 5

    # eval columns reached the machine-readable log, and each tier got its own event dir
    header = (run_dir / "logs" / "progress.csv").read_text().splitlines()[0]
    for tier in ("easy", "medium", "hard", "elite"):
        assert f"eval/win_rate_{tier}" in header
        assert list((run_dir / "logs" / f"eval_{tier}").glob("events.out.tfevents.*"))

    # continuous training-side visibility (TrainingMonitorCallback) reached the same log
    for col in ("train/win_rate", "train/mean_rank", "train/episodes_total",
                "train/kills_mean", "train/damage_dealt_mean"):
        assert col in header

    # ...and the checkpoint is watchable
    from scripts.watch import main as watch_main
    assert watch_main([str(run_dir / "best_model.zip"), "--tier", "hard", "--no-view",
                       "--save", str(tmp_path / "match.npz")]) == 0
    assert (tmp_path / "match.npz").exists()


@pytest.mark.slow
def test_watch_builds_a_real_matplotlib_view(tmp_path):
    """`watch.py`'s viewer path, exercised without a GUI: the frames it records must be exactly
    what ReplayViewer consumes (the same contract record_rollout produces)."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("sb3_contrib")
    import matplotlib
    matplotlib.use("Agg")
    from brawl_sim.render.viewer import ReplayViewer
    from scripts.watch import build_watch_env, play_match

    tcfg = _eval_tcfg()
    sim, venv, env_cfg = build_watch_env(tcfg, "hard", seed=0, device="cpu")
    frames, summary = play_match(_StubModel(), venv, sim, uses_masks=False,
                                 deterministic=True, max_steps=env_cfg.max_agent_steps)

    # One frame per SIM TICK (+1 for the post-reset frame), NOT one per decision: watch.py
    # captures through BrawlVecEnv.tick_hook so replays stay at the sim's own 20 Hz whatever
    # action_repeat is. `summary["steps"]` counts decisions, hence the multiplier.
    assert frames["ent_pos"].shape[0] == summary["steps"] * env_cfg.action_repeat + 1
    assert frames["ent_pos"].shape[1] == env_cfg.n_entities
    assert set(summary) >= {"steps", "kills", "damage_dealt", "death_cause", "alive"}
    assert sim.tick_hook is None, "play_match must uninstall its hook before returning"

    viewer = ReplayViewer(frames, sim.bank, env_cfg, fps=20)
    viewer._draw_frame(0)
    viewer._draw_frame(frames["ent_pos"].shape[0] - 1)


@pytest.mark.slow
def test_watch_rejects_a_mismatched_train_config(tmp_path):
    """A model trained under a different observation layout must fail with a pointer at the
    config, not a bare shape error from inside the policy's first Linear layer."""
    from scripts.watch import find_train_config
    with pytest.raises(FileNotFoundError, match="no train.yaml"):
        find_train_config(tmp_path / "nowhere" / "model.zip")
