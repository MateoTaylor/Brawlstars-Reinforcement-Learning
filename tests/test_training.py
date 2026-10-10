"""Tests for brawl_sim/training/ -- config, schedules, shaped reward, and the curriculum.

Follows CONVENTIONS.md's testing rule: CPU by default, `n_envs = 8` with
`presets/debug_tiny.yaml` for anything that needs a real simulator. The curriculum-sampling
tests deliberately break the n_envs=8 rule and use a few thousand envs instead -- they assert a
*distribution* matches its configured weights, which needs a sample size, and the tiny-map sim
is cheap enough on CPU that it costs a fraction of a second.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.constants import N_KINDS
from brawl_sim.env import BrawlVecEnv
from brawl_sim.training import schedules
from brawl_sim.training.config import (
    TIER_FIELDS, CurriculumConfig, CurriculumStage, DifficultyTier, RewardConfig,
    ScheduleConfig, load_train_config, parse_overrides,
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


def test_shipped_tiers_are_listed_weakest_to_strongest(tcfg):
    """The walk test below reads "stronger" off the ORDER the tiers are listed in, so hold every
    knob to that order: HP, damage, speed, leading, aggression and hero focus never fall, noise,
    reaction and decision period never rise, and each tier moves at least one of them."""
    tiers = list(tcfg.curriculum.tiers.values())
    for weaker, stronger in zip(tiers, tiers[1:]):
        pair = f"{weaker.name!r} -> {stronger.name!r}"
        for knob in ("hp", "damage", "move_speed", "lead_target", "aggression", "hero_focus"):
            assert getattr(stronger, knob) >= getattr(weaker, knob), f"{knob} falls at {pair}"
        for knob in ("aim_noise", "reaction_delay", "decision_period"):
            assert getattr(stronger, knob) <= getattr(weaker, knob), f"{knob} rises at {pair}"
        assert stronger != replace(weaker, name=stronger.name), f"{pair} changes nothing"


def test_shipped_elite_tier_is_the_operator_spec(tcfg):
    """1.5x health and damage and no more (BRAWL_SIM_DESIGN.md §11, 2026-09-18; was 2x / 1.75x), with
    the difficulty carried by accuracy, aggression and hero focus instead. Asked for, not tuned."""
    elite = tcfg.curriculum.tiers["elite"]
    assert (elite.hp, elite.damage) == (1.5, 1.5)
    assert (elite.aim_noise, elite.reaction_delay) == (0.20, 0.30)
    assert (elite.aggression, elite.hero_focus) == (1.7, 1.7)
    assert elite.lead_target >= 1.0
    assert list(tcfg.curriculum.tiers)[-1] == "elite", "elite is the strongest tier"
    # The other half: every lower tier sits at or below the 1.5x cap.
    for tier in tcfg.curriculum.tiers.values():
        assert tier.hp <= 1.5 and tier.damage <= 1.5, f"{tier.name!r} is past the 1.5x cap"


def test_shipped_tier_table_is_the_plan_table(tcfg):
    """The shipped tier table as literals, so a retune is a deliberate edit here too. Column order:
    aim_noise, reaction_delay, lead_target, decision_period, move_speed, hp, damage, aggression,
    hero_focus."""
    table = {
        "easy":    (2.5, 2.2,  0.25, 2.0,  0.90, 0.75, 0.70, 0.6, 0.0),
        "medium":  (1.8, 1.6,  0.60, 1.5,  0.95, 0.90, 0.85, 0.8, 0.4),
        "hard":    (1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0),
        "veteran": (0.6, 0.6,  1.0,  1.0,  1.0,  1.15, 1.15, 1.2, 1.3),
        "expert":  (0.4, 0.45, 1.25, 0.75, 1.0,  1.30, 1.30, 1.4, 1.5),
        "elite":   (0.2, 0.30, 1.5,  0.5,  1.0,  1.50, 1.50, 1.7, 1.7),
    }
    assert list(tcfg.curriculum.tiers) == list(table)
    for name, row in table.items():
        t = tcfg.curriculum.tiers[name]
        got = (t.aim_noise, t.reaction_delay, t.lead_target, t.decision_period, t.move_speed,
               t.hp, t.damage, t.aggression, t.hero_focus)
        assert got == row, f"tier {name!r} is {got}, the pinned row is {row}"


def test_shipped_stage_walk_is_the_plan_walk(tcfg):
    """The shipped walk as literals: stage names, every mixture, the 0.15 gates, the 2000-episode
    window and the 125M cap.

    The two structural tests around this one (thresholds non-decreasing, shares never falling)
    read every expectation off the loaded config, so a bumped gate or weight that keeps the walk
    monotone passes both. These ARE tuning knobs: a deliberate retune edits this table in the same
    change, exactly as it edits the tier table above. What this stops is an accidental one."""
    curriculum = tcfg.curriculum
    walk = [(s.name, s.tier_weights, s.advance_win_rate) for s in curriculum.stages]
    assert walk == [
        ("hard_intro",    {"medium": 0.3, "hard": 0.7}, 0.15),
        ("veteran_intro", {"hard": 0.6, "veteran": 0.4}, 0.15),
        ("veteran",       {"hard": 0.3, "veteran": 0.5, "expert": 0.2}, 0.15),
        ("expert",        {"hard": 0.1, "veteran": 0.3, "expert": 0.4, "elite": 0.2}, 0.15),
        ("elite",         {"hard": 0.1, "veteran": 0.15, "expert": 0.25, "elite": 0.5}, None),
    ]
    assert curriculum.enabled is True
    assert (curriculum.window_episodes, curriculum.min_episodes_at_stage) == (2000, 2000)
    assert curriculum.max_timesteps_at_stage == 125_000_000
    assert curriculum.demote_win_rate is None


def test_tier_rejects_an_aggression_of_zero():
    """The bots read an aggression of 0 as the neutral 1.0 (it is what a missing brawlers.yaml
    key resolves to), so a tier that multiplied by 0 would come out as aggressive as `hard` --
    the opposite of what it says. hero_focus 0 is fine: 0 genuinely means "no preference"."""
    with pytest.raises(ValueError, match="aggression must be > 0"):
        DifficultyTier("timid", aggression=0.0)
    with pytest.raises(ValueError, match="hero_focus must be >= 0"):
        DifficultyTier("shy", hero_focus=-0.1)
    assert DifficultyTier("easy", aggression=0.6, hero_focus=0.0).hero_focus == 0.0


def test_shipped_curriculum_walks_easy_to_hard(tcfg):
    """The user-facing shape of the default curriculum: every stage's mixture is at least as
    strong as the one before it, and the terminal stage is mostly the strongest tier.

    "At least as strong" is checked at every tier boundary: for each tier, the share of bots at
    that tier OR ABOVE never falls from one stage to the next. That covers the two conditions this
    test used to spell out with tier names -- the weakest tier's share never grows, and the
    strongest tiers' combined share never falls -- for any number of tiers.

    Deliberately does NOT require the terminal stage to be a single pure tier. Keeping some
    weaker opponents in the final stage is an anti-overfitting choice, not a regression.
    """
    names = list(tcfg.curriculum.tiers)
    stages = tcfg.curriculum.stages

    def at_or_above(stage, k):
        total = sum(stage.tier_weights.values())
        return sum(stage.tier_weights.get(n, 0.0) for n in names[k:]) / total

    for k, name in enumerate(names[1:], start=1):
        shares = [round(at_or_above(s, k), 9) for s in stages]
        assert shares == sorted(shares), (
            f"the share of bots at `{name}` or above must never fall across the walk, got {shares}"
        )
    last = stages[-1]
    assert max(last.tier_weights, key=last.tier_weights.get) == names[-1], (
        f"the terminal stage's largest share must be the strongest tier, `{names[-1]}`"
    )
    assert names[0] not in last.tier_weights, f"the terminal stage must contain no `{names[0]}` bots"


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


def test_a_min_episodes_floor_the_window_cannot_hold_is_rejected():
    """The callback needs `min_episodes_at_stage` outcomes IN the window as well as at the stage,
    so a floor above the window size would leave every stage to its timestep budget."""
    with pytest.raises(ValueError, match="must not exceed"):
        CurriculumConfig(
            window_episodes=100, min_episodes_at_stage=101,
            tiers={"easy": DifficultyTier("easy")},
            stages=(CurriculumStage("s", {"easy": 1.0}),),
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
        "attack_in_reach_tick": torch.zeros(n, dtype=torch.int32, device=device),
        "gadget_hit_tick": torch.zeros(n, dtype=torch.int32, device=device),
        "move_reversal_tick": torch.zeros(n, dtype=torch.int32, device=device),
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
                               decision_period=2.0, move_speed=0.9, hp=0.5, damage=0.5,
                               aggression=0.6, hero_focus=0.4),
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
                        "reaction_delay", "lead_target_fraction", "decision_period",
                        "aggression", "hero_focus")}
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
    # The two behaviour axes, as literals: brawlers.yaml authors `aggression: 1.0` and
    # `hero_focus: 0.5` on every bot kind and neither on the hero, and the `easy` tier above
    # multiplies them by 0.6 and 0.4.
    aggression, focus = env.params.aggression, env.params.hero_focus
    assert torch.allclose(aggression[:, 1:], torch.full_like(aggression[:, 1:], 0.6))
    assert torch.allclose(focus[:, 1:], torch.full_like(focus[:, 1:], 0.2))
    assert torch.equal(aggression[:, 0], torch.zeros(64))
    assert torch.equal(focus[:, 0], torch.zeros(64))


def _flat_params(n_envs, **values):
    """A stand-in for SimParams holding every per-kind tensor `apply_tiers` writes, each one a
    single value across ALL columns -- the hero's included. The real env cannot show that the
    hero column is left alone on these two axes (the hero's own aggression and hero_focus are 0,
    and 0 times anything is still 0), so this builds a hero column a leak WOULD move."""
    columns = {"aim_noise_std_rad": 0.1, "aim_noise_tiles": 0.5, "reaction_delay": 0.2,
               "move_speed": 2.5, "base_hp": 6000.0, "base_damage": 1000.0,
               "lead_target_fraction": 0.7, "aggression": 1.0, "hero_focus": 0.5}
    columns.update(values)
    params = SimpleNamespace(**{name: torch.full((n_envs, N_KINDS), float(value))
                                for name, value in columns.items()})
    params.decision_period = torch.full((n_envs, N_KINDS), 4, dtype=torch.int64)
    return params


def test_manager_scales_aggression_and_hero_focus_on_bot_columns_only():
    """The elite multipliers on the authored bases: aggression 1.0 -> 1.7, hero_focus 0.5 -> 0.85,
    and a hero column that holds the SAME non-zero bases comes through untouched."""
    tiers = {"hard": DifficultyTier("hard"),
             "elite": DifficultyTier("elite", aggression=1.7, hero_focus=1.7)}
    mgr, _ = _manager({"elite": 1.0}, tiers=tiers)
    params = _flat_params(4)
    mgr(params, torch.ones(4, dtype=torch.bool))
    assert torch.allclose(params.aggression[:, 1:], torch.full((4, N_KINDS - 1), 1.7))
    assert torch.allclose(params.hero_focus[:, 1:], torch.full((4, N_KINDS - 1), 0.85))
    assert torch.equal(params.aggression[:, 0], torch.full((4,), 1.0))
    assert torch.equal(params.hero_focus[:, 0], torch.full((4,), 0.5))


def test_manager_clamps_hero_focus_to_one():
    """`select_target` scales the hero's distance by (1 - focus). A base of 0.7 under elite's 1.7
    is 1.19, which would make that distance NEGATIVE and pick the hero from anywhere, ahead of a
    bot standing on the observer; the product clamps to 1.0 ("the hero whenever visible")."""
    tiers = {"hard": DifficultyTier("hard"),
             "elite": DifficultyTier("elite", aggression=1.7, hero_focus=1.7)}
    mgr, _ = _manager({"elite": 1.0}, tiers=tiers)
    params = _flat_params(4, hero_focus=0.7, aggression=2.0)
    params.hero_focus[:, 2] = 0.0      # an unset (neutral) base stays unset under any multiplier
    params.aggression[:, 2] = 0.0
    # The hero column holds values a clamp to [0, 1] WOULD move (0.7 would come through a
    # whole-tensor clamp unchanged and prove nothing), on both fractions that share the loop.
    params.hero_focus[:, 0] = 1.3
    params.lead_target_fraction[:, 0] = 1.3
    mgr(params, torch.ones(4, dtype=torch.bool))
    assert torch.equal(params.hero_focus[:, 1], torch.full((4,), 1.0))
    assert torch.equal(params.hero_focus[:, 2], torch.zeros(4))
    assert torch.equal(params.hero_focus[:, 0], torch.full((4,), 1.3))   # hero: not even clamped
    assert torch.equal(params.lead_target_fraction[:, 0], torch.full((4,), 1.3))
    # aggression is NOT a fraction and is not clamped: 2.0 x 1.7 = 3.4 (personality.py clamps
    # each of its reads). A 0 base stays 0, which the bots read as the neutral 1.0.
    assert torch.allclose(params.aggression[:, 1], torch.full((4,), 3.4))
    assert torch.equal(params.aggression[:, 2], torch.zeros(4))


def test_every_tier_field_reaches_the_applier():
    """A knob added to TIER_FIELDS but not to curriculum.py's target lists would load, validate,
    and then do nothing at all. Every field must be a DifficultyTier attribute AND have a
    multiplier table in the applier."""
    mgr, _ = _manager({"easy": 1.0})
    assert set(TIER_FIELDS) == set(mgr._tables)
    assert set(TIER_FIELDS) == set(DifficultyTier.__dataclass_fields__) - {"name"}
    assert {"aggression", "hero_focus"} <= set(TIER_FIELDS)
    assert TIER_FIELDS["aggression"] == 1.0 and TIER_FIELDS["hero_focus"] == 1.0


def test_a_tier_that_names_no_knob_is_the_bots_as_authored(tmp_path):
    """B4.1's "default 1.0", pinned where it is LIVE. Nothing reads TIER_FIELDS' values (only its
    keys), so the assertion above cannot see the real default: the dataclass's own. A default of
    0.5 there would halve the aggression and hero focus of every tier that omits the two keys --
    `hard: {}`, which DifficultyTier's docstring advertises, and every tier in a pre-B4 config."""
    tier = DifficultyTier("x")
    assert (tier.aggression, tier.hero_focus) == (1.0, 1.0)
    assert [getattr(tier, key) for key in TIER_FIELDS] == [1.0] * 9

    path = tmp_path / "t.yaml"
    path.write_text("curriculum:\n  enabled: false\n  tiers:\n    hard: {}\n"
                    "    old: {hp: 2.0, aim_noise: 0.25}\neval:\n  enabled: false\n")
    tiers = load_train_config(path).curriculum.tiers
    assert (tiers["hard"].aggression, tiers["hard"].hero_focus) == (1.0, 1.0)
    assert (tiers["old"].aggression, tiers["old"].hero_focus, tiers["old"].hp) == (1.0, 1.0, 2.0)

    # ...and applied: the authored bases come through `hard: {}` as the same literals.
    mgr, _ = _manager({"hard": 1.0}, tiers={"hard": tiers["hard"]})
    params = _flat_params(4)
    mgr(params, torch.ones(4, dtype=torch.bool))
    assert torch.equal(params.aggression, torch.full((4, N_KINDS), 1.0))
    assert torch.equal(params.hero_focus, torch.full((4, N_KINDS), 0.5))


def test_manager_writes_aggression_and_hero_focus_on_reset_rows_only():
    """Every env's autoreset calls the hook with a PARTIAL mask. A write that ignored it would
    re-multiply every live env's bots on every reset anywhere in the batch (1.7, 2.89, ...)."""
    tiers = {"hard": DifficultyTier("hard"),
             "elite": DifficultyTier("elite", aggression=1.7, hero_focus=1.7)}
    mgr, _ = _manager({"elite": 1.0}, tiers=tiers)
    params = _flat_params(4)
    mgr(params, torch.tensor([True, False, True, False]))
    for row in (1, 3):      # not reset: every column still the base
        assert torch.equal(params.aggression[row], torch.full((N_KINDS,), 1.0))
        assert torch.equal(params.hero_focus[row], torch.full((N_KINDS,), 0.5))
    for row in (0, 2):      # reset: bots scaled, hero not
        assert torch.allclose(params.aggression[row, 1:], torch.full((N_KINDS - 1,), 1.7))
        assert torch.allclose(params.hero_focus[row, 1:], torch.full((N_KINDS - 1,), 0.85))
        assert float(params.aggression[row, 0]) == 1.0
        assert float(params.hero_focus[row, 0]) == 0.5


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
    written = ("aim_noise_std_rad", "aim_noise_tiles", "reaction_delay", "move_speed", "base_hp",
               "base_damage", "aggression", "lead_target_fraction", "hero_focus", "decision_period")
    snapshot = {attr: getattr(env.params, attr).clone() for attr in written}
    mask = torch.zeros(8, dtype=torch.bool)
    mask[3] = True
    env.reset(reset_mask=mask)
    untouched = [i for i in range(8) if i != 3]
    for attr in written:    # every tensor apply_tiers rebinds, not base_hp alone
        assert torch.equal(getattr(env.params, attr)[untouched], snapshot[attr][untouched]), attr
    # Literals for the two behaviour axes under `_manager`'s easy tier (0.6 x 1.0, 0.4 x 0.5): an
    # un-gated write would leave the seven un-reset rows at 0.36 / 0.08 after this second reset,
    # and the reset row compounds to neither (resample_params rewrites its base first).
    assert torch.allclose(env.params.aggression[:, 1:], torch.full((8, N_KINDS - 1), 0.6))
    assert torch.allclose(env.params.hero_focus[:, 1:], torch.full((8, N_KINDS - 1), 0.2))


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
    _feed(cb, wins=0, losses=10)     # each env's first finish at "b" still had "a"'s bots
    assert mgr.stage.name == "b"
    _feed(cb, wins=0, losses=10)
    assert mgr.stage.name == "a" and cb.history[-1]["how"] == "demoted"


def test_callback_skips_episodes_whose_bots_came_from_the_previous_stage():
    """Tiers are drawn at reset, so the episodes in flight when a stage changes finish under the
    OLD stage's bots. Each env's first finish after a transition is skipped; after that it counts.
    An env that has not finished yet stays owed, however many times the others finish."""
    ccfg = _cc(); mgr = _mgr_for(ccfg); cb = _callback(ccfg, mgr)
    # 10 envs. All win at "a", which advances it.
    _feed(cb, wins=10, losses=0)
    assert mgr.stage.name == "b" and cb.episodes_at_stage == 0

    # Envs 0-7 finish (still "a"'s bots), then 0-7 finish again at "b"; envs 8-9 have not finished.
    for _ in range(2):
        cb.locals = {"infos": [{"outcome": {"won": True, "rank": 0}}] * 8 + [{}, {}]}
        cb._on_step()
    assert cb.episodes_at_stage == 8 and cb.total_episodes == 26
    assert int(cb._stale.sum()) == 2
    assert mgr.stage.name == "b", "8 counted episodes are below min_episodes_at_stage=10"

    # Envs 8-9 finally finish. It is their first finish since the transition, so it is skipped.
    cb.locals = {"infos": [{}] * 8 + [{"outcome": {"won": True, "rank": 0}}] * 2}
    cb._on_step()
    assert cb.episodes_at_stage == 8 and not cb._stale.any()
    assert mgr.stage.name == "b"


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
from brawl_sim.training.config import deep_merge, resolved_training_maps  # noqa: E402
from brawl_sim.training.evaluation import METRICS, TierEvaluator, build_evaluators  # noqa: E402


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


@pytest.mark.parametrize("tier, aggression, hero_focus", [
    ("easy",    0.6, 0.00),
    ("medium",  0.8, 0.20),
    ("hard",    1.0, 0.50),
    ("veteran", 1.2, 0.65),
    ("expert",  1.4, 0.75),
    ("elite",   1.7, 0.85),
])
def test_each_shipped_tier_reaches_the_bots_as_the_plan_tables_effective_value(
        tcfg, tier, aggression, hero_focus):
    """The plan table's "->" column, end to end: configs/train.yaml's SHIPPED multipliers times
    configs/brawlers.yaml's SHIPPED bases (aggression 1.0, hero_focus 0.5), through the hook eval
    and `watch.py --tier` pin bots with, read off a real env's SimParams.

    Every other pin holds one factor of this product: the tier table above pins the multipliers,
    tests/test_configs_files.py the bases, and the manager tests the multiplication on a
    test-local tier. None applies a shipped tier past index 1 of the six by name, and none would
    notice the two files drifting apart -- a base moved to 0.6 makes elite 1.7 x 0.6 = 1.02,
    clamped to 1.0: "always the hero", with every other pin updated in good faith."""
    cfg, env = _tiny_env(n_envs=4)
    env.params_hook = FixedTierHook.uniform(tcfg.curriculum.tiers, "cpu", 4, tier)
    for reset in (1, 2):      # the second reset must not multiply the first one's product again
        env.reset()
        got_aggression, got_focus = env.params.aggression, env.params.hero_focus
        assert torch.allclose(got_aggression[:, 1:], torch.full_like(got_aggression[:, 1:], aggression)), (
            f"{tier}: bot aggression after reset {reset} is {got_aggression[0, 1:].tolist()}")
        assert torch.allclose(got_focus[:, 1:], torch.full_like(got_focus[:, 1:], hero_focus)), (
            f"{tier}: bot hero_focus after reset {reset} is {got_focus[0, 1:].tolist()}")
        assert torch.equal(got_aggression[:, 0], torch.zeros(4))      # the hero authors neither
        assert torch.equal(got_focus[:, 0], torch.zeros(4))


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
    # No holdout eval here: debug_tiny's world is 20x20 and `blank` is the only map that size, so
    # the shipped 60x60 holdout maps could not load. `_holdout_tcfg` below is the 60x60 twin.
    return load_train_config(TRAIN_CONFIG, overrides={
        "run": {"device": "cpu", "env_overrides": yaml.safe_load(DEBUG_TINY.read_text())},
        "eval": {"episodes_per_tier": episodes_per_tier, "tiers": list(tiers), "holdout_maps": None},
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
    # no holdout evaluator was given, so nothing may claim to be a holdout number
    assert not [k for k in recorded if "holdout" in k]
    assert "holdout_mean_win_rate" not in cb.history[-1]


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


# ---- gadget throws per episode ----------------------------------------------------------------

class _GadgetOnceModel(_StubModel):
    """Throws the gadget on the first decision it is asked for, then idles."""
    def __init__(self):
        super().__init__()
        self.decisions = 0

    def predict(self, obs, deterministic=True, action_masks=None):
        action, state = super().predict(obs, deterministic, action_masks)
        if self.decisions == 0:
            action[:, 1] = 3
        self.decisions += 1
        return action, state


@pytest.mark.parametrize("model_cls, per_episode", [(_StubModel, 0.0), (_GadgetOnceModel, 1.0)])
def test_evaluator_counts_the_gadgets_a_real_rollout_throws(model_cls, per_episode):
    """Every hero spawns with its gadget charged, so one press on the first decision is one throw
    in every slot, and the idle policy throws none. This reads the REAL env's mask: a count keyed
    to the super's column would find an uncharged super there and report 0.0 for both."""
    ev = TierEvaluator(_eval_tcfg(episodes_per_tier=3))
    results = ev.evaluate(model_cls())
    ev.close()
    assert {name: r["gadgets_used"] for name, r in results.items()} == {
        "easy": per_episode, "hard": per_episode}


class _ScriptedVenv:
    """Four slots with scripted gadget legality and dones, for the count's two gates. The mask is
    the wrapper's real layout, written out: 17 move columns, then [no-fire, attack, super, gadget],
    so the gadget is column 20 of 21 (brawl_sim/wrappers/sb3_vecenv.action_masks)."""
    def __init__(self, gadget_legal, dones):
        self.gadget_legal, self.dones, self.t = gadget_legal, dones, 0

    def _obs(self):
        return {"x": np.zeros((4, 1), dtype=np.float32)}

    def reset(self):
        self.t = 0
        return self._obs()

    def action_masks(self):
        masks = np.ones((4, 21), dtype=bool)
        masks[:, 20] = self.gadget_legal[self.t]
        return masks

    def step(self, action):
        done = np.array(self.dones[self.t], dtype=bool)
        self.t += 1
        infos = [{"outcome": {"won": False, "rank": 1}, "episode": {"l": self.t, "r": 0.0}}
                 if d else {} for d in done]
        return self._obs(), np.zeros(4, dtype=np.float32), done, infos


class _AlwaysGadgetModel(_StubModel):
    def predict(self, obs, deterministic=True, action_masks=None):
        action, state = super().predict(obs, deterministic, action_masks)
        action[:, 1] = 3
        return action, state


def test_evaluator_counts_a_gadget_only_where_legal_and_only_in_the_first_episode():
    """The policy presses the gadget in every slot at every decision. Slot 0 finishes at decision
    0 and slot 1 at decision 1, so their later presses belong to autoreset episodes and must not
    count; slot 2's first press and slot 3's last are masked out, which the sim does not throw.
    Per slot that is 1, 2, 2, 2, and easy holds slots 0-1, hard 2-3."""
    ev = TierEvaluator(_eval_tcfg(episodes_per_tier=2))
    ev.venv = _ScriptedVenv(gadget_legal=[[1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0]],
                            dones=[[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]])
    results = ev.evaluate(_AlwaysGadgetModel())
    assert {name: r["gadgets_used"] for name, r in results.items()} == {"easy": 1.5, "hard": 2.0}
    assert {name: r["episodes"] for name, r in results.items()} == {"easy": 2, "hard": 2}


# ---------------------------------------------------------------------------
# map-overfitting eval: training maps vs holdout maps
# ---------------------------------------------------------------------------

TRAINING_MAPS = (
    "open", "bushy", "skull_creek", "feast_or_famine", "scorched_stone", "island_invasion",
    "broken_wall", "stone_fort", "twin_ponds", "cross_creek", "narrow_pass",
    "dry_gulch", "thorn_field", "reed_marsh",
    "hot_maze", "ghost_point", "shadow_spirits", "crescent_lakes", "twisting_vines",
    "pond_maze", "canal_maze", "picket_maze", "lagoon_ring", "square_lakes", "bush_halo",
    "bramble_ponds", "bramble_bend", "bramble_stars", "moon_gate", "half_moon", "moon_pools",
    "vine_springs", "vine_canal", "vine_hollow",
)
HOLDOUT_MAPS = ("split_river", "hollow_ring")


def _holdout_tcfg(episodes_per_tier=2, tiers=("easy", "hard"), world=None, holdout=HOLDOUT_MAPS):
    """debug_tiny's entity count and view on a 60x60 world, because every map but `blank` is
    60x60: trains on (open, bushy), holds out the shipped pair. 100 ticks = 20 decisions per
    episode keeps one evaluation near 2 s on CPU."""
    env = yaml.safe_load(DEBUG_TINY.read_text())
    env["world"] = {"map_h": 60, "map_w": 60, "maps": ["open", "bushy"],
                    "map_selection": "uniform", "fixed_map": "open", **(world or {})}
    env["sim"] = {"max_episode_steps": 100}
    return load_train_config(TRAIN_CONFIG, overrides={
        "run": {"device": "cpu", "env_overrides": env},
        "eval": {"episodes_per_tier": episodes_per_tier, "tiers": list(tiers),
                 "holdout_maps": None if holdout is None else list(holdout)},
    })


def test_an_override_list_replaces_the_base_list():
    """The thirty-four-map rotation only works if `world.maps` in an override REPLACES
    configs/default.yaml's thirty-six. A merge that unioned lists would quietly put the holdout
    maps back into training. Both merges on the path are pinned: train.yaml's own, and the one
    `load_config` applies to `run.env_overrides`."""
    merged = deep_merge({"world": {"maps": ["a", "b", "c"], "map_h": 60}},
                        {"world": {"maps": ["b"]}})
    assert merged == {"world": {"maps": ["b"], "map_h": 60}}

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      overrides={"world": {"maps": ["bushy", "open"]}})
    assert cfg.map_names == ("bushy", "open")


def test_shipped_config_trains_on_thirty_four_maps_and_holds_out_two(tcfg):
    from brawl_sim.training.builder import build_spec

    training = resolved_training_maps(tcfg.run)
    assert training == TRAINING_MAPS
    assert len(training) == 34
    assert tcfg.eval.holdout_maps == ("split_river", "hollow_ring")
    assert tcfg.eval.has_holdout
    assert not set(training) & set(tcfg.eval.holdout_maps)

    # the SimParams view of the same overrides agrees with the EnvConfig view
    assert tuple(build_spec(tcfg)["world"]["maps"]) == TRAINING_MAPS

    # and together they are exactly the sim's thirty-six: nothing was dropped by accident
    loaded = load_config(REPO_ROOT / "configs" / "default.yaml").map_names
    assert len(loaded) == 36
    assert set(training) | set(tcfg.eval.holdout_maps) == set(loaded)


def test_a_holdout_map_that_is_also_trained_on_is_rejected(tcfg):
    with pytest.raises(ValueError, match=r"holdout_maps \['open'\] are also in the training"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"holdout_maps": ["split_river", "open"]}})

    # What counts is the RESOLVED rotation, not what train.yaml happens to list: strip the
    # thirty-four-map override and the env falls back to default.yaml's thirty-six, holdouts included.
    from brawl_sim.training.config import validate_train_config
    with pytest.raises(ValueError, match=r"\['split_river', 'hollow_ring'\] are also in the training"):
        validate_train_config(replace(tcfg, run=replace(tcfg.run, env_overrides={})))


def test_a_config_without_the_rotation_override_cannot_hold_maps_out(tmp_path):
    """The mistake this validation exists for: naming a holdout and forgetting to take it out of
    `world.maps`. Nothing else would notice -- the run would just report a holdout score on a
    map it trained on."""
    path = tmp_path / "t.yaml"
    path.write_text(
        "eval: {holdout_maps: [hollow_ring]}\n"
        "curriculum:\n"
        "  tiers: {hard: {}}\n"
        "  stages: [{name: only, tier_weights: {hard: 1.0}}]\n"
    )
    # The OVERLAP message, not just the map's name: `hollow_ring` also appears in the unknown-map
    # message (it prints the whole registry) and in the wrong-size one, so matching the name alone
    # could not tell which check refused the config.
    with pytest.raises(ValueError, match=r"holdout_maps \['hollow_ring'\] are also in the training"):
        load_train_config(path)


def test_an_unknown_holdout_map_is_rejected():
    # `quiet_lake` is in the plan's list of candidate names and was never generated.
    with pytest.raises(ValueError, match=r"unknown map\(s\) \['quiet_lake'\]"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"holdout_maps": ["quiet_lake"]}})


def test_holdout_maps_must_be_a_list_without_repeats():
    with pytest.raises(ValueError, match="must be a list"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"holdout_maps": "split_river"}})
    with pytest.raises(ValueError, match="more than once"):
        load_train_config(TRAIN_CONFIG, overrides={
            "eval": {"holdout_maps": ["split_river", "split_river"]}})


def test_a_holdout_map_of_the_wrong_size_is_rejected_at_load():
    """`blank` is registered and in no 60x60 rotation, so the name and overlap checks both pass
    it -- but it is 20x20. Left to the evaluator it failed only when the holdout env was built."""
    with pytest.raises(ValueError, match=r"'blank' cannot be loaded into this run's 60x60 world"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"holdout_maps": ["blank"]}})
    # the same check guards a config edited after it was loaded
    from brawl_sim.training.config import validate_train_config
    tcfg = load_train_config(TRAIN_CONFIG)
    with pytest.raises(ValueError, match="map is 20x20, cfg expects 60x60"):
        validate_train_config(replace(tcfg, eval=replace(tcfg.eval, holdout_maps=("hollow_ring", "blank"))))


@pytest.mark.parametrize("value", [None, []])
def test_no_holdout_maps_is_valid_and_builds_no_second_evaluator(value):
    tcfg = _holdout_tcfg(holdout=value)
    assert not tcfg.eval.has_holdout
    evaluator, holdout = build_evaluators(tcfg)
    evaluator.close()
    assert holdout is None
    assert evaluator.sim.cfg.map_names == ("open", "bushy")


def test_a_config_written_before_holdout_maps_existed_still_loads(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "eval: {episodes_per_tier: 4}\n"
        "curriculum:\n"
        "  tiers: {hard: {}}\n"
        "  stages: [{name: only, tier_weights: {hard: 1.0}}]\n"
    )
    cfg = load_train_config(path)
    assert cfg.eval.holdout_maps is None
    assert not cfg.eval.has_holdout
    assert EvalConfig().holdout_maps is None


def test_an_archived_run_still_deploys_after_its_holdout_map_leaves_the_registry(tmp_path, monkeypatch):
    """`eval.holdout_maps` is validated against the repo as it is TODAY (the map registry, the
    CSVs on disk). That is right for a run about to start and wrong for a reader of an archived
    `runs/<name>/train.yaml` that never builds the holdout evaluator. Deployment is the reader
    that matters: nothing else on `DeployedPolicy.from_run`'s path looks at a map name
    (`load_config` does not check them; `config.validate` does, and only `BrawlVecEnv` calls it),
    so without the opt-out an eval-only key is the one thing tying a deployed checkpoint to the
    registry."""
    import brawl_sim.config as sim_config
    from brawl_deployment.policy import DeployedPolicy

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train.yaml").write_text(TRAIN_CONFIG.read_text())   # archives the shipped pair
    registry = sim_config._KNOWN_MAP_NAMES
    assert "hollow_ring" in registry
    monkeypatch.setattr(sim_config, "_KNOWN_MAP_NAMES",
                        tuple(name for name in registry if name != "hollow_ring"))

    # a run about to START (or resume) from that file is still refused...
    with pytest.raises(ValueError, match=r"unknown map\(s\) \['hollow_ring'\]"):
        load_train_config(run_dir / "train.yaml")
    # ...a read-only consumer can ask not to be, and gets the archived list back as written
    archived = load_train_config(run_dir / "train.yaml", check_holdout=False)
    assert archived.eval.holdout_maps == ("split_river", "hollow_ring")
    assert archived.eval.has_holdout
    # ...and deployment does ask: it gets past the config, to the checkpoint that is not there
    with pytest.raises(FileNotFoundError, match="best_model"):
        DeployedPolicy.from_run(run_dir)


def test_skipping_the_holdout_check_skips_nothing_else(tmp_path):
    """`check_holdout=False` drops the checks that read the repo's current state, and only those:
    what the FILE itself says is still validated, holdout keys included."""
    with pytest.raises(ValueError, match="must be a list"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"holdout_maps": "split_river"}},
                          check_holdout=False)
    with pytest.raises(ValueError, match="more than once"):
        load_train_config(TRAIN_CONFIG, check_holdout=False, overrides={
            "eval": {"holdout_maps": ["split_river", "split_river"]}})
    with pytest.raises(ValueError, match="eval.tiers names undefined tier"):
        load_train_config(TRAIN_CONFIG, overrides={"eval": {"tiers": ["nightmare"]}},
                          check_holdout=False)
    # the default is the strict one
    from brawl_sim.training.config import validate_train_config
    loose = load_train_config(TRAIN_CONFIG, check_holdout=False,
                              overrides={"eval": {"holdout_maps": ["split_river", "open"]}})
    assert loose.eval.holdout_maps == ("split_river", "open")
    with pytest.raises(ValueError, match=r"holdout_maps \['open'\] are also in the training"):
        validate_train_config(loose)


def test_holdout_evaluator_env_holds_only_the_holdout_maps():
    """Asserted on the BUILT env, not on the config handed in: the map bank is what episodes
    are drawn from, and it is built from `EnvConfig.map_names`."""
    from brawl_sim.maps.loader import CSV_DIR, load_map_csv

    tcfg = _holdout_tcfg(episodes_per_tier=8)
    evaluator, holdout = build_evaluators(tcfg)
    try:
        assert evaluator.sim.cfg.map_names == ("open", "bushy")
        assert holdout.sim.cfg.map_names == ("split_river", "hollow_ring")
        assert holdout.map_names == ("split_river", "hollow_ring")
        assert tuple(holdout.sim.bank.tiles.shape) == (2, 60, 60)
        for i, name in enumerate(("split_river", "hollow_ring")):
            on_disk = torch.as_tensor(load_map_csv(CSV_DIR / f"{name}.csv", holdout.sim.cfg))
            assert torch.equal(holdout.sim.bank.tiles[i], on_disk), f"bank slot {i} is not {name}"
        assert not torch.equal(holdout.sim.bank.tiles, evaluator.sim.bank.tiles)

        # both are drawn, and nothing else can be: map_id indexes a two-map bank
        holdout.sim.gen.manual_seed(holdout.seed)
        holdout.venv.reset()
        assert sorted(set(holdout.sim.state.map_id.tolist())) == [0, 1]

        # a twin in every other respect
        assert holdout.tier_names == evaluator.tier_names == ("easy", "hard")
        assert holdout.n_envs == evaluator.n_envs == 16
        assert holdout.seed == evaluator.seed == 999983
        assert holdout.sim is not evaluator.sim
        assert isinstance(holdout.sim.params_hook, FixedTierHook)

        # Each evaluator's `tcfg` describes the env THAT evaluator built. The holdout one used to
        # keep the run's un-patched config, so `holdout.tcfg` named the training maps while its
        # bank held the other two: a spec or env rebuilt from it would have been the wrong one.
        from brawl_sim.training.builder import build_spec
        world = holdout.tcfg.run.env_overrides["world"]
        assert world["maps"] == ["split_river", "hollow_ring"]
        assert (world["map_selection"], world["fixed_map"]) == ("uniform", "split_river")
        assert build_spec(holdout.tcfg)["world"]["maps"] == ["split_river", "hollow_ring"]
        assert holdout.tcfg.eval == tcfg.eval           # same tiers, episodes, seed, determinism
        assert evaluator.tcfg.run.env_overrides["world"]["maps"] == ["open", "bushy"]
        # ...and deriving the twin did not write into the run's own config
        assert tcfg.run.env_overrides["world"]["maps"] == ["open", "bushy"]
        assert tcfg.run.env_overrides["world"]["map_selection"] == "uniform"
        assert tcfg.run.env_overrides["world"]["fixed_map"] == "open"
    finally:
        evaluator.close()
        holdout.close()


def test_holdout_evaluator_ignores_a_training_run_pinned_to_one_map():
    """A run trained with `map_selection: fixed` names a `fixed_map` that is, by validation, not
    a holdout map; carried over unchanged it would crash the holdout env's first reset."""
    tcfg = _holdout_tcfg(world={"map_selection": "fixed", "fixed_map": "bushy"})
    holdout = TierEvaluator(tcfg, maps=tcfg.eval.holdout_maps)
    try:
        assert holdout.sim.cfg.map_selection == "uniform"
        assert holdout.sim.cfg.fixed_map == "split_river"
        holdout.venv.reset()
        assert int(holdout.sim.state.map_id.max()) <= 1
    finally:
        holdout.close()


def _drive_eval_callback(cb, model, recorded):
    model.logger = SimpleNamespace(record=lambda k, v, **kw: recorded.__setitem__(k, v))
    cb.model = model
    cb.num_timesteps = 0
    cb.locals = {}
    cb._on_training_start()
    cb._on_step()


def test_eval_callback_logs_training_and_holdout_win_rates(tmp_path):
    """The M4 acceptance run: real envs, both evaluators, every key the run's log must carry."""
    from brawl_sim.training.callbacks import TierEvalCallback
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tcfg = _holdout_tcfg()
    evaluator, holdout = build_evaluators(tcfg)
    recorded = {}
    cb = TierEvalCallback(evaluator, every_timesteps=1000, log_dir=tmp_path, at_start=True,
                          best_model_path=tmp_path / "best_model.zip", verbose=0,
                          holdout_evaluator=holdout)
    _drive_eval_callback(cb, _StubModel(), recorded)
    for writer in cb._writers.values():
        writer.close()
    evaluator.close()
    holdout.close()

    for tier in ("easy", "hard"):
        assert f"eval/win_rate_{tier}" in recorded
        assert f"eval/holdout_win_rate_{tier}" in recorded
        assert 0.0 <= recorded[f"eval/holdout_win_rate_{tier}"] <= 1.0
    assert "eval/win_rate_mean" in recorded
    assert "eval/holdout_win_rate" in recorded
    assert "eval/holdout_gap" in recorded

    assert set(cb.history[-1]) == {"timesteps", "mean_win_rate", "tiers",
                                   "holdout_mean_win_rate", "holdout_tiers"}
    assert set(cb.history[-1]["holdout_tiers"]) == {"easy", "hard"}

    # the per-tier overlay chart exists for the holdout maps too, in the SAME tier directories
    for tier in ("easy", "hard"):
        tags = EventAccumulator(str(tmp_path / f"eval_{tier}")).Reload().Tags()["scalars"]
        assert "eval/win_rate" in tags and "eval/holdout_win_rate" in tags


class _CannedEvaluator:
    """Hands back scripted win rates, one dict per call -- for pinning the callback's arithmetic
    and its checkpoint choice, which a real rollout of an idle policy (0% everywhere) cannot."""
    def __init__(self, *calls, map_names=("a",)):
        self.calls = list(calls)
        self.map_names = map_names
        self.venv = None

    def evaluate(self, model):
        return {tier: {"win_rate": w, "mean_rank": 1.0, "mean_ep_length": 10.0, "mean_reward": 0.0,
                       "gadgets_used": 2 * w, "episodes": 4}
                for tier, w in self.calls.pop(0).items()}


def test_eval_callback_means_and_gap_are_per_evaluator(tmp_path):
    from brawl_sim.training.callbacks import TierEvalCallback
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    recorded = {}
    cb = TierEvalCallback(_CannedEvaluator({"easy": 0.5, "hard": 0.25}), 1000, tmp_path, verbose=0,
                          holdout_evaluator=_CannedEvaluator({"easy": 0.25, "hard": 0.0}))
    _drive_eval_callback(cb, _StubModel(), recorded)
    for writer in cb._writers.values():
        writer.close()

    assert recorded["eval/win_rate_mean"] == 0.375
    assert recorded["eval/holdout_win_rate"] == 0.125
    assert recorded["eval/holdout_gap"] == 0.25
    assert recorded["eval/win_rate_easy"] == 0.5
    assert recorded["eval/holdout_win_rate_easy"] == 0.25
    assert recorded["eval/holdout_win_rate_hard"] == 0.0
    assert recorded["eval/best_win_rate_mean"] == 0.375
    assert cb.history == [{"timesteps": 0, "mean_win_rate": 0.375,
                           "tiers": {"easy": 0.5, "hard": 0.25},
                           "holdout_mean_win_rate": 0.125,
                           "holdout_tiers": {"easy": 0.25, "hard": 0.0}}]

    # The per-tier TensorBoard overlays, by VALUE: the tag being present says nothing about what
    # was written under it (the training-map mean under the holdout tag passed every tag check).
    # Each line carries its own tier's number from its own evaluator; all four are exact in the
    # float32 an event file stores.
    def line(tier, tag):
        events = EventAccumulator(str(tmp_path / f"eval_{tier}")).Reload().Scalars(tag)
        return [(e.step, e.value) for e in events]

    assert line("easy", "eval/win_rate") == [(0, 0.5)]
    assert line("hard", "eval/win_rate") == [(0, 0.25)]
    assert line("easy", "eval/holdout_win_rate") == [(0, 0.25)]
    assert line("hard", "eval/holdout_win_rate") == [(0, 0.0)]
    # The canned gadget count is twice the win rate, so each tier's line is its own number too.
    assert recorded["eval/gadgets_used_easy"] == 1.0
    assert line("easy", "eval/gadgets_used") == [(0, 1.0)]
    assert line("hard", "eval/gadgets_used") == [(0, 0.5)]


def test_best_model_is_chosen_on_the_training_maps_never_the_holdout(tmp_path):
    """Selecting the checkpoint by its holdout score would fit the selection to the holdout
    maps. Second eval: holdout jumps 0.0 -> 1.0 while the training maps slip -> NO save. Third:
    the training maps improve while the holdout collapses -> save."""
    from brawl_sim.training.callbacks import TierEvalCallback

    training = _CannedEvaluator({"hard": 0.5}, {"hard": 0.25}, {"hard": 0.75})
    holdout = _CannedEvaluator({"hard": 0.0}, {"hard": 1.0}, {"hard": 0.0})
    model, recorded = _StubModel(), {}
    cb = TierEvalCallback(training, 1000, tmp_path, best_model_path=tmp_path / "best_model.zip",
                          verbose=0, holdout_evaluator=holdout)
    _drive_eval_callback(cb, model, recorded)
    assert len(model.saved) == 1

    cb._evaluate()
    assert len(model.saved) == 1, "a better HOLDOUT score saved a new best model"
    assert recorded["eval/best_win_rate_mean"] == 0.5
    assert recorded["eval/holdout_win_rate"] == 1.0

    cb._evaluate()
    assert len(model.saved) == 2
    assert recorded["eval/best_win_rate_mean"] == 0.75
    for writer in cb._writers.values():
        writer.close()


def test_eval_callback_copies_observation_statistics_onto_both_evaluators(tmp_path):
    """`normalize.obs: true`: a policy scored on raw observations looks like a failed run. Both
    evaluators must carry the training env's statistics, re-copied at EVERY eval (they keep
    moving). Until 2026-09-18 this path died at the first eval: SB3's sync helper asserts the two
    wrapper stacks are equally deep, and training has a VecMonitor layer the evaluators lack."""
    from stable_baselines3.common.vec_env import VecNormalize
    from brawl_sim.training.builder import build_env
    from brawl_sim.training.callbacks import TierEvalCallback

    env = yaml.safe_load(DEBUG_TINY.read_text())
    env["sim"] = {"max_episode_steps": 50}      # 10 decisions a rollout: four rollouts below
    tcfg = load_train_config(TRAIN_CONFIG, overrides={
        "run": {"device": "cpu", "n_envs": 8, "env_overrides": env},
        "ppo": {"n_steps": 16, "batch_size": 32},
        "normalize": {"obs": True},
        "eval": {"episodes_per_tier": 2, "tiers": ["easy", "hard"], "holdout_maps": None},
    })
    train_venv, _ = build_env(tcfg)
    assert isinstance(train_venv, VecNormalize) and train_venv.obs_rms

    def set_statistics(mean, var):
        for rms in train_venv.obs_rms.values():
            rms.mean[...] = mean
            rms.var[...] = var

    model = _StubModel()
    model.get_vec_normalize_env = lambda: train_venv
    evaluators = (TierEvaluator(tcfg), TierEvaluator(tcfg))   # the second stands in for the holdout
    cb = TierEvalCallback(evaluators[0], 1000, tmp_path, verbose=0,
                          holdout_evaluator=evaluators[1])

    set_statistics(0.25, 4.0)
    _drive_eval_callback(cb, model, {})
    set_statistics(0.5, 9.0)
    cb._evaluate()
    for writer in cb._writers.values():
        writer.close()

    for ev in evaluators:
        assert isinstance(ev.venv, VecNormalize)
        assert ev.venv.training is False and ev.venv.norm_reward is False
        assert set(ev.venv.obs_rms) == set(train_venv.obs_rms)
        for key, rms in ev.venv.obs_rms.items():
            assert rms is not train_venv.obs_rms[key], "shared, not copied: eval would mutate it"
            assert np.all(rms.mean == 0.5) and np.all(rms.var == 9.0), key
        ev.close()
    train_venv.close()


def test_holdout_maps_survive_the_archived_train_yaml(tmp_path, tcfg):
    """scripts/train.py archives `tcfg.raw` as runs/<name>/train.yaml, and scripts/watch.py and
    brawl_deployment load that file back. The tuple has to come back as the same tuple, and the
    rotation it is disjoint from has to come back with it."""
    archived = tmp_path / "train.yaml"
    archived.write_text(yaml.safe_dump(tcfg.raw, sort_keys=False))
    back = load_train_config(archived)
    assert back.eval.holdout_maps == ("split_river", "hollow_ring")
    assert resolved_training_maps(back.run) == TRAINING_MAPS
    assert back.eval == tcfg.eval


def test_watch_can_pin_a_map_the_run_held_out():
    """`scripts/watch.py --map split_river` on a run whose rotation excludes it: the map joins
    the END of the bank, so the maps the policy trained on keep their indices."""
    from scripts.watch import build_watch_env

    tcfg = _holdout_tcfg()
    sim, venv, env_cfg = build_watch_env(tcfg, "hard", seed=0, device="cpu", map_name="split_river")
    venv.reset()
    assert env_cfg.map_names == ("open", "bushy", "split_river")
    assert env_cfg.fixed_map == "split_river"
    assert sim.state.map_id.tolist() == [2]
    venv.close()

    sim, venv, env_cfg = build_watch_env(tcfg, "hard", seed=0, device="cpu", map_name="bushy")
    venv.reset()
    assert env_cfg.map_names == ("open", "bushy")
    assert sim.state.map_id.tolist() == [1]
    venv.close()


def test_a_saved_watch_match_replays_over_the_map_it_was_played_on(tmp_path, monkeypatch):
    """A frame stores `map_id`, an index into the RECORDING env's map list, and the viewer CLI
    builds its bank from configs/default.yaml unless given a preset. Once a run's rotation is
    not default.yaml's (configs/train.yaml: thirty-four maps, another order) a bare replay draws the
    wrong terrain. Pinned on a SHIFTED index: narrow_pass is map 1 here, where default.yaml has
    bushy -- on `open` (0 in both) the bug is invisible."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from brawl_sim.maps.loader import CSV_DIR, load_map_csv
    from brawl_sim.render import viewer as viewer_mod
    from scripts.watch import build_watch_env, play_match, save_rollout

    tcfg = _holdout_tcfg(world={"maps": ["open", "narrow_pass"]})
    sim, venv, env_cfg = build_watch_env(tcfg, "hard", seed=0, device="cpu", map_name="narrow_pass")
    frames, _ = play_match(_StubModel(), venv, sim, uses_masks=False, deterministic=True,
                           max_steps=2)
    venv.close()
    assert set(frames["map_id"].tolist()) == {1}

    npz, preset = tmp_path / "match.npz", tmp_path / "match.preset.yaml"
    command = save_rollout(npz, frames, tcfg, "narrow_pass")
    assert command == f"python -m brawl_sim.render.viewer {npz} --preset {preset}"
    world = yaml.safe_load(preset.read_text())["world"]
    assert world["maps"] == ["open", "narrow_pass"]
    assert (world["map_selection"], world["fixed_map"]) == ("fixed", "narrow_pass")

    seen = {}

    class _CaptureViewer:       # the real CLI path, minus the window
        def __init__(self, frames, bank, cfg, fps=20):
            seen.update(frames=frames, bank=bank, cfg=cfg)

        def show(self):
            pass

    monkeypatch.setattr(viewer_mod, "ReplayViewer", _CaptureViewer)
    narrow_pass = torch.as_tensor(load_map_csv(CSV_DIR / "narrow_pass.csv", env_cfg))

    assert viewer_mod.main([str(npz), "--preset", str(preset)]) == 0
    assert seen["cfg"].map_names == ("open", "narrow_pass")
    assert torch.equal(seen["bank"].tiles[int(seen["frames"]["map_id"][0])], narrow_pass)

    # What watch.py printed before: no preset. Same npz, map 1 is now a different map.
    assert viewer_mod.main([str(npz)]) == 0
    assert seen["cfg"].map_names[1] == "bushy"
    assert not torch.equal(seen["bank"].tiles[int(seen["frames"]["map_id"][0])], narrow_pass)

    # A held-out map pinned with --map is appended to the bank, and the preset must say so too.
    save_rollout(tmp_path / "held.npz", frames, tcfg, "split_river")
    held = yaml.safe_load((tmp_path / "held.preset.yaml").read_text())["world"]
    assert held["maps"] == ["open", "narrow_pass", "split_river"]
    assert held["fixed_map"] == "split_river"

    # A run on some other env config: the viewer's --config default would be the wrong file.
    other = tmp_path / "env.yaml"
    other.write_text((REPO_ROOT / "configs" / "default.yaml").read_text())
    moved = replace(tcfg, run=replace(tcfg.run, env_config=str(other)))
    assert save_rollout(tmp_path / "moved.npz", frames, moved).endswith(f" --config {other}")


def test_watch_builds_its_env_and_its_spec_from_the_same_overrides(monkeypatch):
    """`BrawlVecEnv` takes two views of the run's env overrides: the EnvConfig and the raw
    `spec=` dict. `--map <holdout>` patched only the first, so the spec still described the
    un-patched rotation (harmless while nothing reads spec['world'], and a trap the day it does)."""
    from scripts import watch

    seen = {}
    real_build_spec = watch.build_spec

    def spy(tcfg):
        seen["world"] = tcfg.run.env_overrides["world"]
        return real_build_spec(tcfg)

    tcfg = _holdout_tcfg()
    monkeypatch.setattr(watch, "build_spec", spy)
    sim, venv, env_cfg = watch.build_watch_env(tcfg, "hard", seed=0, device="cpu",
                                               map_name="split_river")
    venv.close()
    assert seen["world"]["maps"] == ["open", "bushy", "split_river"]
    assert (seen["world"]["map_selection"], seen["world"]["fixed_map"]) == ("fixed", "split_river")
    # ...and the caller's config is not mutated by the patch
    assert tcfg.run.env_overrides["world"]["maps"] == ["open", "bushy"]
    assert tcfg.run.env_overrides["world"]["map_selection"] == "uniform"


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
    # --save also writes the preset the viewer needs to rebuild THIS run's map bank
    saved = yaml.safe_load((tmp_path / "match.preset.yaml").read_text())
    assert saved["world"]["maps"] == ["blank"]
    assert (saved["world"]["map_h"], saved["world"]["map_w"]) == (20, 20)

    # --smoke switches the holdout eval off (its 20x20 world cannot load the 60x60 holdout maps)
    assert "holdout" not in header


@pytest.mark.slow
def test_train_script_builds_the_evaluators_before_the_run_directory(tmp_path, monkeypatch):
    """A config the builder refuses must not leave an empty run directory behind, and the
    evaluators are the last thing that can refuse one: the second (holdout) eval env is where a
    run is likeliest to run out of VRAM."""
    pytest.importorskip("sb3_contrib")
    import scripts.train as train_script

    def refuse(*args, **kwargs):
        raise RuntimeError("no room for a second eval env")

    monkeypatch.setattr(train_script, "build_evaluators", refuse)
    with pytest.raises(RuntimeError, match="no room for a second eval env"):
        train_script.main(["--smoke", "--out-dir", str(tmp_path / "runs")])
    assert not (tmp_path / "runs" / "smoke").exists()
    assert not list(tmp_path.rglob("train.yaml"))



# ---- SB3's image heuristic must not transpose the grid ------------------------------------------

def _tiny_run_tcfg(extra=None):
    return load_train_config(TRAIN_CONFIG, overrides=deep_merge({
        "run": {"device": "cpu", "n_envs": 8, "tensorboard": False,
                "env_overrides": yaml.safe_load(DEBUG_TINY.read_text())},
        "ppo": {"n_steps": 16, "batch_size": 32},
        "eval": {"holdout_maps": None},
    }, extra or {}))


def test_the_model_sees_the_grid_channels_first_as_the_env_builds_it():
    """SB3 takes a uint8 [0, 255] Box of rank 3 for an image, guesses its channel axis from the
    smallest dimension, and wraps the env in a transposing VecTransposeImage when it guesses
    channels-last. debug_tiny's view is 10 x 14, so the shipped spec's grid is (13, 10, 14):
    until 2026-09-21 every model built on it, `train.py --smoke` included, trained on
    (14, 13, 10), view columns for channels. The real 13 x 21 view escapes it only on a tie."""
    from brawl_sim.training.builder import build_run

    tcfg = _tiny_run_tcfg()
    model, venv, _ = build_run(tcfg)
    # deploy5 since 2026-09-25 (the operator moved train.yaml to it); its grid is deploy4's 13 channels
    assert tcfg.run.agent_obs == "configs/agent_obs_deploy5.yaml"
    assert venv.observation_space["grid"].shape == (13, 10, 14)
    assert model.observation_space["grid"].shape == (13, 10, 14)
    assert model.policy.features_extractor.cnn[0].in_channels == 13


def test_a_resumed_model_keeps_the_grid_channels_first(tmp_path):
    """`scripts/train.py --resume` hands SB3 the env twice, at load and again with the saved
    VecNormalize statistics, and either hand-off alone would transpose this grid. The saved
    model holds the untransposed space, so a transposed env would not even load."""
    import scripts.train as train_script
    from brawl_sim.training.builder import build_run

    tcfg = _tiny_run_tcfg()
    model, venv, _ = build_run(tcfg)
    model.save(tmp_path / "model.zip")
    venv.save(str(tmp_path / "vecnormalize.pkl"))
    loaded = train_script._resume(model, venv, tcfg, tmp_path / "model.zip", None, tmp_path)
    assert loaded.get_vec_normalize_env() is not None, "the statistics branch did not run"
    assert loaded.observation_space["grid"].shape == (13, 10, 14)
    assert loaded.get_env().observation_space["grid"].shape == (13, 10, 14)


def test_a_resumed_run_saves_the_statistics_it_trained_through(tmp_path):
    """A resume trains through the VecNormalize `_resume` restores, not the fresh one build_run
    made. Until 2026-09-27 train.py saved the fresh one as final_vecnormalize.pkl, so the 450M
    run's continuation left count 1e-4 and var 1 where the model had trained against ~46, and a
    resume of its final_model.zip would have restarted the reward scaling."""
    import scripts.train as train_script
    from brawl_sim.training.builder import build_run
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import VecNormalize

    tcfg = _tiny_run_tcfg()
    model, venv, _ = build_run(tcfg)
    model.save(tmp_path / "model.zip")
    venv.ret_rms.count, venv.ret_rms.var = 4.5e8, np.array(45.8)    # a long run's statistics...
    venv.save(str(tmp_path / "vecnormalize.pkl"))
    venv.ret_rms = RunningMeanStd(shape=())                         # ...and a fresh build's
    loaded = train_script._resume(model, venv, tcfg, tmp_path / "model.zip", None, tmp_path)

    run_dir = tmp_path / "resumed"
    run_dir.mkdir()
    train_script._save_final(loaded, run_dir)
    saved = VecNormalize.load(str(run_dir / "final_vecnormalize.pkl"), venv.venv)
    assert saved.ret_rms.count == pytest.approx(4.5e8)
    assert float(saved.ret_rms.var) == pytest.approx(45.8)


def test_restart_schedules_runs_the_config_schedules_over_the_resumed_steps_only(tmp_path):
    """A resume keeps the checkpoint's schedules on SB3's whole-model progress; that is why a
    100M extension of the 450M model starts at 7.7e-5 whatever train.yaml says. With
    `restart_schedules` the config's schedules span the resumed steps alone: 768 done, 256 more
    as two rollouts of 128, so the two updates land halfway through the linear 5e-5 -> 1e-5 and
    at its end. The rates are read off the optimizer, so SB3's own progress count is what's
    checked, not this file's arithmetic."""
    import scripts.train as train_script
    from brawl_sim.training.builder import build_run

    tcfg = _tiny_run_tcfg()
    model, venv, _ = build_run(tcfg)
    model.num_timesteps = 768
    model.learning_rate = schedules.constant_schedule(1.234e-4)     # the checkpoint's own
    model.save(tmp_path / "model.zip")
    venv.save(str(tmp_path / "vecnormalize.pkl"))
    fine_tune = _tiny_run_tcfg({
        "run": {"total_timesteps": 256},
        "learning_rate": {"schedule": "linear", "initial": 5e-5, "final": 1e-5},
        "clip_range": {"schedule": "linear", "initial": 0.15, "final": 0.1},
    })

    kept = train_script._resume(model, venv, fine_tune, tmp_path / "model.zip", None, tmp_path)
    assert kept.lr_schedule(0.25) == kept.lr_schedule(0.0) == pytest.approx(1.234e-4)

    restarted = train_script._resume(model, venv, fine_tune, tmp_path / "model.zip", None,
                                     tmp_path, restart_schedules=True)
    assert restarted.clip_range(0.25) == pytest.approx(0.15)
    assert restarted.clip_range(0.0) == pytest.approx(0.1)
    rates, train = [], restarted.train

    def recording_train():
        train()
        rates.append(restarted.policy.optimizer.param_groups[0]["lr"])

    restarted.train = recording_train
    restarted.learn(total_timesteps=256, reset_num_timesteps=False)
    del restarted.train     # it holds the model itself, which save() would try to pickle
    assert rates == pytest.approx([3e-5, 1e-5])

    # A crash at 896 is recovered by a plain resume of the remaining 128: the checkpoint carries
    # the squeezed schedule, so it continues from the halfway rate instead of starting over.
    restarted.num_timesteps = 896
    (tmp_path / "crashed").mkdir()
    restarted.save(tmp_path / "crashed" / "model.zip")
    recovered = train_script._resume(model, venv, _tiny_run_tcfg({"run": {"total_timesteps": 128}}),
                                     tmp_path / "crashed" / "model.zip", None, tmp_path)
    assert recovered.lr_schedule(128 / 1024) == pytest.approx(3e-5)
    assert recovered.lr_schedule(0.0) == pytest.approx(1e-5)


@pytest.mark.slow
def test_train_script_logs_holdout_columns_next_to_the_training_map_ones(tmp_path, monkeypatch):
    """scripts/train.py's own wiring of the second evaluator, end to end: --smoke's sizes on a
    60x60 world that trains on two maps and holds the shipped pair out. 1024 decisions is two
    rollouts and two evals (at_start + end of training); nothing here needs the policy to learn.

    The column NAMES alone cannot show the wiring is right: hand the callback the training-map
    evaluator twice and every eval/holdout_* column still appears, as a copy of the training-map
    number, with eval/holdout_gap a flat 0.0 -- "no map overfitting", for the whole run. So the
    callback train.py builds is kept hold of, and what is asserted is the map list of the env
    each of its two evaluators actually BUILT."""
    pytest.importorskip("sb3_contrib")
    import scripts.train as train_script

    built = []

    class _KeptCallback(train_script.TierEvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(train_script, "TierEvalCallback", _KeptCallback)
    train_main = train_script.main

    assert train_main([
        "--smoke", "--out-dir", str(tmp_path),
        "--set", "run.name=smoke_holdout", "--set", "run.total_timesteps=1024",
        "--set", "run.env_overrides.world.map_h=60", "--set", "run.env_overrides.world.map_w=60",
        "--set", "run.env_overrides.world.maps=[open, bushy]",
        "--set", "run.env_overrides.world.map_selection=uniform",
        "--set", "run.env_overrides.world.fixed_map=open",
        "--set", "run.env_overrides.sim.max_episode_steps=100",
        "--set", "eval.holdout_maps=[split_river, hollow_ring]",
    ]) == 0
    run_dir = tmp_path / "smoke_holdout"

    # `evaluator` / `holdout_evaluator` are the attributes `_evaluate` and `_evaluate_holdout`
    # score, so this holds however train.py passed them (positionally or by keyword).
    (callback,) = built
    assert callback.evaluator.sim.cfg.map_names == ("open", "bushy")
    assert callback.holdout_evaluator.sim.cfg.map_names == ("split_river", "hollow_ring")
    assert callback.holdout_evaluator is not callback.evaluator
    # ...and every eval of the run scored the holdout maps, on all six tiers
    assert callback.history
    for entry in callback.history:
        assert set(entry["holdout_tiers"]) == {"easy", "medium", "hard", "veteran", "expert", "elite"}

    header = (run_dir / "logs" / "progress.csv").read_text().splitlines()[0].split(",")
    for tier in ("easy", "medium", "hard", "veteran", "expert", "elite"):
        assert f"eval/win_rate_{tier}" in header
        assert f"eval/holdout_win_rate_{tier}" in header
    for column in ("eval/win_rate_mean", "eval/holdout_win_rate", "eval/holdout_gap"):
        assert column in header
    assert (run_dir / "best_model.zip").exists()

    # the archived config is the one a later `watch.py` reloads: holdout and rotation both intact
    archived = load_train_config(run_dir / "train.yaml")
    assert archived.eval.holdout_maps == ("split_river", "hollow_ring")
    assert resolved_training_maps(archived.run) == ("open", "bushy")


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
