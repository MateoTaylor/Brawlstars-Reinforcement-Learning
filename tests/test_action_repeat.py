"""Action repeat: one agent decision spans `cfg.action_repeat` sim ticks (env.py
`_run_decision`).

Three properties make this knob safe to turn, and each has a test below:

  1. The WORLD's clock is unchanged -- `dt` per sub-tick, `step_count`/`time`/`max_episode_steps`
     still in sim ticks. Only things counted in `step()` CALLS move to decisions.
  2. Per-tick reward terms INTEGRATE over the window, so total episode return is invariant to the
     decision rate (the per-term half of this lives in tests/test_training.py).
  3. An episode that ends mid-window has its outcome LATCHED at the sub-tick it actually ended,
     rather than being overwritten by the leftover sub-ticks that still run afterwards.

(3) is the one that would be a silent training bug rather than a loud crash: without it a hero
that becomes last-alive on sub-tick 2 of 5 keeps taking zone damage through sub-tick 5, and a win
is reported as a death.
"""
import pytest
import torch
import yaml

from brawl_sim.config import EnvConfig, build_params, load_config, validate
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _env(n_envs=4, seed=0, overrides=None):
    merged = {name: dict(section) if isinstance(section, dict) else section
              for name, section in CONFIGS_TINY.items()}
    for name, section in (overrides or {}).items():
        merged[name] = {**merged.get(name, {}), **section} if isinstance(section, dict) else section
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed, verbose=False)


def _idle(n_envs):
    return torch.zeros(n_envs, 2, dtype=torch.int64)


def test_hp_healed_tick_sums_regen_across_the_whole_decision():
    """The per-tick heal delta is a summed delta like damage and cubes, not a snapshot of the
    last sub-tick -- otherwise `hp_healed` reward would silently shrink by `action_repeat` while
    `damage_taken` (which env.py already sums) did not, and the two would stop cancelling."""
    healed_per_repeat = {}
    for repeat in (1, 5):
        env = _env(n_envs=1, overrides={"sim": {"action_repeat": repeat},
                                        "zone": {"enabled": False},
                                        "regen": {"enabled": True}})
        env.reset()
        env.state.ent_hp.fill_(1.0)              # far from max, so nothing clips
        env.state.ent_out_of_combat_t.fill_(99.0)  # regen is already unlocked
        # Every bot forced idle and non-firing (override[..., 0] == 0), so nothing can damage
        # the hero mid-decision and reset the out-of-combat stopwatch this test depends on.
        idle_all = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
        _, _, _, _, info = env.step(_idle(1), idle_all)
        healed_per_repeat[repeat] = float(info["hp_healed_tick"][0, 0])

    per_tick = env.params.regen_fraction_per_second[0] * env.state.ent_max_hp[0, 0] * env.cfg.dt
    assert healed_per_repeat[1] == pytest.approx(float(per_tick), rel=1e-4)
    assert healed_per_repeat[5] == pytest.approx(5 * healed_per_repeat[1], rel=1e-3)


# ---- (1) the world's clock still runs at dt per SUB-tick -----------------------------------

@pytest.mark.parametrize("repeat", [1, 2, 5])
def test_one_step_advances_the_sim_by_action_repeat_ticks(repeat):
    env = _env(n_envs=4, overrides={"sim": {"action_repeat": repeat}})
    env.reset()
    env.step(_idle(4))

    assert torch.all(env.state.step_count == repeat)
    assert torch.allclose(env.state.time, torch.full_like(env.state.time, repeat * env.cfg.dt))


def test_action_repeat_one_leaves_step_count_at_one_tick_per_step():
    """The default-off path: action_repeat=1 must be exactly the pre-existing behavior."""
    env = _env(n_envs=2, overrides={"sim": {"action_repeat": 1}})
    env.reset()
    for expected in (1, 2, 3):
        env.step(_idle(2))
        assert torch.all(env.state.step_count == expected)


def test_max_agent_steps_converts_the_episode_cap_from_ticks_to_decisions():
    assert EnvConfig(max_episode_steps=3000, action_repeat=1).max_agent_steps == 3000
    assert EnvConfig(max_episode_steps=3000, action_repeat=5).max_agent_steps == 600
    # Rounds UP: a remainder makes the last decision SHORT, it does not drop it.
    assert EnvConfig(max_episode_steps=301, action_repeat=5).max_agent_steps == 61
    assert EnvConfig(dt=0.05, action_repeat=5).agent_dt == pytest.approx(0.25)


def test_truncation_still_fires_at_the_sim_tick_cap_not_the_decision_count():
    env = _env(n_envs=2, overrides={"sim": {"max_episode_steps": 20, "action_repeat": 5}})
    env.reset()

    # 10 -> 15 ticks: short of the 20-tick cap, so no truncation yet.
    env.state.step_count.fill_(10)
    *_, truncated, _ = env.step(_idle(2))
    assert not bool(truncated.any())

    # 15 -> 20 ticks: the cap is crossed on the last sub-tick of this decision.
    env.state.step_count.fill_(15)
    *_, truncated, _ = env.step(_idle(2))
    assert bool(truncated.all())


def test_validate_rejects_action_repeat_below_one():
    cfg = load_config(CONFIGS_DEFAULT, overrides={"sim": {"action_repeat": 0}})
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    params = build_params(cfg, n_envs=1, device="cpu", gen=torch.Generator(), spec=spec)
    with pytest.raises(ValueError, match="action_repeat must be >= 1"):
        validate(cfg, params)


# ---- (2) per-tick counters cover the whole window ------------------------------------------

def test_tick_counters_cover_every_sub_tick_of_a_live_decision():
    """`alive_ticks`/`in_zone_ticks` are what the per-tick reward terms integrate over, so for an
    env that survives the whole decision they must equal action_repeat, not 1."""
    repeat = 5
    env = _env(n_envs=1, overrides={"sim": {"action_repeat": repeat}, "zone": {"enabled": True}})
    env.reset()
    env.state.ent_hp[0, 0] = 1e9          # survives whatever the zone does to it
    env.state.ent_max_hp[0, 0] = 1e9
    _park_hero_outside_the_safe_rect(env)

    *_, info = env.step(_idle(1))
    assert int(info["n_ticks"][0]) == repeat
    assert int(info["alive_ticks"][0]) == repeat
    assert int(info["in_zone_ticks"][0]) == repeat


# ---- (3) outcomes are latched at the sub-tick the episode ended ----------------------------

def _park_hero_outside_the_safe_rect(env) -> None:
    """Shrinks the safe rect to a 1x1 patch in a corner of the map the hero is not standing in,
    so every sub-tick deals zone damage to it. Cheaper and far more deterministic than waiting
    out the real shrink schedule."""
    env.state.zone_lo[:, 0] = env.cfg.map_w - 1.0
    env.state.zone_lo[:, 1] = env.cfg.map_h - 1.0
    env.state.zone_hi[:, 0] = float(env.cfg.map_w)
    env.state.zone_hi[:, 1] = float(env.cfg.map_h)
    env.state.ent_pos[:, 0, 0] = 1.0
    env.state.ent_pos[:, 0, 1] = 1.0


def _hp_that_survives_exactly_one_zone_tick(env) -> float:
    """HP high enough to live through one sub-tick in the zone and die on the next.

    Derived, not hardcoded. This was `60.0`, chosen when the zone dealt a flat
    `dps * dt = 1000 * 0.05 = 50` per tick. Step B3 made the rate a FRACTION of max HP, so it is
    now `0.20 * 8000 * 0.05 = 80` per tick for the hero -- and 60 HP stopped surviving the first
    sub-tick at all, which silently destroyed the scenario these two tests are built on (the hero
    must WIN on sub-tick 1 and only then be killed). Both failed on the latched-outcome assertion,
    which reads as a latching bug rather than a stale constant.

    1.5x the per-tick damage is the whole safe band: strictly more than one tick's worth (so it
    survives the first) and strictly less than two (so the second is lethal).
    """
    per_tick = (
        float(env.params.zone_hp_fraction[0])
        * float(env.state.ent_max_hp[0, 0])
        * env.cfg.dt
    )
    return 1.5 * per_tick


def test_a_win_mid_decision_is_not_undone_by_the_remaining_sub_ticks():
    """The load-bearing case. The hero becomes last-alive on sub-tick 1, then the zone kills it
    on sub-tick 2 while sub-ticks 2-5 keep running. The OUTCOME fields must all describe the
    moment it won; only the observation is allowed to show the later, staler state."""
    env = _env(n_envs=1, overrides={"sim": {"action_repeat": 5}, "zone": {"enabled": True}})
    env.reset()

    # Enemies die on sub-tick 1 (hp<=0 is resolved by phase 13), with no ammo to fire first.
    env.state.ent_hp[0, 1:] = 0.0
    env.state.ent_ammo[0, 1:] = 0.0
    env.state.ent_last_hit_by[0, 1:] = -1
    _park_hero_outside_the_safe_rect(env)
    env.state.ent_hp[0, 0] = _hp_that_survives_exactly_one_zone_tick(env)

    *_, info = env.step(_idle(1))

    assert bool(info["terminated"][0])
    assert bool(info["hero_alive"][0]), "hero_alive must be LATCHED at the winning sub-tick"
    assert int(info["hero_rank"][0]) == 0, "rank 0 == last one standing == won"
    # The world genuinely kept ticking: the observation shows the hero the zone finished off.
    assert not bool(info["final_observation"]["hero"]["alive"][0])


def test_counters_stop_at_the_sub_tick_the_episode_ended():
    """A finished env must stop accruing per-tick reward for the rest of the decision -- the
    other half of the same freeze. Same scenario as above: the episode ends on sub-tick 1, so
    exactly one tick may be counted even though five ran."""
    env = _env(n_envs=1, overrides={"sim": {"action_repeat": 5}, "zone": {"enabled": True}})
    env.reset()
    env.state.ent_hp[0, 1:] = 0.0
    env.state.ent_ammo[0, 1:] = 0.0
    env.state.ent_last_hit_by[0, 1:] = -1
    _park_hero_outside_the_safe_rect(env)
    env.state.ent_hp[0, 0] = _hp_that_survives_exactly_one_zone_tick(env)

    *_, info = env.step(_idle(1))

    assert int(info["n_ticks"][0]) == 1
    assert int(info["alive_ticks"][0]) == 1
    assert int(info["in_zone_ticks"][0]) == 1


# ---- (4) movement is held across the window, fire is not ----------------------------------

def test_held_keeps_the_move_bin_and_drops_only_the_fire_bit():
    """`_held` is what sub-ticks 2..K are driven with. Movement must survive it (that IS the
    action repeat); fire must not (one decision = at most one attack)."""
    action = torch.tensor([[7, 1], [0, 1], [3, 0]], dtype=torch.int64)
    held = BrawlVecEnv._held(action)
    assert held[:, 0].tolist() == [7, 0, 3]      # move bins untouched
    assert held[:, 1].tolist() == [0, 0, 0]      # fire cleared
    assert action[:, 1].tolist() == [1, 1, 0]    # and the caller's tensor is not mutated

    # (N,E,2) override: the -1 "no override for this slot" sentinel lives in column 0 and must
    # survive, or an unoverridden entity would start being driven at bin -1.
    override = torch.tensor([[[-1, -1], [5, 1]]], dtype=torch.int64)
    held_override = BrawlVecEnv._held(override)
    assert held_override[0, 0].tolist() == [-1, 0]
    assert held_override[0, 1].tolist() == [5, 0]
    assert BrawlVecEnv._held(None) is None


def test_one_fire_decision_produces_exactly_one_shot():
    """The guard itself, exercised through the override channel.

    The cooldown is forced to 0.05s HERE rather than relying on the shipped value: every
    attack_cooldown in configs/brawlers.yaml is now 0.30-0.50s, comfortably longer than a 0.25s
    decision, so with shipped numbers no brawler can physically fire twice in one window and this
    test would pass whether or not the guard existed. Pinning a fast weapon locally is what keeps
    it a test of `_held` instead of a test of the current balance -- and a fast weapon is exactly
    the case the guard is retained for."""
    from brawl_sim.constants import Kind

    env = _env(n_envs=1, overrides={"sim": {"action_repeat": 5}, "entities": {"n_enemies": 2}})
    env.reset()
    env.state.ent_kind[0, 1] = int(Kind.BOT_SNIPER)
    env.state.ent_hp[0, 1] = 1e9                 # must survive the whole window to keep firing
    env.state.ent_max_hp[0, 1] = 1e9
    env.state.ent_ammo[0, 1] = 3.0
    env.state.ent_attack_cd[0, 1] = 0.0
    # A hypothetical fast-firing brawler: 0.05s = 1 tick, so an ungated held bit would fire on
    # every one of the five sub-ticks and empty the clip.
    env.params.attack_cooldown[:, int(Kind.BOT_SNIPER)] = 0.05
    assert float(env.params.attack_cooldown[0, int(Kind.BOT_SNIPER)]) < env.cfg.agent_dt

    shots_before = int(env.state.ent_shots_fired[0, 1])
    override = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([5, 1])        # entity 1: move bin 5, FIRE
    env.step(_idle(1), override=override)

    assert int(env.state.ent_shots_fired[0, 1]) - shots_before == 1
    # Exactly one round left the clip. Not `== 2.0`: ammo also RELOADS continuously
    # (reload_seconds=1.8, so ~0.11 ammo trickles back over the window's remaining 4 ticks).
    # Two shots would land near 1.1 and three near 0.1, so `> 2.0` separates the cases cleanly.
    assert 2.0 < float(env.state.ent_ammo[0, 1]) < 3.0


def test_shipped_cooldowns_are_at_least_one_decision_window():
    """No brawler may fire more than once per agent decision, so the agent always gets a say
    between its own shots and an enemy's shots always arrive with a dodgeable gap.

    **Rewritten in Step C1 (bot_overhaul.md D7), and one of its three original assertions was
    deleted rather than re-valued.** It used to require `0.30 <= cooldown <= 0.50` and, crucially,
    `cooldown < reload_seconds` -- the latter justified in configs/brawlers.yaml as "this costs no
    sustained DPS, because ammo regen is still the binding constraint". That is now false BY
    DESIGN: `attack_cooldown` also pauses the reload (core/hero.tick_timers), so sustained fire is
    `attack_cooldown + reload_seconds` and the cooldown deliberately does cap it. Keeping that
    assertion would have pinned a philosophy the design moved away from -- and it was numerically
    false for Buzz anyway (1.00 cooldown vs 1.00 reload).

    What survives is the invariant that never depended on the philosophy: a cooldown is at least
    one decision long. `>=` rather than `>`, because the 0.25 s floor is exactly `agent_dt` -- and
    that is sufficient, since `env._held` independently guarantees one attack per decision by
    clearing the fire bit on sub-ticks 2..K.
    """
    spec = yaml.safe_load(open("configs/brawlers.yaml").read())
    cfg = load_config(CONFIGS_DEFAULT)
    for name, brawler in spec.items():
        cooldown = float(brawler["attack_cooldown"])
        assert cooldown >= cfg.agent_dt, (
            f"{name}: attack_cooldown {cooldown} < one decision ({cfg.agent_dt}s), so it could "
            "fire more than once per agent decision"
        )


def test_shipped_cooldown_floor_is_honoured():
    """D7's "always a minimum of 1/4 second" -- the floor is a stated requirement, not an accident
    of the current numbers, so it gets its own assertion."""
    spec = yaml.safe_load(open("configs/brawlers.yaml").read())
    for name, brawler in spec.items():
        assert float(brawler["attack_cooldown"]) >= 0.25, (
            f"{name}: attack_cooldown {brawler['attack_cooldown']} is below the 0.25s floor"
        )


def test_mortis_cooldown_exceeds_his_dash_duration():
    """A dash IS Mortis's attack, and `can_attack` requires `dash_t <= 0` on top of the cooldown.
    A cooldown at or below `dash_duration` would therefore be entirely masked by the dash and the
    reload pause would never actually fire -- the cooldown has to outlast the dash to mean
    anything."""
    spec = yaml.safe_load(open("configs/brawlers.yaml").read())
    mortis = spec["hero_mortis"]
    assert float(mortis["attack_cooldown"]) > float(mortis["dash_duration"])


# ---- (5) tick_hook: sampling the world at the SIM rate -------------------------------------

def test_tick_hook_fires_once_per_sub_tick_with_the_world_mid_decision():
    """What `scripts/watch.py` and `scripts/record_rollout.py` capture replay frames through, so
    a recording stays at the sim's 20 Hz instead of the decision rate."""
    repeat = 5
    env = _env(n_envs=2, overrides={"sim": {"action_repeat": repeat}})
    env.reset()

    seen = []
    env.tick_hook = lambda e: seen.append(int(e.state.step_count[0]))
    env.step(_idle(2))
    env.step(_idle(2))

    # Once per sub-tick, in order, and observing the world BETWEEN ticks -- not just the
    # post-decision state a caller could already see from step()'s return value.
    assert seen == list(range(1, 2 * repeat + 1))


def test_tick_hook_defaults_to_none_and_is_not_installed_by_training():
    env = _env(n_envs=1)
    assert env.tick_hook is None
    env.reset()
    env.step(_idle(1))       # the default hot path must not need one


def test_freezing_a_finished_env_does_not_freeze_its_neighbours():
    """The freeze is per-env and mask-driven, like every other per-env branch in the sim: env 0
    ends on sub-tick 1 and stops counting, env 1 runs the full window."""
    env = _env(n_envs=2, overrides={"sim": {"action_repeat": 5}, "zone": {"enabled": True}})
    env.reset()
    env.state.ent_hp[:, 0] = 1e9
    env.state.ent_max_hp[:, 0] = 1e9
    _park_hero_outside_the_safe_rect(env)
    # Only env 0's enemies die, so only env 0's episode ends this decision.
    env.state.ent_hp[0, 1:] = 0.0
    env.state.ent_ammo[0, 1:] = 0.0
    env.state.ent_last_hit_by[0, 1:] = -1

    *_, terminated, _, info = env.step(_idle(2))

    assert bool(terminated[0]) and not bool(terminated[1])
    assert int(info["n_ticks"][0]) == 1
    assert int(info["n_ticks"][1]) == 5
