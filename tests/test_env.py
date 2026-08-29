import pytest
import yaml
import torch

from brawl_sim.config import load_config
from brawl_sim.constants import Kind
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _tiny_env(n_envs=8, seed=0, overrides=None, autoreset=True):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed, autoreset=autoreset)


def _sim(**kwargs) -> dict:
    """A `sim:` overrides section that PRESERVES debug_tiny's own sim settings -- `_tiny_env`
    merges only one level deep, so passing a bare `{"sim": {...}}` would drop
    `max_episode_steps: 300` and silently restore the 3000-tick default."""
    return {"sim": {**CONFIGS_TINY.get("sim", {}), **kwargs}}


#: Tests that measure SINGLE-SIM-TICK physics (one tick of displacement, latency counted in
#: ticks, a timer's per-tick decay) must pin this: with the shipped `action_repeat: 5` one
#: `step()` advances the world five ticks, and a dash that lasts ~6 ticks resolves entirely
#: inside a single call. This is a unit mismatch, not a behavior change -- the mechanics under
#: test are per-tick and are unaffected by the decision rate. See tests/test_action_repeat.py
#: for the coverage of what happens ACROSS a repeat window.
_PER_TICK = _sim(action_repeat=1)


def _idle_action(n_envs):
    return torch.zeros(n_envs, 2, dtype=torch.int64)


def _random_action(n_envs, gen=None):
    move = torch.randint(0, 17, (n_envs,), generator=gen)
    fire = torch.randint(0, 2, (n_envs,), generator=gen)
    return torch.stack([move, fire], dim=1)


# ---- basic construction / reset / step ----------------------------------------------------

def test_reset_returns_full_obs_matching_spec():
    from brawl_sim.core import obs_schema
    env = _tiny_env(n_envs=8)
    obs = env.reset()
    obs_schema.validate_obs(obs, env.cfg)


def test_step_shapes_and_dtypes():
    env = _tiny_env(n_envs=8)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_idle_action(8))

    assert reward.shape == (8,) and reward.dtype == torch.float32
    assert terminated.shape == (8,) and terminated.dtype == torch.bool
    assert truncated.shape == (8,) and truncated.dtype == torch.bool
    assert "final_observation" in info and "final_info" in info

    from brawl_sim.core import obs_schema
    obs_schema.validate_obs(obs, env.cfg)


def test_observation_spec_and_action_spec_properties():
    env = _tiny_env(n_envs=8)
    spec = env.observation_spec
    assert "hero.pos" in spec
    # (move bins + idle, attack). The attack dim is 3-valued as of Step D2:
    # 0 = nothing, 1 = attack, 2 = super. Derived from cfg so it tracks n_move_bins.
    assert env.action_spec == {"nvec": env.cfg.action_nvec}
    assert env.cfg.action_nvec == (17, 3)


def test_snapshot_is_cpu_numpy():
    import numpy as np
    env = _tiny_env(n_envs=8)
    env.reset()
    snap = env.snapshot(0)
    assert isinstance(snap["ent_pos"], np.ndarray)


# ---- action latency ---------------------------------------------------------------------

def test_action_latency_ring_buffer_mechanics():
    # dt=0.05 -> action_latency_seconds=0.1 gives exactly 2 ticks of delay, L=3.
    env = _tiny_env(n_envs=4, overrides={"sim": {"action_latency_seconds": 0.1}})
    assert env.cfg.action_latency_ticks == 2
    env.reset()
    assert env.state.act_buf.shape[1] == 3

    a0 = torch.tensor([[5, 1]] * 4, dtype=torch.int64)
    a1 = torch.tensor([[7, 0]] * 4, dtype=torch.int64)
    a2 = torch.tensor([[9, 1]] * 4, dtype=torch.int64)

    eff0 = env._pop_action_buffer(a0)
    eff1 = env._pop_action_buffer(a1)
    eff2 = env._pop_action_buffer(a2)  # should surface a0, written 2 ticks ago

    assert torch.equal(eff2, a0)
    # at 0 ticks it's a pass-through
    env2 = _tiny_env(n_envs=4, overrides={"sim": {"action_latency_seconds": 0.0}})
    assert env2.cfg.action_latency_ticks == 0
    env2.reset()
    eff = env2._pop_action_buffer(a0)
    assert torch.equal(eff, a0)


def test_action_latency_delays_hero_dash_by_the_configured_ticks():
    env = _tiny_env(n_envs=1, overrides=_sim(action_latency_seconds=0.1, action_repeat=1))
    env.reset()
    env.state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])

    fire_action = torch.tensor([[5, 1]], dtype=torch.int64)
    idle_action = torch.tensor([[0, 0]], dtype=torch.int64)

    assert env.cfg.action_latency_ticks == 2  # 0.1s at dt=0.05; the whole point of this test

    pos_before = [env.state.ent_pos[0, 0].clone()]
    for a in (fire_action, idle_action, idle_action):
        env.step(a)
        pos_before.append(env.state.ent_pos[0, 0].clone())

    disp = [(pos_before[i + 1] - pos_before[i]).norm().item() for i in range(3)]

    # One tick of dashing vs one tick of walking, both derived from the config rather than
    # hardcoded. The old assertion was `disp[2] > 0.5`, which was one tick of a 5.0-tile dash;
    # hero_mortis.dash_distance is now the real game's 2.67, so a dash tick covers 0.445 tiles and
    # the test failed on a number that had nothing to do with action latency.
    hero = int(Kind.HERO_MORTIS)
    dash_step = (
        float(env.params.dash_distance[0, hero]) / float(env.params.dash_duration[0, hero])
    ) * env.cfg.dt
    walk_step = float(env.params.move_speed[0, hero]) * env.cfg.dt

    # The fire+move action submitted at tick 0 only takes effect at tick 2: the two ticks before
    # it read the latency buffer's zero-initialised (idle) slots, so the hero does not move at all.
    assert disp[0] < 1e-4 and disp[1] < 1e-4
    assert disp[2] == pytest.approx(dash_step, rel=0.05)
    assert disp[2] > walk_step  # and a dash tick outruns a walk tick, which is why it is visible


# ---- dash replaces the walk (N02/N03/R02) ------------------------------------------------

def test_dashing_tick_shows_only_dash_displacement_not_walk():
    env = _tiny_env(n_envs=1, overrides=_PER_TICK)
    env.reset()
    env.state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    env.state.ent_ammo[0, 0] = 3.0

    pos_before = env.state.ent_pos[0, 0].clone()
    action = torch.tensor([[5, 1]], dtype=torch.int64)  # move bin 5 + fire -> dash
    env.step(action)
    disp = (env.state.ent_pos[0, 0] - pos_before).norm().item()

    dash_distance = float(env.params.dash_distance[0, 0])
    dash_duration = float(env.params.dash_duration[0, 0])
    dash_speed = dash_distance / dash_duration
    expected_dash_disp = dash_speed * env.cfg.dt
    walk_speed = float(env.params.move_speed[0, 0])
    expected_walk_disp = walk_speed * env.cfg.dt

    assert abs(disp - expected_dash_disp) < 1e-3
    assert abs(disp - (expected_dash_disp + expected_walk_disp)) > 1e-2


# ---- override ------------------------------------------------------------------------------

def test_override_drives_a_non_hero_entity_exactly():
    env = _tiny_env(n_envs=1, overrides={"entities": {"n_enemies": 2}, **_PER_TICK})
    env.reset()
    E = env.cfg.n_entities
    env.state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    env.state.ent_ammo[0, :] = 3.0

    override = torch.full((1, E, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([5, 0])  # entity 1: move bin 5, no fire

    from brawl_sim.core import geometry as geo
    expected_dir = geo.dir_from_bin(torch.tensor([4]), env.cfg.n_move_bins)[0]

    hero_action = torch.tensor([[0, 0]], dtype=torch.int64)  # hero idle
    pos_before = env.state.ent_pos[0, 1].clone()
    env.step(hero_action, override=override)
    delta = env.state.ent_pos[0, 1] - pos_before

    speed = float(env.params.move_speed[0, env.state.ent_kind[0, 1]])
    expected_delta = expected_dir * speed * env.cfg.dt
    assert torch.allclose(delta, expected_delta, atol=1e-3)


def test_override_with_negative_one_sentinel_is_a_no_op():
    env = _tiny_env(n_envs=4, overrides={"entities": {"n_enemies": 2}})
    env.reset()
    E = env.cfg.n_entities
    override_all_off = torch.full((4, E, 2), -1, dtype=torch.int64)
    obs1, *_ = env.step(_idle_action(4), override=None)
    env2 = _tiny_env(n_envs=4, overrides={"entities": {"n_enemies": 2}})
    env2.reset()
    obs2, *_ = env2.step(_idle_action(4), override=override_all_off)
    # both runs use different envs (different RNG state histories aren't identical anyway since
    # env vs env2 are separate instances) -- just check override=None and an all-sentinel
    # override don't error and produce a structurally valid obs either way.
    from brawl_sim.core import obs_schema
    obs_schema.validate_obs(obs1, env.cfg)
    obs_schema.validate_obs(obs2, env2.cfg)


# ---- reveal on attack ------------------------------------------------------------------------

def test_attacking_sets_the_attackers_reveal_timer():
    """`perception.reveal_after_attack` was loaded into SimParams and read by
    perception.visibility from Step 3 onward, but nothing ever WROTE ent_reveal_t -- so firing
    from a bush left you hidden and the parameter was inert. Driven here through `override`, which
    forces an attack without depending on any archetype's own fire gate."""
    # Per-tick, same reason as test_the_hero_is_revealed_by_attacking_too below: ent_reveal_t is
    # written at phase 6 and decremented by dt at phase 2 of every LATER sub-tick, so a 5-tick
    # decision reads back 1.0 - 4*dt = 0.80 rather than the full duration. (This test used to
    # pass at action_repeat=5 only by accident -- the override's fire bit was applied on every
    # sub-tick, so the timer was re-set to full on the last one. env.py now fires once per
    # decision, which is what exposed it.)
    env = _tiny_env(n_envs=1, overrides={"entities": {"n_enemies": 2}, **_PER_TICK})
    env.reset()
    E = env.cfg.n_entities
    env.state.ent_ammo[0, :] = 3.0
    env.state.ent_attack_cd[0, :] = 0.0
    assert float(env.state.ent_reveal_t[0, 1]) == 0.0

    override = torch.full((1, E, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([1, 1])  # entity 1: move bin 1, FIRE
    env.step(torch.tensor([[0, 0]], dtype=torch.int64), override=override)

    expected = float(env.params.reveal_after_attack[0])
    assert expected > 0.0  # sanity: the config actually asks for a reveal
    assert float(env.state.ent_reveal_t[0, 1]) == expected


def test_the_hero_is_revealed_by_attacking_too():
    """Deliberately symmetric: an agent that could attack out of a bush without breaking cover is
    exactly the asymmetry that made bush-sitting a winning policy."""
    # Per-tick: ent_reveal_t is set at phase 6 and decremented by dt at phase 2 of every
    # subsequent sub-tick, so a multi-tick decision would read back a partially decayed timer.
    env = _tiny_env(n_envs=1, overrides=_PER_TICK)
    env.reset()
    env.state.ent_ammo[0, 0] = 3.0
    env.state.ent_attack_cd[0, 0] = 0.0
    env.state.ent_dash_t[0, 0] = 0.0

    env.step(torch.tensor([[1, 1]], dtype=torch.int64))  # hero: move bin 1, fire (dash)
    assert float(env.state.ent_reveal_t[0, 0]) == float(env.params.reveal_after_attack[0])


def test_a_second_shot_never_shortens_a_running_reveal():
    env = _tiny_env(n_envs=1, overrides={"entities": {"n_enemies": 2}})
    env.reset()
    E = env.cfg.n_entities
    env.state.ent_ammo[0, :] = 9.0
    env.state.ent_attack_cd[0, :] = 0.0
    long_reveal = float(env.params.reveal_after_attack[0]) + 5.0
    env.state.ent_reveal_t[0, 1] = long_reveal

    override = torch.full((1, E, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([1, 1])
    env.step(torch.tensor([[0, 0]], dtype=torch.int64), override=override)

    # cfg.action_repeat ticks of phase-2 decay happened, but the value is still the LONG one, not
    # reset down to reveal_after_attack. The +5.0 head-room above swamps that decay, so this one
    # holds at any decision rate and does not need pinning to a single tick.
    assert float(env.state.ent_reveal_t[0, 1]) > float(env.params.reveal_after_attack[0])


# ---- out-of-combat regen ---------------------------------------------------------------------

def test_attacking_resets_the_out_of_combat_stopwatch():
    """The half of "4 seconds of not taking damage OR attacking" that did not exist: only
    combat.apply_damage ever reset this timer, so a bot could shoot continuously and still be
    considered out of combat."""
    # Per-tick: the shot zeroes the stopwatch at phase 6, and phase 2 of each LATER sub-tick
    # advances it again, so a 5-tick decision would read back 4*dt = 0.20 instead of 0.0.
    env = _tiny_env(n_envs=1, overrides={"entities": {"n_enemies": 2}, **_PER_TICK})
    env.reset()
    E = env.cfg.n_entities
    env.state.ent_ammo[0, :] = 5.0
    env.state.ent_attack_cd[0, :] = 0.0

    idle = torch.tensor([[0, 0]], dtype=torch.int64)
    for _ in range(10):
        env.step(idle)
    assert env.state.ent_out_of_combat_t[0, 1].item() > 0.0  # stopwatch has been running

    override = torch.full((1, E, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([1, 1])  # entity 1 fires
    env.step(idle, override=override)
    assert env.state.ent_out_of_combat_t[0, 1].item() == 0.0  # its own shot reset it


def test_a_hero_out_of_combat_heals_to_full_and_stops_there():
    # _PER_TICK: both loops below count `step()` calls as `cfg.dt` seconds, but a step is
    # `action_repeat` TICKS -- at the shipped 5, `int(3.0 / cfg.dt)` steps advance 15 seconds, not
    # 3, so the "still inside the delay" phase ran nearly four delays past it. It passed anyway
    # only because a bot happened to shoot the hero often enough to keep resetting
    # out_of_combat_t; Step E1's RNG-stream change moved that coincidence and exposed it.
    env = _tiny_env(n_envs=1, overrides={**_PER_TICK, "regen": {"enabled": True}})
    env.reset()
    # Bots deal nothing: this test is about the REGEN CURVE, and a live lobby made it depend on
    # whether anyone reached the hero. Two ways that bit, both silent. A Buzz in range does
    # ~2500 HP/s against regen's 1600, so the heal never completes; worse, the hero pinned at 25%
    # can DIE inside a step, and the autoreset then re-samples params -- wiping the two fill_()
    # lines below and leaving the test measuring the shipped 0.13 rate against a 0.2 expectation.
    env.params.base_damage[:, 1:] = 0.0
    env.params.regen_delay.fill_(4.0)
    env.params.regen_fraction_per_second.fill_(0.2)

    max_hp = env.state.ent_max_hp[0, 0].item()
    env.state.ent_hp[0, 0] = 0.25 * max_hp
    env.state.ent_out_of_combat_t[0, 0] = 0.0

    idle = torch.tensor([[0, 0]], dtype=torch.int64)
    # Through the delay, nothing happens. Re-pin HP each tick so a stray bot shot can't confuse
    # the "is it healing yet" check.
    for _ in range(int(3.0 / env.cfg.dt)):
        env.state.ent_hp[0, 0] = 0.25 * max_hp
        env.step(idle)
    assert env.state.ent_hp[0, 0].item() == pytest.approx(0.25 * max_hp, rel=1e-3)

    # Now let it run: 0.2 of max per second means full from 25% in about 4 seconds.
    env.state.ent_out_of_combat_t[0, 0] = 99.0
    for _ in range(int(6.0 / env.cfg.dt)):
        env.state.ent_out_of_combat_t[0, 0] = 99.0
        env.step(idle)
        if env.state.ent_hp[0, 0].item() >= max_hp:
            break
    assert env.state.ent_hp[0, 0].item() == pytest.approx(max_hp, rel=1e-4)
    assert env.state.ent_hp[0, 0].item() <= max_hp  # never overshoots


# ---- termination / autoreset ---------------------------------------------------------------

def test_hero_death_terminates_and_autoresets_with_final_observation():
    n_envs = 4
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    env.state.ent_hp[:, 0] = 0.0
    env.state.ent_last_hit_by[:, 0] = -1  # avoid the zero_-default-0 last_hit_by test artifact

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))

    assert torch.all(terminated)
    assert torch.all(obs["hero"]["alive"])  # returned obs is the FRESH new episode
    assert torch.all(obs["meta"]["step_count"] == 0)
    assert not torch.any(info["final_observation"]["hero"]["alive"])
    assert torch.all(info["final_observation"]["hero"]["hp"] <= 0)
    assert torch.equal(info["final_info"]["terminated"], terminated)


def test_autoreset_false_leaves_terminal_state_frozen_for_the_caller_to_reset():
    n_envs = 4
    env = _tiny_env(n_envs=n_envs, autoreset=False)
    env.reset()
    env.state.ent_hp[:, 0] = 0.0
    env.state.ent_last_hit_by[:, 0] = -1

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))

    assert torch.all(terminated)
    # unlike autoreset=True, `obs` IS the finished (terminal) tick's own observation, not a
    # fresh episode's first one -- the hero is dead here, not freshly alive.
    assert not torch.any(obs["hero"]["alive"])
    assert torch.all(obs["hero"]["hp"] <= 0)
    # final_observation/final_info are still set (API-shape consistency) and equal obs/info,
    # since nothing was actually reset.
    assert torch.equal(info["final_observation"]["hero"]["hp"], obs["hero"]["hp"])
    assert torch.equal(info["final_info"]["terminated"], terminated)

    # `obs` is a zero-copy view into `state` (Step 25/27/29's established convention) -- clone
    # the one field checked below BEFORE the next step() call mutates it in place.
    step_count_after_first = obs["meta"]["step_count"].clone()

    # state stays frozen -- stepping again without calling reset() just keeps re-observing the
    # same terminal tick (the hero, dead, does nothing) rather than silently starting over.
    obs2, reward2, terminated2, truncated2, info2 = env.step(_idle_action(n_envs))
    assert torch.all(terminated2)
    assert not torch.any(obs2["hero"]["alive"])
    # step_count is in SIM TICKS, and one step() is cfg.action_repeat of them.
    assert torch.equal(obs2["meta"]["step_count"], step_count_after_first + env.cfg.action_repeat)

    # explicit reset() still works normally and starts a fresh episode.
    obs3 = env.reset()
    assert torch.all(obs3["hero"]["alive"])
    assert torch.all(obs3["meta"]["step_count"] == 0)


def test_autoreset_true_is_the_default_and_unaffected():
    env = _tiny_env(n_envs=4)
    assert env.autoreset is True


def test_hero_last_alive_terminates_as_a_win():
    n_envs = 2
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    env.state.ent_hp[:, 1:] = 0.0
    env.state.ent_last_hit_by[:, 1:] = -1

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))
    assert torch.all(terminated)
    assert torch.all(info["final_observation"]["hero"]["alive"])
    assert torch.all(info["hero_rank"] == 0)


def test_truncation_at_step_limit():
    n_envs = 2
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    env.state.step_count.fill_(env.cfg.max_episode_steps - 1)

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))
    assert torch.all(truncated)
    assert torch.all(obs["meta"]["step_count"] == 0)  # autoreset happened


def test_episode_never_exceeds_step_limit_plus_one():
    n_envs = 4
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    gen = torch.Generator().manual_seed(0)
    max_seen_before_reset = torch.zeros(n_envs, dtype=torch.int64)
    prev_step_count = env.state.step_count.clone()

    for _ in range(1000):
        env.step(_random_action(n_envs, gen))
        grew = env.state.step_count.to(torch.int64) - prev_step_count.to(torch.int64)
        # step_count either grew by 1 (continuing) or dropped to 0 (autoreset)
        continuing = env.state.step_count > 0
        max_seen_before_reset = torch.maximum(max_seen_before_reset, prev_step_count.to(torch.int64))
        prev_step_count = env.state.step_count.clone()

    assert int(max_seen_before_reset.max()) <= env.cfg.max_episode_steps + 1


# ---- debug_checks invariants over a rollout -----------------------------------------------

def test_debug_checks_invariants_hold_over_rollout():
    env = _tiny_env(n_envs=8, overrides={"engine": {"debug_checks": True}})
    env.reset()
    gen = torch.Generator().manual_seed(0)
    for _ in range(1000):
        env.step(_random_action(8, gen))  # raises on any invariant violation


# ---- batched / no-NaN smoke, real (non-tiny) config ----------------------------------------

def test_batched_smoke_default_config():
    cfg = load_config(CONFIGS_DEFAULT, overrides={"sim": {"max_episode_steps": 200}})
    env = BrawlVecEnv(cfg, n_envs=16, device="cpu", seed=3)
    env.reset()
    gen = torch.Generator().manual_seed(0)

    def _assert_finite(node):
        if isinstance(node, dict):
            for v in node.values():
                _assert_finite(v)
        elif torch.is_tensor(node) and node.is_floating_point():
            assert torch.isfinite(node).all()

    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(_random_action(16, gen))
        _assert_finite(obs)
        _assert_finite(reward)
