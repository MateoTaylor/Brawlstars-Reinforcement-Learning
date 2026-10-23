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


def test_the_default_action_space_is_17_move_values_by_4_attack_values():
    env = _tiny_env(n_envs=8)
    # (move bins + idle, attack): 0 = nothing, 1 = attack, 2 = super, 3 = gadget.
    assert env.cfg.action_nvec == (17, 4)


def test_snapshot_is_cpu_numpy():
    import numpy as np
    env = _tiny_env(n_envs=8)
    env.reset()
    snap = env.snapshot(0)
    assert isinstance(snap["ent_pos"], np.ndarray)


# ---- action latency ---------------------------------------------------------------------

def test_action_latency_ring_buffer_mechanics():
    # dt=0.05 -> action_latency_seconds=0.1 gives exactly 2 ticks of delay, L=3.
    env = _tiny_env(n_envs=4, overrides=_sim(action_latency_seconds=0.1))
    assert env.cfg.action_latency_ticks == 2
    env.reset()
    assert env.state.act_buf.shape[1] == 3

    a0 = torch.tensor([[5, 1]] * 4, dtype=torch.int64)
    a1 = torch.tensor([[7, 0]] * 4, dtype=torch.int64)
    a2 = torch.tensor([[9, 1]] * 4, dtype=torch.int64)

    eff0 = env._pop_action_buffer(a0)
    eff1 = env._pop_action_buffer(a1)
    eff2 = env._pop_action_buffer(a2)  # should surface a0, written 2 ticks ago

    # the first two reads hit the slots reset zeroed: idle, never a not-yet-due action
    assert not eff0.any() and not eff1.any()
    assert torch.equal(eff2, a0)
    assert torch.equal(env._pop_action_buffer(a0), a1)  # the head wrapped: still 2 behind
    # at 0 ticks it's a pass-through
    env2 = _tiny_env(n_envs=4, overrides=_sim(action_latency_seconds=0.0))
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


# ---- dash replaces the walk --------------------------------------------------------------

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
    """Two envs built alike step bit-identically (tests/test_integration.py pins that), so an
    all-sentinel override must leave every state tensor exactly where `override=None` does."""
    env = _tiny_env(n_envs=4, overrides={"entities": {"n_enemies": 2}})
    env2 = _tiny_env(n_envs=4, overrides={"entities": {"n_enemies": 2}})
    env.reset()
    env2.reset()
    override_all_off = torch.full((4, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        action = _random_action(4, gen)
        env.step(action.clone(), override=None)
        env2.step(action.clone(), override=override_all_off)
    for name in env.state.__slots__:
        assert torch.equal(getattr(env.state, name), getattr(env2.state, name)), name


# ---- reveal on attack ------------------------------------------------------------------------

def test_attacking_sets_the_attackers_reveal_timer():
    """`perception.reveal_after_attack` was loaded into SimParams and read by
    perception.visibility, but nothing ever WROTE ent_reveal_t -- so firing
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
    # out_of_combat_t; an RNG-stream change moved that coincidence and exposed it.
    env = _tiny_env(n_envs=1, overrides={**_PER_TICK, "regen": {"enabled": True}})
    env.reset()
    # Bots deal nothing: this test is about the REGEN CURVE, and a live lobby made it depend on
    # whether anyone reached the hero. Two ways that bit, both silent. A Buzz in range does
    # ~2500 HP/s against regen's 1600, so the heal never completes; worse, the hero pinned at 25%
    # can DIE inside a step, and the autoreset then re-samples params -- wiping the two fill_()
    # lines below and leaving the test measuring the shipped 0.13 rate against a 0.2 expectation.
    # A short delay and a fast rate (they were 4.0 s and 0.2/s, 135 ticks; 2026-10-20): the
    # curve's shape -- nothing inside the delay, a linear climb, a clamp at max -- is the same.
    env.params.base_damage[:, 1:] = 0.0
    env.params.regen_delay.fill_(0.5)
    env.params.regen_fraction_per_second.fill_(1.0)

    max_hp = env.state.ent_max_hp[0, 0].item()
    env.state.ent_hp[0, 0] = 0.25 * max_hp
    env.state.ent_out_of_combat_t[0, 0] = 0.0

    idle = torch.tensor([[0, 0]], dtype=torch.int64)
    # Through the delay, nothing happens. Re-pin HP each tick so a stray bot shot can't confuse
    # the "is it healing yet" check.
    for _ in range(int(0.4 / env.cfg.dt)):
        env.state.ent_hp[0, 0] = 0.25 * max_hp
        env.step(idle)
    assert env.state.ent_hp[0, 0].item() == pytest.approx(0.25 * max_hp, rel=1e-3)

    # Now let it run: all of max per second means full from 25% in about 0.75 seconds.
    env.state.ent_out_of_combat_t[0, 0] = 99.0
    for _ in range(int(1.0 / env.cfg.dt)):
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

    # `obs` is a zero-copy view into `state` (BRAWL_SIM_DESIGN.md §3) -- clone
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


def test_truncation_latches_when_the_cap_is_crossed_on_a_decisions_first_tick():
    """The cap is crossed on the FIRST of the decision's five ticks, which a count stepped from 0
    only meets when max_episode_steps is not a multiple of action_repeat. The clock test below
    covers caps met on a decision's last tick, and tests/test_action_repeat.py the boundary."""
    n_envs = 2
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    env.state.step_count.fill_(env.cfg.max_episode_steps - 1)

    obs, reward, terminated, truncated, info = env.step(_idle_action(n_envs))
    assert torch.all(truncated)
    assert torch.all(obs["meta"]["step_count"] == 0)  # autoreset happened


def _full_width_action(env, gen, without=()):
    """Random moves and the attack column over its WHOLE width (`cfg.action_nvec`), so supers
    and gadgets fire too -- `_random_action` draws attacks from {0, 1} only. `without` takes
    attack values out of the draw."""
    n_move, n_attack = env.cfg.action_nvec
    values = torch.tensor([v for v in range(n_attack) if v not in without])
    return torch.stack([torch.randint(0, n_move, (env.n_envs,), generator=gen),
                        values[torch.randint(0, len(values), (env.n_envs,), generator=gen)]], dim=1)


def test_episode_never_runs_past_the_decision_that_crosses_the_step_limit():
    """Every decision runs its whole `action_repeat` window, and only a finished env is reset
    to 0: no episode outlasts the decision that crosses `max_episode_steps`, and a clock-only
    end (truncated, not terminated) lands exactly at that decision's last tick. Two episode
    lengths of decisions, so every env finishes at least twice."""
    n_envs = 4
    env = _tiny_env(n_envs=n_envs)
    env.reset()
    repeat = env.cfg.action_repeat
    last_tick = env.cfg.max_agent_steps * repeat
    gen = torch.Generator().manual_seed(0)
    prev = env.state.step_count.to(torch.int64).clone()
    n_done = torch.zeros(n_envs, dtype=torch.int64)
    clock_only = 0

    for _ in range(2 * env.cfg.max_agent_steps):
        _obs, _reward, terminated, truncated, info = env.step(_random_action(n_envs, gen))
        done = terminated | truncated
        ended = info["final_observation"]["meta"]["step_count"].to(torch.int64)
        now = env.state.step_count.to(torch.int64)
        assert torch.equal(ended, prev + repeat)
        assert torch.equal(now, torch.where(done, torch.zeros_like(now), ended))
        assert torch.all(ended[done] <= last_tick)
        assert torch.all(ended[truncated & ~terminated] == last_tick)
        n_done += done.to(torch.int64)
        clock_only += int((truncated & ~terminated).sum())
        prev = now.clone()

    assert int(n_done.min()) >= 2, n_done.tolist()
    assert clock_only > 0, "every episode ended in a death or a win: the clock assert never ran"  # 3 at seed 0


# ---- debug_checks invariants over a rollout -----------------------------------------------

def test_debug_checks_invariants_hold_over_rollout():
    """`check_invariants` after every decision raises on any violation. Wide rather than long:
    64 envs for two episode lengths of decisions take every env through two full episodes and
    their autoresets -- about the env-decisions 8 envs for 1000 decisions did, at a third of
    the cost, since a decision costs nearly the same at 64 envs as at 8.

    The attack column draws the gadget (value 3) as well as the attack, and the test counts the
    throws so its path provably ran under the checks (~190 here, measured 2026-10-20). The super
    (value 2) stays out: its piercing bolt can fly past the map border before its range runs
    out, which `check_invariants` flags. The user accepted that bolt, unfixed (2026-10-21,
    BRAWL_SIM_DESIGN.md §2's accepted divergences), so the super stays out of this draw."""
    n_envs = 64
    env = _tiny_env(n_envs=n_envs, overrides={"engine": {"debug_checks": True}})
    assert env.cfg.debug_checks is True  # a misspelled override would make this test vacuous
    assert env.cfg.action_nvec[1] >= 4  # 3 is the gadget: it must be drawable below
    env.reset()
    gen = torch.Generator().manual_seed(0)
    throws = 0
    cd_prev = env.state.ent_gadget_cd[:, 0].clone()
    for _ in range(2 * env.cfg.max_agent_steps):
        env.step(_full_width_action(env, gen, without=(2,)))
        cd = env.state.ent_gadget_cd[:, 0]
        throws += int((cd > cd_prev).sum())  # a throw is the one write that raises the timer
        cd_prev = cd.clone()
    assert throws > 0


# ---- batched / no-NaN smoke, real (non-tiny) config ----------------------------------------

def test_batched_smoke_default_config():
    """Every float in the observation and the reward stays finite over two full episodes of the
    real config (the whole map rotation, zone on). The reward is train.yaml's shaped one: the
    env's default ZeroReward returned zeros, so its finiteness check could not fail."""
    from brawl_sim.training.config import load_train_config
    from brawl_sim.training.reward import ShapedReward

    cfg = load_config(CONFIGS_DEFAULT, overrides={"sim": {"max_episode_steps": 200}})
    reward_fn = ShapedReward(load_train_config("configs/train.yaml").reward)
    env = BrawlVecEnv(cfg, n_envs=16, device="cpu", seed=3, reward_fn=reward_fn)
    env.reset()
    gen = torch.Generator().manual_seed(0)
    nonzero = 0

    def _assert_finite(node):
        if isinstance(node, dict):
            for v in node.values():
                _assert_finite(v)
        elif torch.is_tensor(node) and node.is_floating_point():
            assert torch.isfinite(node).all()

    for _ in range(2 * cfg.max_agent_steps):
        obs, reward, terminated, truncated, info = env.step(_full_width_action(env, gen))
        _assert_finite(obs)
        _assert_finite(reward)
        nonzero += int((reward != 0).sum())
    assert nonzero > 0   # ~490 of the 1280 env-decisions, measured 2026-10-20


# ---- super charge: dash hits charge it -----------------------------------------------------

def _one_enemy_duel(enemy_at):
    """A per-tick tiny env with bot 2 dead, the hero at (10,10), bot 1 at `enemy_at` at its full
    HP, and the hero's super at 0. Returns (env, bot 1's HP before anything happens): the tests
    pin the DELTA, because the tick clamps HP to the kind's effective max, so an inflated HP set
    here would silently be undone (a 100000 became Shelly's 7800 before the hit landed)."""
    env = _tiny_env(n_envs=1, overrides=_PER_TICK)
    env.reset()
    st = env.state
    st.ent_super_charge.fill_(0)
    st.ent_alive[0, 2] = False
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    return env, st.ent_hp[0, 1].item()


# A gadget hit's damage without super charge is pinned through the real throw (attack value 3)
# by tests/test_gadget.py's test_env_revealed_enemy_at_one_and_a_half_tiles...; this file's copy
# spawned the spinner by hand and was removed on 2026-10-20.

def test_a_dash_hit_still_charges_the_super():
    """The other half of test_gadget.py's no-charge pin: a dash hit does charge the super. Move
    bin 1 is +x (geo.dir_from_bin(0)), straight into the enemy 0.3 tiles east."""
    env, hp_before = _one_enemy_duel([10.3, 10.0])
    st = env.state
    st.ent_ammo[0, 0] = 3.0
    st.ent_attack_cd[0, 0] = 0.0
    st.ent_dash_t[0, 0] = 0.0

    env.step(torch.tensor([[1, 1]], dtype=torch.int64))

    assert hp_before - st.ent_hp[0, 1].item() == 2000.0  # one dash hit at base_damage 2000
    assert int(st.ent_super_charge[0, 0]) == 1


# ---- the auto-aimed attack (action.auto_aim, attack value 4; the lead, 2026-09-26) ------------

def test_an_auto_aimed_attack_dashes_at_the_nearest_target_not_along_the_move_bin():
    """Attack value 4 dashes at the nearest enemy or crate in reach whatever the move bin says,
    and with nothing in reach it dashes along the move bin, as value 1 does. The dash costs
    what a plain dash costs (one ammo), and `move_dir` itself is untouched: the walk after the
    dash still follows the bin."""
    env = _tiny_env(n_envs=2, overrides={**_PER_TICK, "action": {"auto_aim": True}},
                    autoreset=False)
    env.reset()
    assert env.cfg.action_nvec == (17, 5)
    st = env.state
    st.ent_pos[:, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[:, 1:] = torch.tensor([17.0, 17.0])       # ~9.9 tiles away, out of reach
    st.box_alive.fill_(False)
    st.ent_ammo[:, 0] = 3.0
    st.ent_attack_cd[:, 0] = 0.0
    st.ent_dash_t[:, 0] = 0.0
    st.ent_pos[0, 1] = torch.tensor([12.0, 10.0])         # env 0: an enemy 2 tiles east
    action = torch.tensor([[5, 4], [5, 4]])               # bin 5 = +y; value 4 = auto-aim
    env.step(action)
    assert torch.allclose(st.ent_dash_dir[0, 0], torch.tensor([1.0, 0.0]), atol=1e-4)
    assert torch.allclose(st.ent_dash_dir[1, 0], torch.tensor([0.0, 1.0]), atol=1e-4)
    assert st.ent_dash_t[:, 0].min().item() > 0
    assert st.ent_ammo[:, 0].tolist() == pytest.approx([2.0, 2.0], abs=1e-5)
    assert st.ent_shots_fired[:, 0].tolist() == [1, 1]


def test_without_the_flag_a_4_in_the_attack_column_is_a_silent_no_op():
    """The default config keeps the 4-wide column, so every run trained before the flag decodes
    exactly as it did; a 4 there is an illegal value and does nothing."""
    env = _tiny_env(n_envs=1, overrides=_PER_TICK, autoreset=False)
    env.reset()
    assert env.cfg.action_nvec == (17, 4)
    st = env.state
    st.ent_pos[:, 1:] = torch.tensor([17.0, 17.0])
    st.ent_ammo[:, 0] = 3.0
    env.step(torch.tensor([[5, 4]]))
    assert st.ent_dash_t[0, 0].item() == 0.0
    assert st.ent_shots_fired[0, 0].item() == 0


def test_the_env_low_passes_the_velocity_bots_lead_by_only_under_lead_velocity_tau():
    """env._run_tick advances `ent_vel_seen` once `ent_vel` is final for the tick, after the dash
    phase (`bots.lead_velocity_tau`, archive/SIM_ISSUES_PLAN.md §5.1); tau 0 never writes it."""
    for tau in (0.0, 0.25):
        env = _tiny_env(n_envs=4, overrides={"bots": {"lead_velocity_tau": tau}})
        env.reset()
        for _ in range(4):
            env.step(_idle_action(4))
        seen, vel = env.state.ent_vel_seen, env.state.ent_vel
        assert vel.abs().sum() > 0   # the bots have been walking
        if tau > 0:
            assert seen.abs().sum() > 0 and not torch.equal(seen, vel)
        else:
            assert not seen.any()
