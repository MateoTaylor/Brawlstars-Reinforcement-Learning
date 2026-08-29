"""Swept multi-hitscan melee (core/melee_sweep.py, bot_overhaul.md Step C2).

One test file per module, matching the repo's convention. The bot-side fire gate that decides
WHEN Buzz starts a sweep lives in tests/test_melee.py with the rest of the archetype's logic.
"""
import math

import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import TILE_BLOCKS_PROJ, TILE_BLOCKS_UNIT, TILE_IS_BUSH, Kind, Tile
from brawl_sim.core import combat, melee_sweep
from brawl_sim.core.state import allocate

CONFIGS = "configs"
_MELEE = int(Kind.BOT_MELEE)


class _FakeBank:
    def __init__(self, tiles):
        self.blocks_unit = TILE_BLOCKS_UNIT[tiles].unsqueeze(0)
        self.blocks_proj = TILE_BLOCKS_PROJ[tiles].unsqueeze(0)
        self.is_bush = TILE_IS_BUSH[tiles].unsqueeze(0)


def _cfg_and_params(n_enemies=1, map_h=20, map_w=20):
    cfg = load_config(
        f"{CONFIGS}/default.yaml",
        overrides={"world": {"map_h": map_h, "map_w": map_w},
                   "entities": {"n_enemies": n_enemies}, "zone": {"enabled": False}},
    )
    spec = {**yaml.safe_load(open(f"{CONFIGS}/default.yaml")),
            **yaml.safe_load(open(f"{CONFIGS}/brawlers.yaml"))}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    return cfg, build_params(cfg, n_envs=1, device="cpu", gen=gen, spec=spec)


def _armed_buzz(cfg, params, facing=0.0):
    """A Buzz on the tick his attack fired: `_attack_phase` has just set ent_attack_cd."""
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_kind[0, 1] = _MELEE
    state.ent_alive.fill_(True)
    state.ent_max_hp.fill_(1e9)
    state.ent_hp.fill_(1e9)
    state.ent_facing.fill_(facing)
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.ent_attack_cd[0, 1] = params.attack_cooldown[0, _MELEE].item()
    return state


def _run_sweep(state, cfg, params, bank=None, extra_ticks=6):
    """Advances the cooldown one tick at a time, collecting every sub-swing that fires.
    Returns [(elapsed_seconds, cone_dir_radians, dmg_to_entity_0), ...]."""
    cooldown = params.attack_cooldown[0, _MELEE].item()
    n_ticks = int(round(cooldown / cfg.dt)) + extra_ticks
    out = []
    for i in range(n_ticks):
        fire, cone_dir = melee_sweep.sweep_cone_dir(state, params, cfg)
        if bool(fire[0, 1]):
            dmg = 0.0
            if bank is not None:
                dmg_ent, _, _ = combat.melee_hitscan(state, fire, bank, params, cfg, cone_dir=cone_dir)
                dmg = dmg_ent[0, 0].item()
            out.append((i * cfg.dt, float(cone_dir[0, 1]), dmg))
        state.ent_attack_cd[0, 1] = max(0.0, float(state.ent_attack_cd[0, 1]) - cfg.dt)
    return out


# ---- schedule ---------------------------------------------------------------------------

def test_sweep_fires_exactly_hitscan_count_sub_swings():
    cfg, params = _cfg_and_params()
    state = _armed_buzz(cfg, params)
    fired = _run_sweep(state, cfg, params)
    assert len(fired) == int(params.hitscan_count[0, _MELEE])


def test_sub_swings_are_evenly_spaced_across_the_cooldown():
    """The cooldown IS the attack's duration (Step C1), so five swings across 1.0s land at
    0.0/0.2/0.4/0.6/0.8 -- the last one strictly inside the window, not on its edge."""
    cfg, params = _cfg_and_params()
    state = _armed_buzz(cfg, params)
    fired = _run_sweep(state, cfg, params)

    cooldown = params.attack_cooldown[0, _MELEE].item()
    interval = cooldown / int(params.hitscan_count[0, _MELEE])
    for k, (t, _dir, _dmg) in enumerate(fired):
        assert abs(t - k * interval) < cfg.dt / 2, f"sub-swing {k} landed at {t}, expected {k * interval}"
    assert fired[-1][0] < cooldown


def test_schedule_never_drops_or_doubles_a_swing_when_dt_does_not_divide_the_interval():
    """The floor()-crossing schedule exists to be robust to a ragged interval. 5 swings across a
    0.30s cooldown gives 0.06s intervals against a 0.05s tick -- deliberately not a whole number
    of ticks. A counter-based schedule would drift here; this must still fire exactly 5 times."""
    cfg, params = _cfg_and_params()
    params.attack_cooldown[0, _MELEE] = 0.30
    state = _armed_buzz(cfg, params)
    fired = _run_sweep(state, cfg, params)
    assert len(fired) == int(params.hitscan_count[0, _MELEE])
    assert len({round(t, 6) for t, _, _ in fired}) == len(fired), "a tick fired twice"


# ---- geometry ---------------------------------------------------------------------------

def test_fan_spans_the_sweep_angle_symmetrically_about_the_anchored_facing():
    cfg, params = _cfg_and_params()
    facing = 0.7
    state = _armed_buzz(cfg, params, facing=facing)
    dirs = [d for _t, d, _dmg in _run_sweep(state, cfg, params)]

    sweep = params.hitscan_sweep_rad[0, _MELEE].item()
    assert abs((dirs[-1] - dirs[0]) - sweep) < 1e-4
    assert abs(((dirs[0] + dirs[-1]) / 2.0) - facing) < 1e-4, "fan is not centred on the facing"
    gaps = [b - a for a, b in zip(dirs, dirs[1:])]
    assert max(gaps) - min(gaps) < 1e-4, "fan is not evenly spaced"


def test_sweep_runs_clockwise():
    """Clockwise is INCREASING angle here: y grows downward (CONVENTIONS.md 'Coordinates'), so a
    positive rotation from +x toward +y reads as clockwise on screen."""
    cfg, params = _cfg_and_params()
    state = _armed_buzz(cfg, params)
    dirs = [d for _t, d, _dmg in _run_sweep(state, cfg, params)]
    assert dirs == sorted(dirs)


# ---- damage (bot_overhaul.md D9) --------------------------------------------------------

def test_at_most_three_sub_swings_can_connect_with_one_target():
    """D9's whole reason for attack_arc_rad 0.65. CHARACTER_DETAILS: the hitscans "should overlap
    slightly, but it is unrealistic players are hit by more than 3". Checked across the full fan,
    not just dead centre -- a bearing between two sub-swings is the case that could catch 4."""
    cfg, params = _cfg_and_params()
    bank = _FakeBank(torch.zeros(20, 20, dtype=torch.int64))
    per_hit = params.base_damage[0, _MELEE].item()

    worst = 0
    for bearing_deg in range(-50, 51, 2):
        state = _armed_buzz(cfg, params)
        radius = 2.0
        theta = math.radians(bearing_deg)
        state.ent_pos[0, 0] = torch.tensor(
            [10.0 + radius * math.cos(theta), 10.0 + radius * math.sin(theta)]
        )
        hits = sum(1 for _t, _d, dmg in _run_sweep(state, cfg, params, bank=bank) if dmg > 0)
        worst = max(worst, hits)

    assert worst == 3, f"max sub-swings connecting was {worst}, expected exactly 3 (D9)"
    assert abs(3 * per_hit - 2520.0) < 1e-3, "D9's 2520 max-damage figure no longer holds"


def test_a_target_dead_centre_takes_the_full_three_hits():
    cfg, params = _cfg_and_params()
    bank = _FakeBank(torch.zeros(20, 20, dtype=torch.int64))
    state = _armed_buzz(cfg, params)
    state.ent_pos[0, 0] = torch.tensor([12.0, 10.0])  # straight ahead of facing 0.0
    total = sum(dmg for _t, _d, dmg in _run_sweep(state, cfg, params, bank=bank))
    assert abs(total - 3 * params.base_damage[0, _MELEE].item()) < 1e-3


# ---- the not-swept path is untouched ----------------------------------------------------

def test_hitscan_count_of_one_never_sweeps():
    """Every non-Buzz kind resolves `hitscan_count` to 0 or 1, and must keep the single
    instantaneous cone melee always had -- the sweep path must not fire for them at all."""
    cfg, params = _cfg_and_params()
    params.hitscan_count[0, _MELEE] = 1
    state = _armed_buzz(cfg, params)
    assert not bool(melee_sweep.is_swept(state, params)[0, 1])
    assert _run_sweep(state, cfg, params) == []


def test_is_swept_is_false_for_every_ranged_kind():
    cfg, params = _cfg_and_params()
    state = _armed_buzz(cfg, params)
    for kind in (Kind.HERO_MORTIS, Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_RIFLE):
        state.ent_kind[0, 1] = int(kind)
        assert not bool(melee_sweep.is_swept(state, params)[0, 1]), f"{kind!r} reports as swept"


# ---- integration through the real env ---------------------------------------------------

def _env_with_a_buzz():
    from brawl_sim.env import BrawlVecEnv
    cfg = load_config(
        f"{CONFIGS}/default.yaml",
        overrides={"entities": {"n_enemies": 2, "randomize_enemy_types": False,
                                "fixed_enemy_types": ["melee", "melee"]},
                   "zone": {"enabled": False}, "regen": {"enabled": False},
                   "observation": {"include_world_grid": False}},
    )
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    return env


def test_a_whole_sweep_costs_exactly_one_ammo():
    """bot_overhaul.md assumption A1, user-confirmed: Buzz spends 1 ammo to fire all five
    hitscans, not 1 per hitscan. If this ever became 5 his sustained DPS would collapse from
    ~1260 to ~250 and he would stop being a threat, so it is worth pinning explicitly."""
    env = _env_with_a_buzz()
    override = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([0, 1])  # entity 1: idle move, FIRE

    env.state.ent_ammo[0, 1] = 3.0
    env.state.ent_attack_cd[0, 1] = 0.0
    before = float(env.state.ent_ammo[0, 1])
    env.step(torch.zeros(1, 2, dtype=torch.int64), override=override)
    after_trigger = float(env.state.ent_ammo[0, 1])

    assert before - after_trigger <= 1.0 + 1e-4, "the trigger pull cost more than one ammo"

    # Drive the REST of the sweep with the fire bit DOWN. Ammo must not drop further (no sub-swing
    # charges ammo) and must not rise either (C1's cooldown pauses the reload).
    #
    # Bounded by the cooldown actually still running, not by a fixed step count: one env.step() is
    # `action_repeat` sim ticks = 0.25s, and Buzz's sweep is 1.0s, so a naive "4 more steps"
    # overshoots the end of the sweep and then correctly observes the reload resuming.
    idle = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    steps = 0
    while float(env.state.ent_attack_cd[0, 1]) > env.cfg.agent_dt and steps < 10:
        env.step(torch.zeros(1, 2, dtype=torch.int64), override=idle)
        steps += 1
        assert abs(float(env.state.ent_ammo[0, 1]) - after_trigger) < 1e-4, (
            "ammo moved mid-sweep -- either a sub-swing charged ammo, or the reload was not paused"
        )
    assert steps > 0, "the sweep was over before it could be observed"


def test_facing_is_frozen_during_a_sweep_but_the_body_still_moves():
    """D6: Buzz keeps moving, but keeps facing his initial attack direction."""
    env = _env_with_a_buzz()
    override = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    override[0, 1] = torch.tensor([3, 1])  # move bin 3 AND fire

    env.state.ent_ammo[0, 1] = 3.0
    env.state.ent_attack_cd[0, 1] = 0.0
    env.step(torch.zeros(1, 2, dtype=torch.int64), override=override)

    facing_at_start = float(env.state.ent_facing[0, 1])
    pos_at_start = env.state.ent_pos[0, 1].clone()

    # Keep walking in a DIFFERENT direction for the rest of the sweep.
    walk = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    walk[0, 1] = torch.tensor([11, 0])
    for _ in range(3):
        env.step(torch.zeros(1, 2, dtype=torch.int64), override=walk)
        assert bool(env.state.ent_attack_cd[0, 1] > 0), "sweep ended earlier than expected"
        assert abs(float(env.state.ent_facing[0, 1]) - facing_at_start) < 1e-5, (
            "facing moved mid-sweep; the fan would smear away from where it was aimed"
        )
    assert not torch.allclose(env.state.ent_pos[0, 1], pos_at_start), "the body did not move at all"

    # Once the cooldown expires, facing tracks movement again.
    while bool(env.state.ent_attack_cd[0, 1] > 0):
        env.step(torch.zeros(1, 2, dtype=torch.int64), override=walk)
    env.step(torch.zeros(1, 2, dtype=torch.int64), override=walk)
    assert abs(float(env.state.ent_facing[0, 1]) - facing_at_start) > 1e-3, (
        "facing stayed frozen after the sweep finished"
    )


def test_a_ranged_kind_still_turns_freely_while_on_cooldown():
    """The facing freeze is gated on the KIND being swept, not on `ent_attack_cd > 0` -- since
    Step C1 every brawler has a cooldown running after every shot, so the looser gate would pin
    all five brawlers' facing for 0.25s after every attack."""
    from brawl_sim.env import BrawlVecEnv
    cfg = load_config(
        f"{CONFIGS}/default.yaml",
        overrides={"entities": {"n_enemies": 2, "randomize_enemy_types": False,
                                "fixed_enemy_types": ["rifle", "rifle"]},
                   "zone": {"enabled": False}, "observation": {"include_world_grid": False}},
    )
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    env.state.ent_ammo[0, 1] = 3.0
    env.state.ent_attack_cd[0, 1] = 0.0

    fire = torch.full((1, cfg.n_entities, 2), -1, dtype=torch.int64)
    fire[0, 1] = torch.tensor([3, 1])
    env.step(torch.zeros(1, 2, dtype=torch.int64), override=fire)
    assert bool(env.state.ent_attack_cd[0, 1] > 0), "shooter is not on cooldown"

    facing_before = float(env.state.ent_facing[0, 1])
    walk = torch.full((1, cfg.n_entities, 2), -1, dtype=torch.int64)
    walk[0, 1] = torch.tensor([11, 0])
    env.step(torch.zeros(1, 2, dtype=torch.int64), override=walk)
    assert abs(float(env.state.ent_facing[0, 1]) - facing_before) > 1e-3, (
        "a ranged kind's facing was frozen by its cooldown -- the freeze leaked past swept kinds"
    )
