import math
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Tile, TILE_BLOCKS_UNIT
from brawl_sim.core import hero, movement, terrain
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class _FakeBank:
    def __init__(self, blocks_unit):
        self.blocks_unit = blocks_unit


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _bank_from_grid(tiles):
    return _FakeBank(blocks_unit=TILE_BLOCKS_UNIT[tiles].unsqueeze(0))


def _cfg_and_params(n_envs=1, extra_overrides=None):
    overrides = {"world": {"map_h": 20, "map_w": 20}}
    if extra_overrides:
        overrides = {**overrides, **extra_overrides}
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _fresh_state(cfg, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    return state


# ---- action_mask / decode_action ---------------------------------------------

def test_action_mask_move_always_true():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    mask = hero.action_mask(state, params, cfg)
    assert mask["move"].shape == (1, cfg.n_move_bins + 1)
    assert torch.all(mask["move"])


def test_action_mask_fire_true_when_ammo_and_cd_ok():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0
    mask = hero.action_mask(state, params, cfg)
    assert bool(mask["attack"][0, 1])
    assert bool(mask["attack"][0, 0])  # "don't fire" always legal


def test_firing_at_ammo_point_nine_does_nothing_and_mask_says_so():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 0.9
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0

    mask = hero.action_mask(state, params, cfg)
    assert not bool(mask["attack"][0, 1])

    action = torch.tensor([[0, 1]], dtype=torch.int64)  # idle move, attempt to fire
    move_dir, fire, _super, _gadget, _auto = hero.decode_action(action, state, params, cfg)
    assert not bool(fire[0])  # illegal fire silently becomes a no-op


def test_action_mask_fire_false_when_dashing():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.1
    mask = hero.action_mask(state, params, cfg)
    assert not bool(mask["attack"][0, 1])


def test_decode_action_idle_gives_zero_move_dir():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    action = torch.tensor([[0, 0]], dtype=torch.int64)
    move_dir, fire, _super, _gadget, _auto = hero.decode_action(action, state, params, cfg)
    assert torch.allclose(move_dir, torch.zeros(1, 2))


def test_decode_action_bin_zero_is_facing_east():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    action = torch.tensor([[1, 0]], dtype=torch.int64)  # move bin k=1 -> dir_from_bin(0,16)
    move_dir, fire, _super, _gadget, _auto = hero.decode_action(action, state, params, cfg)
    assert torch.allclose(move_dir, torch.tensor([[1.0, 0.0]]), atol=1e-5)


def test_decode_action_legal_fire_passes_through():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0
    action = torch.tensor([[0, 1]], dtype=torch.int64)
    move_dir, fire, _super, _gadget, _auto = hero.decode_action(action, state, params, cfg)
    assert bool(fire[0])


# ---- the gadget column ---------------------------------------------------------

def test_action_mask_is_twenty_one_wide_with_the_gadget_as_the_fourth_attack_column():
    """17 move bins + [no-fire, attack, super, gadget]. Literal widths: `cfg.action_nvec` is the
    thing under test, so deriving the expectation from it would compare the change to itself."""
    cfg, params = _cfg_and_params(n_envs=3)
    state = _fresh_state(cfg, n_envs=3)
    state.ent_ammo[:, 0] = 3.0
    mask = hero.action_mask(state, params, cfg)
    assert mask["move"].shape == (3, 17)
    assert mask["attack"].shape == (3, 4)
    assert torch.cat([mask["move"], mask["attack"]], dim=-1).shape == (3, 21)
    assert cfg.action_nvec == (17, 4)
    # Fresh hero: gadget starts charged (gadget_cd 0), no super charge yet.
    assert mask["attack"][0].tolist() == [True, True, False, True]


def test_the_gadget_is_legal_while_dashing_on_cooldown_and_with_an_empty_clip():
    """The gate is `alive & gadget_cd <= 0 & the kind has a gadget` and NOTHING else -- it is
    deliberately not ANDed with the `ready` term the attack and the super share."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 0.0
    state.ent_attack_cd[:, 0] = 0.30
    state.ent_dash_t[:, 0] = 0.15
    state.ent_super_charge[:, 0] = 99
    mask = hero.action_mask(state, params, cfg)["attack"]
    assert mask[0].tolist() == [True, False, False, True]

    move_dir, fire, sup, gadget, _auto = hero.decode_action(torch.tensor([[0, 3]]), state, params, cfg)
    assert bool(gadget[0]) and not bool(fire[0]) and not bool(sup[0])

    # The move column decodes on its own, whatever the attack column holds: bin 5 is straight
    # down the screen (+y), thrown gadget or not.
    move_dir, _fire, _sup, gadget, _auto = hero.decode_action(torch.tensor([[5, 3]]), state, params, cfg)
    assert bool(gadget[0])
    assert torch.allclose(move_dir, torch.tensor([[0.0, 1.0]]), atol=1e-6)


def test_the_gadget_is_illegal_while_its_cooldown_runs_and_for_the_dead():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0

    state.ent_gadget_cd[:, 0] = 0.05          # one tick left is still "not ready"
    assert hero.action_mask(state, params, cfg)["attack"][0].tolist() == [True, True, False, False]
    _mv, _fire, _sup, gadget, _auto = hero.decode_action(torch.tensor([[0, 3]]), state, params, cfg)
    assert not bool(gadget[0]), "a masked gadget request is a silent no-op"

    state.ent_gadget_cd[:, 0] = 17.95
    assert not bool(hero.action_mask(state, params, cfg)["attack"][0, 3])

    state.ent_gadget_cd[:, 0] = 0.0
    assert bool(hero.action_mask(state, params, cfg)["attack"][0, 3])
    state.ent_alive[:, 0] = False
    assert hero.action_mask(state, params, cfg)["attack"][0].tolist() == [True, False, False, False]


def test_a_kind_without_a_gadget_never_gets_the_column():
    """`gadget_cooldown: 0` is "this kind has no gadget" -- every bot. `gadget_cd <= 0`
    alone would read a bot's permanently-zero timer as READY, which is the bug the third term of
    the gate exists to prevent."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    assert float(params.gadget_cooldown[0, int(Kind.HERO_MORTIS)]) == 18.0
    assert float(params.gadget_cooldown[0, int(Kind.BOT_SNIPER)]) == 0.0

    ready = hero.gadget_ready(state, params)
    assert ready.shape == (1, cfg.n_entities)
    assert ready[0].tolist() == [True] + [False] * (cfg.n_entities - 1)

    state.ent_kind[:, 0] = int(Kind.BOT_SNIPER)   # a gadget-less kind in the hero's own slot
    state.ent_ammo[:, 0] = 3.0
    assert hero.action_mask(state, params, cfg)["attack"][0].tolist() == [True, True, False, False]
    _mv, _fire, _sup, gadget, _auto = hero.decode_action(torch.tensor([[0, 3]]), state, params, cfg)
    assert not bool(gadget[0])


def test_decode_action_yields_at_most_one_of_attack_super_gadget():
    cfg, params = _cfg_and_params(n_envs=4)
    state = _fresh_state(cfg, n_envs=4)
    state.ent_ammo[:, 0] = 3.0
    state.ent_super_charge[:, 0] = 99
    action = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.int64)
    _mv, fire, sup, gadget, _auto = hero.decode_action(action, state, params, cfg)
    assert fire.tolist() == [False, True, False, False]
    assert sup.tolist() == [False, False, True, False]
    assert gadget.tolist() == [False, False, False, True]


# ---- tick_timers ---------------------------------------------------------------

def test_ammo_regen_zero_to_full_in_max_ammo_times_reload_seconds():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo.fill_(0.0)
    max_ammo = params.max_ammo[0, int(Kind.HERO_MORTIS)].item()
    reload_seconds = params.reload_seconds[0, int(Kind.HERO_MORTIS)].item()
    total_time = max_ammo * reload_seconds
    n_ticks = int(round(total_time / cfg.dt))

    for _ in range(n_ticks):
        hero.tick_timers(state, params, cfg)

    assert torch.allclose(state.ent_ammo[:, 0], torch.tensor([max_ammo]), atol=1e-3)


def test_gadget_cooldown_counts_down_and_clamps_at_zero():
    """`ent_gadget_cd` is a countdown like ent_attack_cd, and 0 means READY -- which
    is also what core/state.zero_ leaves on reset, so "starts fully charged" costs nothing. A
    full 18 s cooldown runs out in 360 ticks at dt 0.05, for every entity (the decrement is not
    hero-only), and then sits at exactly 0 rather than going negative."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    assert torch.all(state.ent_gadget_cd == 0)  # fresh = ready
    cooldown = 18.0
    state.ent_gadget_cd.fill_(cooldown)
    n_ticks = int(round(cooldown / cfg.dt))
    for _ in range(n_ticks - 1):
        hero.tick_timers(state, params, cfg)
        assert torch.all(state.ent_gadget_cd > 0)
    for _ in range(2):  # the 360th tick lands on ~0 (float32 drift either side); the 361st clamps
        hero.tick_timers(state, params, cfg)
    assert torch.all(state.ent_gadget_cd == 0)


def test_firing_pauses_the_reload_for_exactly_attack_cooldown():
    """BRAWL_SIM_DESIGN.md §5: for `attack_cooldown` seconds after an attack you can neither
    attack nor accrue ammo.

    Asserted as a tick count rather than a duration so an off-by-one in the gate-vs-decrement
    ordering inside tick_timers is visible: the gate is read BEFORE the decrement, so a cooldown of
    k*dt must block exactly k ticks."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    kind = int(Kind.HERO_MORTIS)
    cooldown = params.attack_cooldown[0, kind].item()
    reload_seconds = params.reload_seconds[0, kind].item()
    expected_frozen_ticks = int(round(cooldown / cfg.dt))

    state.ent_ammo.fill_(0.0)
    state.ent_attack_cd[:, 0] = cooldown

    frozen = 0
    for _ in range(expected_frozen_ticks + 5):
        before = state.ent_ammo[0, 0].item()
        hero.tick_timers(state, params, cfg)
        if state.ent_ammo[0, 0].item() == before:
            frozen += 1
        else:
            break

    assert frozen == expected_frozen_ticks, (
        f"reload was frozen for {frozen} ticks, expected {expected_frozen_ticks} "
        f"(attack_cooldown {cooldown} / dt {cfg.dt})"
    )
    # ...and resumes at the ordinary rate immediately afterwards.
    before = state.ent_ammo[0, 0].item()
    hero.tick_timers(state, params, cfg)
    assert abs((state.ent_ammo[0, 0].item() - before) - cfg.dt / reload_seconds) < 1e-6


def test_reload_is_not_paused_when_no_cooldown_is_running():
    """The other half of the gate: an idle entity reloads at the full rate; the pause covers the
    FIRING case only -- `test_ammo_regen_zero_to_full_in_max_ammo_times_reload_seconds`
    passes for this reason and would keep passing even if the gate were inverted, so check the
    per-tick rate directly."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo.fill_(0.0)
    state.ent_attack_cd.fill_(0.0)
    reload_seconds = params.reload_seconds[0, int(Kind.HERO_MORTIS)].item()

    hero.tick_timers(state, params, cfg)
    assert abs(state.ent_ammo[0, 0].item() - cfg.dt / reload_seconds) < 1e-6


def test_ammo_never_exceeds_max():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo.fill_(0.0)
    max_ammo = params.max_ammo[0, int(Kind.HERO_MORTIS)].item()
    for _ in range(10000):
        hero.tick_timers(state, params, cfg)
    assert torch.all(state.ent_ammo[:, 0] <= max_ammo + 1e-5)


def test_countdowns_clamp_at_zero():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_attack_cd.fill_(0.01)
    state.ent_invuln_t.fill_(0.01)
    hero.tick_timers(state, params, cfg)
    hero.tick_timers(state, params, cfg)
    assert torch.all(state.ent_attack_cd >= 0)
    assert torch.all(state.ent_invuln_t >= 0)


def test_reload_continues_during_a_dash():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.2  # currently dashing
    hero.tick_timers(state, params, cfg)
    assert state.ent_ammo[0, 0].item() > 0.0


def test_tick_timers_does_not_touch_dash_t():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_dash_t.fill_(0.2)
    before = state.ent_dash_t.clone()
    hero.tick_timers(state, params, cfg)
    assert torch.equal(state.ent_dash_t, before)


# ---- start_dash: clipping ---------------------------------------------------

def test_dash_at_wall_clips_short_of_full_distance():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 5] = Tile.WALL  # wall face at x=5.0
    bank = _bank_from_grid(tiles)

    dash_distance = params.dash_distance[0, int(Kind.HERO_MORTIS)].item()
    dash_duration = params.dash_duration[0, int(Kind.HERO_MORTIS)].item()

    # Stand half a dash away from the wall face at x=5.0, so the wall always clips the dash to
    # roughly half its length no matter what brawlers.yaml's dash_distance happens to be. The
    # original version of this test hardcoded a 2.5-tile clearance and asserted
    # `dash_distance == 5.0`; when hero_mortis.dash_distance was corrected to the real game's 2.67
    # (CastingRange 8 / 3) the fixed clearance stopped being a meaningful obstruction and the
    # test went red while start_dash was behaving perfectly.
    wall_face_x = 5.0
    clearance = dash_distance / 2.0
    state.ent_pos[0, 0] = torch.tensor([wall_face_x - clearance, 5.5])
    state.ent_facing[0, 0] = 0.0
    state.ent_ammo[0, 0] = 3.0

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])  # dash east

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    traveled = state.ent_dash_speed[0, 0].item() * dash_duration

    assert traveled < dash_distance  # the wall shortened it
    # The dash stops where the BODY meets the face: its centre `radius` short of it, to within
    # one bisection step of the 0.5-tile sample grid plus the millitile clearance, never past.
    radius = params.unit_radius[0].item()
    contact = clearance - radius
    assert contact - cfg.los_step_tiles / 16 - 2e-3 <= traveled < contact
    # the dasher's full body -- not just its center -- must clear the wall face
    assert state.ent_pos[0, 0, 0].item() + traveled + radius < wall_face_x


def test_dashing_into_a_wall_does_not_permanently_trap_the_entity():
    # Regression for a real symptom: dashing at a wall used to be able to land the entity's
    # body (not just its center point) inside the wall tile. Once dash_t hit 0, apply_movement's
    # resolve_move rejects a candidate step outright unless the ENTIRE circle at the destination
    # is clear -- from inside the wall, a step back out was rejected every tick for the exact
    # same reason, forever, since nothing about the situation changes between ticks. Confirm the
    # dash-clip leaves enough clearance that normal movement can always walk the entity back away
    # from the wall afterward.
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 5] = Tile.WALL  # wall face at x=5.0
    bank = _bank_from_grid(tiles)

    # Stand right up against the wall face -- the worst case for landing embedded in it.
    state.ent_pos[0, 0] = torch.tensor([5.0 - params.unit_radius[0].item() - 1e-3, 5.5])
    state.ent_facing[0, 0] = 0.0
    state.ent_ammo[0, 0] = 3.0

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])  # dash east, straight into the wall

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    n_ticks = int(round(params.dash_duration[0, int(Kind.HERO_MORTIS)].item() / cfg.dt)) + 1
    for _ in range(n_ticks):
        hero.advance_dash(state, params, cfg)
    assert state.ent_dash_t[0, 0].item() == 0.0  # dash finished, normal movement active again

    pos_after_dash = state.ent_pos[0, 0, 0].item()

    walk_dir = torch.zeros(1, cfg.n_entities, 2)
    walk_dir[0, 0] = torch.tensor([-1.0, 0.0])  # try to walk back away from the wall
    for _ in range(5):
        movement.apply_movement(state, walk_dir, bank, params, cfg)

    assert state.ent_pos[0, 0, 0].item() < pos_after_dash  # actually moved, not permanently stuck


def test_dash_at_water_stops_at_shoreline():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 5] = Tile.WATER
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([2.5, 5.5])
    state.ent_facing[0, 0] = 0.0
    state.ent_ammo[0, 0] = 3.0

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    dash_duration = params.dash_duration[0, int(Kind.HERO_MORTIS)].item()
    traveled = state.ent_dash_speed[0, 0].item() * dash_duration
    radius = params.unit_radius[0].item()
    # water blocks bodies like a wall: the body stops at the shoreline, x = 5.0
    assert 2.5 - radius - cfg.los_step_tiles / 16 - 2e-3 <= traveled < 2.5 - radius


def test_dash_open_floor_travels_full_distance():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_ammo[0, 0] = 3.0
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    dash_distance = params.dash_distance[0, int(Kind.HERO_MORTIS)].item()
    dash_duration = params.dash_duration[0, int(Kind.HERO_MORTIS)].item()
    traveled = state.ent_dash_speed[0, 0].item() * dash_duration
    assert abs(traveled - dash_distance) < cfg.los_step_tiles + 1e-3


def test_start_dash_consumes_ammo_and_sets_timers():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_ammo[0, 0] = 3.0
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([0.0, 1.0])

    hero.start_dash(state, fire, move_dir, bank, params, cfg)

    dash_duration = params.dash_duration[0, int(Kind.HERO_MORTIS)].item()
    attack_cooldown = params.attack_cooldown[0, int(Kind.HERO_MORTIS)].item()
    assert state.ent_ammo[0, 0].item() == 2.0
    assert abs(state.ent_dash_t[0, 0].item() - dash_duration) < 1e-5
    assert state.ent_invuln_t[0, 0].item() == 0.0  # the dash grants no i-frames (2026-09-25)
    assert abs(state.ent_attack_cd[0, 0].item() - attack_cooldown) < 1e-5
    assert state.ent_shots_fired[0, 0].item() == 1
    assert not torch.any(state.ent_dash_hits[0, 0])


def test_start_dash_idle_uses_facing():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_ammo[0, 0] = 3.0
    state.ent_facing[0, 0] = 1.5707963  # facing north (pi/2)
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)  # idle move input

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    assert torch.allclose(state.ent_dash_dir[0, 0], torch.tensor([0.0, 1.0]), atol=1e-4)


def test_start_dash_gated_on_dash_distance_zero():
    # bots (kind=1 sniper) currently have no dash_distance defined -> defaults to 0 -> never dashes
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    before_pos = state.ent_dash_t[0, 1].clone()

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 1] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 1] = torch.tensor([1.0, 0.0])

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    assert state.ent_dash_t[0, 1].item() == before_pos.item() == 0.0


def test_dash_on_idle_block_suppresses_dash():
    cfg, params = _cfg_and_params(extra_overrides={"action": {"dash_on_idle": "block"}})
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_ammo[0, 0] = 3.0
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)  # idle

    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    assert state.ent_dash_t[0, 0].item() == 0.0
    assert state.ent_ammo[0, 0].item() == 3.0  # nothing consumed


# ---- advance_dash --------------------------------------------------------------

def _run_full_dash(cfg, params, state, dash_dir_xy):
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor(dash_dir_xy)
    hero.start_dash(state, fire, move_dir, bank, params, cfg)

    n_ticks = int(round(params.dash_duration[0, int(Kind.HERO_MORTIS)].item() / cfg.dt)) + 1
    total_dmg_ent = torch.zeros(1, cfg.n_entities)
    for _ in range(n_ticks):
        dmg_ent, dmg_by, dmg_box = hero.advance_dash(state, params, cfg)
        total_dmg_ent += dmg_ent
    return total_dmg_ent


def test_advance_dash_zero_walk_displacement_is_purely_dash_moved():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    start_pos = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 0] = start_pos.clone()
    state.ent_ammo[0, 0] = 3.0

    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    hero.start_dash(state, fire, move_dir, bank, params, cfg)

    expected_step = state.ent_dash_speed[0, 0].item() * cfg.dt
    hero.advance_dash(state, params, cfg)
    displacement = (state.ent_pos[0, 0] - start_pos).norm().item()
    assert abs(displacement - expected_step) < 1e-4


def test_advance_dash_hits_stationary_enemy_exactly_once():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([2.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([4.5, 10.0])  # sits on the dash line
    state.ent_hp[0, 1] = 999999.0
    state.ent_max_hp[0, 1] = 999999.0

    total_dmg = _run_full_dash(cfg, params, state, [1.0, 0.0])

    expected_hit_dmg = params.base_damage[0, int(Kind.HERO_MORTIS)].item()
    # HERO_MORTIS has no proj/attack stats beyond dash -- its "damage" for the dash hit is
    # effective_damage(HERO kind, cubes=0, params), i.e. base_damage unscaled.
    hits = (total_dmg[0, 1] / max(expected_hit_dmg, 1e-6)).round().item()
    assert hits == 1, f"expected exactly one hit, got {total_dmg[0, 1]} (~{hits}x base damage)"


def test_advance_dash_does_not_hit_dead_or_self():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([2.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([4.5, 10.0])
    state.ent_alive[0, 1] = False  # dead, should never be hit

    total_dmg = _run_full_dash(cfg, params, state, [1.0, 0.0])
    assert total_dmg[0, 1].item() == 0.0
    assert total_dmg[0, 0].item() == 0.0  # never hits self


def test_advance_dash_clears_state_on_completion():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    _run_full_dash(cfg, params, state, [1.0, 0.0])
    assert state.ent_dash_t[0, 0].item() == 0.0
    assert torch.allclose(state.ent_dash_dir[0, 0], torch.zeros(2))
    assert state.ent_dash_speed[0, 0].item() == 0.0
    assert not torch.any(state.ent_dash_hits[0, 0])


def test_advance_dash_damages_box_in_capsule():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([2.0, 10.0])
    state.box_pos[0, 0] = torch.tensor([4.5, 10.0])
    state.box_alive[0, 0] = True
    state.box_hp[0, 0] = 999999.0
    state.box_max_hp[0, 0] = 999999.0

    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    hero.start_dash(state, fire, move_dir, bank, params, cfg)

    n_ticks = int(round(params.dash_duration[0, int(Kind.HERO_MORTIS)].item() / cfg.dt)) + 1
    total_box_dmg = 0.0
    for _ in range(n_ticks):
        _, _, dmg_box = hero.advance_dash(state, params, cfg)
        total_box_dmg += dmg_box[0, 0].item()
    assert total_box_dmg > 0.0


def test_advance_dash_non_dashers_unaffected():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 1] = torch.tensor([5.0, 5.0])
    before = state.ent_pos[0, 1].clone()
    dmg_ent, dmg_by, dmg_box = hero.advance_dash(state, params, cfg)
    assert torch.equal(state.ent_pos[0, 1], before)
    assert torch.all(dmg_ent == 0)


# ---- long dash -------------------------------------------------------------------

def _mortis_state(cfg, params, idle_seconds=0.0):
    state = _fresh_state(cfg)
    state.ent_alive.fill_(True)
    state.ent_ammo.fill_(3.0)
    state.ent_attack_cd.fill_(0.0)
    state.ent_attack_idle_t.fill_(idle_seconds)
    return state


def test_long_dash_charges_only_by_not_attacking_never_reset_by_damage():
    """The reason ent_attack_idle_t exists rather than reusing ent_out_of_combat_t: a Mortis being
    shot at while repositioning must still build his long dash."""
    cfg, params = _cfg_and_params()
    state = _mortis_state(cfg, params)
    threshold = params.long_dash_seconds[0, int(Kind.HERO_MORTIS)].item()

    for _ in range(int(round(threshold / cfg.dt)) + 1):
        hero.tick_timers(state, params, cfg)
        # Simulate taking damage every tick, which resets the OTHER stopwatch.
        state.ent_out_of_combat_t.fill_(0.0)

    assert bool(hero.long_dash_ready(state, params)[0, 0]), (
        "long dash failed to charge while taking damage -- it must key off attacking only"
    )


def test_long_dash_charge_fraction_ramps_then_saturates():
    cfg, params = _cfg_and_params()
    state = _mortis_state(cfg, params)
    threshold = params.long_dash_seconds[0, int(Kind.HERO_MORTIS)].item()

    state.ent_attack_idle_t.fill_(0.0)
    assert abs(float(hero.long_dash_charge_frac(state, params)[0, 0])) < 1e-6
    state.ent_attack_idle_t.fill_(threshold / 2.0)
    assert abs(float(hero.long_dash_charge_frac(state, params)[0, 0]) - 0.5) < 1e-4
    state.ent_attack_idle_t.fill_(threshold * 10.0)
    assert abs(float(hero.long_dash_charge_frac(state, params)[0, 0]) - 1.0) < 1e-6
    assert bool(hero.long_dash_ready(state, params)[0, 0])


def test_a_kind_without_the_ability_never_charges_it():
    """Every bot resolves long_dash_seconds to 0. The gate is on that field being > 0, not on a
    threshold nobody reaches -- so an enormous idle time must still leave them unready."""
    cfg, params = _cfg_and_params()
    state = _mortis_state(cfg, params)
    state.ent_attack_idle_t.fill_(1e6)
    for kind in (Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_MELEE, Kind.BOT_RIFLE):
        state.ent_kind[0, 0] = int(kind)
        assert not bool(hero.long_dash_ready(state, params)[0, 0]), f"{kind!r} charged a long dash"
        assert float(hero.long_dash_charge_frac(state, params)[0, 0]) == 0.0
        assert float(hero.long_dash_scale(state, params)[0, 0]) == 1.0


def test_charged_dash_covers_exactly_the_multiplier_times_the_distance():
    # A 60x60 map, because a charged dash reaches 5.34 tiles and the default 20x20 fixture leaves
    # too little room to be sure neither dash is clipped by the border wall.
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    bank = _bank_from_grid(_grid(60, 60))   # wide open, so nothing clips either dash
    kind = int(Kind.HERO_MORTIS)
    distance = params.dash_distance[0, kind].item()
    multiplier = params.long_dash_multiplier[0, kind].item()
    duration = params.dash_duration[0, kind].item()

    def dash_speed(idle_seconds):
        state = _mortis_state(cfg, params, idle_seconds=idle_seconds)
        state.ent_pos.fill_(30.0)
        fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
        fire[0, 0] = True
        move = torch.zeros(1, cfg.n_entities, 2)
        move[0, 0] = torch.tensor([1.0, 0.0])
        hero.start_dash(state, fire, move, bank, params, cfg)
        return float(state.ent_dash_speed[0, 0])

    short = dash_speed(0.0)
    long = dash_speed(params.long_dash_seconds[0, kind].item() + 1.0)
    assert abs(short - distance / duration) < 1e-3
    assert abs(long - (distance * multiplier) / duration) < 1e-3
    # dash_duration is NOT scaled, so a long dash is also multiplier-times FASTER.
    assert abs(long / short - multiplier) < 1e-4


def test_a_charged_dash_into_a_wall_still_lands_the_body_clear_of_it():
    """The doubled distance doubles how many wall configurations the dash clip has to handle, and
    landing a hitbox inside a wall is a ONE-WAY TRAP: resolve_move rejects any step whose whole
    circle is not clear, so from inside a wall every escape attempt fails identically forever.
    Sweeps the approach distance so the clip is exercised across its whole range."""
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    kind = int(Kind.HERO_MORTIS)
    reach = params.dash_distance[0, kind].item() * params.long_dash_multiplier[0, kind].item()
    radius = params.unit_radius[0].item()

    tiles = _grid(60, 60)
    tiles[:, 30] = Tile.WALL
    bank = _bank_from_grid(tiles)

    for start_x in [30.0 - reach - 0.5 + 0.13 * i for i in range(24)]:
        state = _mortis_state(cfg, params, idle_seconds=999.0)
        state.ent_pos[0, 0] = torch.tensor([start_x, 30.5])
        fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
        fire[0, 0] = True
        move = torch.zeros(1, cfg.n_entities, 2)
        move[0, 0] = torch.tensor([1.0, 0.0])
        hero.start_dash(state, fire, move, bank, params, cfg)
        for _ in range(int(round(params.dash_duration[0, kind].item() / cfg.dt)) + 2):
            hero.advance_dash(state, params, cfg)

        landed_x = float(state.ent_pos[0, 0, 0])

        # The invariant that matters is NOT "the hitbox never overlaps the wall tile" -- a dash
        # that ends a hair short can leave the circle grazing it, and `movement.resolve_move`
        # simply refuses that one axis next tick. What must never happen is the entity's CENTRE
        # landing inside a blocking tile: from there every axis is blocked, resolve_move rejects
        # every candidate step identically, and it is stuck for the rest of the episode.
        assert landed_x < 30.0, (
            f"charged dash from x={start_x:.2f} landed its CENTRE at {landed_x:.3f}, inside the "
            f"wall tile at x=30 -- resolve_move can never move it out again"
        )

        # ...and prove it is genuinely not stuck, rather than inferring it from the geometry.
        before = state.ent_pos[0, 0].clone()
        away = torch.zeros(1, cfg.n_entities, 2)
        away[0, 0] = torch.tensor([-1.0, 0.0])
        movement.apply_movement(state, away, bank, params, cfg)
        assert not torch.allclose(state.ent_pos[0, 0], before), (
            f"entity that dashed from x={start_x:.2f} to {landed_x:.3f} cannot move away from the wall"
        )
        _ = radius  # kept for the failure messages above


def _dash_batch(cfg, params, bank, starts, direction, charged):
    """One hero dash per env from `starts` (K,2) along `direction` (2,), run to completion.
    Returns the landing positions (K,2) and the state, so the caller can walk them on."""
    k = starts.shape[0]
    state = _fresh_state(cfg, n_envs=k)
    state.ent_ammo.fill_(3.0)
    state.ent_attack_cd.fill_(0.0)
    state.ent_attack_idle_t.fill_(999.0 if charged else 0.0)
    state.ent_pos[:, 0] = starts
    fire = torch.zeros(k, cfg.n_entities, dtype=torch.bool)
    fire[:, 0] = True
    move = torch.zeros(k, cfg.n_entities, 2)
    move[:, 0] = direction
    hero.start_dash(state, fire, move, bank, params, cfg)
    for _ in range(10):
        hero.advance_dash(state, params, cfg)
    return state.ent_pos[:, 0].clone(), state


def test_no_dash_approach_angle_can_leave_an_entity_stuck_against_a_wall():
    """Regression sweep for the dash wedge, over approach ANGLE as well as distance, for both dash
    lengths (BRAWL_SIM_DESIGN.md §4).

    `terrain.circle_blocked` probes all 8 compass points, so a body that merely OVERLAPS a wall is
    blocked in every direction, including away from it, and stays stuck until its next dash. The
    old clip backed a centre-line hit off along the dash line only, which clears the body head-on
    but not beside a wall: at 30 degrees to the face 57 % of wall-meeting dashes landed inside it,
    at 10 degrees all of them. This test swept only head-on then, so it could not see that.

    For each angle, 120 clear start points spread over every gap from touching to out of reach,
    so the landing falls at every offset against the 0.5-tile sample grid.
    """
    k = 120
    cfg, params = _cfg_and_params(n_envs=k, extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    kind = int(Kind.HERO_MORTIS)
    radius = params.unit_radius[0].item()
    tiles = _grid(60, 60)
    tiles[:, 30] = Tile.WALL                    # the face is x = 30
    bank = _bank_from_grid(tiles)

    for charged in (False, True):
        reach = params.dash_distance[0, kind].item()
        if charged:
            reach *= params.long_dash_multiplier[0, kind].item()
        for deg in (5, 10, 20, 30, 45, 60, 75, 90):
            th = math.radians(deg)
            direction = torch.tensor([math.sin(th), math.cos(th)])   # +x into the wall, +y along it
            gap = 1e-3 + (reach + 1.0) * math.sin(th) * (torch.arange(k) + 0.5) / k
            starts = torch.stack([30.0 - radius - gap, torch.full((k,), 20.0)], dim=-1)
            landed, state = _dash_batch(cfg, params, bank, starts, direction, charged)

            inside = terrain.circle_blocked(bank.blocks_unit, state.map_id, landed,
                                            params.unit_radius, cfg)
            assert not inside.any(), (
                f"{'charged' if charged else 'uncharged'} dash at {deg} deg to the face landed "
                f"the body inside the wall from gap {gap[inside][0].item():.3f}"
            )
            away = torch.zeros(k, cfg.n_entities, 2)
            away[:, 0, 0] = -1.0
            movement.apply_movement(state, away, bank, params, cfg)
            assert ((state.ent_pos[:, 0] - landed).norm(dim=-1) > 0).all(), (
                f"a dash at {deg} deg left an entity unable to walk away from the wall"
            )


def test_a_glancing_dash_stops_on_its_line_at_the_wall_instead_of_sliding():
    """The lead's call (2026-09-25): walls stop momentum, they do not redirect it. At 20 degrees
    to the face with the body 0.3 tiles off it, the body meets the wall 0.3 / sin 20 = 0.88 tiles
    along the line; the dash ends there, on its own line, rather than turning along the wall."""
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    radius = params.unit_radius[0].item()
    tiles = _grid(60, 60)
    tiles[:, 30] = Tile.WALL
    bank = _bank_from_grid(tiles)
    th = math.radians(20)
    direction = torch.tensor([math.sin(th), math.cos(th)])
    start = torch.tensor([[30.0 - radius - 0.3, 20.0]])

    landed, _ = _dash_batch(cfg, params, bank, start, direction, charged=False)
    step = landed[0] - start[0]
    travelled = float(step.norm())
    contact = 0.3 / math.sin(th)
    assert contact - cfg.los_step_tiles / 16 - 2e-3 <= travelled < contact
    cross = float(step[0] * direction[1] - step[1] * direction[0])
    assert abs(cross) < 1e-4, "the landing left the dash line: it slid"


def test_a_dash_alongside_a_wall_with_the_body_clear_is_not_shortened():
    """The body test must not clip what does not touch: running parallel to a face with the body
    0.05 tiles off it, the dash covers its full distance."""
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    radius = params.unit_radius[0].item()
    tiles = _grid(60, 60)
    tiles[:, 30] = Tile.WALL
    bank = _bank_from_grid(tiles)
    start = torch.tensor([[30.0 - radius - 0.05, 20.0]])

    landed, _ = _dash_batch(cfg, params, bank, start, torch.tensor([0.0, 1.0]), charged=False)
    expected = params.dash_distance[0, int(Kind.HERO_MORTIS)].item()
    assert abs(float((landed[0] - start[0]).norm()) - expected) < 1e-3


def test_dash_travels_exactly_its_clipped_distance():
    """A dash must cover `dash_speed * dash_duration` and not one tick more.

    It used to overshoot by exactly one tick -- `dash_duration: 0.30` is not representable in
    float32, so after six subtractions of dt it left 1.19e-8 seconds, `is_dashing` stayed True,
    and every dash in the game travelled 7/6 = 1.167x its configured distance. That silently spent
    the wall-clearance margin `start_dash` computes, which is what made the trap above reachable.
    """
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    bank = _bank_from_grid(_grid(60, 60))
    kind = int(Kind.HERO_MORTIS)

    for charged in (False, True):
        expected = params.dash_distance[0, kind].item()
        if charged:
            expected *= params.long_dash_multiplier[0, kind].item()
        state = _mortis_state(cfg, params, idle_seconds=999.0 if charged else 0.0)
        state.ent_pos[0, 0] = torch.tensor([10.0, 30.5])
        start = state.ent_pos[0, 0].clone()
        fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
        fire[0, 0] = True
        move = torch.zeros(1, cfg.n_entities, 2)
        move[0, 0] = torch.tensor([1.0, 0.0])
        hero.start_dash(state, fire, move, bank, params, cfg)
        for _ in range(20):
            hero.advance_dash(state, params, cfg)

        travelled = float((state.ent_pos[0, 0] - start).norm())
        assert abs(travelled - expected) < 1e-3, (
            f"{'charged' if charged else 'uncharged'} dash travelled {travelled:.4f}, "
            f"expected {expected:.4f}"
        )


def test_attacking_spends_the_long_dash_charge():
    cfg, params = _cfg_and_params()
    bank = _bank_from_grid(_grid(60, 60))
    state = _mortis_state(cfg, params, idle_seconds=999.0)
    state.ent_pos.fill_(30.0)
    assert bool(hero.long_dash_ready(state, params)[0, 0])

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move = torch.zeros(1, cfg.n_entities, 2)
    move[0, 0] = torch.tensor([1.0, 0.0])
    hero.start_dash(state, fire, move, bank, params, cfg)
    # env._attack_phase performs the reset (after start_dash, so the dash itself still reads it).
    state.ent_attack_idle_t.copy_(torch.where(fire, torch.zeros_like(state.ent_attack_idle_t),
                                              state.ent_attack_idle_t))
    assert not bool(hero.long_dash_ready(state, params)[0, 0])


# ---- the auto-aimed attack (action.auto_aim, attack value 4; the lead, 2026-09-26) ------------

def _auto_cfg_and_params(n_envs=1):
    return _cfg_and_params(n_envs=n_envs, extra_overrides={"action": {"auto_aim": True}})


def test_the_auto_aim_flag_adds_a_fifth_attack_value_that_decodes_as_a_dash():
    """`action_nvec[1]` is 5 with the flag and 4 without. The fifth mask column equals the
    attack's (the same ammo, cooldown and dashing gates: it is the same dash), a 4 decodes to
    `fire` AND `auto` under the flag, and without the flag a 4 is a no-op like any illegal
    value, so a run trained before the flag is untouched."""
    cfg, params = _auto_cfg_and_params()
    assert cfg.action_nvec == (17, 5)
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0
    mask = hero.action_mask(state, params, cfg)["attack"]
    assert mask.shape == (1, 5)
    assert bool(mask[0, 1]) and bool(mask[0, 4])
    state.ent_ammo[:, 0] = 0.5
    mask = hero.action_mask(state, params, cfg)["attack"]
    assert not bool(mask[0, 1]) and not bool(mask[0, 4])
    state.ent_ammo[:, 0] = 3.0
    state.ent_dash_t[:, 0] = 0.2
    mask = hero.action_mask(state, params, cfg)["attack"]
    assert not bool(mask[0, 1]) and not bool(mask[0, 4])
    state.ent_dash_t[:, 0] = 0.0

    _mv, fire, sup, gadget, auto = hero.decode_action(torch.tensor([[3, 4]]), state, params, cfg)
    assert bool(fire) and bool(auto) and not bool(sup) and not bool(gadget)
    _mv, fire, _sup, _gadget, auto = hero.decode_action(torch.tensor([[3, 1]]), state, params, cfg)
    assert bool(fire) and not bool(auto)
    _mv, fire, _sup, _gadget, auto = hero.decode_action(torch.tensor([[3, 0]]), state, params, cfg)
    assert not bool(fire) and not bool(auto)

    plain, plain_params = _cfg_and_params()
    assert plain.action_nvec == (17, 4)
    state = _fresh_state(plain)
    state.ent_ammo[:, 0] = 3.0
    assert hero.action_mask(state, plain_params, plain)["attack"].shape == (1, 4)
    _mv, fire, _sup, _gadget, auto = hero.decode_action(torch.tensor([[3, 4]]), state, plain_params, plain)
    assert not bool(fire) and not bool(auto)
    assert hero.ATTACK_AUTO == 4 and hero._BOX_RADIUS == 0.5   # projectiles.BOX_RADIUS restated


def test_auto_aim_target_is_the_nearest_enemy_or_crate_in_reach_and_ignores_visibility():
    """Nearest by centre distance across BOTH lists, alive enemies and unbroken crates only,
    with no visibility term at all (the lead: the flag must reach an enemy out of view). With
    nothing in reach there is no target and the facing comes back for the caller to ignore."""
    cfg, params = _auto_cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_facing[0, 0] = math.pi / 2                      # facing +y
    state.ent_pos[0, 1:] = torch.tensor([50.0, 50.0])          # everyone else far away
    state.box_alive.fill_(False)
    direction, has = hero.auto_aim_target(state, params, cfg)
    assert direction.shape == (1, 2) and has.shape == (1,)
    assert not bool(has[0])
    assert torch.allclose(direction[0], torch.tensor([0.0, 1.0]), atol=1e-5)

    # An enemy 2 tiles east, inside the 3.77-tile reach: the direction is +x.
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
    direction, has = hero.auto_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([1.0, 0.0]), atol=1e-5)

    # A crate 1.5 tiles north is nearer than the enemy: the crate wins.
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.0, 8.5])
    direction, has = hero.auto_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([0.0, -1.0]), atol=1e-5)

    # A dead enemy one tile away does not count, nor does a broken crate at the feet.
    state.ent_alive[0, 2] = False
    state.ent_pos[0, 2] = torch.tensor([11.0, 10.0])
    state.box_pos[0, 1] = torch.tensor([10.3, 10.0])           # box_alive[0, 1] is False
    direction, has = hero.auto_aim_target(state, params, cfg)
    assert bool(has[0]) and torch.allclose(direction[0], torch.tensor([0.0, -1.0]), atol=1e-5)

    # A target ON the hero has no direction and counts as no target.
    state.box_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([50.0, 50.0])
    _direction, has = hero.auto_aim_target(state, params, cfg)
    assert not bool(has[0])


def test_auto_aim_reach_is_the_dash_reach_and_grows_with_the_long_dash():
    """The reach is `dash_distance * long_dash_scale + dash_radius + the target's body`:
    `unit_radius` for an enemy, the crate's radius for a crate. Just outside is no target, just
    inside is; a charged long dash multiplies the distance, and the reach grows with it."""
    cfg, params = _auto_cfg_and_params()
    kind = int(Kind.HERO_MORTIS)
    state = _fresh_state(cfg)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1:] = torch.tensor([50.0, 50.0])
    state.box_alive.fill_(False)
    dash = float(params.dash_distance[0, kind])
    enemy_reach = dash + float(params.dash_radius[0, kind]) + float(params.unit_radius[0])
    crate_reach = dash + float(params.dash_radius[0, kind]) + hero._BOX_RADIUS
    assert enemy_reach == pytest.approx(3.77, abs=0.01)

    state.ent_pos[0, 1] = torch.tensor([10.0 + enemy_reach + 0.05, 10.0])
    assert not bool(hero.auto_aim_target(state, params, cfg)[1][0])
    state.ent_pos[0, 1] = torch.tensor([10.0 + enemy_reach - 0.05, 10.0])
    assert bool(hero.auto_aim_target(state, params, cfg)[1][0])

    state.ent_pos[0, 1] = torch.tensor([50.0, 50.0])
    state.box_alive[0, 0] = True
    state.box_pos[0, 0] = torch.tensor([10.0, 10.0 - crate_reach - 0.05])
    assert not bool(hero.auto_aim_target(state, params, cfg)[1][0])
    state.box_pos[0, 0] = torch.tensor([10.0, 10.0 - crate_reach + 0.05])
    assert bool(hero.auto_aim_target(state, params, cfg)[1][0])

    # The long dash: `long_dash_seconds` without attacking, then the distance is multiplied.
    state.box_alive.fill_(False)
    mult = float(params.long_dash_multiplier[0, kind])
    assert mult > 1.0
    long_reach = enemy_reach + dash * (mult - 1.0)
    state.ent_attack_idle_t[0, 0] = float(params.long_dash_seconds[0, kind]) + 1.0
    state.ent_pos[0, 1] = torch.tensor([10.0 + long_reach - 0.05, 10.0])
    assert bool(hero.auto_aim_target(state, params, cfg)[1][0])
    state.ent_pos[0, 1] = torch.tensor([10.0 + long_reach + 0.05, 10.0])
    assert not bool(hero.auto_aim_target(state, params, cfg)[1][0])
