from pathlib import Path

import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Tile, TILE_BLOCKS_UNIT
from brawl_sim.core import hero, movement
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
    move_dir, fire, _super = hero.decode_action(action, state, params, cfg)
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
    move_dir, fire, _super = hero.decode_action(action, state, params, cfg)
    assert torch.allclose(move_dir, torch.zeros(1, 2))


def test_decode_action_bin_zero_is_facing_east():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    action = torch.tensor([[1, 0]], dtype=torch.int64)  # move bin k=1 -> dir_from_bin(0,16)
    move_dir, fire, _super = hero.decode_action(action, state, params, cfg)
    assert torch.allclose(move_dir, torch.tensor([[1.0, 0.0]]), atol=1e-5)


def test_decode_action_legal_fire_passes_through():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    state.ent_ammo[:, 0] = 3.0
    state.ent_attack_cd[:, 0] = 0.0
    state.ent_dash_t[:, 0] = 0.0
    action = torch.tensor([[0, 1]], dtype=torch.int64)
    move_dir, fire, _super = hero.decode_action(action, state, params, cfg)
    assert bool(fire[0])


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


def test_firing_pauses_the_reload_for_exactly_attack_cooldown():
    """Step C1 / bot_overhaul.md D7: for `attack_cooldown` seconds after an attack you can neither
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
    """The other half of the gate: an idle entity reloads exactly as it always did. Pins that C1
    changed the FIRING case only -- `test_ammo_regen_zero_to_full_in_max_ammo_times_reload_seconds`
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
    # march() only samples every los_step_tiles, and start_dash backs off one further step PLUS
    # unit_radius so the dasher's own body -- not just its center point -- clears the wall, so
    # the landing point sits within one step-plus-radius of the true clearance, on the safe side.
    radius = params.unit_radius[0].item()
    assert traveled <= clearance + 1e-6
    assert abs(traveled - clearance) <= cfg.los_step_tiles + radius + 1e-3
    assert state.ent_pos[0, 0, 0].item() + traveled < wall_face_x  # never lands inside the wall
    # the dasher's full body -- not just its center -- must clear the wall face
    assert state.ent_pos[0, 0, 0].item() + traveled + radius <= wall_face_x + 1e-6


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
    assert abs(traveled - 2.5) < cfg.los_step_tiles + radius + 1e-3


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
    assert abs(state.ent_invuln_t[0, 0].item() - dash_duration) < 1e-5
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


# ---- long dash (Step D1) ---------------------------------------------------------

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


def test_no_dash_approach_angle_can_leave_an_entity_stuck_against_a_wall():
    """Regression sweep for the trap Step D1 uncovered, checked for BOTH dash lengths.

    `terrain.circle_blocked` probes all 8 compass points at a candidate destination, so an entity
    whose circle merely OVERLAPS a wall is blocked in every direction -- including away from it --
    and is stuck for the rest of the episode. `start_dash` backs a clipped dash off to prevent
    that, but it used to march only as far as `dash_distance`, so a wall sitting between the last
    sample and the landing point was never detected and no back-off happened at all.

    Sweeps approach positions at 0.01-tile resolution so the landing point lands at every possible
    offset relative to both the wall face and march's 0.5-tile sample grid. Only positions that
    are themselves legal are dashed from -- an entity already overlapping the wall is stuck for
    reasons that have nothing to do with dashing.
    """
    cfg, params = _cfg_and_params(extra_overrides={"world": {"map_h": 60, "map_w": 60}})
    kind = int(Kind.HERO_MORTIS)
    radius = params.unit_radius[0].item()

    tiles = _grid(60, 60)
    tiles[:, 30] = Tile.WALL
    bank = _bank_from_grid(tiles)

    for charged in (False, True):
        reach = params.dash_distance[0, kind].item()
        if charged:
            reach *= params.long_dash_multiplier[0, kind].item()
        checked = 0
        for i in range(300):
            start_x = 30.0 - reach - 2.0 + i * 0.01
            if start_x + radius >= 30.0:
                continue
            state = _mortis_state(cfg, params, idle_seconds=999.0 if charged else 0.0)
            state.ent_pos[0, 0] = torch.tensor([start_x, 30.5])
            fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
            fire[0, 0] = True
            move = torch.zeros(1, cfg.n_entities, 2)
            move[0, 0] = torch.tensor([1.0, 0.0])
            hero.start_dash(state, fire, move, bank, params, cfg)
            for _ in range(10):
                hero.advance_dash(state, params, cfg)
            checked += 1

            before = state.ent_pos[0, 0].clone()
            away = torch.zeros(1, cfg.n_entities, 2)
            away[0, 0] = torch.tensor([-1.0, 0.0])
            movement.apply_movement(state, away, bank, params, cfg)
            assert not torch.allclose(state.ent_pos[0, 0], before), (
                f"{'charged' if charged else 'uncharged'} dash from x={start_x:.3f} landed at "
                f"{float(before[0]):.4f} and cannot move away from the wall at x=30"
            )
        assert checked > 100, "sweep did not exercise enough approach positions"


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
