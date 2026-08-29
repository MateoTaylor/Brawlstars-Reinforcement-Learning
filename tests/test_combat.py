import dataclasses
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import DeathCause, Kind, Tile, TILE_BLOCKS_PROJ
from brawl_sim.core import combat, hero, stats, terrain
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


class _FakeBank:
    def __init__(self, blocks_proj, blocks_unit=None):
        self.blocks_proj = blocks_proj
        self.blocks_unit = blocks_proj if blocks_unit is None else blocks_unit


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _bank_from_grid(tiles):
    from brawl_sim.constants import TILE_BLOCKS_UNIT
    return _FakeBank(
        blocks_proj=TILE_BLOCKS_PROJ[tiles].unsqueeze(0),
        blocks_unit=TILE_BLOCKS_UNIT[tiles].unsqueeze(0),
    )


def _cfg_and_params(n_envs=1, map_h=20, map_w=20, extra_overrides=None):
    overrides = {"world": {"map_h": map_h, "map_w": map_w}}
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


def _fresh_state(cfg, params, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_MELEE)
    state.map_id.fill_(0)
    # sane non-zero HP by default -- allocate() zero-inits everything, and resolve_deaths
    # treats hp<=0 as dead, so leaving this at 0 would make every entity dead-on-arrival in
    # any test that doesn't explicitly override hp.
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_max_hp.copy_(max_hp)
    state.ent_hp.copy_(max_hp)
    state.ent_last_hit_by.fill_(-1)
    return state


# ---- apply_damage -------------------------------------------------------------

def test_apply_damage_basic_updates():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 1000.0
    dmg = torch.zeros(1, cfg.n_entities)
    dmg[0, 0] = 100.0
    attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)
    attacker[0, 0] = 2

    combat.apply_damage(state, dmg, int(DeathCause.COMBAT), attacker, params, cfg)

    assert state.ent_hp[0, 0].item() == 900.0
    assert state.ent_damage_taken[0, 0].item() == 100.0
    assert state.ent_last_hit_by[0, 0].item() == 2
    assert state.ent_out_of_combat_t[0, 0].item() == 0.0


def test_apply_damage_hp_clamps_at_zero():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 50.0
    dmg = torch.zeros(1, cfg.n_entities)
    dmg[0, 0] = 500.0
    attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)

    combat.apply_damage(state, dmg, int(DeathCause.COMBAT), attacker, params, cfg)
    assert state.ent_hp[0, 0].item() == 0.0


def test_dashing_hero_takes_zero_combat_damage_but_full_zone_damage():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_ammo[0, 0] = 3.0
    state.ent_hp[0, 0] = 1000.0

    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, 0] = True
    move_dir = torch.zeros(1, cfg.n_entities, 2)
    move_dir[0, 0] = torch.tensor([1.0, 0.0])
    hero.start_dash(state, fire, move_dir, bank, params, cfg)
    assert state.ent_invuln_t[0, 0].item() > 0.0  # sanity: dash grants invuln

    combat_dmg = torch.zeros(1, cfg.n_entities)
    combat_dmg[0, 0] = 500.0
    attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)
    attacker[0, 0] = 1
    combat.apply_damage(state, combat_dmg, int(DeathCause.COMBAT), attacker, params, cfg)
    assert state.ent_hp[0, 0].item() == 1000.0  # untouched

    zone_dmg = torch.zeros(1, cfg.n_entities)
    zone_dmg[0, 0] = 300.0
    zone_attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)
    combat.apply_damage(state, zone_dmg, int(DeathCause.ZONE), zone_attacker, params, cfg)
    assert state.ent_hp[0, 0].item() == 700.0  # zone damage still applied


def test_iframes_block_zone_when_configured():
    cfg, params = _cfg_and_params(extra_overrides={"zone": {"iframes_block_zone": True}})
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 1000.0
    state.ent_invuln_t[0, 0] = 1.0

    zone_dmg = torch.zeros(1, cfg.n_entities)
    zone_dmg[0, 0] = 300.0
    attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)
    combat.apply_damage(state, zone_dmg, int(DeathCause.ZONE), attacker, params, cfg)
    assert state.ent_hp[0, 0].item() == 1000.0


def test_apply_damage_does_not_touch_already_dead_entities():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_alive[0, 0] = False
    state.ent_hp[0, 0] = 0.0
    before_last_hit = state.ent_last_hit_by[0, 0].item()

    dmg = torch.zeros(1, cfg.n_entities)
    dmg[0, 0] = 100.0
    attacker = torch.full((1, cfg.n_entities), -1, dtype=torch.int64)
    attacker[0, 0] = 3
    combat.apply_damage(state, dmg, int(DeathCause.COMBAT), attacker, params, cfg)

    assert state.ent_last_hit_by[0, 0].item() == before_last_hit  # not credited


# ---- melee_hitscan --------------------------------------------------------------

def test_melee_hits_target_in_range_arc_and_los():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.ent_facing[0, 1] = 0.0  # facing east
    state.ent_pos[0, 0] = torch.tensor([11.0, 10.0])  # hero directly east, within melee range

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    dmg_ent, dmg_by, _dmg_box = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
    assert dmg_ent[0, 0].item() > 0.0


def test_melee_misses_target_outside_arc():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.ent_facing[0, 1] = 0.0  # facing east
    state.ent_pos[0, 0] = torch.tensor([9.0, 10.0])  # hero to the WEST, behind the attacker

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    dmg_ent, dmg_by, _dmg_box = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
    assert dmg_ent[0, 0].item() == 0.0


def test_melee_misses_target_out_of_range():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)

    state.ent_pos[0, 1] = torch.tensor([5.0, 10.0])
    state.ent_facing[0, 1] = 0.0
    state.ent_pos[0, 0] = torch.tensor([15.0, 10.0])  # far away

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire_mask[0, 1] = True
    dmg_ent, dmg_by, _dmg_box = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
    assert dmg_ent[0, 0].item() == 0.0


def test_melee_blocked_by_wall_but_not_by_water_bush_fence():
    cfg, params = _cfg_and_params()
    # Asserted, not asserted-in-a-comment: the victim-placement line below read "within melee
    # range 2.0" while the shipped BOT_MELEE attack_range was 3.0, so it had been lying for some
    # time. This test is about TILE BLOCKING, so the victim being inside the attacker's reach is
    # a precondition -- if a restat ever breaks it, every case would report "no hit" and the test
    # would pass for three of the four tiles for entirely the wrong reason.
    melee_range = params.attack_range[0, int(Kind.BOT_MELEE)].item()
    assert 1.8 < melee_range, f"test precondition: BOT_MELEE attack_range {melee_range} <= 1.8"

    for tile_type, expect_hit in (
        (Tile.WALL, False), (Tile.WATER, True), (Tile.BUSH, True), (Tile.FENCE, True),
    ):
        state = _fresh_state(cfg, params)
        tiles = _grid(20, 20)
        tiles[10, 10] = tile_type  # sits between attacker (col 9) and victim (col 11)
        bank = _bank_from_grid(tiles)

        state.ent_pos[0, 1] = torch.tensor([9.6, 10.5])
        state.ent_facing[0, 1] = 0.0
        state.ent_pos[0, 0] = torch.tensor([11.4, 10.5])  # distance 1.8 (see precondition above)
        fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
        fire_mask[0, 1] = True
        dmg_ent, _, _ = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
        hit = dmg_ent[0, 0].item() > 0.0
        assert hit == expect_hit, f"tile {tile_type!r}"


def test_melee_los_budget_matches_full_budget_ray():
    """bot_overhaul.md Step A1: melee_hitscan caps its LOS march at `params.cone_ray_tiles`
    instead of the full `cfg.ray_steps`. That deliberately gives a WRONG `los` for pairs further
    apart than the budget, so this pins the actual claim -- that the wrongness is unobservable,
    because every pair it affects is one `in_cone` already rejected.

    Compares real melee_hitscan output against a full-budget reference over randomized states on
    a wall-heavy map (walls are what make a short ray disagree with a long one at all).
    """
    cfg, params = _cfg_and_params(n_envs=4)
    torch.manual_seed(0)

    assert params.cone_ray_tiles > 0, "no kind has attack_arc_rad > 0; this test proves nothing"
    assert terrain.ray_steps_for(params.cone_ray_tiles, cfg) < cfg.ray_steps, (
        "cone budget is not actually shorter than the full budget -- no saving, and this test "
        "would pass vacuously"
    )

    for trial in range(12):
        state = _fresh_state(cfg, params, n_envs=4)
        tiles = _grid(20, 20)
        # Scatter interior walls so plenty of pairs have something between them.
        wall_mask = torch.rand(20, 20) < 0.18
        wall_mask[0, :] = wall_mask[-1, :] = wall_mask[:, 0] = wall_mask[:, -1] = False
        tiles[wall_mask] = int(Tile.WALL)
        bank = _bank_from_grid(tiles)

        state.ent_pos.copy_(1.5 + torch.rand(4, cfg.n_entities, 2) * 17.0)
        state.ent_facing.copy_((torch.rand(4, cfg.n_entities) * 2.0 - 1.0) * 3.14159)
        state.box_pos.copy_(1.5 + torch.rand(4, state.box_pos.shape[1], 2) * 17.0)
        state.box_alive.fill_(True)
        fire_mask = torch.rand(4, cfg.n_entities) < 0.7

        got = combat.melee_hitscan(state, fire_mask, bank, params, cfg)

        # Full-budget reference: same params object with the budget widened past any reachable
        # distance, so the march can never miss a wall.
        saved = params.cone_ray_tiles
        params.cone_ray_tiles = cfg.max_ray_tiles
        try:
            want = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
        finally:
            params.cone_ray_tiles = saved

        for name, a, b in zip(("dmg_ent", "dmg_by", "dmg_box"), got, want):
            assert torch.equal(a, b), f"trial {trial}: {name} differs under the shortened ray"


def test_melee_never_hits_self_or_non_firing():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    tiles = _grid(20, 20)
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 2] = torch.tensor([10.5, 10.0])

    fire_mask = torch.zeros(1, cfg.n_entities, dtype=torch.bool)  # nobody firing
    dmg_ent, _, _ = combat.melee_hitscan(state, fire_mask, bank, params, cfg)
    assert torch.all(dmg_ent == 0)


# ---- resolve_deaths --------------------------------------------------------------

def test_entity_at_exactly_zero_hp_is_dead():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 0.0
    newly_dead = combat.resolve_deaths(state, cfg)
    assert bool(newly_dead[0, 0])
    assert not bool(state.ent_alive[0, 0])


def test_entity_above_zero_hp_stays_alive():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 1.0
    newly_dead = combat.resolve_deaths(state, cfg)
    assert not bool(newly_dead[0, 0])
    assert bool(state.ent_alive[0, 0])


def test_resolve_deaths_records_step_and_combat_cause_and_credits_killer():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.step_count[0] = 42
    state.ent_hp[0, 0] = 0.0
    state.ent_last_hit_by[0, 0] = 3

    combat.resolve_deaths(state, cfg)
    assert state.ent_death_step[0, 0].item() == 42
    assert state.ent_death_cause[0, 0].item() == int(DeathCause.COMBAT)
    assert state.ent_kills[0, 3].item() == 1
    assert state.n_alive[0].item() == cfg.n_entities - 1


def test_resolve_deaths_zone_cause_credits_nobody():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 0.0
    state.ent_last_hit_by[0, 0] = -1  # zone sentinel

    combat.resolve_deaths(state, cfg)
    assert state.ent_death_cause[0, 0].item() == int(DeathCause.ZONE)
    assert torch.all(state.ent_kills == 0)


# ---- drop_cubes_on_death --------------------------------------------------------

def test_killing_bot_holding_three_cubes_drops_four():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_cubes[0, 1] = 3
    state.ent_pos[0, 1] = torch.tensor([7.0, 7.0])
    newly_dead = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    newly_dead[0, 1] = True

    combat.drop_cubes_on_death(state, newly_dead, params, cfg)

    alive_idx = torch.nonzero(state.pku_alive[0], as_tuple=True)[0]
    assert len(alive_idx) == 1
    p = alive_idx.item()
    assert state.pku_cubes[0, p].item() == 4  # cubes_on_kill_base(1) + victim's 3
    assert torch.allclose(state.pku_pos[0, p], torch.tensor([7.0, 7.0]))


def test_drop_victim_cubes_false_only_drops_base():
    cfg, params = _cfg_and_params(extra_overrides={"cubes": {"drop_victim_cubes": False}})
    state = _fresh_state(cfg, params)
    state.ent_cubes[0, 1] = 5
    newly_dead = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    newly_dead[0, 1] = True

    combat.drop_cubes_on_death(state, newly_dead, params, cfg)
    alive_idx = torch.nonzero(state.pku_alive[0], as_tuple=True)[0]
    assert state.pku_cubes[0, alive_idx.item()].item() == 1  # just cubes_on_kill_base


def test_zone_deaths_drop_cubes_too():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 1] = 0.0
    state.ent_last_hit_by[0, 1] = -1  # zone-caused
    newly_dead = combat.resolve_deaths(state, cfg)
    combat.drop_cubes_on_death(state, newly_dead, params, cfg)
    assert int(state.pku_alive.sum()) == 1


# ---- collect_pickups --------------------------------------------------------------

def test_pickup_with_three_entities_consumed_once():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.pku_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.pku_cubes[0, 0] = 2
    state.pku_alive[0, 0] = True
    for e in (0, 1, 2):
        state.ent_pos[0, e] = torch.tensor([10.0, 10.0])
        state.ent_cubes[0, e] = 0

    gained = combat.collect_pickups(state, params, cfg)
    assert gained.sum().item() == 2  # only one winner claims the 2 cubes
    assert int((gained[0] > 0).sum()) == 1
    assert not bool(state.pku_alive[0, 0])  # consumed


def test_lowest_index_wins_ties():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.pku_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.pku_cubes[0, 0] = 1
    state.pku_alive[0, 0] = True
    for e in (2, 1, 0):  # deliberately out of order
        state.ent_pos[0, e] = torch.tensor([10.0, 10.0])

    gained = combat.collect_pickups(state, params, cfg)
    winner = torch.nonzero(gained[0], as_tuple=True)[0]
    assert winner.item() == 0  # lowest entity index


def test_capped_entity_does_not_claim_and_leaves_pickup():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    max_cubes = int(params.max_cubes[0].item())
    state.pku_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.pku_cubes[0, 0] = 1
    state.pku_alive[0, 0] = True
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_cubes[0, 0] = max_cubes  # already capped

    gained = combat.collect_pickups(state, params, cfg)
    assert gained[0, 0].item() == 0
    assert bool(state.pku_alive[0, 0])  # left behind


def test_collect_pickups_preserves_hp_deficit():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.pku_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.pku_cubes[0, 0] = 4
    state.pku_alive[0, 0] = True
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_cubes[0, 0] = 0
    max_hp_before = params.base_hp[0, int(Kind.HERO_MORTIS)].item()
    state.ent_max_hp[0, 0] = max_hp_before
    state.ent_hp[0, 0] = max_hp_before - 100.0  # already missing 100 hp

    combat.collect_pickups(state, params, cfg)
    new_deficit = state.ent_max_hp[0, 0].item() - state.ent_hp[0, 0].item()
    assert abs(new_deficit - 100.0) < 1e-3


def test_out_of_radius_does_not_claim():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.pku_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.pku_cubes[0, 0] = 1
    state.pku_alive[0, 0] = True
    state.ent_pos[0, 0] = torch.tensor([15.0, 15.0])  # far away

    gained = combat.collect_pickups(state, params, cfg)
    assert torch.all(gained == 0)
    assert bool(state.pku_alive[0, 0])


# ---- apply_regen --------------------------------------------------------------

def test_apply_regen_noop_when_disabled():
    cfg, params = _cfg_and_params(extra_overrides={"regen": {"enabled": False}})
    assert cfg.regen_enabled is False
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 10.0
    state.ent_max_hp[0, 0] = 1000.0
    state.ent_out_of_combat_t[0, 0] = 999.0
    healed = combat.apply_regen(state, params, cfg)
    assert state.ent_hp[0, 0].item() == 10.0
    assert healed.shape == state.ent_hp.shape and float(healed.abs().sum()) == 0.0


def test_apply_regen_only_after_delay_and_clamped_at_max():
    # regen_delay/regen_fraction_per_second are SimParams (per-env tensors), not EnvConfig fields
    # -- load_config's `overrides` can't reach them, so set them directly on params instead.
    cfg, params = _cfg_and_params(extra_overrides={"regen": {"enabled": True}})
    params.regen_delay.fill_(3.0)
    params.regen_fraction_per_second.fill_(1.0)  # a full max-HP heal every second
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 0] = 10.0
    state.ent_max_hp[0, 0] = 50.0

    state.ent_out_of_combat_t[0, 0] = 1.0  # not yet past the delay
    combat.apply_regen(state, params, cfg)
    assert state.ent_hp[0, 0].item() == 10.0

    state.ent_out_of_combat_t[0, 0] = 5.0  # past the delay now
    healed = combat.apply_regen(state, params, cfg)
    assert state.ent_hp[0, 0].item() == pytest.approx(10.0 + 50.0 * cfg.dt)
    assert healed[0, 0].item() == pytest.approx(50.0 * cfg.dt)

    for _ in range(200):  # long enough to top up from anywhere
        healed = combat.apply_regen(state, params, cfg)
    assert state.ent_hp[0, 0].item() == 50.0  # clamped at max, never overshoots
    # ...and what it REPORTS healing is clamped with it: reward shaping (training/reward.py's
    # hp_healed) must not be paid for regen ticks that added nothing to a full HP bar.
    assert healed[0, 0].item() == 0.0


def test_apply_regen_rate_is_proportional_to_each_entitys_own_max_hp():
    """The point of the fraction: two brawlers with very different HP pools take the SAME time to
    heal, and a power cube (which raises max_hp) doesn't make topping up slower."""
    cfg, params = _cfg_and_params(extra_overrides={"regen": {"enabled": True}})
    params.regen_delay.fill_(0.0)
    params.regen_fraction_per_second.fill_(0.2)
    state = _fresh_state(cfg, params)
    state.ent_out_of_combat_t.fill_(99.0)

    state.ent_max_hp[0, 0] = 2000.0
    state.ent_max_hp[0, 1] = 8000.0
    state.ent_hp[0, 0] = 1.0
    state.ent_hp[0, 1] = 1.0

    ticks = 0
    while ticks < 1000 and not (
        state.ent_hp[0, 0].item() >= 2000.0 and state.ent_hp[0, 1].item() >= 8000.0
    ):
        combat.apply_regen(state, params, cfg)
        ticks += 1
    # 0.2 of max per second == 5 seconds from empty, whatever the pool size.
    assert ticks * cfg.dt == pytest.approx(5.0, abs=cfg.dt * 2)
