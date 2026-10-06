"""The gadget spinner.

The first half drives `hero.gadget_target` and `projectiles.spawn_gadget` by hand and steps
the projectile pipeline itself -- the mechanics, with no env. The second half (below the
"through env.step" banner) throws the gadget the only way the agent can: attack-column value 3
into `BrawlVecEnv.step`, so it covers the decode, the mask, `_attack_phase`'s wiring and the real
tick order. Numbers are pinned as literals from the shipped hero block (`gadget_cooldown 18.0`,
`gadget_range 2.0`, `gadget_flight_seconds 0.2`, `gadget_damage 2000`, `gadget_radius 1.0`,
`dt 0.05`, `los_step_tiles 0.5`); a test that derived them from `params` could not fail.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Tile, TILE_BLOCKS_PROJ
from brawl_sim.core import hero, stats
from brawl_sim.core import projectiles as proj
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

_HERO = 0
_INF = float("inf")


class _FakeBank:
    def __init__(self, blocks_proj):
        self.blocks_proj = blocks_proj


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _bank_from_grid(tiles):
    return _FakeBank(blocks_proj=TILE_BLOCKS_PROJ[tiles].unsqueeze(0))


def _cfg_and_params(n_envs=1, map_h=20, map_w=20, max_projectiles=16):
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "limits": {"max_projectiles": max_projectiles, "max_boxes": 8, "max_pickups": 8},
    })
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _fresh_state(cfg, n_envs=1):
    """Hero at (10,10) facing +x, every bot parked far away in the opposite corner (so a test that
    places one or two enemies deliberately is not surprised by the rest), all immortal."""
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, _HERO] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_pos.fill_(2.0)
    state.ent_pos[:, _HERO] = torch.tensor([10.0, 10.0])
    state.ent_facing.zero_()  # +x
    state.ent_hp.fill_(1e6)
    state.ent_max_hp.fill_(1e6)
    return state


def _vis_all_revealed(state):
    """The fair (N,E,E) mask with every living entity revealed to every other -- the open-floor,
    no-bush case of `bots/perception.visibility`, built by hand so this file does not depend on
    the bots package. `hero.gadget_target` strips the diagonal itself."""
    alive = state.ent_alive
    return alive.unsqueeze(1) & alive.unsqueeze(2)


def _target(state, params, bank, cfg, vis=None):
    if vis is None:
        vis = _vis_all_revealed(state)
    direction, travel = hero.gadget_target(state, vis, params, bank, cfg)
    return direction[0, _HERO], float(travel[0, _HERO])


def _throw(state, params, cfg, bank, vis=None):
    """The env's fire path in miniature: target, then spawn one spinner from the hero. Returns the
    (N,E) damage the spinner will carry."""
    if vis is None:
        vis = _vis_all_revealed(state)
    direction, travel = hero.gadget_target(state, vis, params, bank, cfg)
    fire = torch.zeros(1, cfg.n_entities, dtype=torch.bool)
    fire[0, _HERO] = True
    damage = stats.effective_gadget_damage(state.ent_kind, state.ent_cubes, params)
    proj.spawn_gadget(state, fire, state.ent_pos.clone(), direction, travel, damage, params, cfg)
    return damage


def _run_until_empty(state, bank, params, cfg, limit=50):
    """Steps projectiles until the buffer is empty. Returns (ticks, total dmg_ent (N,E), total
    dmg_box (N,B), per-tick list of dmg_ent[0])."""
    total_ent = torch.zeros(1, cfg.n_entities)
    total_box = torch.zeros(1, state.box_pos.shape[1])
    per_tick = []
    for tick in range(1, limit + 1):
        dmg_ent, _dmg_by, dmg_box, _heal, _charge = proj.step_projectiles(state, bank, params, cfg)
        total_ent += dmg_ent
        total_box += dmg_box
        per_tick.append(dmg_ent[0].clone())
        if not bool(state.prj_alive.any()):
            return tick, total_ent, total_box, per_tick
    return None, total_ent, total_box, per_tick


# ---- target selection --------------------------------------------------------------------

def test_revealed_enemy_inside_range_travel_is_the_distance_to_it():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([10.0, 11.5])  # 1.5 tiles straight up (+y)

    direction, travel = _target(state, params, bank, cfg)

    assert travel == pytest.approx(1.5, abs=1e-6)
    assert direction.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_revealed_enemy_past_range_travel_caps_at_gadget_range():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([12.8, 10.0])  # 2.8 tiles east

    direction, travel = _target(state, params, bank, cfg)

    assert travel == pytest.approx(2.0, abs=1e-6)
    assert direction.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_nothing_revealed_flies_full_range_along_facing():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.0, 10.0])  # an enemy 1 tile away, but NOT revealed
    state.ent_facing[0, _HERO] = torch.tensor(torch.pi / 2)  # facing +y
    vis = _vis_all_revealed(state)
    vis[0, _HERO, :] = False  # the hero sees nobody

    direction, travel = _target(state, params, bank, cfg, vis=vis)

    assert travel == pytest.approx(2.0, abs=1e-6)
    assert direction.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)  # facing, not the enemy


def test_hidden_close_enemy_is_skipped_for_a_revealed_farther_one():
    """The spinner homes only on what the agent can see. A concealed enemy one tile away must
    not pull the throw off the revealed one two tiles away."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.0, 10.0])  # 1 tile east, hidden
    state.ent_pos[0, 2] = torch.tensor([10.0, 12.0])  # 2 tiles north, revealed
    vis = _vis_all_revealed(state)
    vis[0, _HERO, 1] = False

    direction, travel = _target(state, params, bank, cfg, vis=vis)

    assert travel == pytest.approx(2.0, abs=1e-6)
    assert direction.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_wall_one_tile_away_clips_travel_below_it():
    """The spinner is an artillery shell that walls cannot stop in flight, so the pre-clip is
    what keeps it on the thrower's side. Wall tiles start at x=11 (one tile east of the hero at
    x=10); march samples at 0.5 (clear) and 1.0 (in the wall), and the landing point backs off
    one sample from the hit: 1.0 - 0.5 = 0.5."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 11] = Tile.WALL
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 1] = torch.tensor([13.0, 10.0])  # revealed, 3 tiles east, behind the wall

    direction, travel = _target(state, params, bank, cfg)

    assert direction.tolist() == pytest.approx([1.0, 0.0], abs=1e-6)
    assert travel == pytest.approx(0.5, abs=1e-6)
    assert travel < 1.0


def test_dead_enemy_is_never_a_target_even_if_the_mask_says_visible():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([10.5, 10.0])  # closest, but dead
    state.ent_alive[0, 1] = False
    state.ent_pos[0, 2] = torch.tensor([10.0, 11.2])  # alive, 1.2 north
    vis = torch.ones(1, cfg.n_entities, cfg.n_entities, dtype=torch.bool)  # a stale mask

    direction, travel = _target(state, params, bank, cfg, vis=vis)

    assert travel == pytest.approx(1.2, abs=1e-6)
    assert direction.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_gadget_target_shapes_and_bot_rows_are_zero_travel():
    """Computed for every entity: bots have `gadget_range 0`, so their travel is 0 and their
    direction is still a unit vector (nothing downstream reads either)."""
    cfg, params = _cfg_and_params(n_envs=3)
    state = _fresh_state(cfg, n_envs=3)
    bank = _bank_from_grid(_grid(20, 20))
    # Distinct spots so every row has a real nearest-target vector; the coincident case has
    # its own test below.
    for e in range(1, cfg.n_entities):
        state.ent_pos[:, e] = torch.tensor([2.0 + e, 2.0])
    direction, travel = hero.gadget_target(state, _vis_all_revealed(state), params, bank, cfg)

    assert direction.shape == (3, cfg.n_entities, 2)
    assert travel.shape == (3, cfg.n_entities)
    assert travel[:, 1:].abs().max().item() == 0.0
    assert torch.allclose(direction.norm(dim=-1), torch.ones(3, cfg.n_entities), atol=1e-5)


def test_coincident_target_falls_back_to_facing_with_zero_travel():
    """A revealed enemy standing ON the hero has no direction to normalize; the spinner is
    dropped in place (travel 0) and `dir` is the facing, so the unit-vector contract holds.
    Facing pi/2 (+y) so the fallback is distinguishable from the +x default."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([10.0, 10.0])
    state.ent_facing[0, _HERO] = 1.5707963705062866  # pi/2 in float32

    direction, travel = _target(state, params, bank, cfg)

    assert travel == 0.0
    assert direction.tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


# ---- the spinner's landing and blast, the parts that need no action wiring -----------------

def test_enemy_at_1_5_tiles_takes_2000_on_tick_4_and_the_hero_takes_0():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.0])

    _throw(state, params, cfg, bank)
    ticks, total, _box, per_tick = _run_until_empty(state, bank, params, cfg)

    assert ticks == 4  # 0.2 s of flight at dt 0.05
    assert per_tick[3][1].item() == 2000.0
    assert sum(float(t[1]) for t in per_tick[:3]) == 0.0  # nothing before it lands
    assert total[0, 1].item() == 2000.0
    assert total[0, _HERO].item() == 0.0


def test_enemy_at_2_8_tiles_is_caught_by_a_blast_landing_at_2_0():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([12.8, 10.0])  # 0.8 from the landing point at x=12

    _throw(state, params, cfg, bank)
    assert torch.allclose(state.prj_target[0, 0], torch.tensor([12.0, 10.0]), atol=1e-5)
    ticks, total, _box, _ = _run_until_empty(state, bank, params, cfg)

    assert ticks == 4
    assert total[0, 1].item() == 2000.0


def test_enemy_at_3_5_tiles_is_outside_the_blast():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([13.5, 10.0])  # 1.5 from the landing point, radius is 1.0

    _throw(state, params, cfg, bank)
    ticks, total, _box, _ = _run_until_empty(state, bank, params, cfg)

    assert ticks == 4
    assert total[0, 1].item() == 0.0


def test_two_enemies_and_a_box_inside_the_radius_all_take_2000():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.0])  # the target: landing point
    state.ent_pos[0, 2] = torch.tensor([11.5, 10.7])  # 0.7 from it
    state.box_alive[0, :] = False
    state.box_alive[0, 0] = True
    state.box_hp[0, 0] = 1e6
    state.box_max_hp[0, 0] = 1e6
    state.box_pos[0, 0] = torch.tensor([12.3, 10.0])  # 0.8 from it

    _throw(state, params, cfg, bank)
    _ticks, total, box_total, _ = _run_until_empty(state, bank, params, cfg)

    assert total[0, 1].item() == 2000.0
    assert total[0, 2].item() == 2000.0
    assert box_total[0, 0].item() == 2000.0
    assert total[0, _HERO].item() == 0.0


def test_enemy_behind_a_wall_at_1_5_tiles_is_not_hit():
    """Enemy behind a wall at 1.5 tiles: landing point clipped at the wall, enemy takes 0.
    Reachable geometry: hero at x=10.7, wall column at x=11, enemy at x=12.2 --
    1.5 tiles away, on the far side of the one-tile wall. (The literal reading with the hero at
    10.0 puts the enemy at 11.5, INSIDE the wall tile, which no entity can occupy; there the
    0.5-clipped landing point is exactly 1.0 from it and `<=` would catch it.) The march's first
    sample, 0.5 along the ray, is x=11.2 -- in the wall -- so `hit_t - los_step` is 0 and travel
    is clamped to 0: the spinner is dropped at the hero's feet and blasts nothing 1.5 tiles away.
    """
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 11] = Tile.WALL
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, _HERO] = torch.tensor([10.7, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.2, 10.0])

    _throw(state, params, cfg, bank)
    assert torch.allclose(state.prj_target[0, 0], torch.tensor([10.7, 10.0]), atol=1e-5)
    ticks, total, _box, _ = _run_until_empty(state, bank, params, cfg)

    assert ticks == 1  # zero travel: detonates in place on the first tick
    assert total[0, 1].item() == 0.0
    assert total[0, _HERO].item() == 0.0


def test_enemy_just_past_a_wall_from_a_hero_one_tile_back_is_not_hit():
    """The pre-review geometry, kept as the tightest legal case: hero at x=10.0, wall column at
    x=11, enemy at x=11.6 (the first floor tile past the wall, 0.1 inside it). The march samples
    0.5 (clear) and 1.0 (in the wall), so the landing point backs off to x=10.5 -- 1.1 from the
    enemy, 0.1 outside the 1.0 radius. An enemy at 11.5 WOULD be caught (`<=` at exactly 1.0),
    but 11.5 is inside the wall tile and unreachable."""
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    tiles = _grid(20, 20)
    tiles[:, 11] = Tile.WALL
    bank = _bank_from_grid(tiles)
    state.ent_pos[0, 1] = torch.tensor([11.6, 10.0])

    _throw(state, params, cfg, bank)
    assert torch.allclose(state.prj_target[0, 0], torch.tensor([10.5, 10.0]), atol=1e-5)
    _ticks, total, _box, _ = _run_until_empty(state, bank, params, cfg)

    assert total[0, 1].item() == 0.0


def test_a_gadget_holds_exactly_one_slot_for_its_four_ticks_of_flight():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.0])

    _throw(state, params, cfg, bank)
    assert int(state.prj_alive.sum()) == 1
    live_after = []
    for _ in range(5):
        proj.step_projectiles(state, bank, params, cfg)
        live_after.append(int(state.prj_alive.sum()))

    # Alive after ticks 1-3, gone on tick 4 (detonated; the hero has no split_count so nothing
    # takes the slot over), still gone on tick 5.
    assert live_after == [1, 1, 1, 0, 0]


def test_cubes_scale_the_spinner_like_they_scale_the_attack():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg)
    bank = _bank_from_grid(_grid(20, 20))
    state.ent_pos[0, 1] = torch.tensor([11.5, 10.0])
    state.ent_cubes[0, _HERO] = 5  # +10% each: 2000 * 1.5

    _throw(state, params, cfg, bank)
    _ticks, total, _box, _ = _run_until_empty(state, bank, params, cfg)

    assert total[0, 1].item() == pytest.approx(3000.0, abs=1e-3)


# ==== the same scenarios THROUGH env.step ======================================================
#
# `debug_tiny` (blank 20x20 map, 2 bots, zone off) at one tick per step, with `debug_checks` on so
# `check_invariants` (which bounds `ent_gadget_cd` to [0, gadget_cooldown]) runs after every step.
# Bots are frozen with an all-idle `override` -- otherwise they walk and shoot, and "the hero
# takes 0" or "the enemy is 0.8 tiles from the landing point" would depend on bot AI. The hero's
# own override slot is the -1 sentinel, so his action is the one passed to `step`.

_GADGET = 3      # attack-column values, as literals: importing them would hide a renumbering
_ATTACK = 1
_SPINNER = 7     # Proj.GADGET_SPINNER
_FAR = [3.0, 3.0]


def _env(action_repeat=1, max_episode_steps=2000, autoreset=True, n_envs=1, device="cpu",
         debug_checks=True):
    from brawl_sim.env import BrawlVecEnv

    tiny = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        **tiny,
        "sim": {**tiny["sim"], "action_repeat": action_repeat, "max_episode_steps": max_episode_steps},
        "engine": {"debug_checks": debug_checks, "compile": False},
    })
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=0, verbose=False, autoreset=autoreset)
    env.reset()
    return env


def _duel(enemy_at, other_at=None, action_repeat=1, autoreset=True):
    """Hero at (10,10) facing +x, bot 1 at `enemy_at`, bot 2 at `other_at` (far away by default),
    every crate dead. Returns `(env, override, hp_before (E,))`. HP is pinned as a DELTA: the
    tick clamps HP to the kind's effective max, so an inflated HP would be silently undone."""
    env = _env(action_repeat=action_repeat, autoreset=autoreset)
    st = env.state
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    st.ent_pos[0, 2] = torch.tensor(other_at if other_at is not None else _FAR)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    hp_before = st.ent_hp[0].clone()
    assert bool((hp_before > 2000.0).all()), "a 2000 hit must not be clipped by a kill"
    return env, override, hp_before


def _act(env, override, attack, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


def _spinners(state) -> int:
    return int((state.prj_alive[0] & (state.prj_kind[0] == _SPINNER)).sum())


def _throw_and_land(env, override, hp_before):
    """Gadget on step 1, idle after. Returns the per-step HP loss (5 rows of (E,))."""
    losses = []
    for step in range(5):
        _act(env, override, _GADGET if step == 0 else 0)
        losses.append((hp_before - env.state.ent_hp[0]).tolist())
    return losses


def test_env_action_nvec_has_the_four_wide_attack_column():
    env = _env()
    assert env.cfg.action_nvec == (17, 4)
    obs = env.reset()
    assert obs["action_mask"]["move"].shape == (1, 17)
    assert obs["action_mask"]["attack"].shape == (1, 4)
    assert obs["action_mask"]["attack"][0].tolist() == [True, True, False, True]


def test_env_revealed_enemy_at_one_and_a_half_tiles_takes_2000_on_tick_4_and_the_hero_takes_0():
    env, override, hp_before = _duel([11.5, 10.0])
    losses = _throw_and_land(env, override, hp_before)
    #                 hero  enemy  bystander
    assert losses == [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 2000.0, 0.0],      # 0.2 s of flight = 4 ticks, the throw tick included
                      [0.0, 2000.0, 0.0]]      # ...and exactly once
    assert env.state.ent_damage_dealt[0, 0].item() == 2000.0
    assert int(env.state.ent_super_charge[0, 0]) == 0   # the gadget charges no super


def test_env_enemy_at_two_point_eight_is_inside_the_blast_of_a_spinner_that_stops_at_two():
    env, override, hp_before = _duel([12.8, 10.0])
    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 2000.0, 0.0]


def test_env_enemy_at_three_and_a_half_is_out_of_reach():
    env, override, hp_before = _duel([13.5, 10.0])
    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 0.0, 0.0]


def test_env_two_enemies_and_a_crate_inside_the_radius_all_take_2000():
    env, override, hp_before = _duel([11.5, 10.0], other_at=[11.5, 10.7])
    st = env.state
    st.box_pos[0, 0] = torch.tensor([11.5, 9.2])     # 0.8 from the landing point
    st.box_alive[0, 0] = True
    st.box_hp[0, 0] = 3000.0
    st.box_pos[0, 1] = torch.tensor([11.5, 8.0])     # 2.0 from it: a control
    st.box_alive[0, 1] = True
    st.box_hp[0, 1] = 3000.0

    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 2000.0, 2000.0]
    assert st.box_hp[0, 0].item() == 1000.0
    assert st.box_hp[0, 1].item() == 3000.0


def _stamp_wall_column(env, x, y0, y1):
    """Writes a WALL column into the env's own (per-instance) map bank, padded copies included."""
    bank, cfg = env.bank, env.cfg
    pad_h, pad_w = cfg.view_h // 2, cfg.view_w // 2
    bank.tiles[0, y0:y1, x] = int(Tile.WALL)
    bank.blocks_unit[0, y0:y1, x] = True
    bank.blocks_proj[0, y0:y1, x] = True
    bank.pad_tiles[0, pad_h + y0:pad_h + y1, pad_w + x] = int(Tile.WALL)
    bank.pad_blocks_unit[0, pad_h + y0:pad_h + y1, pad_w + x] = True
    bank.pad_blocks_proj[0, pad_h + y0:pad_h + y1, pad_w + x] = True


def test_env_enemy_behind_a_wall_at_one_and_a_half_tiles_takes_0():
    """Reachable geometry: hero x 10.7, wall column 11, enemy x 12.2. The
    fair visibility has no wall term, so the enemy IS the target; the pre-throw terrain clip is
    what keeps the blast on the hero's side. The control below removes the wall and nothing else."""
    env, override, hp_before = _duel([12.2, 10.0])
    env.state.ent_pos[0, 0] = torch.tensor([10.7, 10.0])
    _stamp_wall_column(env, 11, 7, 14)
    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 0.0, 0.0]

    env, override, hp_before = _duel([12.2, 10.0])
    env.state.ent_pos[0, 0] = torch.tensor([10.7, 10.0])
    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 2000.0, 0.0]


def test_env_firing_sets_the_cooldown_to_18_and_the_mask_refuses_for_360_ticks():
    env, override, _hp = _duel([11.5, 10.0])
    st = env.state
    obs, *_ = _act(env, override, _GADGET)
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert st.ent_gadget_cd[0, 1:].tolist() == [0.0, 0.0]          # nobody else's timer moves
    assert obs["action_mask"]["attack"][0].tolist() == [True, True, False, False]

    obs, *_ = _act(env, override, 0)
    assert st.ent_gadget_cd[0, 0].item() == pytest.approx(17.95, abs=1e-4)

    legal = [bool(obs["action_mask"]["attack"][0, 3])]
    for _ in range(360):
        obs, *_ = _act(env, override, 0)
        legal.append(bool(obs["action_mask"]["attack"][0, 3]))
    # Index k = the observation k+1 ticks after the throw. 18.0 s / 0.05 s = 360 ticks.
    assert legal[:359] == [False] * 359
    assert legal[359] is True and legal[360] is True
    assert st.ent_gadget_cd[0, 0].item() == 0.0

    # ...and ready means it really can be thrown again.
    _act(env, override, _GADGET)
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert _spinners(st) == 1


def test_env_the_gadget_breaks_concealment_and_combat_but_not_the_long_dash_or_the_clip():
    """`reveal_after_attack` is 1.0 s; the long dash needs 4.5 s of not ATTACKING and a
    gadget is not an attack. Phase 2 (timers) runs before phase 6 (the throw), so the stopwatch
    that is NOT reset reads one tick more and the ones that are read exactly 0 / 1.0."""
    env, override, _hp = _duel([11.5, 10.0])
    st = env.state
    st.ent_attack_idle_t[0, 0] = 5.0
    st.ent_out_of_combat_t[0, 0] = 7.0
    assert st.ent_reveal_t[0, 0].item() == 0.0 and st.ent_ammo[0, 0].item() == 3.0

    _act(env, override, _GADGET)

    assert st.ent_reveal_t[0, 0].item() == 1.0
    assert st.ent_out_of_combat_t[0, 0].item() == 0.0
    assert st.ent_attack_idle_t[0, 0].item() == pytest.approx(5.05, abs=1e-5)
    assert bool(hero.long_dash_ready(st, env.params)[0, 0]), "the banked long dash survives"
    assert st.ent_ammo[0, 0].item() == 3.0
    assert st.ent_attack_cd[0, 0].item() == 0.0
    assert int(st.ent_shots_fired[0, 0]) == 0

    # Contrast: an ordinary attack (a dash along +x) DOES spend the long-dash charge.
    env, override, _hp = _duel([16.0, 16.0])
    st = env.state
    st.ent_attack_idle_t[0, 0] = 5.0
    _act(env, override, _ATTACK, move=1)
    assert st.ent_attack_idle_t[0, 0].item() == 0.0


def test_env_the_gadget_goes_mid_dash_and_on_an_empty_clip():
    """Through the tick: dash first, then throw while `dash_t > 0` and `attack_cd > 0`."""
    env, override, _hp = _duel([11.5, 13.0])
    st = env.state
    _act(env, override, _ATTACK, move=1)                 # dash along +x
    assert st.ent_dash_t[0, 0].item() > 0 and st.ent_attack_cd[0, 0].item() > 0
    st.ent_ammo[0, 0] = 0.0

    obs, *_ = _act(env, override, _GADGET)
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert _spinners(st) == 1
    assert st.ent_dash_t[0, 0].item() > 0, "the throw did not interrupt the dash"


def test_env_a_gadget_chosen_while_masked_is_a_no_op():
    env, override, hp_before = _duel([11.5, 10.0])
    st = env.state
    st.ent_gadget_cd[0, 0] = 5.0
    st.ent_out_of_combat_t[0, 0] = 7.0

    for _ in range(6):
        _act(env, override, _GADGET)

    assert int(st.prj_alive.sum()) == 0
    assert st.ent_gadget_cd[0, 0].item() == pytest.approx(4.7, abs=1e-4)   # 6 ticks of countdown only
    assert st.ent_reveal_t[0, 0].item() == 0.0
    assert st.ent_out_of_combat_t[0, 0].item() == pytest.approx(7.3, abs=1e-4)
    assert (hp_before - st.ent_hp[0]).tolist() == [0.0, 0.0, 0.0]


def test_attack_phase_safety_net_refuses_a_gadget_from_any_source():
    """`decode_action` already masks the hero's request, so the only way to reach
    `_attack_phase`'s own gate is to hand it an all-True `gadget_fire`, which is what an override
    or a future bot rule could do. Bots have `gadget_cooldown 0`: exactly one spinner, the hero's."""
    from brawl_sim.bots import perception

    env, _override, _hp = _duel([11.5, 10.0])
    st, cfg = env.state, env.cfg
    E = cfg.n_entities
    none = torch.zeros(1, E, dtype=torch.bool)
    zeros2 = torch.zeros(1, E, 2)
    everyone = torch.ones(1, E, dtype=torch.bool)
    vis = perception.visibility(st, env.bank, env.params, cfg)

    env._attack_phase(zeros2, none, none, zeros2, st.ent_pos.clone(), everyone, vis)
    assert int(st.prj_alive.sum()) == 1
    assert st.prj_owner[0][st.prj_alive[0]].tolist() == [0]
    assert st.ent_gadget_cd[0].tolist() == [18.0, 0.0, 0.0]
    assert st.ent_reveal_t[0].tolist() == [1.0, 0.0, 0.0]     # bots did not "attack" either

    env._attack_phase(zeros2, none, none, zeros2, st.ent_pos.clone(), everyone, vis)
    assert int(st.prj_alive.sum()) == 1, "cooldown running: the hero is refused too"


def test_env_one_gadget_per_decision_under_action_repeat():
    """The `_held` rule for column value 3. The 18 s cooldown would hide a broken `_held` on its
    own (sub-ticks 2..5 are masked anyway), so the tick hook zeroes the cooldown after every
    sub-tick: if the held action still carried the 3, every sub-tick would throw another one."""
    env, override, hp_before = _duel([11.5, 10.0], action_repeat=5)
    st = env.state
    spinners = []

    def _hook(e):
        spinners.append(_spinners(e.state))
        e.state.ent_gadget_cd.zero_()

    env.tick_hook = _hook
    _act(env, override, _GADGET)
    env.tick_hook = None

    # One decision = 5 sub-ticks; one spinner, alive for ticks 1-3 and detonated on tick 4.
    assert spinners == [1, 1, 1, 0, 0]
    assert (hp_before - st.ent_hp[0]).tolist() == [0.0, 2000.0, 0.0]

    # Without the hook the cooldown is 18.0 minus the 4 sub-ticks that followed the throw.
    env, override, _hp = _duel([11.5, 10.0], action_repeat=5)
    _act(env, override, _GADGET)
    assert env.state.ent_gadget_cd[0, 0].item() == pytest.approx(17.8, abs=1e-4)


def test_env_a_gadget_holds_one_projectile_slot_for_four_ticks():
    env, override, _hp = _duel([11.5, 10.0])
    st = env.state
    live = []
    for step in range(6):
        _act(env, override, _GADGET if step == 0 else 0)
        live.append(int(st.prj_alive.sum()))
    # Counted AFTER each tick: in flight after ticks 1-3, freed by the detonation on tick 4.
    assert live == [1, 1, 1, 0, 0, 0]


def test_env_with_nothing_revealed_the_spinner_flies_two_tiles_along_the_facing():
    """Both bots dead -> nothing to home on. Facing is +y here so the landing point is
    unmistakably the facing's and not a leftover +x default. `autoreset=False` because a hero
    alone on the map has WON: the episode ends on this very tick, and an autoreset would wipe
    the projectile buffer before the test could read it."""
    env, override, _hp = _duel([11.5, 10.0], autoreset=False)
    st = env.state
    st.ent_alive[0, 1:] = False
    st.ent_hp[0, 1:] = 0.0          # check_invariants: a dead entity has no HP
    st.ent_facing[0, 0] = 3.14159265 / 2

    _act(env, override, _GADGET)
    slot = st.prj_alive[0] & (st.prj_kind[0] == _SPINNER)
    assert int(slot.sum()) == 1
    target = st.prj_target[0][slot][0]
    assert target[0].item() == pytest.approx(10.0, abs=1e-4)
    assert target[1].item() == pytest.approx(12.0, abs=1e-4)


# =====================================================================================================
# The wiring the scenarios above cannot see
# =====================================================================================================
# Every `_duel` scenario above has the enemy on +x with the hero FACING +x, on a map with no bush.
# The facing fallback then lands inside the blast radius of every pinned target, so a throw that
# never homed, one that homed on a concealed enemy, and one that ignored power cubes all passed.
# Each test below was checked against the mutant of `_attack_phase` it is named for.

def _spinner_target(state) -> list[float]:
    slot = state.prj_alive[0] & (state.prj_kind[0] == _SPINNER)
    assert int(slot.sum()) == 1
    return [round(v, 4) for v in state.prj_target[0][slot][0].tolist()]


def test_env_the_spinner_homes_on_an_enemy_that_is_not_along_the_facing():
    """Hero faces +x, the enemy is 1.5 tiles away on +y. The facing fallback would land at
    (12, 10), 2.5 tiles from him. Kills `gadget_target(state, zeros_like(vis), ...)`."""
    env, override, hp_before = _duel([10.0, 11.5])
    _act(env, override, _GADGET)
    assert _spinner_target(env.state) == [10.0, 11.5]
    for _ in range(4):
        _act(env, override, 0)
    assert (hp_before - env.state.ent_hp[0]).tolist() == [0.0, 2000.0, 0.0]


def _bush_scene(bush: bool):
    """Hero at (10.5, 10) facing +x; bot 1 at (10.5, 12.6), which is 2.6 tiles away -- past the
    2.0 `bush_reveal_radius` and past the 2.0 gadget range, but 0.6 from where a spinner thrown AT
    him would land. Bot 2 is dead: alive at `_FAR` he is always revealed and would be the target."""
    env, override, _hp = _duel([10.5, 12.6])
    st = env.state
    st.ent_pos[0, 0] = torch.tensor([10.5, 10.0])
    st.ent_alive[0, 2] = False
    st.ent_hp[0, 2] = 0.0
    if bush:
        env.bank.is_bush[0, 12, 10] = True        # the tile bot 1 stands on, and only that one
    return env, override, st.ent_hp[0].clone()


def test_env_the_spinner_does_not_home_on_an_enemy_concealed_in_a_bush():
    """The FAIR visibility reaches `gadget_target`. Kills `ones_like(vis)`."""
    env, override, hp_before = _bush_scene(bush=True)
    _act(env, override, _GADGET)
    assert _spinner_target(env.state) == [12.5, 10.0]          # two tiles along the facing
    for _ in range(4):
        _act(env, override, 0)
    assert (hp_before - env.state.ent_hp[0]).tolist() == [0.0, 0.0, 0.0]

    # Control -- the same scene without the bush: he is seen, homed on, and caught by the blast.
    env, override, hp_before = _bush_scene(bush=False)
    _act(env, override, _GADGET)
    assert _spinner_target(env.state) == [10.5, 12.0]          # range-capped at 2.0 toward him
    for _ in range(4):
        _act(env, override, 0)
    assert (hp_before - env.state.ent_hp[0]).tolist() == [0.0, 2000.0, 0.0]


def test_env_power_cubes_scale_the_spinner_thrown_through_the_env():
    """Cube-scaled damage, through `_attack_phase` rather than this file's `_throw` helper
    (which calls `effective_gadget_damage` itself and so cannot see what the env calls).
    5 cubes at +10 % each: 2000 -> 3000. Kills the raw `gather_kind(params.gadget_damage, ...)`."""
    env, override, hp_before = _duel([11.5, 10.0])
    assert hp_before[1].item() > 3000.0
    env.state.ent_cubes[0, 0] = 5
    assert _throw_and_land(env, override, hp_before)[-1] == [0.0, 3000.0, 0.0]


def test_env_an_override_on_the_hero_slot_neither_cancels_nor_throws_a_gadget():
    """`_override_phase`'s contract, all three halves. An override drives movement and the
    ORDINARY attack only: the gadget always comes from the action, and an override's attack
    column is a bool, so a 3 there is an ordinary dash."""
    env, override, _hp = _duel([16.0, 16.0])
    st = env.state
    override[:, 0] = torch.tensor([0, 0])                      # hero overridden: idle, no fire
    _act(env, override, _GADGET)
    assert _spinners(st) == 1
    assert st.ent_gadget_cd[0, 0].item() == 18.0

    env, override, _hp = _duel([16.0, 16.0])
    st = env.state
    override[:, 0] = torch.tensor([1, 3])                      # move bin 1 (+x), attack column 3
    _act(env, override, 0)
    assert _spinners(st) == 0
    assert st.ent_gadget_cd[0, 0].item() == 0.0
    assert st.ent_ammo[0, 0].item() == 2.0                     # it was a dash: one ammo spent
    assert st.ent_dash_t[0, 0].item() > 0

    # The third half: an override that FIRES, under an action that asks for the gadget. The two
    # come from different sources, so BOTH happen on the one tick -- the only way a dash and a
    # gadget ever share one ("one column, one value" is a property of the ACTION). Reachable from
    # a test alone, since nothing overrides slot 0; pinned so it stays a decision, not an accident.
    env, override, _hp = _duel([16.0, 16.0])
    st = env.state
    st.ent_attack_idle_t[0, 0] = 2.0
    override[:, 0] = torch.tensor([1, 1])                      # move bin 1 (+x), ordinary attack
    _act(env, override, _GADGET)
    assert _spinners(st) == 1
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert st.ent_ammo[0, 0].item() == 2.0
    assert st.ent_dash_t[0, 0].item() > 0
    assert st.ent_attack_idle_t[0, 0].item() == 0.0            # reset by the dash, not the gadget


def test_env_an_episode_that_ends_with_the_cooldown_running_restarts_charged():
    """"Starts charged" holds across autoreset. The spinner kills the last enemy on tick
    4, the episode ends there, and the observation `step` returns is the NEXT episode's first
    one. Without the reset the timer would read 17.85 and column 3 would be False."""
    env, override, _hp = _duel([11.5, 10.0])
    st = env.state
    st.ent_hp[0, 1] = 100.0
    st.ent_alive[0, 2] = False
    st.ent_hp[0, 2] = 0.0

    ended = []
    for step in range(4):
        obs, _reward, terminated, _truncated, _info = _act(env, override, _GADGET if step == 0 else 0)
        ended.append(bool(terminated[0]))
    assert ended == [False, False, False, True]
    assert st.ent_gadget_cd[0].tolist() == [0.0, 0.0, 0.0]
    assert obs["action_mask"]["attack"][0].tolist() == [True, True, False, True]


# =====================================================================================================
# What one env, one tick per step and a standing hero cannot see
# =====================================================================================================
# Every scenario above runs ONE env at one tick per decision with the hero standing still and an
# empty projectile buffer. Each test below was checked against the mutant of the wiring it names
# (scratch copies only).

def _duel_batch(n_envs):
    """`_duel([11.5, 10.0])` in every env of a batch. The scene is identical across rows, so any
    difference between rows afterwards is the per-env ACTION's doing -- or a leak between rows."""
    env = _env(n_envs=n_envs)
    st = env.state
    st.ent_pos[:, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[:, 1] = torch.tensor([11.5, 10.0])
    st.ent_pos[:, 2] = torch.tensor(_FAR)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(n_envs, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    hp_before = st.ent_hp.clone()
    assert bool((hp_before > 2000.0).all())
    return env, override, hp_before


def test_env_each_env_of_a_batch_throws_only_its_own_gadget():
    """Env 0 idles with its gadget READY, env 1 throws, env 2 asks while 5.0 s of cooldown remain.
    Kills `gadget_fire[:, 0] = hero_gadget[0]` and `= hero_gadget.max()` in `_bot_phase`, and a
    cooldown written from `gadget_fire.amax(0)` in `_attack_phase` -- all three survived the
    single-env scenarios."""
    env, override, hp_before = _duel_batch(3)
    st = env.state
    st.ent_gadget_cd[2, 0] = 5.0
    action = torch.tensor([[0, 0], [0, _GADGET], [0, _GADGET]], dtype=torch.int64)

    obs, *_ = env.step(action, override)

    assert (st.prj_alive & (st.prj_kind == _SPINNER)).sum(dim=1).tolist() == [0, 1, 0]
    assert st.ent_gadget_cd[:, 0].tolist() == pytest.approx([0.0, 18.0, 4.95], abs=1e-4)
    assert st.ent_reveal_t[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert obs["action_mask"]["attack"][:, 3].tolist() == [True, False, False]

    idle = torch.zeros(3, 2, dtype=torch.int64)
    for _ in range(4):
        env.step(idle, override)
    #                                          hero  enemy  bystander
    assert (hp_before - st.ent_hp).tolist() == [[0.0, 0.0, 0.0],
                                                [0.0, 2000.0, 0.0],
                                                [0.0, 0.0, 0.0]]


def test_env_a_gadget_thrown_on_the_move_does_not_stop_the_walk():
    """The move column is independent of the attack column's value. Mortis walks 2.73 tiles/s,
    0.1365 a tick; bin 5 is +y while he FACES +x, so neither "a throw cancels the move" nor "a
    throw moves him along his facing" passes. Every throw above was made standing still."""
    env, override, _hp = _duel([16.0, 16.0])
    st = env.state
    _act(env, override, _GADGET, move=5)
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert _spinners(st) == 1
    assert st.ent_pos[0, 0].tolist() == pytest.approx([10.0, 10.1365], abs=1e-4)


def test_env_a_full_projectile_buffer_spends_the_cooldown_and_throws_nothing():
    """`_attack_phase`'s documented overflow contract: the cooldown is written from `gadget_fire`,
    not from `alloc_slots`' `ok`, so a full buffer costs the 18 s (and the concealment) and throws
    nothing -- the same silent drop as a super's charge or a thinned volley. `debug_tiny` keeps
    default.yaml's 192 slots. The fillers are all-zero slots: inert (no damage, no velocity), and
    they expire in this tick's projectile phase, AFTER the throw was refused in phase 6."""
    # Control: one free slot among 192 is enough, and it is the slot the spinner takes.
    env, override, _hp = _duel([11.5, 10.0])
    st = env.state
    assert st.prj_alive.shape == (1, 192)
    st.prj_alive.fill_(True)
    st.prj_alive[0, 77] = False
    _act(env, override, _GADGET)
    assert st.prj_alive[0].nonzero().flatten().tolist() == [77]
    assert int(st.prj_kind[0, 77]) == _SPINNER and int(st.prj_owner[0, 77]) == 0

    env, override, hp_before = _duel([11.5, 10.0])
    st = env.state
    st.prj_alive.fill_(True)
    _act(env, override, _GADGET)
    assert _spinners(st) == 0 and int(st.prj_alive.sum()) == 0
    assert st.ent_gadget_cd[0, 0].item() == 18.0
    assert st.ent_reveal_t[0, 0].item() == 1.0
    for _ in range(4):
        _act(env, override, 0)
    assert (hp_before - st.ent_hp[0]).tolist() == [0.0, 0.0, 0.0]


def test_env_at_the_shipped_decision_rate_the_gadget_rearms_on_the_73rd_observation():
    """The decision-boundary effect, for the gadget. The timer still runs out 360 ticks after the
    throw, but at `action_repeat: 5` the mask is sampled once per 5-tick decision and the throw
    lands on sub-tick 1: the observation after decision k reads 17.8 - 0.25 k, which is 0.05 (still
    masked) at k = 71 and 0 at k = 72. So the earliest second throw is decision 73, 365 ticks =
    18.25 s after the first -- the figure a decision-rate mirror (the deployment shadow) must
    reproduce, not 18.0."""
    env, override, _hp = _duel([11.5, 10.0], action_repeat=5)
    st = env.state
    obs, *_ = _act(env, override, _GADGET)
    legal = [bool(obs["action_mask"]["attack"][0, 3])]
    for _ in range(72):
        obs, *_ = _act(env, override, 0)
        legal.append(bool(obs["action_mask"]["attack"][0, 3]))

    assert legal[:72] == [False] * 72            # the throw's own observation and the 71 after it
    assert legal[72] is True
    assert int(st.step_count[0]) == 365          # 73 decisions x 5 ticks x 0.05 s = 18.25 s

    _act(env, override, _GADGET)                 # ...and legal means it really goes again
    assert st.ent_gadget_cd[0, 0].item() == pytest.approx(17.8, abs=1e-4)
    assert float(st.ent_damage_dealt[0, 0]) == 4000.0


def test_a_gadget_thrown_every_decision_is_sync_free_on_cuda():
    """tests/test_integration.py's sync test draws the attack column from {0, 1}, so it never
    throws a gadget: it sees `_attack_phase`'s gadget block only with `gadget_fire` all False.
    Here every env throws on every decision (the cooldown is zeroed between steps, on device)
    under `set_sync_debug_mode("error")`, so a data-dependent branch on the firing path raises.
    The throw counts are accumulated ON DEVICE and read once, after the window closes.
    Bots are frozen by the override, so nobody dies and 43 decisions fit one 2000-tick episode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    n_envs, decisions = 64, 40
    env = _env(action_repeat=5, n_envs=n_envs, device="cuda", debug_checks=False)  # check_invariants syncs
    st = env.state
    action = torch.tensor([[0, _GADGET]], dtype=torch.int64, device="cuda").repeat(n_envs, 1)
    override = torch.zeros(n_envs, env.cfg.n_entities, 2, dtype=torch.int64, device="cuda")
    override[:, 0, 0] = -1
    throws = torch.zeros((), dtype=torch.int64, device="cuda")
    sightings = torch.zeros((), dtype=torch.int64, device="cuda")

    def _hook(e):
        sightings.add_((e.state.prj_alive & (e.state.prj_kind == _SPINNER)).sum())

    for _ in range(3):                           # warm-up: lazy first-call allocations may sync
        st.ent_gadget_cd.zero_()
        env.step(action, override)
    torch.cuda.synchronize()

    env.tick_hook = _hook
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(decisions):
            st.ent_gadget_cd.zero_()
            env.step(action, override)
            throws.add_((st.ent_gadget_cd[:, 0] > 17.0).sum())
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("default")
        env.tick_hook = None

    assert int(throws) == 2560                   # 64 envs x 40 decisions, every one a throw
    assert int(sightings) == 7680                # each spinner is alive after sub-ticks 1-3 of 5
