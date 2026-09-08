"""`perception/assemble.py` -- the last stage before the policy.

The test that matters here is PARITY, not shape. This module's whole job is to arrange perception
outputs into the tensor the checkpoint expects, and every way of getting that wrong is silent: a
column swapped, an enemy slot shifted by the hero row, a grid plane permuted, a divisor missed.
None of it raises and none of it shows up in a shape assertion.

So `test_assemble_reproduces_the_sims_own_agent_obs` runs a real `BrawlVecEnv`, pulls out of its
observation exactly the quantities the deployed suppliers would produce, feeds them through
`assemble`, and demands the result equal what `obs_select.build_agent_obs` makes of the same
frame. If the two agree column-for-column on a real observation, the wiring is right -- and that
is an assertion about the deployed path, since `build_agent_obs` is the training path.

The rest of the file covers the decisions the module makes that the sim has no opinion about: the
`entities.hp` promotion rule (6.5), the refusal to invent a missing supplier (9.8), and the
12-channel view scatter.
"""
import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_deployment.perception.assemble import MapFrame, ObservationAssembler

CONFIGS = "configs/default.yaml"
TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
# The preset disables the zone; both deploy specs have a zone group, so put it back. The small
# map, two enemies and short episode are all still wanted -- this is the fast-suite config.
TINY["zone"] = {"enabled": True}
SPECS = ("configs/agent_obs_deploy.yaml", "configs/agent_obs_deploy2.yaml")


def _assembler(spec_path, cfg=None):
    """Origin (0, 0) and the sim's own extent, so `MapFrame` is the identity `pos / (map_w,
    map_h)` the sim uses. The quarantine only has to be neutral for parity to mean something --
    what the deployed frame should actually be is a separate, measured question."""
    cfg = cfg if cfg is not None else load_config(CONFIGS)
    spec = obs_select.load_agent_spec(spec_path, cfg)
    frame = MapFrame(origin=(0.0, 0.0), extent=(float(cfg.map_w), float(cfg.map_h)))
    return ObservationAssembler(spec, cfg, map_frame=frame)


def _env(n_envs=1, seed=0):
    from brawl_sim.env import BrawlVecEnv

    cfg = load_config(CONFIGS, overrides=TINY)
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)
    return env, cfg


class _Track:
    """What `EntityTracker` hands `assemble`: a world position, a velocity and `seen_now`."""

    def __init__(self, pos, vel, seen_now):
        self.pos, self.vel, self.seen_now = pos, vel, seen_now


def _suppliers_from(full, cfg, spec, env_i=0):
    """The deployed suppliers' outputs, read off a sim observation instead of off a screen.

    Every line here is a claim about which real supplier produces which sim field, and they are
    the claims BRAWL_DEPLOYMENT_DESIGN.md 6.3--6.5 make: the shadow owns the hero's timers, the
    tracker owns enemy identity and velocity, `HealthTracker` owns both HP columns, `hud.py` owns
    the brawler count, and the terrain map owns the two `in_bush` lookups.
    """
    at = lambda t: t[env_i]                                              # noqa: E731
    hero, ent, prj, zone, meta = (full["hero"], full["entities"], full["projectiles"],
                                  full["zone"], full["meta"])

    shadow = {
        "facing_vec": tuple(at(hero["facing_vec"]).tolist()),
        "ammo_frac": float(at(hero["ammo_frac"])),
        "ammo_whole": float(at(hero["ammo_whole"])),
        "attack_cd": float(at(hero["attack_cd"])),
        "can_attack": bool(at(hero["can_attack"])),
        "dashing": bool(at(hero["dashing"])),
        "dash_t": float(at(hero["dash_t"])),
        "dash_dir": tuple(at(hero["dash_dir"]).tolist()),
        "invuln": bool(at(hero["invuln"])),
        "long_dash_ready": bool(at(hero["long_dash_ready"])),
        "long_dash_frac": float(at(hero["long_dash_frac"])),
        "super_ready": bool(at(hero["super_ready"])),
        "super_charge_frac": float(at(hero["super_charge_frac"])),
    }
    hero_pos = tuple(at(hero["pos"]).tolist())
    hero_vel = tuple(at(hero["vel"]).tolist())

    # Enemy slot k of the tracker is entities index k + 1; only alive enemies exist as tracks.
    enemies, enemy_hp, enemy_bush = [], {}, {}
    for slot in range(cfg.n_entities - 1):
        i = slot + 1
        if not bool(at(ent["alive"])[i]):
            enemies.append(None)
            continue
        enemies.append(_Track(tuple(at(ent["pos"])[i].tolist()),
                              tuple(at(ent["vel"])[i].tolist()),
                              bool(at(ent["revealed_to_hero"])[i])))
        enemy_hp[slot] = float(at(ent["hp"])[i])
        enemy_bush[slot] = bool(at(ent["in_bush"])[i])

    alive_p = at(prj["alive"]).nonzero().flatten().tolist()
    projectiles = [(tuple(at(prj["rel_pos"])[j].tolist()), tuple(at(prj["vel"])[j].tolist()),
                    float(at(prj["time_to_closest"])[j])) for j in alive_p]

    zone_kw = {"hero_margin": tuple(at(zone["hero_margin"]).tolist()),
               "hero_margin_local": tuple(at(zone["hero_margin_local"]).tolist()),
               "active": bool(at(zone["active"])),
               "safe_area_frac": float(at(zone["safe_area_frac"])),
               "next_shrink_in": float(at(zone["next_shrink_in"]))}

    grid_group = next(g for g in spec.groups if g.view_channels is not None)
    grid = at(full["view"])[list(grid_group.channel_idx)].numpy().astype(np.uint8)

    return dict(
        hero_pos=hero_pos, hero_vel=hero_vel, shadow=shadow,
        hero_hp=float(at(hero["hp"])), n_enemies_alive=float(at(meta["n_enemies_alive"])),
        elapsed_s=float(at(meta["time_frac"])) * float(cfg.max_episode_steps) * float(cfg.dt),
        hero_in_bush=bool(at(hero["in_bush"])), hero_in_zone=bool(at(hero["in_zone"])),
        enemies=enemies, enemy_hp=enemy_hp, enemy_in_bush=enemy_bush,
        projectiles=projectiles, zone=zone_kw, grid=grid,
    )


# ---- parity ---------------------------------------------------------------------------------

@pytest.mark.parametrize("spec_path", SPECS)
def test_assemble_reproduces_the_sims_own_agent_obs(spec_path):
    """The load-bearing test. Same frame, two paths, every column equal.

    Tolerance is exact for the grid (uint8) and 1e-5 for the float groups -- the only arithmetic
    between the two paths is `assemble` round-tripping through Python floats, which is lossless at
    float32 for everything here except `dist`, recomputed with `math.hypot` where the sim used
    `safe_norm`.
    """
    env, cfg = _env()
    asm = _assembler(spec_path, cfg)    # the tiny preset, so both paths see one config
    spec, buffers = asm.spec, asm.buffers   # and one set of buffers, so `want` must be cloned

    full = env.reset()
    gen = torch.Generator().manual_seed(0)
    for step in range(6):
        want = obs_select.build_agent_obs(full, spec, cfg, buffers)
        want = {k: v[0].clone().numpy() for k, v in want.items()}
        got = asm.assemble(**_suppliers_from(full, cfg, spec))

        assert got.keys() == want.keys()
        for name in want:
            if name == "grid":
                assert np.array_equal(got[name], want[name]), f"{name} differs at step {step}"
            else:
                np.testing.assert_allclose(got[name], want[name], atol=1e-5,
                                           err_msg=f"{name} differs at step {step}")
        action = torch.stack([torch.randint(0, cfg.n_move_bins + 1, (1,), generator=gen),
                              torch.randint(0, 2, (1,), generator=gen)], dim=1)
        full, *_ = env.step(action)


def test_parity_survives_a_frame_with_live_projectiles():
    """The projectile group is the one place the two paths select rather than copy: `max_slots`
    takes the nearest 12 by `time_to_closest` out of `max_projectiles` slots, and deployment's
    live projectiles sit at slots 0..n-1 while the sim's sit wherever they were spawned. The
    selection has to be index-independent for that to be safe, which is why it is left to
    `obs_select` on both sides -- this is the test that says it stayed there."""
    env, cfg = _env(n_envs=1, seed=3)
    asm = _assembler(SPECS[0], cfg)
    spec, buffers = asm.spec, asm.buffers

    full = env.reset()
    gen = torch.Generator().manual_seed(1)
    seen = 0
    for _ in range(40):
        n_live = int(full["projectiles"]["alive"][0].sum())
        if n_live:
            seen = max(seen, n_live)
            want = obs_select.build_agent_obs(full, spec, cfg, buffers)["projectiles"][0].numpy()
            got = asm.assemble(**_suppliers_from(full, cfg, spec))["projectiles"]
            np.testing.assert_allclose(got, want, atol=1e-5)
        action = torch.stack([torch.randint(0, cfg.n_move_bins + 1, (1,), generator=gen),
                              torch.ones(1, dtype=torch.int64)], dim=1)   # attack every step
        full, *_ = env.step(action)
    assert seen > 0, "no projectile ever went live; this test asserted nothing"


# ---- the decisions the sim has no opinion about ----------------------------------------------

def test_a_tracked_enemy_with_no_hp_yet_is_not_promoted_to_a_slot():
    """§6.5's rule, and the reason it is safe. 84% of enemy boxes yield an HP and every track that
    NEVER yielded one lived four frames or fewer, so a slot held back for want of an HP is one
    that would have been a glancing detection anyway -- and a track that will persist gets its
    number within 1 frame at p50, 3 at p90, inside one decision.

    The alternative is a made-up HP in a column the policy was trained to trust. An empty slot is
    a state the sim produces constantly; a fabricated HP is one it never produces at all."""
    asm = _assembler(SPECS[0])
    tracks = [None] * (asm.cfg.n_entities - 1)
    tracks[0] = _Track((12.0, 9.0), (1.0, 0.0), True)
    tracks[1] = _Track((14.0, 9.0), (1.0, 0.0), True)

    obs = asm.assemble(**_base_kwargs(asm, enemies=tracks, enemy_hp={0: 5400}))
    alive = obs["enemies"][:, 0]
    assert alive[0] == 1.0, "slot 0 has a committed HP and must be visible"
    assert alive[1] == 0.0, "slot 1 has no HP yet, so it is not an enemy the policy is told about"
    assert not obs["enemies"][1].any(), "an unpromoted slot must be entirely zero, not partly filled"


def test_an_unrevealed_enemy_is_masked_exactly_as_fairness_masked_it_in_training():
    """`fair: true` multiplies each enemy row by `revealed_to_hero`, so a coasting track
    contributes nothing. Left to `obs_select` rather than reimplemented -- this pins that it is
    still reached, because a tracker that set `seen_now` wrong would otherwise leak a position the
    trained policy was never allowed to see."""
    asm = _assembler(SPECS[0])
    tracks = [None] * (asm.cfg.n_entities - 1)
    tracks[0] = _Track((12.0, 9.0), (1.0, 0.0), False)
    obs = asm.assemble(**_base_kwargs(asm, enemies=tracks, enemy_hp={0: 5400}))
    assert not obs["enemies"][0].any()


@pytest.mark.parametrize("missing", ["hero_hp", "n_enemies_alive", "hero_in_bush", "zone"])
def test_a_missing_supplier_raises_rather_than_becoming_a_zero(missing):
    """The rule the whole `agent_obs_deploy*.yaml` family exists to enforce, at the last point it
    can still be enforced. A field in the spec has a supplier; if that supplier produced nothing
    this tick, the loop's answer is to fail closed, not to substitute a plausible number."""
    asm = _assembler(SPECS[0])
    kwargs = _base_kwargs(asm)
    kwargs[missing] = None
    with pytest.raises(ValueError, match="no value supplied"):
        asm.assemble(**kwargs)


def test_a_zone_field_the_spec_needs_cannot_be_silently_omitted():
    """`_put_zone` is driven by the loaded spec, because two spec versions are live at once and
    they want different zone fields (9.16). Handing deploy2's dict to a deploy assembler must
    raise, not quietly produce a shorter group."""
    asm = _assembler(SPECS[0])
    kwargs = _base_kwargs(asm)
    kwargs["zone"] = {"hero_margin_local": (1.0, 1.0, 1.0, 1.0), "active": True}
    with pytest.raises(ValueError, match="zone.hero_margin"):
        asm.assemble(**kwargs)


def test_the_view_scatter_puts_each_plane_at_its_canonical_sim_channel():
    """The grid group is `index_select` over the sim's 12-channel ordering, so a plane written to
    the wrong index is read as a different channel entirely -- walls arriving as bushes. Marking
    each plane with its own value makes a permutation visible; equal planes would not."""
    asm = _assembler(SPECS[0])
    grid = np.zeros((len(asm._grid_channels), asm.cfg.view_h, asm.cfg.view_w), np.uint8)
    for i in range(grid.shape[0]):
        grid[i, 0, 0] = i + 1
    obs = asm.assemble(**_base_kwargs(asm, grid=grid))
    assert list(obs["grid"][:, 0, 0]) == list(range(1, grid.shape[0] + 1))
    # and the four channels nothing supplies were never touched
    unfilled = set(range(12)) - set(asm._grid_channels)
    assert unfilled == {5, 7, 9, 10}, "enemy_any / enemy_hidden / box / pickup"
    assert not asm._view[sorted(unfilled)].any()


def test_time_frac_is_clamped_because_the_sim_truncates_at_one():
    """Running short is harmless -- the policy just never sees the top of the range, exactly as
    when a sim episode ends early. Running LONG is out-of-distribution: the column's entire
    training range is bounded by 1, and a 200-second match would feed it 1.33 (9.6)."""
    asm = _assembler(SPECS[0])
    idx = obs_select.agent_obs_index_map(asm.spec, asm.cfg)["self"]["meta.time_frac"][0]
    for elapsed, want in ((0.0, 0.0), (75.0, 0.5), (150.0, 1.0), (400.0, 1.0)):
        obs = asm.assemble(**_base_kwargs(asm, elapsed_s=elapsed))
        assert obs["self"][idx] == pytest.approx(want, abs=1e-6)


def test_the_map_frame_is_the_only_thing_that_reads_absolute_position():
    """`hero.pos_norm` needs an origin and an extent the deployed loop does not have, and
    `MapFrame` is where that is quarantined. Moving it must change those two columns and NOTHING
    else -- if it ever moves another column, absolute position has leaked into a relative field
    and the leak is worth finding immediately."""
    asm = _assembler(SPECS[0])
    a = asm.assemble(**_base_kwargs(asm))
    asm.map_frame = MapFrame(origin=(7.0, -3.0), extent=(80.0, 80.0))
    b = asm.assemble(**_base_kwargs(asm))

    assert not np.allclose(a["self"][:2], b["self"][:2])
    np.testing.assert_allclose(a["self"][2:], b["self"][2:], atol=0)
    for name in ("enemies", "projectiles", "zone", "grid"):
        assert np.array_equal(a[name], b[name]), f"{name} moved with the map frame"


def test_the_default_map_frame_centres_odometry_and_keeps_the_sims_scale():
    """The measured choice (see `MapFrame`'s docstring). A constant origin error is free -- 0.734
    at +0.15 and 0.743 at +0.35 against a 0.734 baseline -- while zeroing the column costs 4.7 pp
    overall and 12 pp on elite. A SCALE error is a different perturbation and was not priced, so
    the extent must stay the sim's own: odometry is in tiles, `hero.pos` is in tiles, and dividing
    by the sim's map size makes one tile of real movement produce exactly the column movement it
    produced in training. This pins both halves of that."""
    asm = ObservationAssembler.from_paths(SPECS[0], CONFIGS)
    cfg = asm.cfg
    assert asm.map_frame.extent == (float(cfg.map_w), float(cfg.map_h))
    assert asm.map_frame.pos_norm((0.0, 0.0)) == (0.5, 0.5), "odometry's start belongs at the centre"

    # And the scale is the sim's: one tile of movement moves the column by exactly 1 / map_w.
    x0, _ = asm.map_frame.pos_norm((0.0, 0.0))
    x1, _ = asm.map_frame.pos_norm((1.0, 0.0))
    assert x1 - x0 == pytest.approx(1.0 / cfg.map_w)


def test_the_map_frame_does_not_clamp_an_impossible_position():
    """Saturating at [0, 1] would hide the one thing that says the assumed extent is wrong. It is
    also not the failure the ablation found: +0.35 pushed the column past 1.0 and cost nothing,
    while the clamp's natural failure mode -- everything pinned at a corner value -- is the -12 pp
    one."""
    frame = MapFrame(origin=(0.0, 0.0), extent=(60.0, 60.0))
    assert frame.pos_norm((90.0, -30.0)) == (1.5, -0.5)


def _base_kwargs(asm, **overrides):
    """A complete, valid supplier set. Deliberately not a fixture: every test that overrides one
    field wants to see the other twelve are present, since the thing under test is often that a
    missing one is fatal."""
    shadow = {"facing_vec": (1.0, 0.0), "ammo_frac": 0.83, "ammo_whole": 2.0, "attack_cd": 0.1,
              "can_attack": True, "dashing": False, "dash_t": 0.0, "dash_dir": (0.0, 0.0),
              "invuln": False, "long_dash_ready": True, "long_dash_frac": 1.0,
              "super_ready": False, "super_charge_frac": 0.4}
    kwargs = dict(
        hero_pos=(10.0, 10.0), hero_vel=(0.5, -0.5), shadow=shadow, hero_hp=8000.0,
        n_enemies_alive=4.0, elapsed_s=42.0, hero_in_bush=False, hero_in_zone=False,
        enemies=[None] * (asm.cfg.n_entities - 1), enemy_hp={}, enemy_in_bush={},
        projectiles=[],
        zone={"hero_margin": (5.0, 12.0, -1.0, 30.0),
              "hero_margin_local": (5.0, 10.0, -1.0, 10.0), "active": True,
              "safe_area_frac": 0.4, "next_shrink_in": 0.0},
        grid=np.zeros((len(asm._grid_channels), asm.cfg.view_h, asm.cfg.view_w), np.uint8),
    )
    kwargs.update(overrides)
    return kwargs
