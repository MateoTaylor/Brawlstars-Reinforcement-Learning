import tempfile
import os
import pathlib

import gymnasium
import pytest
import torch
import yaml

from brawl_sim.config import _deep_merge, load_config
from brawl_sim.core import obs_select
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"
# The deploy spec in use. The five before it are in archive/configs/.
DEPLOY6_YAML = "configs/agent_obs_deploy6.yaml"


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = _deep_merge(merged, overrides)  # the preset's other keys in a section stay
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def _env_and_obs(n_envs=4, seed=0, overrides=None):
    cfg = _cfg(overrides)
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)
    full_obs = env.reset()
    return env, cfg, full_obs


_WRITTEN_SPECS = []


@pytest.fixture(autouse=True)
def _remove_written_specs():
    """Removes every spec `_write_spec` wrote during the test: a dozen callers never removed
    theirs, and each run left that many .yaml files in the temp dir."""
    start = len(_WRITTEN_SPECS)
    yield
    for path in _WRITTEN_SPECS[start:]:
        pathlib.Path(path).unlink(missing_ok=True)
    del _WRITTEN_SPECS[start:]


def _write_spec(spec_dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    _WRITTEN_SPECS.append(path)
    with open(path, "w") as f:
        yaml.dump(spec_dict, f)
    return path


# ---- agent_space is a flat Dict, flatdim succeeds -----------------------------------------

def test_agent_space_is_flat_dict_and_flatdim_succeeds():
    cfg = _cfg()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    space = obs_select.agent_space(spec, cfg)
    assert isinstance(space, gymnasium.spaces.Dict)
    for sub in space.spaces.values():
        assert not isinstance(sub, gymnasium.spaces.Dict)  # flat -- no nesting
    dim = gymnasium.spaces.utils.flatdim(space)
    assert dim > 0


# ---- build_agent_obs output shapes match the space, every value finite --------------------

def test_build_agent_obs_shapes_match_space_and_are_finite():
    env, cfg, full_obs = _env_and_obs(n_envs=6)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    space = obs_select.agent_space(spec, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=6, device="cpu")

    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        action = torch.stack([torch.randint(0, 17, (6,), generator=gen), torch.randint(0, 2, (6,), generator=gen)], dim=1)
        full_obs, *_rest = env.step(action)
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
        for name, box in space.spaces.items():
            t = out[name]
            assert tuple(t.shape) == (6, *box.shape), f"{name}: {tuple(t.shape)} != {(6, *box.shape)}"
            assert t.dtype == (torch.uint8 if box.dtype.name == "uint8" else torch.float32)
            assert torch.isfinite(t.to(torch.float32)).all(), f"{name} has non-finite values"


# ---- privileged fields are rejected at load time, with a clear message --------------------

def test_privileged_field_raises_at_load_time():
    cfg = _cfg()
    bad_spec = {
        "fair": True, "normalize": False,
        "groups": [{"name": "leak", "per_entity": True, "dtype": "float32",
                    "fields": ["entities.privileged.target_id"]}],
    }
    path = _write_spec(bad_spec)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "entities.privileged" in str(e)
    finally:
        os.remove(path)


def test_unknown_field_raises_at_load_time():
    cfg = _cfg()
    bad_spec = {
        "fair": True, "normalize": False,
        "groups": [{"name": "x", "per_entity": False, "dtype": "float32", "fields": ["hero.not_a_real_field"]}],
    }
    path = _write_spec(bad_spec)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "not_a_real_field" in str(e)
    finally:
        os.remove(path)


def test_unfair_grid_channels_rejected_under_fair_true():
    cfg = _cfg()
    for bad_channel in ("enemy_any", "enemy_hidden"):
        spec_dict = {
            "fair": True, "normalize": False,
            "groups": [{"name": "grid", "dtype": "uint8", "view_channels": [bad_channel]}],
        }
        path = _write_spec(spec_dict)
        try:
            try:
                obs_select.load_agent_spec(path, cfg)
                assert False, f"expected a ValueError for {bad_channel!r}"
            except ValueError as e:
                assert bad_channel in str(e)
        finally:
            os.remove(path)
    # the same channel is fine under fair: false
    spec_dict = {"fair": False, "normalize": False, "groups": [{"name": "grid", "dtype": "uint8", "view_channels": ["enemy_any"]}]}
    path = _write_spec(spec_dict)
    try:
        obs_select.load_agent_spec(path, cfg)  # must not raise
    finally:
        os.remove(path)


def test_too_many_groups_rejected():
    cfg = _cfg()
    spec_dict = {
        "fair": False, "normalize": False,
        "groups": [{"name": f"g{i}", "per_entity": False, "dtype": "float32", "fields": ["hero.hp_frac"]} for i in range(7)],
    }
    path = _write_spec(spec_dict)
    try:
        try:
            obs_select.load_agent_spec(path, cfg)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "6" in str(e)
    finally:
        os.remove(path)


# ---- fairness gating: bush-hidden enemy contributes all-zero fields, revealed=0 -----------

def test_fair_true_zeroes_a_bush_hidden_enemys_row():
    env, cfg, full_obs = _env_and_obs(n_envs=1, overrides={"entities": {"n_enemies": 3}})
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    assert spec.fair

    full_obs["entities"]["revealed_to_hero"][0, 1] = False
    # full obs still has the true (non-zeroed) position for the hidden entity.
    true_pos = full_obs["entities"]["pos"][0, 1].clone()
    # sanity (2026-10-20: was `... or True`, which could not fail): a live enemy off the origin,
    # so the all-zero row below is the mask's doing and not an empty slot's
    assert bool(full_obs["entities"]["alive"][0, 1]) and bool(true_pos.abs().sum() > 0)

    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)

    row = out["enemies"][0, 0]  # 'enemies' slot 0 == entity index 1 (hero dropped)
    assert torch.equal(row, torch.zeros_like(row))


def test_fair_false_does_not_zero_hidden_enemies():
    env, cfg, full_obs = _env_and_obs(n_envs=1, overrides={"entities": {"n_enemies": 3}})
    path = _write_spec({
        "fair": False, "normalize": False,
        "groups": [{"name": "enemies", "per_entity": True, "dtype": "float32",
                    "fields": ["entities.alive", "entities.revealed_to_hero", "entities.hp_frac"]}],
    })
    try:
        spec = obs_select.load_agent_spec(path, cfg)
    finally:
        os.remove(path)

    full_obs["entities"]["revealed_to_hero"][0, 1] = False
    full_obs["entities"]["hp_frac"][0, 1] = 0.5  # force a nonzero value we can check survives
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
    row = out["enemies"][0, 0]
    assert row[2].item() == 0.5  # hp_frac column untouched despite revealed_to_hero == False


# ---- max_slots projectile group: 12 slots, all zero before anything fires -----------------

def test_a_fresh_reset_fills_all_12_projectile_slots_with_zeros():
    # 2026-10-20: renamed from "..._selects_k_nearest_by_time_to_closest_and_zero_pads"; nothing
    # is alive at reset, so this checks K and the empty fill only. Selection, tie order and
    # padding a part-filled selection are the next test's.
    env, cfg, full_obs = _env_and_obs(n_envs=1)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    proj_group = next(g for g in spec.groups if g.name == "projectiles")
    assert proj_group.max_slots == 12

    # no projectiles alive yet (fresh reset) -- every one of the 12 slots must be all-zero.
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
    assert torch.equal(out["projectiles"], torch.zeros_like(out["projectiles"]))


def test_tied_projectiles_go_nearest_first_whichever_slots_hold_them():
    """Everything moving away from the hero reads `time_to_closest` exactly 0, so ties are the
    common case, and `topk` left their order to the slot layout: the sim's scattered slots and
    deployment's packed list came out different. Ties now go nearest the
    hero first. The nearest projectile of all still ranks after every tie when its time is later,
    dead slots, which keep stale values in the sim, never rank, and a row left with no live
    projectile reads zero."""
    cfg = _cfg()
    path = _write_spec({"fair": False, "normalize": False, "groups": [{
        "name": "projectiles", "per_entity": True, "max_slots": 3,
        "fields": ["projectiles.rel_pos", "projectiles.time_to_closest"]}]})
    try:
        spec = obs_select.load_agent_spec(path, cfg)
    finally:
        os.remove(path)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    P = cfg.max_projectiles
    # (rel_pos, time_to_closest): four ties at 0, at distances 4, 1, 3 and 2, then the nearest of
    # all at a later time. Rows come out as (rel_pos x, rel_pos y, time_to_closest).
    live = [((4.0, 0.0), 0.0), ((0.0, -1.0), 0.0), ((3.0, 0.0), 0.0), ((0.0, 2.0), 0.0),
            ((0.5, 0.0), 0.2)]
    want = torch.tensor([[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
    cases = [(slots, live, want) for slots in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [P - 1, 7, 2, P - 5, 11])]
    # 2026-10-20: two live projectiles leave the third row to a dead slot's stale values. Every
    # case above fills all 3 rows, so a padding that zeroed only an all-dead selection passed.
    two = torch.tensor([[0.0, -1.0, 0.0], [4.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cases += [(slots, live[:2], two) for slots in ([0, 1], [P - 1, 7])]
    for slots, live_k, want_k in cases:
        alive = torch.zeros(1, P, dtype=torch.bool)
        rel_pos = torch.full((1, P, 2), 0.1)       # dead slots: stale, and nearer than any live one
        ttc = torch.zeros(1, P)
        for slot, (rel, t) in zip(slots, live_k):
            alive[0, slot] = True
            rel_pos[0, slot] = torch.tensor(rel)
            ttc[0, slot] = t
        full_obs = {"projectiles": {"alive": alive, "rel_pos": rel_pos, "time_to_closest": ttc}}
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["projectiles"][0]
        assert torch.equal(out, want_k), f"live projectiles in slots {slots}:\n{out}"


def test_a_fair_projectile_group_admits_only_what_is_on_screen():
    """deploy6's projectile group is gated by `projectiles.in_view`, and that is the
    camera window, not the crop. A projectile 20 tiles east flying at the hero has the smallest
    time_to_closest on the map and still gets an all-zero row; 5 tiles east it fills one."""
    env, cfg, _ = _env_and_obs(n_envs=1, overrides={
        "world": {"map_h": 60, "map_w": 60, "maps": ["walled"], "map_selection": "fixed",
                  "fixed_map": "walled"}})
    spec = obs_select.load_agent_spec(DEPLOY6_YAML, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    st = env.state
    st.ent_pos[0, 0] = torch.tensor([30.5, 30.5])
    st.prj_alive[0] = False
    st.prj_alive[0, 0] = True
    st.prj_owner[0, 0] = 1
    st.prj_vel[0, 0] = torch.tensor([-8.0, 0.0])
    rows = {}
    for dx in (20.0, 5.0):
        st.prj_pos[0, 0] = torch.tensor([30.5 + dx, 30.5])
        out = obs_select.build_agent_obs(env._build_observation(), spec, cfg, buffers)
        rows[dx] = out["projectiles"][0].clone()
    assert not rows[20.0].any(), "off screen: no row at all, whatever its rank"
    assert rows[5.0][0].any() and not rows[5.0][1:].any()


def test_a_projectile_under_the_static_speed_reads_still_and_ranks_on_zero():
    """`observation.projectile_static_speed` (user decision, 2026-10-07). A Grom shell lobbed short
    crawls: 1.5 tiles/s, 3 tiles out, straight-line time 2.0 s. Under a 2.0 floor it reads the way
    the live tracker reports it, vel 0 and time 0, so it ranks among the zeros, nearest first,
    ahead of a fast shot due in 0.125 s. A shot at exactly 2.0 keeps its velocity, as live, where
    `speed < STATIC_TILES_S` is still. Off, values and order are the old ones, and either way
    `full_obs` keeps the true values, since the reward and `info` read it."""
    path = _write_spec({"fair": False, "normalize": False, "groups": [{
        "name": "projectiles", "per_entity": True, "max_slots": 4,
        "fields": ["projectiles.rel_pos", "projectiles.vel", "projectiles.time_to_closest"]}]})
    # (rel_pos, vel, time_to_closest): fast and moving away, the crawl, fast and incoming, and one
    # at the floor. Rows come out as (rel x, rel y, vel x, vel y, time_to_closest).
    live = [((0.0, 2.0), (0.0, 4.0), 0.0), ((3.0, 0.0), (-1.5, 0.0), 2.0),
            ((1.0, 0.0), (-8.0, 0.0), 0.125), ((5.0, 0.0), (-2.0, 0.0), 2.5)]
    want = {
        0.0: [[0, 2, 0, 4, 0], [1, 0, -8, 0, 0.125], [3, 0, -1.5, 0, 2.0], [5, 0, -2, 0, 2.5]],
        2.0: [[0, 2, 0, 4, 0], [3, 0, 0, 0, 0], [1, 0, -8, 0, 0.125], [5, 0, -2, 0, 2.5]],
    }
    try:
        for floor, rows in want.items():
            cfg = load_config(CONFIGS_DEFAULT, overrides={"observation": {"projectile_static_speed": floor}})
            spec = obs_select.load_agent_spec(path, cfg)
            buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
            P = cfg.max_projectiles
            alive = torch.zeros(1, P, dtype=torch.bool)
            rel_pos, vel, ttc = torch.zeros(1, P, 2), torch.zeros(1, P, 2), torch.zeros(1, P)
            for slot, (rel, v, t) in zip((7, 2, P - 1, 0), live):
                alive[0, slot] = True
                rel_pos[0, slot], vel[0, slot], ttc[0, slot] = torch.tensor(rel), torch.tensor(v), t
            prj = {"alive": alive, "rel_pos": rel_pos, "vel": vel, "time_to_closest": ttc}
            out = obs_select.build_agent_obs({"projectiles": prj}, spec, cfg, buffers)["projectiles"][0]
            assert torch.equal(out, torch.tensor(rows, dtype=torch.float32)), f"floor {floor}:\n{out}"
            assert vel[0, 2].tolist() == [-1.5, 0.0] and ttc[0, 2] == 2.0, "full_obs is not touched"
    finally:
        os.remove(path)


# ---- build_agent_obs allocates nothing per call (steady-state memory doesn't grow) --------

def test_build_agent_obs_reuses_the_same_out_buffer_tensors():
    env, cfg, full_obs = _env_and_obs(n_envs=4)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=4, device="cpu")
    data_ptrs_before = {name: t.data_ptr() for name, t in buffers.items()}

    gen = torch.Generator().manual_seed(0)
    for _ in range(2):
        action = torch.stack([torch.randint(0, 17, (4,), generator=gen), torch.randint(0, 2, (4,), generator=gen)], dim=1)
        full_obs, *_rest = env.step(action)
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
        assert out is buffers  # same dict object returned
        for name, t in out.items():
            assert t.data_ptr() == data_ptrs_before[name]  # same underlying storage, every call


# ---- docs/AGENT_OBS*.md match a fresh render of their spec -------------------------------

@pytest.mark.parametrize("spec_path", [AGENT_OBS_YAML, DEPLOY6_YAML])
def test_each_agent_obs_doc_matches_a_fresh_render_of_its_spec(spec_path):
    # 2026-10-20: this rendered the default spec twice and compared the renders, which a stale
    # doc passes; each committed doc is now compared with a fresh render of its own spec.
    # 2026-10-21: deploy4's and deploy5's docs went to archive/ and their specs to
    # archive/configs/, frozen, so only the docs of the specs in use are held to a render.
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    fresh = dump_mod.render_agent_obs(spec_path)
    doc = dump_mod.agent_docs_path(spec_path)
    assert fresh == doc.read_text(encoding="utf-8"), (
        f"{doc} is stale: rerun `python scripts/dump_obs_schema.py --spec {spec_path}`")
    assert "## `self`" in fresh


def test_each_spec_renders_to_a_doc_named_after_itself():
    """`--spec` gives every sibling spec its own doc, because each narrowing
    is a different width and a different from-scratch run, so a column index only means something
    next to the spec it came from. The default path is the one the constant already names."""
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    assert dump_mod.agent_docs_path("configs/agent_obs.yaml") == dump_mod.AGENT_DOCS_PATH
    assert dump_mod.agent_docs_path("configs/agent_obs.yaml").name == "AGENT_OBS.md"
    assert dump_mod.agent_docs_path(DEPLOY6_YAML).name == "AGENT_OBS_DEPLOY6.md"
    assert dump_mod.agent_docs_path("configs/agent_obs_lowinfo.yaml").name == "AGENT_OBS_LOWINFO.md"


def test_the_deploy6_spec_renders_its_own_layout_not_the_full_one():
    """deploy6 is the spec configs/train.yaml trains, and its doc is the deployment mirrors' one
    readable reference for a column index: it has to name its file, show the pinned widths -- the
    27-column self with the gadget pair, the history group, thirteen grid planes -- say its enemy
    rows are in tracker-slot order with memory rows that outlive the sighting, and mark both
    critic groups as critic-only."""
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    deploy6 = dump_mod.render_agent_obs(DEPLOY6_YAML)
    assert "`configs/agent_obs_deploy6.yaml`" in deploy6.splitlines()[2]
    assert "shape: `(27,)`" in deploy6 and "`hero.gadget_ready`" in deploy6
    assert "## `history`" in deploy6
    assert "shape: `(13, 13, 21)`" in deploy6             # ten single-frame planes + three history
    assert "| 12 | `enemy_hist3` |" in deploy6
    assert "**Tracked-slot order**" in deploy6
    assert "A `memory.*` row is masked by `memory.tracked`" in deploy6
    assert "## `critic_enemies`" in deploy6 and "shape: `(9, 25)`" in deploy6
    assert "## `critic_world`" in deploy6 and "shape: `(6,)`" in deploy6
    assert deploy6.count("**Critic only**") == 2
    assert deploy6 != dump_mod.render_agent_obs()         # --spec really changes the render


# ---- normalize: true, per unit ------------------------------------------------------------
#
# These tests pin `normalize: true` for every unit, so an `hp` or `count` field cannot sit in a
# normalized spec undivided.

def test_every_normalized_unit_divides_by_its_documented_scale():
    cfg = _cfg()
    assert obs_select._norm_divisor("tiles", cfg) == float(max(cfg.map_w, cfg.map_h))
    # Literals, not the module's constants (2026-10-20): a trained checkpoint's inputs were
    # divided by these numbers, so changing one must fail here rather than agree with itself.
    assert obs_select._norm_divisor("tiles/s", cfg) == 20.0
    assert obs_select._norm_divisor("hp", cfg) == 20000.0
    assert obs_select._norm_divisor("count", cfg) == 20.0
    # Bounded by construction; a divisor here would shrink a field that is already in range.
    for units in ("fraction", "bool", "onehot", "unitless", "radians"):
        assert obs_select._norm_divisor(units, cfg) == 1.0


def _norm_self_row(field, raw):
    """`raw` (one value per env) through a one-field normalized "self" group."""
    cfg = _cfg()
    path = _write_spec({"fair": False, "normalize": True, "groups": [
        {"name": "self", "per_entity": False, "dtype": "float32", "fields": [field]}]})
    try:
        spec = obs_select.load_agent_spec(path, cfg)
    finally:
        os.remove(path)
    group, key = field.split(".")
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, len(raw), "cpu")
    return obs_select.build_agent_obs({group: {key: raw}}, spec, cfg, buffers)["self"][:, 0]


def test_the_hp_scale_passes_the_largest_hp_training_reaches_through_unclipped():
    """The HP a run can reach goes past the scale on purpose, and nothing clips it.

    2026-10-20: this asserted the default config's ceiling normalizes to <= 1.05, but the elite
    tier and the cube cap in configs/train.yaml (the user retunes both) already put it past 1.
    The ceilings are an enemy at the roster's highest base HP, times train.yaml's elite `hp`
    (core/curriculum scales base HP only), plus default.yaml's and then train.yaml's
    `max_cubes` at `hp_per_cube`. Read at run time, they must come out of an hp field as
    exactly raw / 20000, finite, and above 1 where they are past the scale. The scale itself is
    the literal: a trained checkpoint's inputs were divided by it.
    """
    import yaml as _yaml

    assert obs_select._NORM_HP_SCALE == 20000.0
    # cubes.* resolve onto SimParams, not EnvConfig, so they are read from the files themselves.
    cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]
    train = _yaml.safe_load(open("configs/train.yaml").read())
    train_cubes = {**cubes, **train["run"]["env_overrides"].get("cubes", {})}
    elite_hp = train["curriculum"]["tiers"]["elite"]["hp"]
    brawlers = _yaml.safe_load(open("configs/brawlers.yaml").read())
    base = [v["base_hp"] for v in brawlers.values() if isinstance(v, dict) and "base_hp" in v]
    raw = torch.tensor([max(base)] + [
        max(base) * elite_hp + c["max_cubes"] * c["hp_per_cube"] for c in (cubes, train_cubes)])

    out = _norm_self_row("hero.hp", raw)
    assert torch.equal(out, raw / 20000.0)
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out > 1.0, raw > 20000.0)
    # Not so large that a real HP reading vanishes into the noise floor either -- a fresh Mortis
    # should land near the ~0.1-0.5 band _NORM_SPEED_SCALE puts a walking entity in.
    assert 0.2 <= brawlers["hero_mortis"]["base_hp"] / obs_select._NORM_HP_SCALE <= 0.6


def test_the_count_scale_bounds_the_default_counts_and_passes_train_cubes_through():
    """default.yaml's largest counts (16 cubes, 9 enemies, 3 ammo) normalize to at most 1, and
    configs/train.yaml's cube cap, read at run time, comes out of a count field as exactly
    raw / 20, finite, and above 1 where it is past the scale (nothing clips it). The scale is
    the literal a trained checkpoint depends on.

    2026-10-20: the enemy count came from the tiny preset (2), not default.yaml's 9, and the
    train.yaml cube cap that goes past the scale was not checked at all.
    """
    import yaml as _yaml

    assert obs_select._NORM_COUNT_SCALE == 20.0
    cfg = load_config(CONFIGS_DEFAULT)
    assert cfg.n_enemies == 9
    max_cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]["max_cubes"]
    for largest in (max_cubes, cfg.n_enemies, 3):  # cubes, brawlers left, ammo
        assert largest / 20.0 <= 1.0

    train = _yaml.safe_load(open("configs/train.yaml").read())
    train_max = train["run"]["env_overrides"].get("cubes", {}).get("max_cubes", max_cubes)
    raw = torch.tensor([max_cubes, train_max])
    out = _norm_self_row("hero.cubes", raw)
    assert torch.equal(out, raw.float() / 20.0)
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out > 1.0, raw > 20)


def test_normalized_hp_and_count_fields_come_out_in_range_on_a_real_env():
    """End to end: the divisor is applied where it is supposed to be, not merely defined."""
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=3)
    spec_dict = {
        "fair": True, "normalize": True,
        "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                    "fields": ["hero.hp", "meta.n_enemies_alive", "hero.ammo_whole"]}],
    }
    path = _write_spec(spec_dict)
    try:
        spec = obs_select.load_agent_spec(path, cfg)
        buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["self"]
        assert out.min() >= 0.0
        assert out.max() <= 1.0, f"a normalized field left [0, 1]: {out.max().item()}"
        # And it is a real division, not a coincidence of small raw values: hero HP is thousands.
        raw_hp = full_obs["hero"]["hp"]
        assert torch.allclose(out[:, 0], raw_hp / obs_select._NORM_HP_SCALE)
        assert raw_hp.max() > 100.0
    finally:
        os.remove(path)


def test_two_specs_with_a_same_named_group_do_not_share_a_normalization_vector():
    """A latent bug found by the test above, worth its own name.

    `_normalize` memoizes its scale vector, and the key used to be the group NAME alone -- but
    the vector depends on the group's FIELDS. Every shipped spec has a group called
    "self", not all at one width, so loading two of them in one process gave the second the
    first's divisors: a broadcast error when the widths differ, and silently wrong scaling when
    they match. Anything that evaluates a checkpoint against more than one spec hits this.
    """
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=5)
    wide = {"fair": True, "normalize": True,
            "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                        "fields": ["hero.hp", "hero.vel", "meta.n_enemies_alive"]}]}
    narrow = {"fair": True, "normalize": True,
              "groups": [{"name": "self", "per_entity": False, "dtype": "float32",
                          "fields": ["meta.n_enemies_alive"]}]}
    paths = [_write_spec(wide), _write_spec(narrow)]
    try:
        outs = []
        for path in paths:
            spec = obs_select.load_agent_spec(path, cfg)
            buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
            outs.append(obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["self"].clone())
        assert outs[0].shape[1] == 4 and outs[1].shape[1] == 1
        # The shared field must normalize identically in both, by the count scale.
        raw_n = full_obs["meta"]["n_enemies_alive"].to(torch.float32)
        expected = raw_n / obs_select._NORM_COUNT_SCALE
        assert torch.allclose(outs[0][:, 3], expected)
        assert torch.allclose(outs[1][:, 0], expected)
    finally:
        for p in paths:
            os.remove(p)


def _grid_spec(channels):
    return {"fair": True, "normalize": True,
            "groups": [{"name": "grid", "dtype": "uint8", "view_channels": list(channels)}]}


@pytest.mark.parametrize("first, second", [
    # The pair that found it: the full suite built deploy3's grid (10 planes), and then every
    # deploy-spec test (8) died on a size mismatch. Both are in archive/configs/ now; deploy6's
    # thirteen planes before agent_obs.yaml's ten is the same mismatch.
    (DEPLOY6_YAML, AGENT_OBS_YAML),
    # The worse half: equal widths, so nothing raises and the second spec just reads the wrong planes.
    (_grid_spec(["hero", "blocks_unit"]), _grid_spec(["blocks_unit", "hero"])),
])
def test_two_specs_with_a_same_named_grid_do_not_share_channel_indices(first, second):
    """The grid half of the bug above. `_build_grid_group` memoized its channel indices under the
    group NAME, and every spec calls its view group "grid".

    The reference is the second spec built from an empty cache, so the expected planes do not come
    from the same channel table the code under test uses.
    """
    env, cfg, full_obs = _env_and_obs(n_envs=4, seed=5)
    paths = [s if isinstance(s, str) else _write_spec(s) for s in (first, second)]
    try:
        specs = [obs_select.load_agent_spec(p, cfg) for p in paths]

        def grid(spec):
            buffers = obs_select.make_agent_obs_buffers(spec, cfg, 4, torch.device("cpu"))
            return obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["grid"].clone()

        obs_select._tensor_cache.clear()
        reference = grid(specs[1])
        obs_select._tensor_cache.clear()
        assert not torch.equal(grid(specs[0]), reference), "the specs must differ for this to tell"
        assert torch.equal(grid(specs[1]), reference)
    finally:
        for s, p in zip((first, second), paths):
            if isinstance(s, dict):
                os.remove(p)


# ---- the history group and planes --------------------------------------------------------------

_HIST_FIELDS = ["hist.valid", "hist.move_onehot", "hist.attack_onehot", "hist.hp", "hist.ammo_frac",
                "hist.displacement"]


def _hist_spec(normalize):
    return {"fair": True, "normalize": normalize,
            "groups": [{"name": "history", "per_entity": False, "dtype": "float32", "fields": _HIST_FIELDS}]}


def _load_spec(spec_dict, cfg):
    path = _write_spec(spec_dict)
    try:
        return obs_select.load_agent_spec(path, cfg)
    finally:
        os.remove(path)


def test_the_six_hist_fields_make_a_78_column_group():
    """3 * (1 + 17 + 4 + 1 + 1 + 2)."""
    cfg = _cfg()
    spec = _load_spec(_hist_spec(normalize=False), cfg)
    assert spec.groups[0].shape == (78,)
    assert obs_select.agent_space(spec, cfg)["history"].shape == (78,)
    assert obs_select.agent_obs_index_map(spec, cfg)["history"] == {
        "hist.valid": (0, 3), "hist.move_onehot": (3, 54), "hist.attack_onehot": (54, 66),
        "hist.hp": (66, 69), "hist.ammo_frac": (69, 72), "hist.displacement": (72, 78),
    }


def test_a_rank_3_field_flattens_slot_by_slot():
    """Row-major: slot 0's 17 move columns, then slot 1's, then slot 2's, and displacement as
    (x0, y0, x1, y1, x2, y2). A hand-made package, so every column's value is known."""
    cfg = _cfg()
    spec = _load_spec(_hist_spec(normalize=False), cfg)
    move = torch.zeros(1, 3, 17, dtype=torch.uint8)
    move[0, 0, 5] = move[0, 1, 0] = move[0, 2, 16] = 1
    attack = torch.zeros(1, 3, 4, dtype=torch.uint8)
    attack[0, 0, 1] = attack[0, 1, 3] = attack[0, 2, 2] = 1
    full_obs = {"hist": {
        "valid": torch.tensor([[True, True, True]]),
        "move_onehot": move,
        "attack_onehot": attack,
        "hp": torch.tensor([[100.0, 200.0, 300.0]]),
        "ammo_frac": torch.tensor([[0.5, 0.25, 1.0]]),
        "displacement": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
    }}
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, 1, torch.device("cpu"))
    row = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["history"][0]
    assert row[0:3].tolist() == [1.0, 1.0, 1.0]
    assert row[3:54].nonzero().flatten().tolist() == [5, 17, 50]  # 17 * slot + bin
    assert row[54:66].nonzero().flatten().tolist() == [1, 7, 10]  # 4 * slot + attack
    assert row[66:72].tolist() == [100.0, 200.0, 300.0, 0.5, 0.25, 1.0]
    assert row[72:78].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_normalize_scales_hist_hp_by_the_hp_scale_and_displacement_by_the_map():
    cfg = _cfg()  # debug_tiny: a 20 x 20 map
    g = _load_spec(_hist_spec(normalize=True), cfg).groups[0]
    assert g.norm_scale == (1.0,) * 66 + (20000.0,) * 3 + (1.0,) * 3 + (20.0,) * 6


def test_a_hist_group_builds_on_a_real_env():
    env, cfg, full_obs = _env_and_obs(n_envs=2)
    spec = _load_spec(_hist_spec(normalize=True), cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, 2, torch.device("cpu"))
    assert not obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["history"].any()  # no past yet
    full_obs, *_ = env.step(torch.tensor([[5, 0], [0, 0]]))
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["history"]
    assert out[:, 0:3].tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert out[:, 3:20].argmax(dim=1).tolist() == [5, 0]


def test_channel_index_appends_one_enemy_hist_plane_per_history_frame():
    base = dict(obs_select._CHANNEL_INDEX)
    assert len(base) == 12 and base["projectile"] == 11
    three = obs_select.channel_index(_cfg())
    assert {k: v for k, v in three.items() if k not in base} == {
        "enemy_hist1": 12, "enemy_hist2": 13, "enemy_hist3": 14}
    assert {k: three[k] for k in base} == base
    one = obs_select.channel_index(_cfg({"observation": {"history_frames": 1}}))
    assert {k: v for k, v in one.items() if k not in base} == {"enemy_hist1": 12}
    assert obs_select._CHANNEL_INDEX == base, "the fixed table is not mutated"


def test_a_grid_spec_selects_history_planes_and_refuses_one_past_k():
    env, cfg, full_obs = _env_and_obs(n_envs=2)
    spec = _load_spec(_grid_spec(["enemy_revealed", "enemy_hist1", "enemy_hist3"]), cfg)
    assert spec.groups[0].channel_idx == (6, 12, 14)
    full_obs["view"][:, 14] = 7  # a marker only the oldest history plane carries
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, 2, torch.device("cpu"))
    grid = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)["grid"]
    assert grid.shape == (2, 3, 10, 14)
    assert (grid[:, 2] == 7).all() and not (grid[:, :2] == 7).any()
    with pytest.raises(ValueError, match="enemy_hist4"):
        _load_spec(_grid_spec(["enemy_hist4"]), cfg)


# ---- slots: tracked --------------------------------------------------------------------------

# deploy6's enemies columns read from `entities.*` instead of `memory.*`, less `age_s`: deploy4's
# enemies group exactly, and deploy5's once it has `slots: tracked`.
_ENTITY_FIELDS = ["entities.alive", "entities.revealed_to_hero", "entities.rel_pos",
                  "entities.dist", "entities.rel_vel", "entities.hp", "entities.in_bush"]


def _entities_variant(**group_keys) -> str:
    """deploy6 written to a temp file with its enemies group read from `entities.*`, without
    `slots`, and with `group_keys` set on it."""
    doc = yaml.safe_load(open(DEPLOY6_YAML).read())
    (g,) = [g for g in doc["groups"] if g["name"] == "enemies"]
    g["fields"] = list(_ENTITY_FIELDS)
    del g["slots"]
    g.update(group_keys)
    return _write_spec(doc)


def test_tracked_slots_order_the_rows_by_the_sims_slot_table():
    """Slot 0 <- entity 3, slot 1 <- entity 1, slot 2 empty: the tracked build's rows are the
    plain build's rows for those entities, and the empty slot is a zero row."""
    env, cfg, full = _env_and_obs(
        n_envs=1, overrides={"entities": {**CONFIGS_TINY.get("entities", {}), "n_enemies": 3}})
    plain = obs_select.load_agent_spec(_entities_variant(), cfg)
    tracked = obs_select.load_agent_spec(_entities_variant(slots="tracked"), cfg)
    assert [g.slots for g in tracked.groups if g.entity_prefix == "entities"] == ["tracked"]
    assert all(g.slots is None for g in plain.groups)
    assert [g.shape for g in tracked.groups] == [g.shape for g in plain.groups], "same width"

    full["entities"]["revealed_to_hero"][:] = True       # every row lit: the gather is what moves them
    entity = torch.tensor([[4, 2, 0]])
    full["slots"] = {"entity": entity, "valid": entity > 0}
    want = obs_select.build_agent_obs(
        full, plain, cfg, obs_select.make_agent_obs_buffers(plain, cfg, n_envs=1, device="cpu"))
    want = want["enemies"][0].clone()
    tbuf = obs_select.make_agent_obs_buffers(tracked, cfg, n_envs=1, device="cpu")
    got = obs_select.build_agent_obs(full, tracked, cfg, tbuf)["enemies"][0].clone()
    assert want.abs().sum(1).gt(0).all(), "the plain rows are non-zero, so the equalities below bite"
    assert torch.equal(got[0], want[2]) and torch.equal(got[1], want[0])
    assert not got[2].any()

    full["entities"]["revealed_to_hero"][0, 3] = False    # the entity in slot 0 goes unrevealed
    got = obs_select.build_agent_obs(full, tracked, cfg, tbuf)["enemies"][0].clone()
    assert not got[0].any() and torch.equal(got[1], want[0])


@pytest.mark.parametrize("group_keys, message", [
    ({"slots": "other"}, "'tracked'"),
    ({"slots": "tracked", "max_slots": 2}, "max_slots"),
])
def test_tracked_slots_refuses_a_bad_option_at_load_time(group_keys, message):
    with pytest.raises(ValueError, match=message):
        obs_select.load_agent_spec(_entities_variant(**group_keys), _cfg())


def test_tracked_slots_is_only_for_the_hero_axis():
    cfg = _cfg()
    for group in ({"name": "p", "per_entity": True, "dtype": "float32", "slots": "tracked",
                   "fields": ["projectiles.rel_pos"]},
                  {"name": "s", "per_entity": False, "dtype": "float32", "slots": "tracked",
                   "fields": ["hero.hp"]}):
        path = _write_spec({"fair": True, "normalize": False, "groups": [group]})
        with pytest.raises(ValueError, match="entities"):
            obs_select.load_agent_spec(path, cfg)


def test_the_slot_bookkeeping_is_not_an_observation():
    path = _write_spec({"fair": True, "normalize": False,
                        "groups": [{"name": "x", "per_entity": True, "dtype": "float32",
                                    "fields": ["slots.entity"]}]})
    with pytest.raises(ValueError, match="slots"):
        obs_select.load_agent_spec(path, _cfg())


# ---- memory.* groups and the critic section (deploy6, archive/CRITIC_AND_MEMORY_PLAN.md) ---------

def _rebuild(env, cfg):
    """The env's observation of its current state, rebuilt WITHOUT a decision of `slots.update`
    (which `env._build_observation` runs), so a slot table set by hand is read as set, with
    the reveal `env._obs_hero_view` holds."""
    from brawl_sim.bots import perception
    from brawl_sim.core import observation
    los = perception.raw_los(env.state, env.bank, cfg) if cfg.obs_include_raw_los else None
    return observation.build_obs(env.state, env.bank, env._obs_vis, los, env.params, cfg,
                                 hero_view=env._obs_hero_view)


def _build(full, spec, cfg, critic=True):
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    return {k: v[0].clone() for k, v in
            obs_select.build_agent_obs(full, spec, cfg, buffers, critic=critic).items()}


def _columns(spec, cfg, group):
    return {f: slice(*c) for f, c in obs_select.agent_obs_index_map(spec, cfg)[group].items()}


def test_a_memory_group_keeps_the_coasting_row_an_entities_group_blanks():
    """Slot 0 <- entity 2, out of sight for 5 decisions, last seen at (4.5, 15.5) with 1234 HP;
    slot 1 <- entity 1, seen now. deploy6's memory rows show entity 2 where and as it was last
    seen, 1.25 s ago, with `seen` 0, and entity 1 as it is now; the same rows read from
    `entities.*` (deploy5's group), gated by revealed_to_hero, blank entity 2's row. Normalized:
    tiles / 20 on this map."""
    env, cfg, _ = _env_and_obs(n_envs=1)
    st = env.state
    st.slot_ent[0] = torch.tensor([3, 2])
    st.ent_slot[0] = torch.tensor([0, 2, 1])
    st.ent_misses[0] = torch.tensor([0, 0, 5], dtype=torch.int32)
    st.ent_seen_pos[0, 1], st.ent_seen_vel[0, 1] = st.ent_pos[0, 1], st.ent_vel[0, 1]
    st.ent_seen_hp[0, 1] = st.ent_hp[0, 1]
    st.ent_seen_pos[0, 2] = torch.tensor([4.5, 15.5])
    st.ent_seen_vel[0, 2] = torch.tensor([1.0, -0.5])
    st.ent_seen_hp[0, 2] = 1234.0
    env._obs_hero_view[0, 1] = True
    env._obs_hero_view[0, 2] = False
    full = _rebuild(env, cfg)

    d6 = obs_select.load_agent_spec(DEPLOY6_YAML, cfg)
    rows = _build(full, d6, cfg)["enemies"]
    c = _columns(d6, cfg, "enemies")
    hero = st.ent_pos[0, 0]
    assert [rows[0, c[f]].tolist() for f in ("memory.tracked", "memory.seen", "memory.age_s")] \
        == [[1.0], [0.0], [1.25]]
    torch.testing.assert_close(rows[0, c["memory.rel_pos"]], (torch.tensor([4.5, 15.5]) - hero) / 20)
    assert rows[0, c["memory.hp"]].item() == pytest.approx(1234.0 / 20000.0)
    assert [rows[1, c[f]].tolist() for f in ("memory.tracked", "memory.seen", "memory.age_s")] \
        == [[1.0], [1.0], [0.0]]
    torch.testing.assert_close(rows[1, c["memory.rel_pos"]], (st.ent_pos[0, 1] - hero) / 20)

    gated = obs_select.load_agent_spec(_entities_variant(slots="tracked"), cfg)
    gated_rows = _build(full, gated, cfg)["enemies"]
    assert not gated_rows[0].any() and gated_rows[1].any()


def test_a_memory_group_without_tracked_slots_is_refused():
    doc = yaml.safe_load(open(DEPLOY6_YAML).read())
    (g,) = [g for g in doc["groups"] if g["name"] == "enemies"]
    del g["slots"]
    with pytest.raises(ValueError, match="slots: tracked"):
        obs_select.load_agent_spec(_write_spec(doc), _cfg())


CRITIC_GROUP = {"name": "critic_x", "per_entity": True, "dtype": "float32",
                "fields": ["entities.alive", "entities.privileged.targets_hero"]}


def _deploy6_critic(*critic_groups) -> str:
    doc = yaml.safe_load(open(DEPLOY6_YAML).read())
    doc["critic"] = list(critic_groups)
    return _write_spec(doc)


def test_a_critic_group_may_read_what_an_actor_group_may_not():
    spec = obs_select.load_agent_spec(_deploy6_critic(CRITIC_GROUP), _cfg())
    (g,) = spec.critic_groups
    assert g.critic and "entities.privileged.targets_hero" in g.fields
    assert not any(a.critic for a in spec.groups)
    assert [a.name for a in spec.all_groups][-1] == "critic_x"


@pytest.mark.parametrize("change, message", [
    ({"name": "x"}, "must start with 'critic_'"),
    ({"slots": "tracked"}, "slots"),
    ({"max_slots": 2}, "max_slots"),
    ({"dtype": "uint8"}, "uint8"),
    ({"view_channels": ["hero"]}, "view_channels"),
])
def test_a_critic_group_refuses_what_its_rows_cannot_hold(change, message):
    with pytest.raises(ValueError, match=message):
        obs_select.load_agent_spec(_deploy6_critic({**CRITIC_GROUP, **change}), _cfg())


@pytest.mark.parametrize("critic_groups, message", [
    ((CRITIC_GROUP, {**CRITIC_GROUP, "name": "critic_y"}, {**CRITIC_GROUP, "name": "critic_z"}),
     "critic groups"),
    ((CRITIC_GROUP, CRITIC_GROUP), "duplicate"),
])
def test_the_critic_section_is_at_most_two_uniquely_named_groups(critic_groups, message):
    with pytest.raises(ValueError, match=message):
        obs_select.load_agent_spec(_deploy6_critic(*critic_groups), _cfg())


def test_deploy6s_critic_reads_every_living_bot_seen_or_not():
    """Entity 1 has never been seen (no slot, out of sight): `enemies` has nothing of it, and
    `critic_enemies` row 0 holds its true position and HP. Entity 2 is dead: a zero critic row,
    the rows being masked by `entities.alive`."""
    env, cfg, _ = _env_and_obs(n_envs=1)
    st = env.state
    for name in ("slot_ent", "ent_slot", "ent_hits", "ent_misses"):
        getattr(st, name).zero_()
    env._obs_hero_view[0, 1] = False
    st.ent_alive[0, 2] = False
    full = _rebuild(env, cfg)

    spec = obs_select.load_agent_spec(DEPLOY6_YAML, cfg)
    out = _build(full, spec, cfg)
    assert not out["enemies"].any()
    rows = out["critic_enemies"]
    c = _columns(spec, cfg, "critic_enemies")
    assert rows[0, c["entities.alive"]].tolist() == [1.0]
    assert rows[0, c["entities.revealed_to_hero"]].tolist() == [0.0]
    torch.testing.assert_close(rows[0, c["entities.rel_pos"]], (st.ent_pos[0, 1] - st.ent_pos[0, 0]) / 20)
    assert rows[0, c["entities.hp"]].item() == pytest.approx(float(st.ent_hp[0, 1]) / 20000.0)
    assert not rows[1].any()


def test_critic_false_leaves_the_critic_buffers_untouched():
    """What the live assembler relies on: a build with critic=False writes the actor groups and
    not one value of the critic groups, which a training build does fill."""
    env, cfg, full = _env_and_obs(n_envs=2)
    spec = obs_select.load_agent_spec(DEPLOY6_YAML, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=2, device="cpu")
    obs_select.build_agent_obs(full, spec, cfg, buffers, critic=False)
    assert buffers["self"].any()
    assert not any(buffers[g.name].any() for g in spec.critic_groups)
    obs_select.build_agent_obs(full, spec, cfg, buffers)
    assert all(buffers[g.name].any() for g in spec.critic_groups)
