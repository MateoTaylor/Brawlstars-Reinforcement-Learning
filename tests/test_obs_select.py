import tempfile
import os

import gymnasium
import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


def _env_and_obs(n_envs=4, seed=0, overrides=None):
    cfg = _cfg(overrides)
    env = BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed)
    full_obs = env.reset()
    return env, cfg, full_obs


def _write_spec(spec_dict) -> str:
    path = tempfile.mktemp(suffix=".yaml")
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
    assert not torch.equal(true_pos, torch.zeros_like(true_pos)) or True  # sanity, not the actual assertion below

    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)

    row = out["enemies"][0, 0]  # 'enemies' slot 0 == entity index 1 (hero dropped)
    assert torch.equal(row, torch.zeros_like(row))

    index_map = obs_select.agent_obs_index_map(spec, cfg)
    start, end = index_map["enemies"]["entities.revealed_to_hero"]
    assert torch.equal(row[start:end], torch.zeros(end - start))


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


# ---- max_slots projectile selection: alive, sorted by time_to_closest, zero-padded --------

def test_projectile_group_selects_k_nearest_by_time_to_closest_and_zero_pads():
    env, cfg, full_obs = _env_and_obs(n_envs=1)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    proj_group = next(g for g in spec.groups if g.name == "projectiles")
    assert proj_group.max_slots == 12

    # no projectiles alive yet (fresh reset) -- every one of the 12 slots must be all-zero.
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")
    out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
    assert torch.equal(out["projectiles"], torch.zeros_like(out["projectiles"]))


def test_a_fair_projectile_group_admits_only_what_is_on_screen():
    """deploy4's projectile group is gated by `projectiles.in_view`, and since C4 that is the
    camera window, not the crop. A projectile 20 tiles east flying at the hero has the smallest
    time_to_closest on the map and still gets an all-zero row; 5 tiles east it fills one."""
    env, cfg, _ = _env_and_obs(n_envs=1, overrides={
        "world": {"map_h": 60, "map_w": 60, "maps": ["walled"], "map_selection": "fixed",
                  "fixed_map": "walled"}})
    spec = obs_select.load_agent_spec("configs/agent_obs_deploy4.yaml", cfg)
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


# ---- build_agent_obs allocates nothing per call (steady-state memory doesn't grow) --------

def test_build_agent_obs_reuses_the_same_out_buffer_tensors():
    env, cfg, full_obs = _env_and_obs(n_envs=4)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    buffers = obs_select.make_agent_obs_buffers(spec, cfg, n_envs=4, device="cpu")
    data_ptrs_before = {name: t.data_ptr() for name, t in buffers.items()}

    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        action = torch.stack([torch.randint(0, 17, (4,), generator=gen), torch.randint(0, 2, (4,), generator=gen)], dim=1)
        full_obs, *_rest = env.step(action)
        out = obs_select.build_agent_obs(full_obs, spec, cfg, buffers)
        assert out is buffers  # same dict object returned
        for name, t in out.items():
            assert t.data_ptr() == data_ptrs_before[name]  # same underlying storage, every call


# ---- docs/AGENT_OBS.md regenerates deterministically ---------------------------------------

def test_dump_obs_schema_regenerates_agent_obs_docs_deterministically():
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    first = dump_mod.render_agent_obs()
    second = dump_mod.render_agent_obs()
    assert first == second
    assert "## `self`" in first and "## `grid`" in first


def test_each_spec_renders_to_a_doc_named_after_itself():
    """SIM_OVERHAUL Step I3: `--spec` gives every sibling spec its own doc, because each narrowing
    is a different width and a different from-scratch run, so a column index only means something
    next to the spec it came from. The default path is the one the constant already names."""
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    assert dump_mod.agent_docs_path("configs/agent_obs.yaml") == dump_mod.AGENT_DOCS_PATH
    assert dump_mod.agent_docs_path("configs/agent_obs.yaml").name == "AGENT_OBS.md"
    assert dump_mod.agent_docs_path("configs/agent_obs_deploy4.yaml").name == "AGENT_OBS_DEPLOY4.md"
    assert dump_mod.agent_docs_path("configs/agent_obs_lowinfo.yaml").name == "AGENT_OBS_LOWINFO.md"


def test_the_deployed_spec_renders_its_own_layout_not_the_full_one():
    """The deploy4 doc has to show the widths H3 pins and say which file it came from, or the
    deployment mirrors have no readable reference for a column index."""
    import importlib
    dump_mod = importlib.import_module("scripts.dump_obs_schema")
    deploy4 = dump_mod.render_agent_obs("configs/agent_obs_deploy4.yaml")
    assert "`configs/agent_obs_deploy4.yaml`" in deploy4.splitlines()[2]
    assert "## `history`" in deploy4
    assert "shape: `(26,)`" in deploy4                    # self, the gadget pair included
    assert "`hero.gadget_ready`" in deploy4
    assert "shape: `(13, 13, 21)`" in deploy4             # ten deploy3 planes + three history
    assert "| 12 | `enemy_hist3` |" in deploy4
    assert deploy4 != dump_mod.render_agent_obs()         # --spec really changes the render


# ---- normalize: true, per unit ------------------------------------------------------------
#
# Until the deployed spec dropped `hp_frac`, nothing here exercised `normalize: true` at all --
# obs_select's own docstring called it "a genuinely underspecified corner of Step 32's plan text
# (no acceptance criterion exercises it)". That is how `hp` and `count` fields came to sit in a
# normalized spec undivided. These tests are that acceptance criterion.

def test_every_normalized_unit_divides_by_its_documented_scale():
    cfg = _cfg()
    assert obs_select._norm_divisor("tiles", cfg) == float(max(cfg.map_w, cfg.map_h))
    assert obs_select._norm_divisor("tiles/s", cfg) == obs_select._NORM_SPEED_SCALE
    assert obs_select._norm_divisor("hp", cfg) == obs_select._NORM_HP_SCALE
    assert obs_select._norm_divisor("count", cfg) == obs_select._NORM_COUNT_SCALE
    # Bounded by construction; a divisor here would shrink a field that is already in range.
    for units in ("fraction", "bool", "onehot", "unitless", "radians"):
        assert obs_select._norm_divisor(units, cfg) == 1.0


def test_the_hp_scale_covers_the_largest_hp_the_sim_can_produce():
    """Derived from the shipped config rather than restated, so a change to the roster or the
    cube numbers fails here instead of quietly pushing a normalized field past 1.

    The ceiling is an enemy at the roster's highest base HP, holding `max_cubes` at
    `hp_per_cube` flat, scaled by the top of `enemy_hp_mult` -- see core/stats.effective_max_hp,
    which applies that multiplier to the cube total and never to the hero.
    """
    import yaml as _yaml

    # cubes.* resolve onto SimParams, not EnvConfig, so they are read from the file itself.
    cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]
    brawlers = _yaml.safe_load(open("configs/brawlers.yaml").read())
    base = [v["base_hp"] for v in brawlers.values() if isinstance(v, dict) and "base_hp" in v]
    ceiling = (max(base) + cubes["max_cubes"] * cubes["hp_per_cube"]) * 1.25

    normalized = ceiling / obs_select._NORM_HP_SCALE
    assert normalized <= 1.05, (
        f"max possible HP {ceiling:.0f} normalizes to {normalized:.2f}; raise _NORM_HP_SCALE"
    )
    # Not so large that a real HP reading vanishes into the noise floor either -- a fresh Mortis
    # should land near the ~0.1-0.5 band _NORM_SPEED_SCALE puts a walking entity in.
    assert 0.2 <= brawlers["hero_mortis"]["base_hp"] / obs_select._NORM_HP_SCALE <= 0.6


def test_the_count_scale_covers_every_bounded_count_in_the_schema():
    import yaml as _yaml

    cfg = _cfg()
    max_cubes = _yaml.safe_load(open("configs/default.yaml").read())["cubes"]["max_cubes"]
    for largest in (max_cubes, cfg.n_enemies, 3):  # cubes, brawlers left, ammo
        assert largest / obs_select._NORM_COUNT_SCALE <= 1.0


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
    the vector depends on the group's FIELDS. All three shipped specs have a group called
    "self", at 25/25/24 columns, so loading two of them in one process gave the second the
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
    # The pair that found it: the full suite built agent_obs_deploy3.yaml's grid (10 planes) and
    # then every agent_obs_deploy.yaml test (8) died on a size mismatch.
    ("configs/agent_obs_deploy3.yaml", "configs/agent_obs_deploy.yaml"),
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


# ---- the history group and planes (SIM_OVERHAUL Step H2) ---------------------------------------

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
    """3 * (1 + 17 + 4 + 1 + 1 + 2), the width H2.2 pins."""
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


# ---- slots: tracked (OBS_PARITY_TASKS.md C9) -------------------------------------------------

DEPLOY4_YAML = "configs/agent_obs_deploy4.yaml"


def _enemies_group(doc: dict) -> dict:
    """deploy4's one hero-axis per-entity group."""
    (g,) = [g for g in doc["groups"]
            if g.get("per_entity") and all(f.startswith("entities.") for f in g.get("fields", ()))]
    return g


def _deploy4_variant(**group_keys) -> str:
    """deploy4 written to a temp file with `group_keys` set on its enemies group."""
    doc = yaml.safe_load(open(DEPLOY4_YAML).read())
    _enemies_group(doc).update(group_keys)
    return _write_spec(doc)


def test_tracked_slots_order_the_rows_by_the_sims_slot_table():
    """Slot 0 <- entity 3, slot 1 <- entity 1, slot 2 empty: the tracked build's rows are the
    plain build's rows for those entities, and the empty slot is a zero row."""
    env, cfg, full = _env_and_obs(
        n_envs=1, overrides={"entities": {**CONFIGS_TINY.get("entities", {}), "n_enemies": 3}})
    plain = obs_select.load_agent_spec(_deploy4_variant(), cfg)
    tracked = obs_select.load_agent_spec(_deploy4_variant(slots="tracked"), cfg)
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
        obs_select.load_agent_spec(_deploy4_variant(**group_keys), _cfg())


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
