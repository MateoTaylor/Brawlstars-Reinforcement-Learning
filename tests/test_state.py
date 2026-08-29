import dataclasses
from pathlib import Path

import numpy as np
import torch
import pytest
import yaml

from brawl_sim.config import EnvConfig, build_params, load_config
from brawl_sim.core.state import (
    SimState,
    allocate,
    check_invariants,
    snapshot,
    zero_,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _small_cfg(**kw):
    base = dict(n_enemies=2, max_projectiles=8, max_boxes=4, max_pickups=4)
    base.update(kw)
    return EnvConfig(**base)


def _default_cfg():
    return load_config(CONFIGS / "default.yaml")


def _spec():
    default = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    brawlers = yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())
    return {**default, **brawlers}


# ---- allocate ---------------------------------------------------------------

def test_allocate_shapes_and_dtypes():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=8, device="cpu", verbose=False)
    N, E, P, B, U, L = 8, cfg.n_entities, cfg.max_projectiles, cfg.max_boxes, cfg.max_pickups, cfg.latency_buf_len

    assert state.ent_pos.shape == (N, E, 2) and state.ent_pos.dtype == torch.float32
    assert state.ent_facing.shape == (N, E) and state.ent_facing.dtype == torch.float32
    assert state.ent_alive.shape == (N, E) and state.ent_alive.dtype == torch.bool
    assert state.ent_kind.shape == (N, E) and state.ent_kind.dtype == torch.int64
    assert state.ent_cubes.dtype == torch.int64
    assert state.ent_dash_hits.shape == (N, E, E) and state.ent_dash_hits.dtype == torch.bool
    assert state.ent_death_step.dtype == torch.int32
    assert state.ent_kills.dtype == torch.int32

    assert state.prj_pos.shape == (N, P, 2)
    assert state.prj_owner.dtype == torch.int64
    assert state.prj_alive.dtype == torch.bool

    assert state.box_pos.shape == (N, B, 2)
    assert state.pku_pos.shape == (N, U, 2)
    assert state.pku_cubes.dtype == torch.int64

    assert state.zone_lo.shape == (N, 2)
    assert state.zone_step.dtype == torch.int32
    assert state.map_id.shape == (N,) and state.map_id.dtype == torch.int64
    assert state.time.shape == (N,) and state.time.dtype == torch.float32

    assert state.act_buf.shape == (N, L, 2) and state.act_buf.dtype == torch.int64
    assert state.act_head.shape == (N,) and state.act_head.dtype == torch.int64

    assert torch.equal(state.env_idx, torch.arange(N, dtype=torch.int64))


def test_allocate_on_device():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
    for name in SimState.__slots__:
        assert getattr(state, name).device.type == "cpu"


def test_allocate_all_zero_initially():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
    for name in SimState.__slots__:
        if name == "env_idx":
            continue
        t = getattr(state, name)
        assert torch.count_nonzero(t) == 0, name


def test_allocate_4096_envs_under_250mb():
    cfg = _default_cfg()
    state = allocate(cfg, n_envs=4096, device="cpu", verbose=False)
    total = 0
    for name in SimState.__slots__:
        t = getattr(state, name)
        total += t.element_size() * t.nelement()
    assert total < 250e6, f"SimState is {total / 1e6:.1f} MB, expected well under 250 MB"


def test_allocate_verbose_prints_report(capsys):
    cfg = _small_cfg()
    allocate(cfg, n_envs=8, device="cpu", verbose=True)
    out = capsys.readouterr().out
    assert "SimState memory report" in out
    assert "TOTAL" in out
    assert "ent_pos" in out


# ---- zero_ --------------------------------------------------------------------

def test_zero_only_touches_masked_rows():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=8, device="cpu", verbose=False)

    state.ent_pos.uniform_(1, 10)
    state.ent_hp.uniform_(1, 100)
    state.ent_alive.fill_(True)
    state.ent_kind.fill_(2)
    before = {name: getattr(state, name).clone() for name in SimState.__slots__}

    mask = torch.zeros(8, dtype=torch.bool)
    mask[[1, 5]] = True
    zero_(state, mask)

    unmasked = [i for i in range(8) if i not in (1, 5)]
    for name in SimState.__slots__:
        after = getattr(state, name)
        assert torch.equal(after[unmasked], before[name][unmasked]), name

    for i in (1, 5):
        assert torch.count_nonzero(state.ent_pos[i]) == 0
        assert torch.count_nonzero(state.ent_hp[i]) == 0
        assert not torch.any(state.ent_alive[i])
        assert torch.count_nonzero(state.ent_kind[i]) == 0

    # env_idx is never touched, even for masked rows
    assert torch.equal(state.env_idx, torch.arange(8, dtype=torch.int64))


def test_zero_all_rows():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=5, device="cpu", verbose=False)
    state.ent_pos.uniform_(1, 10)
    state.ent_hp.uniform_(1, 100)
    mask = torch.ones(5, dtype=torch.bool)
    zero_(state, mask)
    assert torch.count_nonzero(state.ent_pos) == 0
    assert torch.count_nonzero(state.ent_hp) == 0


def test_zero_mutates_in_place_same_tensor_identity():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
    ref = state.ent_pos
    zero_(state, torch.ones(4, dtype=torch.bool))
    assert state.ent_pos is ref


# ---- check_invariants ---------------------------------------------------------

def _valid_state_and_params(n_envs=4):
    cfg = _default_cfg()
    cfg = dataclasses.replace(cfg, debug_checks=True)
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=_spec())

    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = 0
    state.ent_kind[:, 1:] = 1
    state.ent_max_hp.fill_(1000.0)
    state.ent_hp.fill_(1000.0)
    state.ent_pos.uniform_(5, cfg.map_w - 5)
    state.box_max_hp.fill_(100.0)
    state.box_hp.fill_(100.0)
    state.act_head.fill_(0)
    return cfg, state, params


def test_check_invariants_passes_on_well_formed_state():
    cfg, state, params = _valid_state_and_params()
    check_invariants(state, cfg, params)  # must not raise


def test_check_invariants_noop_when_debug_checks_false():
    cfg, state, params = _valid_state_and_params()
    cfg = dataclasses.replace(cfg, debug_checks=False)
    state.ent_hp[0, 0] = float("nan")  # would fail if checked
    check_invariants(state, cfg, params)  # must not raise -- it's a no-op


def test_check_invariants_catches_nan():
    cfg, state, params = _valid_state_and_params()
    state.ent_hp[0, 0] = float("nan")
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_pos_out_of_bounds():
    cfg, state, params = _valid_state_and_params()
    state.ent_pos[0, 0, 0] = cfg.map_w + 10.0
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_dead_with_nonzero_hp():
    cfg, state, params = _valid_state_and_params()
    state.ent_alive[0, 0] = False
    state.ent_hp[0, 0] = 50.0
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_cubes_over_max():
    cfg, state, params = _valid_state_and_params()
    state.ent_cubes[0, 0] = int(params.max_cubes[0].item()) + 5
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_dash_t_over_duration():
    cfg, state, params = _valid_state_and_params()
    kind0_dash_duration = params.dash_duration[0, 0].item()
    state.ent_dash_t[0, 0] = kind0_dash_duration + 10.0
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_bad_act_head():
    cfg, state, params = _valid_state_and_params()
    state.act_head[0] = state.act_buf.shape[1] + 5
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


def test_check_invariants_catches_hp_over_max_hp():
    cfg, state, params = _valid_state_and_params()
    state.ent_hp[0, 0] = state.ent_max_hp[0, 0] + 500.0
    with pytest.raises(ValueError):
        check_invariants(state, cfg, params)


# ---- snapshot -----------------------------------------------------------------

def test_snapshot_returns_cpu_numpy_for_one_env():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=4, device="cpu", verbose=False)
    state.ent_pos[2].uniform_(1, 10)
    state.ent_hp[2].fill_(500.0)
    state.map_id[2] = 1

    snap = snapshot(state, 2)
    assert isinstance(snap["ent_pos"], np.ndarray)
    assert snap["ent_pos"].shape == (cfg.n_entities, 2)
    assert np.allclose(snap["ent_pos"], state.ent_pos[2].numpy())
    assert np.all(snap["ent_hp"] == 500.0)
    assert snap["map_id"] == 1


def test_snapshot_is_a_real_copy_not_a_view_on_cpu():
    """Regression test (found by scripts/record_rollout.py, Step 37): on a CPU-device state,
    `.cpu()` is a no-op and `.numpy()` shares memory with the live tensor -- without an explicit
    `.copy()`, a snapshot taken before a later in-place mutation would silently change value
    underneath the caller. This is exactly the bug: take two snapshots, mutate state in between,
    and confirm the FIRST snapshot did not change."""
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=2, device="cpu", verbose=False)
    state.ent_hp[0].fill_(111.0)
    state.step_count[0] = 0

    first = snapshot(state, 0)
    assert np.all(first["ent_hp"] == 111.0)
    assert int(first["step_count"]) == 0

    state.ent_hp[0].fill_(999.0)
    state.step_count[0] = 7

    assert np.all(first["ent_hp"] == 111.0), "snapshot mutated in place after a later state change"
    assert int(first["step_count"]) == 0, "snapshot mutated in place after a later state change"


def test_snapshot_covers_every_field():
    cfg = _small_cfg()
    state = allocate(cfg, n_envs=2, device="cpu", verbose=False)
    snap = snapshot(state, 0)
    assert set(snap.keys()) == set(SimState.__slots__)
