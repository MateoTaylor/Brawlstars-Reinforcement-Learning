from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import Kind, N_KINDS
from brawl_sim.core import stats

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _real_params(n_envs=4):
    cfg = load_config(CONFIGS / "default.yaml")
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    return cfg, build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)


def _synthetic_params(n_envs=3, n_kinds=N_KINDS):
    # base_hp/base_damage/move_speed distinct per kind so a gather bug (e.g. always kind 0)
    # is unmistakable; values distinct per env too, so a batching bug is also unmistakable.
    base_hp = torch.arange(1, n_envs * n_kinds + 1, dtype=torch.float32).reshape(n_envs, n_kinds) * 100
    base_damage = torch.arange(1, n_envs * n_kinds + 1, dtype=torch.float32).reshape(n_envs, n_kinds) * 10
    move_speed = torch.arange(1, n_envs * n_kinds + 1, dtype=torch.float32).reshape(n_envs, n_kinds) * 0.1
    return SimpleNamespace(
        base_hp=base_hp,
        base_damage=base_damage,
        move_speed=move_speed,
        cube_hp_flat=torch.full((n_envs,), 400.0),
        cube_damage_bonus=torch.full((n_envs,), 0.2),
        enemy_hp_mult=torch.full((n_envs,), 2.0),
        enemy_damage_mult=torch.full((n_envs,), 3.0),
    )


# ---- gather_kind --------------------------------------------------------------

def test_gather_kind_matches_naive_loop():
    params = _synthetic_params(n_envs=5, n_kinds=N_KINDS)
    kind = torch.randint(0, N_KINDS, (5, 6), dtype=torch.int64)
    result = stats.gather_kind(params.base_hp, kind)

    naive = torch.zeros_like(result)
    for n in range(5):
        for e in range(6):
            naive[n, e] = params.base_hp[n, kind[n, e]]
    assert torch.equal(result, naive)


def test_gather_kind_does_not_collapse_to_hero_stats():
    # every entity a different, non-hero kind -- a "gather_kind always returns kind 0" bug
    # would make every column equal params.base_hp[:, 0], which this directly contradicts.
    params = _synthetic_params(n_envs=2, n_kinds=N_KINDS)
    kind = torch.arange(N_KINDS, dtype=torch.int64).unsqueeze(0).expand(2, N_KINDS).clone()
    result = stats.gather_kind(params.base_hp, kind)
    assert torch.equal(result, params.base_hp)  # column k should equal base_hp[:, k], not [:, 0]
    for k in range(1, N_KINDS):
        assert not torch.equal(result[:, k], params.base_hp[:, 0])


# ---- is_hero --------------------------------------------------------------------

def test_is_hero_matches_kind_enum():
    kind = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    expected = torch.tensor([[True, False, False, False, False]])
    assert torch.equal(stats.is_hero(kind), expected)


# ---- effective_max_hp / effective_damage (acceptance) --------------------------

def test_hero_cube_bonus_scales_linearly_with_each_configured_rate():
    """HP and damage cube bonuses have DIFFERENT SHAPES, not just different rates: HP is a FLAT
    `cubes.hp_per_cube` added to base (400, the real game's number, Step E2) while damage is a
    FRACTION of base. Each expectation is derived from its own param, so a rebalance of either
    moves the test with the code.

    Its two previous versions are why: it first hardcoded 1.30x for both and went red when the
    damage rate was set to the real 10%, then derived both as fractions and would have gone red
    again when HP became flat -- both times with the code doing exactly the right thing."""
    cfg, params = _real_params(n_envs=2)
    kind = torch.zeros(2, 1, dtype=torch.int64)  # hero only
    n_cubes = 2
    cubes = torch.full((2, 1), n_cubes, dtype=torch.int64)

    hp = stats.effective_max_hp(kind, cubes, params)
    dmg = stats.effective_damage(kind, cubes, params)
    hp_flat = params.cube_hp_flat.unsqueeze(-1)
    dmg_rate = params.cube_damage_bonus.unsqueeze(-1)

    assert torch.allclose(hp, params.base_hp[:, 0:1] + hp_flat * n_cubes, rtol=1e-5)
    assert torch.allclose(dmg, params.base_damage[:, 0:1] * (1.0 + dmg_rate * n_cubes), rtol=1e-5)


def test_cube_hp_is_flat_so_every_brawler_gains_the_same_amount():
    """The property that distinguishes the real mechanic from the fraction it replaced: cubes
    NARROW the HP spread rather than widening it. Asserted across two kinds with genuinely
    different base HP (Buzz is the tankiest, Brock among the squishiest), so a regression to a
    percentage bonus fails here even if the per-cube number happens to look right."""
    cfg, params = _real_params(n_envs=2)
    tanky, squishy = int(Kind.BOT_MELEE), int(Kind.BOT_SNIPER)
    kind = torch.tensor([[tanky, squishy]] * 2, dtype=torch.int64)
    n_cubes = 7

    zero = stats.effective_max_hp(kind, torch.zeros(2, 2, dtype=torch.int64), params)
    full = stats.effective_max_hp(kind, torch.full((2, 2), n_cubes, dtype=torch.int64), params)
    gain = full - zero

    assert float(zero[0, 0]) > float(zero[0, 1])          # precondition: the bases really differ
    assert torch.allclose(gain[:, 0], gain[:, 1])          # ...and both gain the same
    assert torch.allclose(gain[:, 0], params.cube_hp_flat * n_cubes, rtol=1e-5)
    # The spread therefore SHRINKS in relative terms, which a fractional bonus could never do.
    assert float(full[0, 0] / full[0, 1]) < float(zero[0, 0] / zero[0, 1])


def test_enemy_hp_mult_does_not_change_the_hero():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.zeros(1, 1, dtype=torch.int64)  # hero
    cubes = torch.zeros(1, 1, dtype=torch.int64)
    hp_with_mult = stats.effective_max_hp(kind, cubes, params)
    params_no_mult = SimpleNamespace(**{**params.__dict__, "enemy_hp_mult": torch.ones(1)})
    hp_without_mult = stats.effective_max_hp(kind, cubes, params_no_mult)
    assert torch.allclose(hp_with_mult, hp_without_mult)
    assert torch.allclose(hp_with_mult, params.base_hp[:, 0:1])


def test_enemy_hp_mult_does_change_bots():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.ones(1, 1, dtype=torch.int64)  # a bot kind
    cubes = torch.zeros(1, 1, dtype=torch.int64)
    hp = stats.effective_max_hp(kind, cubes, params)
    expected = params.base_hp[:, 1:2] * params.enemy_hp_mult.unsqueeze(-1)
    assert torch.allclose(hp, expected)


def test_effective_damage_applies_cube_bonus_and_enemy_mult():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.full((1, 1), 2, dtype=torch.int64)
    cubes = torch.full((1, 1), 3, dtype=torch.int64)
    dmg = stats.effective_damage(kind, cubes, params)
    expected = params.base_damage[:, 2:3] * (1 + params.cube_damage_bonus.unsqueeze(-1) * 3) * params.enemy_damage_mult.unsqueeze(-1)
    assert torch.allclose(dmg, expected)


def test_effective_max_hp_zero_cubes_is_just_base_times_mult():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.full((1, 1), 3, dtype=torch.int64)
    cubes = torch.zeros(1, 1, dtype=torch.int64)
    hp = stats.effective_max_hp(kind, cubes, params)
    expected = params.base_hp[:, 3:4] * params.enemy_hp_mult.unsqueeze(-1)
    assert torch.allclose(hp, expected)


# ---- effective_speed --------------------------------------------------------------

def test_effective_speed_matches_gather_no_bonus():
    params = _synthetic_params(n_envs=2, n_kinds=N_KINDS)
    kind = torch.randint(0, N_KINDS, (2, 4), dtype=torch.int64)
    speed = stats.effective_speed(kind, params)
    assert torch.equal(speed, stats.gather_kind(params.move_speed, kind))


# ---- apply_cube_gain --------------------------------------------------------------

def test_apply_cube_gain_raises_hp_by_max_hp_delta():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.full((1, 1), 1, dtype=torch.int64)
    old_cubes = torch.zeros(1, 1, dtype=torch.int64)
    new_cubes = torch.full((1, 1), 4, dtype=torch.int64)
    hp = torch.tensor([[50.0]])  # some damage already taken relative to old max

    old_max = stats.effective_max_hp(kind, old_cubes, params)
    new_max = stats.effective_max_hp(kind, new_cubes, params)
    new_hp = stats.apply_cube_gain(hp, kind, old_cubes, new_cubes, params)

    assert torch.allclose(new_hp - hp, new_max - old_max)
    # deficit relative to max is preserved, not healed to full
    assert torch.allclose((new_max - new_hp), (old_max - hp))


def test_apply_cube_gain_no_change_when_cubes_unchanged():
    params = _synthetic_params(n_envs=1, n_kinds=N_KINDS)
    kind = torch.full((1, 1), 2, dtype=torch.int64)
    cubes = torch.full((1, 1), 3, dtype=torch.int64)
    hp = torch.tensor([[123.0]])
    new_hp = stats.apply_cube_gain(hp, kind, cubes, cubes, params)
    assert torch.allclose(new_hp, hp)


# ---- batched leading-dim smoke test -------------------------------------------

def test_functions_support_n_e_batch():
    n, e = 8, 7
    params = _synthetic_params(n_envs=n, n_kinds=N_KINDS)
    kind = torch.randint(0, N_KINDS, (n, e), dtype=torch.int64)
    cubes = torch.randint(0, 5, (n, e), dtype=torch.int64)
    hp = torch.rand(n, e) * 1000

    for out in (
        stats.effective_max_hp(kind, cubes, params),
        stats.effective_damage(kind, cubes, params),
        stats.effective_speed(kind, params),
        stats.is_hero(kind).to(torch.float32),
        stats.apply_cube_gain(hp, kind, cubes, cubes + 1, params),
    ):
        assert out.shape == (n, e)
        assert not torch.any(torch.isnan(out))
