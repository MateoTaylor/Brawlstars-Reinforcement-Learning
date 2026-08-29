from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import (
    ARCHETYPE_SHORT_NAMES,
    EnvConfig,
    SimParams,
    _dset,
    apply_randomization,
    build_params,
    check_removed_keys,
    load_config,
    load_randomization,
    resample_params,
    validate,
)
from brawl_sim.constants import N_KINDS, Proj

# These mirror BRAWL_SIM_BUILD_PLAN.md Step 4's configs/default.yaml and configs/brawlers.yaml
# verbatim (section/key names must match exactly -- Step 4 adds no code, only these files).
DEFAULT_YAML = """
seed: 0
device: cpu

world: {map_h: 60, map_w: 60, maps: [open, bushy, walled],
        map_selection: uniform, fixed_map: open}
view:  {height: 20, width: 40}

sim:
  dt: 0.05
  max_episode_steps: 3000
  action_repeat: 1
  action_latency_seconds: 0.001

action: {n_move_bins: 16, dash_on_idle: facing}

observation:
  include_world_grid: true
  include_privileged: true
  grid_dtype: uint8

entities:
  n_enemies: 6
  randomize_enemy_types: true
  enemy_type_weights: {sniper: 0.2, artillery: 0.2, melee: 0.2, rifle: 0.2, edgar: 0.2, spike: 0.2,
                       bull: 0.2}
  fixed_enemy_types: [sniper, artillery, melee, rifle, sniper, rifle]
  enemy_hp_mult: 1.0
  enemy_damage_mult: 1.0
  unit_radius: 0.4

limits: {max_projectiles: 128, max_boxes: 16, max_pickups: 32}

cubes:
  hp_per_cube: 400
  damage_bonus_per_cube: 0.15
  max_cubes: 12
  cubes_per_box: 1
  cubes_on_kill_base: 1
  drop_victim_cubes: true
  pickup_radius: 0.6

boxes: {n_boxes: 8, hp: 3000}

zone:
  enabled: true
  mode: rect
  start_fraction: 0.2
  step_seconds: 4.0
  tiles_per_step: 1
  # Step B3 renamed these from the flat `dps` / `dps_growth_per_step`. This fixture is exactly the
  # stale-config case validate() now rejects -- it was carrying `dps: 1000.0`, which the new
  # schema does not read, leaving the zone silently inert. Caught by the new check, as intended.
  max_hp_fraction_per_second: 0.20
  fraction_growth_per_step: 0.04
  iframes_block_zone: false

regen: {enabled: false, delay_seconds: 3.0, max_hp_fraction_per_second: 0.13}

bots: {break_boxes: true, collect_cubes: true, avoid_zone: true, fight_each_other: true}

perception:
  bush_reveal_radius: 2.0
  reveal_after_attack: 1.0
  los_step_tiles: 0.5
  max_ray_tiles: 24.0

engine: {debug_checks: false, compile: false}
"""

BRAWLERS_YAML = """
hero_mortis:
  base_hp: 3800
  base_damage: 1300
  move_speed: 2.56
  max_ammo: 3
  reload_seconds: 1.7
  attack_cooldown: 0.30
  dash_distance: 5.0
  dash_duration: 0.30
  dash_radius: 0.70

bot_sniper:
  base_hp: 2600
  base_damage: 1600
  move_speed: 2.40
  attack_range: 11.0
  max_ammo: 3
  reload_seconds: 1.3
  attack_cooldown: 0.25
  proj_kind: SNIPER_BOLT
  proj_speed: 14.0
  proj_radius: 0.25
  proj_count: 1
  proj_spread_rad: 0.0
  aoe_radius: 0.0
  aim_noise_std_rad: 0.06
  reaction_delay: 0.15
  lead_target_fraction: 0.9
  decision_period_ticks: 4
  fire_needs_los: true
  aim_model: LEAD
  desired_range_fraction: 0.85

bot_artillery:
  base_hp: 3400
  base_damage: 1100
  move_speed: 2.30
  attack_range: 10.0
  max_ammo: 3
  reload_seconds: 1.8
  attack_cooldown: 0.40
  proj_kind: ARTILLERY_SHELL
  proj_speed: 6.0
  proj_radius: 0.30
  proj_count: 1
  proj_spread_rad: 0.0
  aoe_radius: 1.5
  aim_noise_tiles: 0.8
  reaction_delay: 0.25
  lead_target_fraction: 0.6
  decision_period_ticks: 4
  fire_needs_los: false
  aim_model: LOB
  desired_range_fraction: 0.7

bot_melee:
  base_hp: 6500
  base_damage: 500
  move_speed: 2.60
  attack_range: 2.0
  attack_arc_rad: 0.9
  max_ammo: 3
  reload_seconds: 0.9
  attack_cooldown: 0.15
  proj_kind: NONE
  aim_noise_std_rad: 0.10
  reaction_delay: 0.10
  lead_target_fraction: 0.0
  decision_period_ticks: 2
  fire_needs_los: false
  aim_model: DIRECT
  desired_range_fraction: 0.5

bot_rifle:
  base_hp: 3800
  base_damage: 800
  move_speed: 2.40
  attack_range: 8.0
  max_ammo: 3
  reload_seconds: 1.5
  attack_cooldown: 0.35
  proj_kind: RIFLE_ARROW
  proj_speed: 9.0
  proj_radius: 0.25
  proj_count: 3
  proj_spread_rad: 0.18
  aoe_radius: 0.0
  aim_noise_std_rad: 0.08
  reaction_delay: 0.20
  lead_target_fraction: 0.7
  decision_period_ticks: 4
  fire_needs_los: true
  fire_range_fraction: 0.9
  fire_lateral_speed_limit: 2.0
  fire_lateral_hold_range_fraction: 0.6
  aim_model: LEAD
  desired_range_fraction: 0.75
"""

RANDOMIZATION_YAML = """
entities.enemy_hp_mult:       {low: 0.8, high: 1.25}
bot_sniper.aim_noise_std_rad: {low: 0.03, high: 0.20}
bot_melee.move_speed:         {low: 0.9, high: 1.15, mode: multiplicative}
hero_mortis.reload_seconds:   {low: 1.5, high: 1.9}
"""

RANDOMIZATION_YAML_COMMENTED = """
# entities.enemy_hp_mult:       {low: 0.8, high: 1.25}
# bot_sniper.aim_noise_std_rad: {low: 0.03, high: 0.20}
"""


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "default.yaml").write_text(DEFAULT_YAML)
    (tmp_path / "brawlers.yaml").write_text(BRAWLERS_YAML)
    (tmp_path / "randomization.yaml").write_text(RANDOMIZATION_YAML)
    (tmp_path / "randomization_commented.yaml").write_text(RANDOMIZATION_YAML_COMMENTED)
    return tmp_path


def merged_spec():
    default = yaml.safe_load(DEFAULT_YAML)
    brawlers = yaml.safe_load(BRAWLERS_YAML)
    return {**default, **brawlers}


def _dummy_params(cfg: EnvConfig) -> SimParams:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    # Minimal but COHERENT: every kind with proj_count > 0 also needs a way for its projectiles to
    # move (or validate rightly rejects it), plus the ammo/range/cooldown that
    # `peak_projectile_demand` reads -- with those at 0 the demand computes as 0 and the
    # max_projectiles check below can never fire. This fixture exists to exercise validate's OTHER
    # branches, so it must not itself be invalid OR vacuous.
    # desired_range_fraction is required alongside attack_range (Step E1): it is the fraction of
    # its own range a KITE bot holds, and 0 would make one charge to point blank.
    shooter = {"max_ammo": 3, "attack_range": 8.0, "attack_cooldown": 0.25,
               "desired_range_fraction": 0.8}
    spec = {
        "bot_sniper": {**shooter, "proj_count": 1, "proj_speed": 10.0},
        "bot_artillery": {**shooter, "proj_count": 1, "proj_flight_seconds": 1.0},
        "bot_rifle": {**shooter, "proj_count": 3, "proj_speed": 8.0},
        "zone": {"start_fraction": 0.2, "step_seconds": 4.0,
                 "max_hp_fraction_per_second": 0.2},
    }
    return build_params(cfg, n_envs=4, device="cpu", gen=gen, spec=spec)


# ---- EnvConfig --------------------------------------------------------------

def test_load_config_maps_fields(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    assert cfg.map_h == 60 and cfg.map_w == 60
    assert cfg.map_names == ("open", "bushy", "walled")
    assert cfg.view_h == 20 and cfg.view_w == 40
    assert cfg.n_enemies == 6
    # One entry per ARCHETYPE_SHORT_NAMES, in that order -- the point of this assertion is the
    # mapping from named keys to a positional tuple, so it is written against the roster's own
    # length rather than a literal that has to be edited for every new brawler.
    assert cfg.enemy_type_weights == (0.2,) * len(ARCHETYPE_SHORT_NAMES)
    assert cfg.fixed_enemy_types == ("sniper", "artillery", "melee", "rifle", "sniper", "rifle")
    assert cfg.dt == 0.05 and cfg.max_episode_steps == 3000
    assert cfg.action_latency_seconds == 0.001
    assert cfg.obs_include_world_grid is True
    assert cfg.bots_break_boxes is True
    assert cfg.zone_enabled is True and cfg.zone_mode == "rect"
    assert cfg.regen_enabled is False
    assert cfg.los_step_tiles == 0.5 and cfg.max_ray_tiles == 24.0
    assert cfg.debug_checks is False
    assert cfg.device == "cpu"


def test_env_config_is_frozen_and_hashable():
    cfg = EnvConfig()
    with pytest.raises(Exception):
        cfg.map_h = 10
    hash(cfg)  # must not raise


def test_load_config_overrides_deep_merge(config_dir):
    cfg = load_config(
        config_dir / "default.yaml",
        overrides={
            "world": {"map_h": 20, "map_w": 20},
            "view": {"height": 10, "width": 14},
            "entities": {"n_enemies": 2},
            "sim": {"max_episode_steps": 300},
            "zone": {"enabled": False},
        },
    )
    assert cfg.map_h == 20 and cfg.map_w == 20
    assert cfg.view_h == 10 and cfg.view_w == 14
    assert cfg.n_enemies == 2
    assert cfg.n_entities == 3
    assert cfg.max_episode_steps == 300
    assert cfg.zone_enabled is False
    assert cfg.dt == 0.05  # untouched field keeps its default.yaml value


def test_derived_properties():
    cfg = EnvConfig(n_enemies=6, n_move_bins=16, max_ray_tiles=24.0, los_step_tiles=0.5,
                     action_latency_seconds=0.001, dt=0.05)
    assert cfg.n_entities == 7
    # 3-valued attack dim as of Step D2: 0 = nothing, 1 = attack, 2 = super.
    assert cfg.action_nvec == (17, 3)
    assert cfg.ray_steps == 48
    assert cfg.action_latency_ticks == 0
    assert cfg.latency_buf_len == 1

    cfg2 = EnvConfig(action_latency_seconds=0.1, dt=0.05)
    assert cfg2.action_latency_ticks == 2
    assert cfg2.latency_buf_len == 3


# ---- RandomizationSpec -------------------------------------------------------

def test_load_randomization_parses_ranges(config_dir):
    spec = load_randomization(config_dir / "randomization.yaml")
    assert spec["entities.enemy_hp_mult"] == {"low": 0.8, "high": 1.25, "mode": "additive"}
    assert spec["bot_melee.move_speed"]["mode"] == "multiplicative"


def test_load_randomization_empty_when_all_commented(config_dir):
    spec = load_randomization(config_dir / "randomization_commented.yaml")
    assert spec == {}


def test_apply_randomization_additive_and_multiplicative(config_dir):
    spec = merged_spec()
    randomization = load_randomization(config_dir / "randomization.yaml")
    merged = apply_randomization(spec, randomization)

    assert merged["entities"]["enemy_hp_mult"] == {"low": 0.8, "high": 1.25}

    base_speed = spec["bot_melee"]["move_speed"]
    result = merged["bot_melee"]["move_speed"]
    assert result["low"] == pytest.approx(base_speed * 0.9)
    assert result["high"] == pytest.approx(base_speed * 1.15)
    assert spec["bot_melee"]["move_speed"] == base_speed  # original spec untouched


def test_apply_randomization_multiplicative_needs_scalar_base():
    with pytest.raises(ValueError):
        apply_randomization({}, {"bot_melee.move_speed": {"low": 0.9, "high": 1.1, "mode": "multiplicative"}})


# ---- SimParams / build_params / resample_params ------------------------------

def test_build_params_shapes_and_device(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=8, device="cpu", gen=gen, spec=spec)

    assert params.base_hp.shape == (8, N_KINDS)
    assert params.base_hp.dtype == torch.float32
    assert params.proj_count.shape == (8, N_KINDS)
    assert params.proj_count.dtype == torch.int64
    assert params.enemy_hp_mult.shape == (8,)
    assert params.zone_start_time.shape == (8,)
    for tensor in (params.base_hp, params.proj_count, params.enemy_hp_mult, params.zone_start_time):
        assert tensor.device.type == "cpu"


def test_build_params_scalars_identical_across_envs(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1)
    params = build_params(cfg, n_envs=16, device="cpu", gen=gen, spec=spec)

    assert torch.all(params.base_hp == params.base_hp[0])
    hero_attack_range = params.attack_range[:, 0]  # not defined for hero_mortis -> default 0
    assert torch.all(hero_attack_range == 0.0)


def test_build_params_proj_kind_resolves_enum(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2)
    params = build_params(cfg, n_envs=4, device="cpu", gen=gen, spec=spec)

    assert torch.all(params.proj_kind[:, 0] == int(Proj.NONE))            # hero_mortis
    assert torch.all(params.proj_kind[:, 1] == int(Proj.SNIPER_BOLT))     # bot_sniper
    assert torch.all(params.proj_kind[:, 2] == int(Proj.ARTILLERY_SHELL))  # bot_artillery
    assert torch.all(params.proj_kind[:, 3] == int(Proj.NONE))            # bot_melee
    assert torch.all(params.proj_kind[:, 4] == int(Proj.RIFLE_ARROW))     # bot_rifle


def test_build_params_zone_start_time_derived(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(3)
    params = build_params(cfg, n_envs=4, device="cpu", gen=gen, spec=spec)
    expected = 0.2 * cfg.max_episode_steps * cfg.dt
    assert torch.allclose(params.zone_start_time, torch.full((4,), expected))


def test_range_produces_variation_across_envs(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    randomization = load_randomization(config_dir / "randomization.yaml")
    merged = apply_randomization(spec, randomization)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(4)
    params = build_params(cfg, n_envs=64, device="cpu", gen=gen, spec=merged)

    assert params.enemy_hp_mult.min() >= 0.8 and params.enemy_hp_mult.max() <= 1.25
    assert not torch.all(params.enemy_hp_mult == params.enemy_hp_mult[0])

    sniper_noise = params.aim_noise_std_rad[:, 1]
    assert sniper_noise.min() >= 0.03 and sniper_noise.max() <= 0.20

    base_speed = spec["bot_melee"]["move_speed"]
    melee_speed = params.move_speed[:, 3]
    assert melee_speed.min() >= base_speed * 0.9 - 1e-5
    assert melee_speed.max() <= base_speed * 1.15 + 1e-5


def test_resample_params_only_touches_masked_rows(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    randomization = load_randomization(config_dir / "randomization.yaml")
    merged = apply_randomization(spec, randomization)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(5)
    params = build_params(cfg, n_envs=8, device="cpu", gen=gen, spec=merged)

    before_hp_mult = params.enemy_hp_mult.clone()
    before_move_speed = params.move_speed.clone()

    mask = torch.zeros(8, dtype=torch.bool)
    mask[[1, 5]] = True
    resample_params(params, mask, cfg, gen, merged)

    unmasked = ~mask
    assert torch.equal(params.enemy_hp_mult[unmasked], before_hp_mult[unmasked])
    assert torch.equal(params.move_speed[unmasked], before_move_speed[unmasked])
    # continuous range -> an exact repeat on the masked rows is vanishingly unlikely
    assert not torch.equal(params.enemy_hp_mult[mask], before_hp_mult[mask])


def test_resample_params_no_sync_ops(config_dir, monkeypatch):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(6)
    params = build_params(cfg, n_envs=8, device="cpu", gen=gen, spec=spec)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[0] = True

    def _boom(*a, **k):
        raise AssertionError("host sync op called from resample_params")

    monkeypatch.setattr(torch.Tensor, "item", _boom)
    monkeypatch.setattr(torch.Tensor, "numpy", _boom)
    monkeypatch.setattr(torch.Tensor, "tolist", _boom)
    resample_params(params, mask, cfg, gen, spec)


def test_range_with_low_greater_than_high_raises(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    spec["entities"]["unit_radius"] = {"low": 1.0, "high": 0.5}
    gen = torch.Generator(device="cpu")
    gen.manual_seed(8)
    with pytest.raises(ValueError):
        build_params(cfg, n_envs=4, device="cpu", gen=gen, spec=spec)


# ---- validate -----------------------------------------------------------

def test_validate_passes_on_default(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    spec = merged_spec()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    params = build_params(cfg, n_envs=8, device="cpu", gen=gen, spec=spec)
    validate(cfg, params)  # must not raise


def test_validate_view_larger_than_map():
    cfg = EnvConfig(map_h=10, map_w=10, view_h=20, view_w=40)
    with pytest.raises(ValueError):
        validate(cfg, _dummy_params(cfg))


def test_validate_n_enemies_too_low():
    cfg = EnvConfig(n_enemies=0)
    with pytest.raises(ValueError):
        validate(cfg, _dummy_params(cfg))


def test_validate_unknown_map_name():
    cfg = EnvConfig(map_names=("open", "not_a_map"))
    with pytest.raises(ValueError):
        validate(cfg, _dummy_params(cfg))


def test_validate_action_latency_negative():
    cfg = EnvConfig(action_latency_seconds=-0.1)
    with pytest.raises(ValueError):
        validate(cfg, _dummy_params(cfg))


def test_build_params_rejects_a_config_using_a_renamed_key(config_dir):
    """`config.check_removed_keys`. Every entry in _REMOVED_KEYS is a rename where the new key
    means something numerically DIFFERENT, so a stale config does not fail -- it resolves the new
    key to 0 and silently disables the mechanic. This check found a genuinely stale
    `regen.per_second` in this file's own DEFAULT_YAML the first time it ran."""
    cfg = load_config(config_dir / "default.yaml")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)

    for stale, expected in (
        ("cubes.hp_bonus_per_cube", "hp_per_cube"),
        ("zone.dps", "max_hp_fraction_per_second"),
        ("regen.per_second", "max_hp_fraction_per_second"),
    ):
        spec = _dset(merged_spec(), stale, 1.0)
        with pytest.raises(ValueError, match=stale.split(".")[-1]) as exc:
            build_params(cfg, n_envs=2, device="cpu", gen=gen, spec=spec)
        assert expected in str(exc.value), stale


def test_shipped_configs_use_no_renamed_keys():
    """The shipped files are the ones that would actually cost a training run."""
    spec = {
        **yaml.safe_load(Path("configs/default.yaml").read_text()),
        **yaml.safe_load(Path("configs/brawlers.yaml").read_text()),
    }
    check_removed_keys(spec)  # must not raise


def test_validate_zone_start_time_too_late():
    cfg = EnvConfig(max_episode_steps=100, dt=0.05)  # episode length 5.0s
    params = _dummy_params(cfg)
    params.zone_start_time = torch.full((4,), 10.0)  # >= 5.0s: invalid
    with pytest.raises(ValueError):
        validate(cfg, params)


def test_validate_rejects_a_zone_that_is_enabled_but_deals_no_damage():
    """The stale-config trap from Step B3's `zone.dps` -> `zone.max_hp_fraction_per_second`
    rename. The old key is simply not read any more, so a config still using it resolves the new
    one to `_dget`'s default of 0 and the zone silently becomes INERT -- no error, no damage.

    A range check cannot catch this on its own: the bad config produces 0, which is a perfectly
    ordinary-looking number. The symptom would surface much later as "episodes never resolve",
    which is exactly what the zone exists to prevent, so it must fail at construction instead.

    `config.check_removed_keys` now also rejects the old SPELLING outright at build_params time
    (Step E2), which catches the same mistake one step earlier and by name. Both layers are worth
    having: the spelling check only sees keys someone left in the file, while this one also
    catches a params object assembled by hand or mutated by a params_hook."""
    cfg = EnvConfig()
    params = _dummy_params(cfg)
    params.zone_hp_fraction = torch.zeros(4)
    with pytest.raises(ValueError, match="no damage"):
        validate(cfg, params)


def test_validate_rejects_a_zone_fraction_above_one():
    cfg = EnvConfig()
    params = _dummy_params(cfg)
    params.zone_hp_fraction = torch.full((4,), 1000.0)  # a flat HP/s left in the fraction field
    with pytest.raises(ValueError, match="FRACTION"):
        validate(cfg, params)


def test_validate_allows_an_inert_zone_when_the_zone_is_disabled():
    cfg = EnvConfig(zone_enabled=False)
    params = _dummy_params(cfg)
    params.zone_hp_fraction = torch.zeros(4)
    validate(cfg, params)  # must not raise -- nothing reads the rate with the zone off


def test_validate_dt_vs_zone_step_seconds():
    cfg = EnvConfig(dt=1.0)
    params = _dummy_params(cfg)
    params.zone_step_seconds = torch.full((4,), 0.5)  # dt >= zone_step_seconds: invalid
    with pytest.raises(ValueError):
        validate(cfg, params)


def test_validate_max_projectiles_too_small():
    cfg = EnvConfig(n_enemies=6, max_projectiles=1)
    params = _dummy_params(cfg)
    with pytest.raises(ValueError):
        validate(cfg, params)
