"""Exercises the actual files under configs/ against brawl_sim.config, per Step 4's acceptance
criteria. Step 3's tests/test_config.py already covers the loader machinery in isolation with
inline fixtures; this file is the end-to-end check that the real files parse correctly.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import (
    build_params,
    load_config,
    load_randomization,
    apply_randomization,
    validate,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
PRESETS = CONFIGS / "presets"


def _merged_spec():
    default = yaml.safe_load((CONFIGS / "default.yaml").read_text())
    brawlers = yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())
    return {**default, **brawlers}


def _build_and_validate(overrides=None, n_envs=8):
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=_merged_spec())
    validate(cfg, params)
    return cfg, params


def test_default_config_loads_and_validates():
    _build_and_validate()


@pytest.mark.parametrize("preset_name", ["debug_tiny", "no_zone", "single_archetype"])
def test_preset_loads_and_validates(preset_name):
    overrides = yaml.safe_load((PRESETS / f"{preset_name}.yaml").read_text())
    _build_and_validate(overrides=overrides)


def test_debug_tiny_yields_three_entities():
    overrides = yaml.safe_load((PRESETS / "debug_tiny.yaml").read_text())
    cfg, _ = _build_and_validate(overrides=overrides)
    assert cfg.n_entities == 3
    assert cfg.map_h == 20 and cfg.map_w == 20
    assert cfg.view_h == 10 and cfg.view_w == 14
    assert cfg.max_episode_steps == 300
    assert cfg.zone_enabled is False
    assert cfg.obs_include_world_grid is False


def test_single_archetype_preset_pins_sniper():
    overrides = yaml.safe_load((PRESETS / "single_archetype.yaml").read_text())
    cfg, _ = _build_and_validate(overrides=overrides)
    assert cfg.randomize_enemy_types is False
    assert cfg.fixed_enemy_types == ("sniper",) * cfg.n_enemies


def test_randomization_file_ships_fully_commented():
    spec = load_randomization(CONFIGS / "randomization.yaml")
    assert spec == {}


def test_uncommenting_one_randomization_line_only_adds_variation():
    cfg = load_config(CONFIGS / "default.yaml")
    base_spec = _merged_spec()

    gen_a = torch.Generator(device="cpu")
    gen_a.manual_seed(42)
    baseline = build_params(cfg, n_envs=64, device="cpu", gen=gen_a, spec=base_spec)
    assert torch.all(baseline.enemy_hp_mult == baseline.enemy_hp_mult[0])

    # Simulate uncommenting exactly the entities.enemy_hp_mult line from the real file.
    raw_text = (CONFIGS / "randomization.yaml").read_text()
    target = "# entities.enemy_hp_mult:       {low: 0.8, high: 1.25}"
    assert target in raw_text, "randomization.yaml's worked example changed out from under this test"
    uncommented_text = raw_text.replace(target, target.lstrip("# "), 1)

    # Parse the mutated text the same way load_randomization does, without needing a temp file.
    from brawl_sim.config import _parse_range_entry  # test-only introspection

    raw = yaml.safe_load(uncommented_text)
    randomization = {k: _parse_range_entry(v) for k, v in raw.items()}
    assert randomization == {"entities.enemy_hp_mult": {"low": 0.8, "high": 1.25, "mode": "additive"}}

    merged = apply_randomization(dict(base_spec), randomization)
    gen_b = torch.Generator(device="cpu")
    gen_b.manual_seed(42)
    randomized = build_params(cfg, n_envs=64, device="cpu", gen=gen_b, spec=merged)

    # The one uncommented field now varies per env...
    assert not torch.all(randomized.enemy_hp_mult == randomized.enemy_hp_mult[0])
    assert randomized.enemy_hp_mult.min() >= 0.8 and randomized.enemy_hp_mult.max() <= 1.25
    # ...and nothing else changed: every other field matches the baseline build exactly
    # (same seed, same spec apart from the one overridden leaf).
    assert torch.equal(randomized.base_hp, baseline.base_hp)
    assert torch.equal(randomized.move_speed, baseline.move_speed)
    assert torch.equal(randomized.enemy_damage_mult, baseline.enemy_damage_mult)
    assert torch.equal(randomized.zone_start_time, baseline.zone_start_time)


def test_agent_obs_yaml_loads_as_a_real_agent_spec():
    # Step 4 only needed the file to exist and parse as a placeholder; Step 32 replaced it with
    # the real schema (brawl_sim.core.obs_select) -- this is the end-to-end check for that file,
    # the same role every other test in this module plays for configs/default.yaml et al.
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs.yaml", cfg)
    assert spec.fair is True
    assert len(spec.groups) >= 1
    assert {g.name for g in spec.groups} >= {"self", "enemies", "projectiles", "zone", "grid"}


# ---- configs/agent_obs_lowinfo.yaml: the masked-information training view --------------------
#
# The low-information regime is a CONFIG, not a code path: it hides enemy archetype and
# projectile type purely by leaving those fields out of the spec. That makes it cheap and fully
# reversible, and it also makes it fragile in one specific way -- a field re-added to the yaml by
# reflex silently restores the signal with nothing to catch it. These three tests are that catch.

# Every field that identifies WHICH brawler an enemy is, or WHICH weapon fired a projectile.
# `damage`/`aoe` are here because they are per-weapon constants, so a policy recovers the type by
# regressing on them (see the yaml header and brawl_sim/constants.py's Proj.BULL_SLUG note);
# `threatens_hero` because core/observation.py derives it from state.prj_radius, a per-weapon stat.
TYPE_IDENTIFYING_FIELDS = frozenset({
    "entities.kind", "entities.kind_onehot", "entities.team",
    "projectiles.kind", "projectiles.kind_onehot",
    "projectiles.class", "projectiles.class_onehot", "projectiles.lobbed",
    "projectiles.owner", "projectiles.owner_kind",
    "projectiles.damage", "projectiles.aoe", "projectiles.radius",
    "projectiles.threatens_hero",
})


def _spec_fields(spec) -> set:
    return {f for g in spec.groups for f in g.fields}


def test_lowinfo_agent_obs_yaml_loads_as_a_real_agent_spec():
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_lowinfo.yaml", cfg)
    assert spec.fair is True
    assert {g.name for g in spec.groups} == {"self", "enemies", "projectiles", "zone", "grid"}


def test_lowinfo_agent_obs_exposes_no_type_identifying_field():
    """The whole point of the file. Fails loudly if any of them is ever put back."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_lowinfo.yaml", cfg)

    leaked = _spec_fields(spec) & TYPE_IDENTIFYING_FIELDS
    assert not leaked, (
        f"configs/agent_obs_lowinfo.yaml exposes {sorted(leaked)}, which identifies an enemy's "
        "archetype or a projectile's weapon -- the one thing that file exists to hide"
    )

    # The grid is shared verbatim with the full-info spec and must stay type-free too: only
    # `enemy_revealed` is a legitimate enemy channel, and it is a bare occupancy count.
    grid = next(g for g in spec.groups if g.name == "grid")
    assert "enemy_revealed" in grid.view_channels
    assert set(grid.view_channels) == set(
        next(g for g in load_agent_spec(CONFIGS / "agent_obs.yaml", cfg).groups if g.name == "grid").view_channels
    )


def test_lowinfo_differs_from_full_info_by_exactly_the_masked_fields():
    """Pins the RELATIONSHIP between the two files, not just each one's contents. Low-info must
    be the full-info view minus a known list -- never the full view plus something, and never
    accidentally missing a field that has nothing to do with masking (dropping
    `entities.hp_frac`, say, would be a different experiment wearing this file's name)."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    full = _spec_fields(load_agent_spec(CONFIGS / "agent_obs.yaml", cfg))
    low = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_lowinfo.yaml", cfg))

    assert not (low - full), f"low-info adds fields the full-info view doesn't have: {sorted(low - full)}"
    assert full - low == {
        "entities.kind_onehot",
        "projectiles.kind_onehot", "projectiles.class_onehot",
        "projectiles.damage", "projectiles.aoe", "projectiles.threatens_hero",
    }


# ---- configs/agent_obs_deploy.yaml: the view the live loop can actually fill ------------------
#
# One more narrowing on top of low-info, for a different reason. Low-info hides what would be a
# TYPE LEAK; this file drops what nothing will SUPPLY. Both failures are silent -- a policy
# trained on a column it is later fed a constant for is being lied to in a place it learned to
# trust -- so both get a test that fails loudly when a field comes back by reflex.

# Dropped because no reader exists or ever will. `hero.cubes` is the odd one: it IS recoverable
# (the pip row sits in the hero's own box stack, next to the HP bar), and it is here by operator
# decision that no reader will be built. Recoverability is not the criterion -- intent to supply
# is. See BRAWL_DEPLOYMENT_DESIGN.md 9.8.
UNSUPPLIABLE_FIELDS = frozenset({
    "entities.cubes", "entities.can_attack", "entities.dashing", "hero.cubes",
})

# Fields the deploy spec REPLACES rather than drops, as `lowinfo name -> deploy name`. The pair
# carries the same information from the same pixels; only the denominator changes. `hp_frac` is
# `hp / max_hp`, and cubes make max_hp unobservable for the hero and for every enemy alike, so
# the deployed agent reads the HP numeral -- which is what brawl_vision's reader has always
# produced. See the yaml header, and BRAWL_DEPLOYMENT_DESIGN.md 6.3.
SUPPLIABLE_SUBSTITUTIONS = {
    "hero.hp_frac": "hero.hp",
    "entities.hp_frac": "entities.hp",
}

# Every spec in the deploy lineage. The properties below hold for all of them -- each new file is
# a change to the last for a stated reason, so a test that only ever checked the first would stop
# guarding the file actually being trained on.
DEPLOY_SPECS = ("agent_obs_deploy.yaml", "agent_obs_deploy2.yaml", "agent_obs_deploy3.yaml")


def test_deploy_agent_obs_yaml_loads_as_a_real_agent_spec():
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_deploy.yaml", cfg)
    assert spec.fair is True
    assert {g.name for g in spec.groups} == {"self", "enemies", "projectiles", "zone", "grid"}


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_deploy_agent_obs_inherits_the_lowinfo_type_masking(spec_name):
    """Narrowing for supply must not quietly UNDO the narrowing for type leakage. This file is a
    child of the low-info view, not an independent edit of the full one."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / spec_name, cfg)

    leaked = _spec_fields(spec) & TYPE_IDENTIFYING_FIELDS
    assert not leaked, f"configs/{spec_name} re-exposes {sorted(leaked)}"


def test_deploy_differs_from_lowinfo_by_exactly_the_unsuppliable_fields():
    """The relationship, pinned the same way low-info's is against full-info. Deploy must be
    low-info minus a known list -- never plus anything, and never missing a field for a reason
    other than "the live loop cannot fill it"."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    low = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_lowinfo.yaml", cfg))
    dep = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_deploy.yaml", cfg))

    added, removed = dep - low, low - dep
    assert added == set(SUPPLIABLE_SUBSTITUTIONS.values()), (
        f"deploy adds fields low-info doesn't have: "
        f"{sorted(added - set(SUPPLIABLE_SUBSTITUTIONS.values()))}"
    )
    assert removed == set(UNSUPPLIABLE_FIELDS) | set(SUPPLIABLE_SUBSTITUTIONS)


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_deploy_keeps_the_fields_the_live_loop_does_supply(spec_name):
    """The other half of the decision, and the easier one to erode. Each of these looks
    unrecoverable at a glance and is not:

    - `entities.rel_vel` -- the tracker produces it, and the odometry error that inflates
      absolute speeds cancels in `ent_vel - hero_vel` (BRAWL_DEPLOYMENT_DESIGN.md 6.1).
    - `entities.revealed_to_hero` -- `Track.seen_now`: detected this tick, not coasting.
    - `entities.in_bush` -- an occupancy lookup at the enemy's tile, not a property of the enemy.
    - `hero.can_attack` / `hero.dashing` -- the same quantities `entities.*` lost, but from the
      side of the camera that ISSUES the actions, so they are dead-reckoned exactly (6.3).
    """
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    fields = _spec_fields(load_agent_spec(CONFIGS / spec_name, cfg))

    for name in ("entities.rel_vel", "entities.revealed_to_hero", "entities.in_bush",
                 "hero.can_attack", "hero.dashing"):
        assert name in fields, (
            f"{name} was dropped from configs/{spec_name}. That file drops fields the "
            "live loop cannot supply, and this one it can -- see the yaml header for how."
        )


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_deploy_reads_absolute_hp_because_cubes_make_max_hp_unobservable(spec_name):
    """`hp_frac` needs max_hp, and a power cube adds a flat +400 to it for the hero and for every
    enemy (configs/default.yaml `cubes.hp_per_cube`) with nothing on screen saying who holds how
    many -- the same fact that removed `entities.cubes`. Dividing by the kit constant in
    configs/brawlers.yaml is right only until the first cube is collected.

    The regression this guards is a quiet revert to `hp_frac` on either side, which would look
    tidier and read as a constant-denominator lie in a column the policy was trained to trust."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    fields = _spec_fields(load_agent_spec(CONFIGS / spec_name, cfg))

    for gone, present in SUPPLIABLE_SUBSTITUTIONS.items():
        assert gone not in fields, (
            f"{gone} is back in configs/{spec_name}. Its denominator is max_hp, which "
            f"cubes change and no screenshot shows -- use {present}."
        )
        assert present in fields


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_the_deploy_spec_has_no_unnormalized_field_of_large_magnitude(spec_name):
    """The property `normalize: true` is supposed to deliver, asserted against the schema rather
    than against a list of field names -- so a spec author adding a raw `hp` or `count` field
    later cannot reintroduce the problem silently.

    This is what caught `meta.n_enemies_alive`: swapping hp_frac for hp drew attention to the
    divisor table, and n_enemies_alive turned out to be sitting in the same vector at up to 9,
    already the largest magnitude in it. Both units have a divisor now."""
    from brawl_sim.core import obs_schema
    from brawl_sim.core.obs_select import _norm_divisor, load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / spec_name, cfg)
    assert spec.normalize is True

    # Units whose values are bounded by construction, or bounded small by the sim's own timings.
    naturally_small = {"fraction", "bool", "unitless", "onehot", "radians", "seconds"}
    for name in sorted(_spec_fields(spec)):
        units = obs_schema.OBS_SCHEMA[name].units
        if units in naturally_small:
            continue
        assert _norm_divisor(units, cfg) > 1.0, (
            f"{name} has units {units!r} and passes through `normalize: true` undivided. Give "
            f"the unit a divisor in obs_select._norm_divisor, sized so the sim's largest "
            f"possible value lands just inside 1.0 -- the rule _NORM_SPEED_SCALE follows."
        )


# Which deploy specs can see a cube lying on the ground. deploy and deploy2 dropped `box`/`pickup`
# as a TEMPORARY operator decision; deploy3 is its reversal (2026-09-11).
SEES_PICKUPS = {
    "agent_obs_deploy.yaml": False,
    "agent_obs_deploy2.yaml": False,
    "agent_obs_deploy3.yaml": True,
}


def test_every_deploy_spec_says_whether_it_sees_pickups():
    assert set(SEES_PICKUPS) == set(DEPLOY_SPECS)


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_the_shipped_cube_pickup_reward_builds_only_with_a_spec_that_sees_cubes(spec_name):
    """One weight in a SHARED file, several specs chosen per run, and nothing structural connecting
    them -- so the pairing is enforced where the two meet, `builder.check_reward_is_observable`,
    and this test runs the REAL configs/train.yaml through it against every deploy spec.

    Paying `reward.cube_pickup` to an agent that cannot see pickups trains an approach behaviour
    that cannot transfer, and it is silent -- the run looks fine. This used to be a static
    comparison of train.yaml against each spec's channels, which could only say "all of them see
    cubes or none do"; deploy3 is the first spec where the answer differs from its siblings.
    """
    from brawl_sim.core.obs_select import load_agent_spec
    from brawl_sim.training.builder import check_reward_is_observable
    from brawl_sim.training.config import load_train_config

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / spec_name, cfg)
    reward = load_train_config(CONFIGS / "train.yaml").reward
    assert reward.cube_pickup != 0.0, (
        "configs/train.yaml no longer pays for cubes. deploy3 exists so that it can -- if that "
        "decision was reversed, say so in agent_obs_deploy3.yaml's header too."
    )

    if SEES_PICKUPS[spec_name]:
        check_reward_is_observable(reward, spec, spec_name)
    else:
        with pytest.raises(ValueError, match="cube_pickup=0"):
            check_reward_is_observable(reward, spec, spec_name)
        # ...and the escape the error names actually works.
        unpaid = load_train_config(CONFIGS / "train.yaml",
                                   overrides={"reward": {"cube_pickup": 0.0}}).reward
        check_reward_is_observable(unpaid, spec, spec_name)


def test_seeing_pickups_means_seeing_where_they_are_not_how_many_the_hero_holds():
    """`hero.cubes` is a count of what the hero already HOLDS -- no help finding the next one --
    so it must not satisfy the guard. A grid `pickup` channel or any `pickups.*` field does."""
    from brawl_sim.core.obs_select import AgentObsSpec, GroupSpec
    from brawl_sim.training.builder import sees_pickups

    def spec(*groups):
        return AgentObsSpec(fair=True, groups=groups, normalize=True)

    holds = GroupSpec(name="self", dtype="float32", shape=(1,), fields=("hero.cubes",))
    channel = GroupSpec(name="grid", dtype="uint8", shape=(1, 13, 21), view_channels=("pickup",))
    crates = GroupSpec(name="grid", dtype="uint8", shape=(1, 13, 21), view_channels=("box",))
    field = GroupSpec(name="pk", dtype="float32", shape=(4,), fields=("pickups.rel_pos",))

    assert not sees_pickups(spec(holds))
    assert not sees_pickups(spec(holds, crates)), "a crate is not a cube until it breaks"
    assert sees_pickups(spec(holds, channel))
    assert sees_pickups(spec(field))


# ---- configs/agent_obs_deploy2.yaml: the same rule, applied to the zone group -----------------
#
# agent_obs_deploy.yaml asked "is it on screen?" and kept the whole zone group because the gas is
# drawn. Measuring the reconstruction (BRAWL_DEPLOYMENT_DESIGN.md 9.14) showed three of the four
# fields are functions of the safe RECT and its SCHEDULE, which an egocentric camera does not
# recover. deploy2 is that correction and nothing else -- so the tests below are mostly about what
# did NOT change.

# lowinfo name -> deploy2 name, same as SUPPLIABLE_SUBSTITUTIONS one level up: the pair carries the
# same four distances, and only the sensing horizon differs.
ZONE_SUBSTITUTION = {"zone.hero_margin": "zone.hero_margin_local"}

# No supplier at any price, for two different reasons -- see the yaml header.
UNSUPPLIABLE_ZONE_FIELDS = frozenset({"zone.next_shrink_in", "zone.safe_area_frac"})


def test_deploy2_agent_obs_yaml_loads_as_a_real_agent_spec():
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_deploy2.yaml", cfg)
    assert spec.fair is True
    assert {g.name for g in spec.groups} == {"self", "enemies", "projectiles", "zone", "grid"}

    zone = next(g for g in spec.groups if g.name == "zone")
    assert zone.shape == (5,), (
        "the zone group is 4 margin columns plus zone.active. A different width means a field was "
        "added or dropped without updating the header that justifies each one."
    )


def test_deploy2_differs_from_deploy_by_exactly_the_zone_group():
    """The relationship, pinned in both directions like every other pair in this file. deploy2
    exists to fix ONE group; a change that leaks into `self`, `enemies`, `projectiles` or `grid`
    is a second decision riding along unannounced, and those groups' justifications live in the
    parent file's header where nobody would think to re-read them."""
    dep = yaml.safe_load((CONFIGS / "agent_obs_deploy.yaml").read_text(encoding="utf-8"))
    dep2 = yaml.safe_load((CONFIGS / "agent_obs_deploy2.yaml").read_text(encoding="utf-8"))

    assert dep2["fair"] == dep["fair"] and dep2["normalize"] == dep["normalize"]

    by_name = lambda spec: {g["name"]: g for g in spec["groups"]}  # noqa: E731
    a, b = by_name(dep), by_name(dep2)
    assert a.keys() == b.keys()
    for name in a:
        if name == "zone":
            continue
        assert a[name] == b[name], (
            f"configs/agent_obs_deploy2.yaml changed the {name!r} group. It is meant to differ "
            f"from its parent in the zone group ONLY -- put an unrelated change in its own file."
        )


def test_deploy2_swaps_the_raw_margin_for_the_horizon_limited_one():
    """`zone.hero_margin` is `hero - rect edge`, which needs the rect. The deployed loop has no
    rect: brawl_deployment/perception/grid.py's GasMap remembers cells it has SEEN gassed and
    answers "how far to gas along -x/+x/-y/+y", saturating past a horizon. `hero_margin_local` is
    that same quantity with the same saturation, so the column means the same thing on both sides
    of the deployment boundary.

    Reverting to the raw field would look like restoring information and would in fact restore a
    range the live estimator can never produce -- silent, and indistinguishable from "the policy
    is bad at avoiding gas"."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    fields = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_deploy2.yaml", cfg))

    for gone, present in ZONE_SUBSTITUTION.items():
        assert gone not in fields, (
            f"{gone} is back in configs/agent_obs_deploy2.yaml. It is unbounded and the deployed "
            f"estimator saturates at cfg.zone_margin_horizon_tiles -- use {present}."
        )
        assert present in fields


def test_deploy2_drops_the_zone_fields_nothing_will_supply():
    """`next_shrink_in` is structurally unobservable -- gas advancing and gas being REVEALED as
    the camera pans are the same pixels, measured at 7 tiles of apparent motion in 3 s from reveal
    alone. `safe_area_frac` needs the map's total extent, which an odometry-anchored canvas never
    learns. Neither is rescued by more footage or a better detector, and the ablation priced a
    plausible-but-wrong `next_shrink_in` at 4.4 pp -- WORSE than not having it (9.15)."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    fields = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_deploy2.yaml", cfg))

    back = fields & UNSUPPLIABLE_ZONE_FIELDS
    assert not back, (
        f"configs/agent_obs_deploy2.yaml restored {sorted(back)}. Nothing supplies them at "
        f"runtime, so they would train as real values and deploy as constants -- read the yaml "
        f"header before deciding this test is wrong."
    )


def test_the_margin_horizon_actually_clamps_something_on_the_real_map():
    """The knob that ties the two halves together, guarded against being turned into a no-op.

    `zone.hero_margin_local` is only a different field from `zone.hero_margin` while the horizon
    is smaller than the distances the map can produce. Set it to 60 on a 60x60 map and every
    margin passes through unclamped: training silently goes back to the unbounded field while the
    deployed estimator keeps saturating, which is exactly the mismatch deploy2 exists to close.
    The failure is invisible -- the spec still names the right field."""
    cfg, _ = _build_and_validate()
    assert cfg.zone_margin_horizon_tiles > 0
    assert cfg.zone_margin_horizon_tiles < max(cfg.map_w, cfg.map_h) / 2, (
        f"zone.margin_horizon_tiles is {cfg.zone_margin_horizon_tiles} on a "
        f"{cfg.map_w}x{cfg.map_h} map, which barely clamps anything. If the deployed gas "
        f"estimator really did sense that far, say so here AND in brawl_deployment -- the two "
        f"numbers are one decision."
    )


def test_a_nonpositive_margin_horizon_is_rejected():
    """A horizon of 0 collapses hero_margin_local to four zeros -- a constant column the policy
    would learn to trust, which is the one failure this whole family of configs exists to prevent.
    It has to fail at construction, not silently train."""
    with pytest.raises(ValueError, match="margin_horizon_tiles"):
        _build_and_validate(overrides={"zone": {"margin_horizon_tiles": 0.0}})



# ---- configs/agent_obs_deploy3.yaml: crates and cubes back on the grid ------------------------
#
# deploy2 dropped `box`/`pickup` as an explicitly TEMPORARY operator decision. deploy3 reverses it
# (2026-09-11) under a rule with two halves -- the agent can SEE crates and cubes on the map before
# they are picked up, and still cannot see how many cubes anyone holds. The tests below pin both
# halves, and that nothing else rode along.

# Where the two channels sit in every non-deploy spec, which deploy3 matches.
CUBE_CHANNELS = ("box", "pickup")


def test_deploy3_agent_obs_yaml_loads_as_a_real_agent_spec():
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_deploy3.yaml", cfg)
    assert spec.fair is True
    assert {g.name for g in spec.groups} == {"self", "enemies", "projectiles", "zone", "grid"}

    grid = next(g for g in spec.groups if g.name == "grid")
    assert grid.shape == (10, cfg.view_h, cfg.view_w), (
        "deploy2's eight channels plus box and pickup. A different count means a channel moved "
        "without the header that justifies it."
    )


def test_deploy3_differs_from_deploy2_by_exactly_the_two_cube_channels():
    """Pinned in both directions like every other pair in this file. deploy3 exists to restore two
    grid channels; a change anywhere else is a second decision riding along unannounced."""
    dep2 = yaml.safe_load((CONFIGS / "agent_obs_deploy2.yaml").read_text(encoding="utf-8"))
    dep3 = yaml.safe_load((CONFIGS / "agent_obs_deploy3.yaml").read_text(encoding="utf-8"))

    assert dep3["fair"] == dep2["fair"] and dep3["normalize"] == dep2["normalize"]

    by_name = lambda spec: {g["name"]: g for g in spec["groups"]}  # noqa: E731
    a, b = by_name(dep2), by_name(dep3)
    assert a.keys() == b.keys()
    for name in a:
        if name == "grid":
            continue
        assert a[name] == b[name], (
            f"configs/agent_obs_deploy3.yaml changed the {name!r} group. It is meant to differ "
            f"from its parent in the grid's box/pickup channels ONLY."
        )

    old, new = a["grid"]["view_channels"], b["grid"]["view_channels"]
    assert [ch for ch in new if ch not in CUBE_CHANNELS] == old, (
        "deploy3's grid must be deploy2's channels, in deploy2's order, plus box and pickup"
    )
    assert set(new) - set(old) == set(CUBE_CHANNELS)


def test_deploy3_puts_the_cube_channels_where_every_other_spec_has_them():
    """Channel ORDER is the grid group's layout -- the CNN's first layer is indexed by it. Matching
    lowinfo's order costs nothing and means a channel index means the same plane in every spec
    that has it."""
    low = yaml.safe_load((CONFIGS / "agent_obs_lowinfo.yaml").read_text(encoding="utf-8"))
    dep3 = yaml.safe_load((CONFIGS / "agent_obs_deploy3.yaml").read_text(encoding="utf-8"))
    grid = lambda spec: next(g for g in spec["groups"] if g["name"] == "grid")  # noqa: E731
    low_ch, dep3_ch = grid(low)["view_channels"], grid(dep3)["view_channels"]

    assert [ch for ch in low_ch if ch in dep3_ch] == dep3_ch


@pytest.mark.parametrize("spec_name", DEPLOY_SPECS)
def test_no_deploy_spec_shows_how_many_cubes_anyone_holds(spec_name):
    """The operator's rule for deploy3, which every deploy spec must satisfy: crates and cubes on the
    ground may be visible, cube COUNTS may not -- not the hero's, not an enemy's, and not what a
    pile on the ground is worth (which would leak what its dead owner was carrying).

    The grid's `pickup` channel is a count of pickup OBJECTS per cell, and a corpse drops exactly
    one however many cubes it held (`combat.drop_cubes_on_death`), so the channel is allowed and
    `pickups.cubes` is not."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    fields = _spec_fields(load_agent_spec(CONFIGS / spec_name, cfg))

    counts = fields & {"hero.cubes", "entities.cubes", "pickups.cubes"}
    assert not counts, f"configs/{spec_name} shows cube counts: {sorted(counts)}"
