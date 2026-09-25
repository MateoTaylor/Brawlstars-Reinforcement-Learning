"""Exercises the actual files under configs/ against brawl_sim.config, per Step 4's acceptance
criteria. Step 3's tests/test_config.py already covers the loader machinery in isolation with
inline fixtures; this file is the end-to-end check that the real files parse correctly.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import (
    KIND_YAML_NAMES,
    build_params,
    load_config,
    load_randomization,
    apply_randomization,
    validate,
)
from brawl_sim.constants import Kind

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


def test_every_shipped_bot_kind_authors_both_difficulty_axes():
    """SIM_OVERHAUL_PLAN.md Phase B (Step B1). `validate()` reads 0 as neutral for `hero_focus`
    and `aggression` so that partial specs keep loading, which means a shipped bot block that
    FORGETS one loads without complaint and then never scales with the curriculum's tier table:
    a tier multiplier on a 0 base is 0. So the shipped file must author both on every bot, and
    this reads them off the BUILT tensors rather than the YAML dict so a misspelled key
    (`agression:`) fails here too. The hero block authors neither -- nothing reads them for it."""
    _, params = _build_and_validate()
    for k, name in enumerate(KIND_YAML_NAMES):
        if k == int(Kind.HERO_MORTIS):
            assert float(params.aggression[:, k].max()) == 0, "the hero block authors no aggression"
            assert float(params.hero_focus[:, k].max()) == 0, "the hero block authors no hero_focus"
            continue
        assert float(params.aggression[:, k].min()) > 0, f"{name}: aggression missing or 0"
        assert 0 < float(params.hero_focus[:, k].min()) <= float(params.hero_focus[:, k].max()) <= 1, (
            f"{name}: hero_focus missing, 0, or outside (0, 1]"
        )


def test_the_shipped_hero_authors_the_gadget_as_specified_and_no_bot_has_one():
    """SIM_OVERHAUL_PLAN.md Phase G (Step G1.3), the operator's numbers: an 18 s cooldown, a
    spinner that flies up to 2 tiles in 0.2 s and deals 2000 in a 1-tile radius. Pinned off the
    built tensors so a renamed or misspelled key fails here rather than as a gadget that never
    fires. `gadget_cooldown: 0` on every bot is what "no gadget" means."""
    _, params = _build_and_validate()
    h = int(Kind.HERO_MORTIS)
    assert float(params.gadget_cooldown[0, h]) == 18.0
    assert float(params.gadget_range[0, h]) == 2.0
    assert abs(float(params.gadget_flight_seconds[0, h]) - 0.2) < 1e-6
    assert float(params.gadget_damage[0, h]) == 2000.0
    assert float(params.gadget_radius[0, h]) == 1.0
    for k, name in enumerate(KIND_YAML_NAMES):
        if k != h:
            assert float(params.gadget_cooldown[:, k].max()) == 0, f"{name} must not have a gadget"


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
DEPLOY_SPECS = ("agent_obs_deploy.yaml", "agent_obs_deploy2.yaml", "agent_obs_deploy3.yaml",
                "agent_obs_deploy4.yaml", "agent_obs_deploy5.yaml")


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
# as a TEMPORARY operator decision; deploy3 is its reversal (2026-09-11), deploy4 keeps
# deploy3's grid planes (2026-09-21) and deploy5 keeps deploy4's (2026-09-24).
SEES_PICKUPS = {
    "agent_obs_deploy.yaml": False,
    "agent_obs_deploy2.yaml": False,
    "agent_obs_deploy3.yaml": True,
    "agent_obs_deploy4.yaml": True,
    "agent_obs_deploy5.yaml": True,
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


# ---- configs/agent_obs_deploy4.yaml: the gadget pair, the history group, three history planes --
#
# deploy4 is deploy3 plus what Phases G and H added to build_obs (2026-09-21): the gadget pair in
# `self`, the six `hist.*` fields as a new `history` group, and `enemy_hist1..3` after deploy3's
# grid planes. The tests below pin exactly that in both directions, and the widths the extractor
# is built on (Step H3 builds it). The parametrized deploy-lineage tests above cover it too.

GADGET_FIELDS = ("hero.gadget_ready", "hero.gadget_charge_frac")
HISTORY_FIELDS = ("hist.valid", "hist.move_onehot", "hist.attack_onehot", "hist.hp",
                  "hist.ammo_frac", "hist.displacement")
HISTORY_CHANNELS = ("enemy_hist1", "enemy_hist2", "enemy_hist3")


def test_deploy4_agent_obs_yaml_loads_as_a_real_agent_spec():
    """Six groups, the most `obs_select` loads (each is its own device-to-host copy), with
    `history` before `grid` so the grid stays last as in every other spec. The column maps are
    the ones the yaml header documents."""
    from brawl_sim.core.obs_select import _MAX_GROUPS, agent_obs_index_map, load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_deploy4.yaml", cfg)
    assert spec.fair is True and spec.normalize is True
    assert [g.name for g in spec.groups] == [
        "self", "enemies", "projectiles", "zone", "history", "grid"]
    assert len(spec.groups) == _MAX_GROUPS == 6, (
        "deploy4 is at the group cap. A seventh group does not load; the next field joins a group.")
    assert {g.name: g.shape for g in spec.groups} == {
        "self": (26,), "enemies": (9, 9), "projectiles": (12, 6), "zone": (5,),
        "history": (78,), "grid": (13, 13, 21),
    }

    index = agent_obs_index_map(spec, cfg)
    assert index["history"] == {
        "hist.valid": (0, 3), "hist.move_onehot": (3, 54), "hist.attack_onehot": (54, 66),
        "hist.hp": (66, 69), "hist.ammo_frac": (69, 72), "hist.displacement": (72, 78),
    }
    assert index["self"]["hero.gadget_ready"] == (22, 23)
    assert index["self"]["hero.gadget_charge_frac"] == (23, 24)

    grid = next(g for g in spec.groups if g.name == "grid")
    assert grid.channel_idx == (0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14), (
        "deploy3's ten sim planes, then the sim's enemy_hist1..3 at 12, 13 and 14")


def test_deploy4_differs_from_deploy3_by_exactly_the_history_and_gadget_additions():
    """Pinned in both directions like every other pair in this file. deploy4 adds what Phases G
    and H put in build_obs and nothing else: a change to `enemies`, `projectiles` or `zone`, or a
    deploy3 field dropped or moved, is a second decision riding along unannounced."""
    dep3 = yaml.safe_load((CONFIGS / "agent_obs_deploy3.yaml").read_text(encoding="utf-8"))
    dep4 = yaml.safe_load((CONFIGS / "agent_obs_deploy4.yaml").read_text(encoding="utf-8"))

    assert dep4["fair"] == dep3["fair"] and dep4["normalize"] == dep3["normalize"]

    by_name = lambda spec: {g["name"]: g for g in spec["groups"]}  # noqa: E731
    a, b = by_name(dep3), by_name(dep4)
    assert a.keys() <= b.keys() and b.keys() - a.keys() == {"history"}
    for name in sorted(a.keys() - {"self", "grid"}):
        assert a[name] == b[name], (
            f"configs/agent_obs_deploy4.yaml changed the {name!r} group. It is meant to differ "
            f"from its parent by the gadget pair, the history group and the history planes ONLY.")

    rest = lambda group, key: {k: v for k, v in group.items() if k != key}  # noqa: E731
    old, new = a["self"]["fields"], b["self"]["fields"]
    assert rest(a["self"], "fields") == rest(b["self"], "fields")
    assert len(new) == len(old) + len(GADGET_FIELDS)
    assert [f for f in new if f not in GADGET_FIELDS] == old, (
        "deploy4's self must be deploy3's fields, in deploy3's order, plus the gadget pair")
    after_super = new.index("hero.super_charge_frac") + 1
    assert tuple(new[after_super:after_super + 2]) == GADGET_FIELDS, (
        "the gadget pair sits right after the super pair: long dash, super, gadget")

    old, new = a["grid"]["view_channels"], b["grid"]["view_channels"]
    assert rest(a["grid"], "view_channels") == rest(b["grid"], "view_channels")
    assert new == old + list(HISTORY_CHANNELS), (
        "deploy4's grid must be deploy3's planes, in deploy3's order, then enemy_hist1..3, so a "
        "deploy3 channel index means the same plane in deploy4")

    assert b["history"] == {"name": "history", "per_entity": False, "dtype": "float32",
                            "fields": list(HISTORY_FIELDS)}


def test_the_shipped_run_trains_deploy4_at_its_pinned_input_widths():
    """SIM_OVERHAUL_STEPS.md Step I2: a bare `python scripts/train.py` trains the deployable spec,
    and the extractor it builds takes 13 grid channels and 262 floats. Built through the run's own
    env config and overrides, as `builder.build_env` does, so an override that moved a width shows
    here; tests/test_sb3_features.py pins the same numbers at the default config."""
    from brawl_sim.core import obs_select
    from brawl_sim.training.config import load_train_config
    from brawl_sim.wrappers.sb3_features import BrawlFeaturesExtractor

    tcfg = load_train_config(CONFIGS / "train.yaml")
    assert tcfg.run.agent_obs == "configs/agent_obs_deploy4.yaml"
    repo = CONFIGS.parent
    env_cfg = load_config(repo / tcfg.run.env_config, overrides=tcfg.run.env_overrides or None)
    spec = obs_select.load_agent_spec(repo / tcfg.run.agent_obs, env_cfg)
    fe = BrawlFeaturesExtractor(obs_select.agent_space(spec, env_cfg))
    assert fe.cnn[0].in_channels == 13
    assert fe.mlp[0].in_features == 262


# ---- configs/agent_obs_deploy5.yaml: hero.near_edge, and the enemies rows in tracked-slot order --
#
# deploy5 is deploy4 plus what OBS_PARITY_TASKS.md C1-C9 put in build_obs (2026-09-24): the
# `hero.near_edge` bit in `self`, right after `hero.in_zone`, and `slots: tracked` on `enemies`.
# The camera window and the zone.active latch change no column, so they leave no trace here; the
# tests below pin the two changes that do, in both directions, and the widths the extractor is
# built on. The parametrized deploy-lineage tests above cover it too.

NEAR_EDGE_FIELD = "hero.near_edge"


def test_deploy5_agent_obs_yaml_loads_as_a_real_agent_spec():
    """Still six groups at the cap, `self` one float wider, `enemies` in tracked-slot order at the
    same shape. The column numbers are the ones the yaml header documents."""
    from brawl_sim.core.obs_select import (_MAX_GROUPS, agent_obs_index_map, agent_space,
                                           load_agent_spec)
    from brawl_sim.wrappers.sb3_features import BrawlFeaturesExtractor

    cfg = load_config(CONFIGS / "default.yaml")
    spec = load_agent_spec(CONFIGS / "agent_obs_deploy5.yaml", cfg)
    assert spec.fair is True and spec.normalize is True
    assert [g.name for g in spec.groups] == [
        "self", "enemies", "projectiles", "zone", "history", "grid"]
    assert len(spec.groups) == _MAX_GROUPS == 6
    assert {g.name: g.shape for g in spec.groups} == {
        "self": (27,), "enemies": (9, 9), "projectiles": (12, 6), "zone": (5,),
        "history": (78,), "grid": (13, 13, 21),
    }

    index = agent_obs_index_map(spec, cfg)
    assert index["self"]["hero.in_zone"] == (17, 18)
    assert index["self"][NEAR_EDGE_FIELD] == (18, 19), (
        "right after in_zone, so the two where-am-I bits read together")
    assert index["self"]["hero.gadget_ready"] == (23, 24)
    assert index["self"]["hero.gadget_charge_frac"] == (24, 25)
    assert index["self"]["meta.n_enemies_alive"] == (26, 27)

    assert {g.name: g.slots for g in spec.groups} == {
        "self": None, "enemies": "tracked", "projectiles": None, "zone": None, "history": None,
        "grid": None}
    enemies = next(g for g in spec.groups if g.name == "enemies")
    assert enemies.max_slots is None, "tracked slots and max_slots are both row orders"

    grid = next(g for g in spec.groups if g.name == "grid")
    assert grid.channel_idx == (0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14), "deploy4's planes"

    fe = BrawlFeaturesExtractor(agent_space(spec, cfg))
    assert fe.cnn[0].in_channels == 13
    assert fe.mlp[0].in_features == 263, "deploy4's 262 plus the near_edge bit"


def test_deploy5_differs_from_deploy4_by_exactly_near_edge_and_tracked_slots():
    """Pinned in both directions like every other pair in this file: a change to any other group,
    or a deploy4 field dropped or moved, is a second decision riding along unannounced."""
    from brawl_sim.core.obs_select import load_agent_spec

    dep4 = yaml.safe_load((CONFIGS / "agent_obs_deploy4.yaml").read_text(encoding="utf-8"))
    dep5 = yaml.safe_load((CONFIGS / "agent_obs_deploy5.yaml").read_text(encoding="utf-8"))

    assert dep5["fair"] == dep4["fair"] and dep5["normalize"] == dep4["normalize"]

    by_name = lambda spec: {g["name"]: g for g in spec["groups"]}  # noqa: E731
    a, b = by_name(dep4), by_name(dep5)
    assert a.keys() == b.keys()
    for name in sorted(a.keys() - {"self", "enemies"}):
        assert a[name] == b[name], (
            f"configs/agent_obs_deploy5.yaml changed the {name!r} group. It is meant to differ "
            f"from its parent by hero.near_edge and the enemies group's slot order ONLY.")

    rest = lambda group, key: {k: v for k, v in group.items() if k != key}  # noqa: E731
    old, new = a["self"]["fields"], b["self"]["fields"]
    assert rest(a["self"], "fields") == rest(b["self"], "fields")
    assert [f for f in new if f != NEAR_EDGE_FIELD] == old, (
        "deploy5's self must be deploy4's fields, in deploy4's order, plus hero.near_edge")
    assert new.index(NEAR_EDGE_FIELD) == new.index("hero.in_zone") + 1

    assert "slots" not in a["enemies"] and b["enemies"]["slots"] == "tracked"
    assert rest(b["enemies"], "slots") == a["enemies"], (
        "deploy5's enemies group is deploy4's, field for field, plus `slots: tracked`")

    cfg = load_config(CONFIGS / "default.yaml")
    f4 = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_deploy4.yaml", cfg))
    f5 = _spec_fields(load_agent_spec(CONFIGS / "agent_obs_deploy5.yaml", cfg))
    assert f5 - f4 == {NEAR_EDGE_FIELD} and not (f4 - f5)


def test_near_edge_is_read_by_deploy5_and_by_nothing_older():
    """OBS_PARITY_TASKS.md C6 and C10: the bit went into deploy5 only. The older deploy specs are
    named by path in shipped runs and keep their widths; the full-information and low-info views
    are not deploy targets and stay as they were."""
    from brawl_sim.core.obs_select import load_agent_spec

    cfg = load_config(CONFIGS / "default.yaml")
    for name in ("agent_obs.yaml", "agent_obs_lowinfo.yaml") + DEPLOY_SPECS[:-1]:
        assert NEAR_EDGE_FIELD not in _spec_fields(load_agent_spec(CONFIGS / name, cfg)), name
    assert DEPLOY_SPECS[-1] == "agent_obs_deploy5.yaml"
