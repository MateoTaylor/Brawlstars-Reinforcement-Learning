import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import (
    AGGRESSIVE_PERSONS,
    N_PERSONS,
    TILE_BLOCKS_UNIT,
    Kind,
    Person,
)
from brawl_sim.core import spawn, stats
from brawl_sim.core.state import allocate
from brawl_sim.maps.loader import build_map_bank

CONFIGS_DEFAULT = "configs/default.yaml"


def _cfg_params_bank(n_envs=1, seed=0, **overrides):
    cfg = load_config(CONFIGS_DEFAULT, overrides=overrides or None)
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    bank = build_map_bank(cfg, device="cpu")
    return cfg, params, bank, gen, spec


def _fresh_state(cfg, n_envs=1):
    return allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)


# ---- sample_map_ids -----------------------------------------------------------------

def test_sample_map_ids_uniform_covers_all_maps_and_in_range():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=2000)
    mask = torch.ones(2000, dtype=torch.bool)
    ids = spawn.sample_map_ids(mask, bank, cfg, gen)
    assert ids.shape == (2000,)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < len(cfg.map_names)
    assert torch.unique(ids).numel() == len(cfg.map_names)  # all 3 maps show up given 2000 draws


def test_sample_map_ids_fixed_is_constant():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=50, world={"map_selection": "fixed", "fixed_map": "bushy"},
    )
    mask = torch.ones(50, dtype=torch.bool)
    ids = spawn.sample_map_ids(mask, bank, cfg, gen)
    expected = cfg.map_names.index("bushy")
    assert torch.all(ids == expected)


# ---- sample_enemy_kinds --------------------------------------------------------------

def test_sample_enemy_kinds_slot_zero_always_hero():
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=200)
    mask = torch.ones(200, dtype=torch.bool)
    kinds = spawn.sample_enemy_kinds(mask, cfg, gen)
    assert kinds.shape == (200, cfg.n_entities)
    assert torch.all(kinds[:, 0] == int(Kind.HERO_MORTIS))


def test_sample_enemy_kinds_degenerate_weights_are_deterministic():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=200,
        entities={"enemy_type_weights": {"sniper": 1.0, "artillery": 0.0, "melee": 0.0,
                                             "rifle": 0.0, "edgar": 0.0, "spike": 0.0,
                                             "bull": 0.0}},
    )
    mask = torch.ones(200, dtype=torch.bool)
    kinds = spawn.sample_enemy_kinds(mask, cfg, gen)
    assert torch.all(kinds[:, 1:] == int(Kind.BOT_SNIPER))


def test_sample_enemy_kinds_fixed_types_matches_config():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=5,
        entities={
            # n_enemies pinned explicitly (not just inherited from default.yaml) so this test's
            # own fixed_enemy_types length stays self-consistent regardless of what the shipped
            # default happens to be set to.
            "n_enemies": 6,
            "randomize_enemy_types": False,
            "fixed_enemy_types": ["sniper", "artillery", "melee", "rifle", "sniper", "rifle"],
        },
    )
    mask = torch.ones(5, dtype=torch.bool)
    kinds = spawn.sample_enemy_kinds(mask, cfg, gen)
    expected = torch.tensor([
        int(Kind.HERO_MORTIS), int(Kind.BOT_SNIPER), int(Kind.BOT_ARTILLERY),
        int(Kind.BOT_MELEE), int(Kind.BOT_RIFLE), int(Kind.BOT_SNIPER), int(Kind.BOT_RIFLE),
    ])
    for row in range(5):
        assert torch.equal(kinds[row], expected)


def test_sample_enemy_kinds_fixed_types_length_mismatch_raises():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=1,
        entities={"randomize_enemy_types": False, "fixed_enemy_types": ["sniper", "rifle"]},
    )
    mask = torch.ones(1, dtype=torch.bool)
    try:
        spawn.sample_enemy_kinds(mask, cfg, gen)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---- sample_spawn_positions (acceptance) ----------------------------------------------

def test_spawn_positions_passable_and_collision_free_and_distinct_across_many_seeds():
    n_envs = 1000
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=n_envs, world={"map_selection": "fixed", "fixed_map": "open"},
    )
    mask = torch.ones(n_envs, dtype=torch.bool)
    map_ids = spawn.sample_map_ids(mask, bank, cfg, gen)
    pos = spawn.sample_spawn_positions(mask, map_ids, bank, cfg, gen)
    E = cfg.n_entities
    assert pos.shape == (n_envs, E, 2)

    ix = pos[..., 0].floor().to(torch.int64)
    iy = pos[..., 1].floor().to(torch.int64)
    tiles = bank.tiles[map_ids.view(n_envs, 1).expand(n_envs, E), iy, ix]
    blocked = TILE_BLOCKS_UNIT[tiles]
    assert not torch.any(blocked)

    unit_radius = 0.4
    for a in range(E):
        for b in range(a + 1, E):
            d = (pos[:, a] - pos[:, b]).norm(dim=-1)
            assert torch.all(d >= 2 * unit_radius)

    diffs = pos.unsqueeze(2) - pos.unsqueeze(1)  # (n_envs,E,E,2)
    dist = diffs.norm(dim=-1)
    eye = torch.eye(E, dtype=torch.bool).unsqueeze(0)
    assert torch.all((dist > 1e-4) | eye)  # no two distinct slots share a position


def test_spawn_positions_only_needs_reset_mask_for_shape():
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=4, world={"map_selection": "fixed", "fixed_map": "open"},
    )
    mask = torch.ones(4, dtype=torch.bool)
    map_ids = spawn.sample_map_ids(mask, bank, cfg, gen)
    pos = spawn.sample_spawn_positions(mask, map_ids, bank, cfg, gen)
    assert pos.shape == (4, cfg.n_entities, 2)
    assert not torch.any(torch.isnan(pos))


# ---- sample_personalities -------------------------------------------------------------

def test_sample_personalities_slot_zero_is_never_assigned_a_bot_personality():
    cfg, _params, _bank, gen, _spec = _cfg_params_bank(n_envs=64)
    mask = torch.ones(64, dtype=torch.bool)
    persons = spawn.sample_personalities(mask, cfg, gen)
    assert persons.shape == (64, cfg.n_entities)
    assert torch.all(persons[:, 0] == int(Person.RUSH))  # the hero's slot, never read
    assert torch.all((persons >= 0) & (persons < N_PERSONS))


def test_sample_personalities_covers_every_personality_over_many_draws():
    cfg, _params, _bank, gen, _spec = _cfg_params_bank(n_envs=512)
    mask = torch.ones(512, dtype=torch.bool)
    persons = spawn.sample_personalities(mask, cfg, gen)[:, 1:]
    seen = set(persons.reshape(-1).tolist())
    assert seen == {int(p) for p in Person}


def test_sample_personalities_honors_the_aggression_floor_in_every_env():
    """A lobby with no rush and no hunter is one the agent beats by standing still, so the floor
    is a hard guarantee, not a statistical tendency. Stacked against it here: weights that would
    otherwise draw NOTHING aggressive."""
    cfg, _params, _bank, gen, _spec = _cfg_params_bank(
        n_envs=256,
        bots={
            "personality_weights": {
                "rush": 0.0, "camper": 1.0, "hunter": 0.0, "trapper": 1.0, "kite": 1.0,
            },
            "min_aggressive": 2,
        },
    )
    mask = torch.ones(256, dtype=torch.bool)
    persons = spawn.sample_personalities(mask, cfg, gen)[:, 1:]

    is_aggressive = torch.zeros_like(persons, dtype=torch.bool)
    for person in AGGRESSIVE_PERSONS:
        is_aggressive |= persons == person
    counts = is_aggressive.sum(dim=1)
    assert torch.all(counts >= 2)
    # Exactly the floor, not more: the weights make every other draw non-aggressive, so this also
    # proves the forcing never converts a slot that was already aggressive (which would waste one).
    assert torch.all(counts == 2)


def test_sample_personalities_does_not_always_force_the_same_entity_slot():
    """Entity slot order is stable for a whole episode and is visible in the observation, so
    "slot 1 is always the dangerous one" would be a pattern the agent could read instead of
    learning to fight. The forced slots must be randomly placed."""
    cfg, _params, _bank, gen, _spec = _cfg_params_bank(
        n_envs=512,
        bots={
            "personality_weights": {
                "rush": 0.0, "camper": 1.0, "hunter": 0.0, "trapper": 0.0, "kite": 0.0,
            },
            "min_aggressive": 1,
        },
    )
    mask = torch.ones(512, dtype=torch.bool)
    persons = spawn.sample_personalities(mask, cfg, gen)[:, 1:]

    is_aggressive = torch.zeros_like(persons, dtype=torch.bool)
    for person in AGGRESSIVE_PERSONS:
        is_aggressive |= persons == person
    assert torch.all(is_aggressive.sum(dim=1) == 1)
    # The one forced slot lands in every position across the batch, not just position 0.
    forced_slot = torch.argmax(is_aggressive.to(torch.int64), dim=1)
    assert len(set(forced_slot.tolist())) == cfg.n_enemies


def test_personalities_disabled_makes_every_bot_a_rush():
    cfg, _params, _bank, gen, _spec = _cfg_params_bank(n_envs=32, bots={"personalities": False})
    mask = torch.ones(32, dtype=torch.bool)
    persons = spawn.sample_personalities(mask, cfg, gen)
    assert torch.all(persons == int(Person.RUSH))


# ---- reset_envs (acceptance) -----------------------------------------------------------

def test_reset_envs_full_reset_sets_expected_fields():
    n_envs = 8
    cfg, params, bank, gen, spec = _cfg_params_bank(
        n_envs=n_envs, world={"map_selection": "fixed", "fixed_map": "open"},
    )
    state = _fresh_state(cfg, n_envs)
    mask = torch.ones(n_envs, dtype=torch.bool)

    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)

    E = cfg.n_entities
    assert torch.all(state.ent_alive)
    assert torch.all(state.ent_cubes == 0)
    assert torch.all(state.ent_death_step == -1)
    assert torch.all(state.ent_target == -1)
    assert torch.all(state.n_alive == E)
    assert torch.all(state.time == 0)
    assert torch.all(state.step_count == 0)
    assert not torch.any(state.prj_alive)
    assert not torch.any(state.pku_alive)

    zero_cubes = torch.zeros_like(state.ent_kind)
    expected_max_hp = stats.effective_max_hp(state.ent_kind, zero_cubes, params)
    assert torch.allclose(state.ent_hp, expected_max_hp, atol=1e-3)
    assert torch.allclose(state.ent_max_hp, expected_max_hp, atol=1e-3)
    expected_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    assert torch.allclose(state.ent_ammo, expected_ammo, atol=1e-3)

    # Personality state: the two fields whose zero_ default is semantically wrong for a spawn.
    assert torch.all(state.ent_hunt_seen == 0)  # bitmask: nowhere visited yet
    assert torch.all(state.ent_hunt_t == cfg.bots_hunt_timeout_seconds)  # not 0 == "already expired"
    assert torch.all((state.ent_person >= 0) & (state.ent_person < N_PERSONS))

    assert torch.all(state.box_alive.sum(dim=1) > 0)
    assert torch.allclose(state.zone_hi[:, 0], torch.full((n_envs,), float(cfg.map_w)))
    assert torch.allclose(state.zone_hi[:, 1], torch.full((n_envs,), float(cfg.map_h)))
    assert torch.allclose(state.zone_lo, torch.zeros(n_envs, 2))

    assert not torch.any(torch.isnan(state.ent_facing))


def test_sim_params_fields_are_tensors_except_the_declared_scalars():
    """SimParams' core contract (N08) and its one explicit exception, both pinned.

    Anything that walks `__slots__` expecting tensors -- `.clone()`, row indexing, torch.where --
    breaks on a plain Python number. `SCALAR_FIELDS` exists so those walks can skip exactly the
    known scalars; this test is what stops that allowlist from silently rotting in either
    direction: a new non-tensor field added without declaring it fails here, and a declared
    scalar that later becomes a tensor fails here too.
    """
    cfg, params, _bank, _gen, _spec = _cfg_params_bank(n_envs=4)

    for attr in params.SCALAR_FIELDS:
        assert attr in params.__slots__, f"{attr} is declared scalar but is not a SimParams field"
        value = getattr(params, attr)
        assert isinstance(value, (int, float)) and not torch.is_tensor(value), (
            f"{attr} is declared in SCALAR_FIELDS but is a {type(value).__name__}"
        )

    for attr in params.__slots__:
        if attr in params.SCALAR_FIELDS:
            continue
        value = getattr(params, attr)
        assert torch.is_tensor(value), (
            f"SimParams.{attr} is a {type(value).__name__}, not a tensor. Every field must be a "
            f"leading-(N,) tensor unless it is structural (sizes a tensor dim / drives Python "
            f"control flow), in which case add it to SimParams.SCALAR_FIELDS and say why."
        )
        assert value.shape[0] == 4, f"SimParams.{attr} has no leading (N,) dim: {tuple(value.shape)}"


def test_reset_envs_partial_mask_leaves_other_rows_bit_identical():
    n_envs = 8
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)
    state = _fresh_state(cfg, n_envs)
    full_mask = torch.ones(n_envs, dtype=torch.bool)
    spawn.reset_envs(state, full_mask, bank, params, cfg, gen, spec)

    from brawl_sim.core.state import _ALL_FIELD_SPECS
    snapshot = {name: getattr(state, name).clone() for name, _, _ in _ALL_FIELD_SPECS}
    # SCALAR_FIELDS are plain Python numbers, not tensors -- snapshot them by value and check
    # them separately below. See SimParams' docstring for why the exception is an enumerated
    # allowlist rather than a duck-type check.
    tensor_attrs = [a for a in params.__slots__ if a not in params.SCALAR_FIELDS]
    params_snapshot = {attr: getattr(params, attr).clone() for attr in tensor_attrs}
    scalar_snapshot = {attr: getattr(params, attr) for attr in params.SCALAR_FIELDS}

    partial_mask = torch.tensor([False, True, False, False, False, True, False, False])
    spawn.reset_envs(state, partial_mask, bank, params, cfg, gen, spec)

    untouched_rows = [0, 2, 3, 4, 6, 7]
    for name, _, _ in _ALL_FIELD_SPECS:
        if name == "env_idx":
            continue
        current = getattr(state, name)
        for row in untouched_rows:
            assert torch.equal(current[row], snapshot[name][row]), f"{name} row {row} changed"
    for attr in tensor_attrs:
        current = getattr(params, attr)
        for row in untouched_rows:
            assert torch.equal(current[row], params_snapshot[attr][row]), f"{attr} row {row} changed"
    # Spec-derived structural scalars are constant for the env's lifetime -- a reset of ANY
    # subset of rows must leave them exactly alone (resample_params never touches them).
    for attr, before in scalar_snapshot.items():
        assert getattr(params, attr) == before, f"scalar {attr} changed across a partial reset"

    # sanity: the masked rows actually did get touched (position resampled)
    changed_rows = [1, 5]
    assert any(
        not torch.equal(state.ent_pos[row], snapshot["ent_pos"][row]) for row in changed_rows
    )


def test_reset_envs_second_reset_of_same_rows_is_consistent():
    # Calling reset_envs twice in a row on the same mask must not error and must still leave
    # a fully self-consistent state (regression guard for the zero_-then-overwrite ordering).
    n_envs = 4
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)
    state = _fresh_state(cfg, n_envs)
    mask = torch.ones(n_envs, dtype=torch.bool)
    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)
    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)
    assert torch.all(state.ent_alive)
    assert torch.all(state.n_alive == cfg.n_entities)


# ---- batched / no-NaN smoke ------------------------------------------------------------

def test_batched_smoke():
    n_envs = 64
    cfg, params, bank, gen, spec = _cfg_params_bank(n_envs=n_envs)
    state = _fresh_state(cfg, n_envs)
    mask = torch.ones(n_envs, dtype=torch.bool)
    spawn.reset_envs(state, mask, bank, params, cfg, gen, spec)

    assert not torch.any(torch.isnan(state.ent_pos))
    assert not torch.any(torch.isnan(state.ent_hp))
    assert not torch.any(torch.isnan(state.zone_lo))
    assert not torch.any(torch.isnan(state.zone_hi))
    assert state.ent_pos.shape == (n_envs, cfg.n_entities, 2)
