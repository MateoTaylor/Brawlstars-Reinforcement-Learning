"""brawl_sim/core/slots.py: tracker-style enemy slots.

The scripts below are one string per decision, one character per ENEMY (entity 1 onwards): `x`
seen by the hero, `.` not. `_run` feeds them to `slots.update` and returns the slot table after
each decision as entity indices (-1 = empty). The parity check against the live `EntityTracker`
is in tests/test_deployment_tracker.py.
"""
from types import SimpleNamespace

import torch

import yaml

from brawl_sim.config import load_config
from brawl_sim.core import slots
from brawl_sim.core.state import allocate, zero_

CONFIGS_DEFAULT = "configs/default.yaml"
TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _cfg(n_enemies=3, **slot_overrides):
    overrides = {"entities": {"n_enemies": n_enemies}}
    if slot_overrides:
        overrides["slots"] = slot_overrides
    return load_config(CONFIGS_DEFAULT, overrides=overrides)


def _state(cfg, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    return state


def _view(row: str, n_entities: int) -> torch.Tensor:
    assert len(row) == n_entities - 1, row
    return torch.tensor([[False] + [ch == "x" for ch in row]])


def _table(state, env=0) -> list:
    return [v - 1 for v in state.slot_ent[env].tolist()]


def _run(state, cfg, script, env=0):
    tables = []
    for row in script:
        hv = torch.zeros(state.ent_alive.shape, dtype=torch.bool)
        hv[env] = _view(row, cfg.n_entities)[0]
        slots.update(state, hv, cfg)
        tables.append(_table(state, env))
    return tables


# ---- promotion ------------------------------------------------------------------------------

def test_a_slot_takes_two_consecutive_sightings():
    cfg = _cfg()
    assert _run(_state(cfg), cfg, ["x..", "x.."]) == [[-1, -1, -1], [1, -1, -1]]


def test_a_miss_between_two_sightings_restarts_the_count():
    cfg = _cfg()
    assert _run(_state(cfg), cfg, ["x..", "...", "x.."])[-1] == [-1, -1, -1]


def test_promote_hits_one_slots_on_the_first_sighting():
    cfg = _cfg(promote_hits=1)
    assert _run(_state(cfg), cfg, ["x.."]) == [[1, -1, -1]]


def test_two_promoted_in_one_decision_take_slots_in_entity_order():
    cfg = _cfg()
    assert _run(_state(cfg), cfg, [".xx", ".xx"])[-1] == [2, 3, -1]


# ---- coasting and retirement ------------------------------------------------------------------

def test_a_slot_is_held_through_max_misses_unseen_decisions_and_freed_on_the_next():
    cfg = _cfg()
    state = _state(cfg)
    tables = _run(state, cfg, ["x..", "x..", "...", "...", "...", "..."])
    assert tables[1:5] == [[1, -1, -1]] * 4, "seen twice, then held through misses 1, 2, 3"
    assert int(state.ent_misses[0, 1]) == 0 and tables[5] == [-1, -1, -1], "miss 4 frees it"


def test_misses_count_the_coast():
    cfg = _cfg()
    state = _state(cfg)
    _run(state, cfg, ["x..", "x..", "...", "..."])
    assert int(state.ent_misses[0, 1]) == 2


def test_max_misses_one_frees_on_the_second_unseen_decision():
    cfg = _cfg(max_misses=1)
    tables = _run(_state(cfg), cfg, ["x..", "x..", "...", "..."])
    assert tables[2] == [1, -1, -1] and tables[3] == [-1, -1, -1]


def test_twelve_misses_hold_the_slot_for_three_seconds():
    """train.yaml's `slots.max_misses: 12` (archive/CRITIC_AND_MEMORY_PLAN.md decision 1): at
    agent_dt 0.25 an unseen enemy keeps its slot, and with it its memory row, for 3.0 s, counting
    misses 1 to 12, and loses both on the 13th unseen decision."""
    cfg = _cfg(max_misses=12)
    state = _state(cfg)
    _run(state, cfg, ["x..", "x.."])
    coast = []
    for _ in range(13):
        _run(state, cfg, ["..."])
        coast.append((int(state.ent_misses[0, 1]), _table(state)[0]))
    assert coast[:12] == [(k, 1) for k in range(1, 13)]
    assert coast[12] == (0, -1)
    assert 12 * cfg.agent_dt == 3.0


def test_a_return_takes_the_lowest_free_slot_not_the_old_one():
    """Entity 1 held slot 0, was freed, and entity 2 took slot 0 meanwhile: entity 1 comes back
    into slot 1."""
    cfg = _cfg()
    script = ["x..", "x..", "...", "...", "...", ".x.", ".x.", "xx.", "xx."]
    tables = _run(_state(cfg), cfg, script)
    assert tables[6] == [2, -1, -1]
    assert tables[-1] == [2, 1, -1]


def test_a_dead_entity_coasts_out_on_the_same_schedule():
    cfg = _cfg()
    state = _state(cfg)
    _run(state, cfg, ["x..", "x.."])
    state.ent_alive[0, 1] = False
    tables = _run(state, cfg, ["x..", "x..", "x..", "x.."])   # "seen" by the view, but dead
    assert tables[:3] == [[1, -1, -1]] * 3 and tables[3] == [-1, -1, -1]


# ---- the last-sighting snapshots ----------------------------------------------------------------

def test_the_snapshot_follows_a_seen_enemy_and_holds_through_the_coast():
    """`ent_seen_*` copy an enemy's position, velocity and HP on every decision it is seen, pending
    or slotted, and keep them while it is not: the last sighting obs["memory"] shows. A death is
    not a sighting, so a dead enemy keeps its last living one. Every value is exact in float32."""
    cfg = _cfg()
    state = _state(cfg)

    def place(x, hp):
        state.ent_pos[0, 1] = torch.tensor([x, 2.0])
        state.ent_vel[0, 1] = torch.tensor([x / 4, -1.0])
        state.ent_hp[0, 1] = hp

    def seen():
        return (state.ent_seen_pos[0, 1].tolist(), state.ent_seen_vel[0, 1].tolist(),
                float(state.ent_seen_hp[0, 1]))

    place(5.0, 3000.0)
    assert _run(state, cfg, ["x.."]) == [[-1, -1, -1]]
    assert seen() == ([5.0, 2.0], [1.25, -1.0], 3000.0), "pending, and already snapshotted"
    place(6.0, 2500.0)
    _run(state, cfg, ["x.."])
    assert seen() == ([6.0, 2.0], [1.5, -1.0], 2500.0)
    place(9.0, 1000.0)                                  # moves on and takes damage unseen
    _run(state, cfg, ["...", "..."])
    assert seen() == ([6.0, 2.0], [1.5, -1.0], 2500.0) and int(state.ent_misses[0, 1]) == 2
    place(10.0, 0.0)
    state.ent_alive[0, 1] = False
    _run(state, cfg, ["x.."])                           # in view, but dead
    assert seen() == ([6.0, 2.0], [1.5, -1.0], 2500.0) and int(state.ent_misses[0, 1]) == 3


def test_zero_clears_every_slot_field():
    """Every field nonzero first, so a field `zero_` skipped would fail here."""
    cfg = _cfg()
    state = _state(cfg)
    state.ent_pos.fill_(4.0)
    state.ent_vel.fill_(1.0)
    state.ent_hp.fill_(3000.0)
    _run(state, cfg, ["xx.", "xx.", "..x"])
    names = ("slot_ent", "ent_slot", "ent_hits", "ent_misses",
             "ent_seen_pos", "ent_seen_vel", "ent_seen_hp")
    assert all(getattr(state, name).any() for name in names)
    zero_(state, torch.ones(1, dtype=torch.bool))
    for name in names:
        assert not getattr(state, name).any(), name


# ---- capacity and batching ------------------------------------------------------------------

def test_more_candidates_than_free_slots_leave_the_highest_index_pending():
    """Ten enemies for nine slots cannot happen with the sim's E - 1 slots, so a bare state with
    a narrower table: the tenth stays pending, keeps counting, and takes the slot that frees."""
    cfg = _cfg(n_enemies=10)
    E, K = 11, 9
    state = SimpleNamespace(
        slot_ent=torch.zeros(1, K, dtype=torch.int64), ent_slot=torch.zeros(1, E, dtype=torch.int64),
        ent_hits=torch.zeros(1, E, dtype=torch.int32), ent_misses=torch.zeros(1, E, dtype=torch.int32),
        ent_alive=torch.ones(1, E, dtype=torch.bool),
        ent_pos=torch.zeros(1, E, 2), ent_vel=torch.zeros(1, E, 2), ent_hp=torch.zeros(1, E),
        ent_seen_pos=torch.zeros(1, E, 2), ent_seen_vel=torch.zeros(1, E, 2),
        ent_seen_hp=torch.zeros(1, E))
    all_seen = torch.tensor([[False] + [True] * 10])
    slots.update(state, all_seen, cfg)
    slots.update(state, all_seen, cfg)
    assert _table(state) == list(range(1, 10)) and int(state.ent_slot[0, 10]) == 0
    assert int(state.ent_hits[0, 10]) == 2
    first_unseen = all_seen.clone()
    first_unseen[0, 1] = False
    for _ in range(4):
        slots.update(state, first_unseen, cfg)
    assert _table(state)[0] == 10, "entity 1 retired on its 4th miss and entity 10 took slot 0"


def test_an_envs_mask_moves_only_the_named_envs_on():
    cfg = _cfg()
    state = _state(cfg, n_envs=2)
    hv = torch.stack([_view("x..", cfg.n_entities)[0]] * 2)
    slots.update(state, hv, cfg)
    slots.update(state, hv, cfg, envs=torch.tensor([True, False]))
    assert _table(state, 0) == [1, -1, -1] and _table(state, 1) == [-1, -1, -1]
    assert int(state.ent_hits[1, 1]) == 1


def test_one_env_step_is_one_decision_of_the_slot_table():
    """With autoreset on, `env.step` builds the observation twice: in `_observe`, and again for
    the whole batch in `_autoreset`, where only the reset envs start a decision. Until 2026-10-12
    both builds ran `slots.update`, so a coast counted two misses per step. A dead enemy is
    unseen, so its slot must count exactly one miss per step."""
    from brawl_sim.env import BrawlVecEnv

    cfg = load_config(CONFIGS_DEFAULT, overrides={**TINY, "slots": {"max_misses": 12}})
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    env.reset()
    idle = torch.zeros((1, 2), dtype=torch.int64)
    env.step(idle)
    assert int(env.state.ent_slot[0, 1]) > 0, "entity 1 holds a slot"
    env.state.ent_alive[0, 1] = False
    for k in range(1, 5):
        env.step(idle)
        assert int(env.state.ent_misses[0, 1]) == k


def test_envs_are_independent():
    cfg = _cfg()
    state = _state(cfg, n_envs=2)
    for row0, row1 in (("x..", ".x."), ("x..", ".x."), ("...", ".x.")):
        hv = torch.stack([_view(row0, cfg.n_entities)[0], _view(row1, cfg.n_entities)[0]])
        slots.update(state, hv, cfg)
    assert _table(state, 0) == [1, -1, -1] and int(state.ent_misses[0, 1]) == 1
    assert _table(state, 1) == [2, -1, -1] and int(state.ent_misses[1, 2]) == 0
