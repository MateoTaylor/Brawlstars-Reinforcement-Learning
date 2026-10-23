"""`perception/assemble.py` -- the last stage before the policy.

The test that matters here is PARITY, not shape. This module's whole job is to arrange perception
outputs into the tensor the checkpoint expects, and every way of getting that wrong is silent: a
column swapped, an enemy slot shifted by the hero row, a grid plane permuted, a divisor missed.
None of it raises and none of it shows up in a shape assertion.

So `test_deploy6_assembles_every_actor_group_and_leaves_the_critic_at_zero` runs a real
`BrawlVecEnv`, pulls out of its observation exactly the quantities the deployed suppliers would
produce, feeds them through `assemble`, and demands the result equal what
`obs_select.build_agent_obs` makes of the same frame. If the two agree column-for-column on a real
observation, the wiring is right -- and that is an assertion about the deployed path, since
`build_agent_obs` is the training path. The sim side always builds into buffers of its own: the
assembler returns copies, but `build_agent_obs` returns views of the buffers it is handed, so a
`want` read out of the assembler's buffers is rewritten by the `assemble` it is compared with.

The `history` group gets the same treatment twice. `test_synthetic_snapshots_give_the_sims_history_
vector_byte_for_byte` writes the sim's rings by hand and demands the same bytes from snapshots of
the same numbers, and `test_deploy6_assembles_whole` runs the spec against a live env, with the
snapshots `DeployLoop` would have taken.

Every test reads `configs/agent_obs_deploy6.yaml`, the one deploy spec in `configs/`; where only a
different spec would reach a branch of the assembler, a temp-file variant of deploy6 reaches it.
The rest of the file covers the decisions the module makes that the sim has no opinion about: the
enemy HP promotion rule (6.5), the refusal to invent a missing supplier (9.8), and the view
scatter into the sim's 15 channels.
"""
from collections import deque

import numpy as np
import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import camera, obs_select
from brawl_deployment.perception.assemble import (DecisionSnapshot, MapFrame,
                                                  ObservationAssembler)

CONFIGS = "configs/default.yaml"
TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
# The preset disables the zone; deploy6 has a zone group, so put it back. The small map, two
# enemies and short episode are all still wanted -- this is the fast-suite config.
TINY["zone"] = {"enabled": True}
# Tracked enemy MEMORY rows, the `history` group and enemy_hist planes, and a privileged critic
# (archive/CRITIC_AND_MEMORY_PLAN.md). The deploy specs before it are in archive/configs/.
DEPLOY6 = "configs/agent_obs_deploy6.yaml"


def _sim_buffers(spec, cfg):
    """Buffers for the sim side's `build_agent_obs`, apart from the assembler's (see the module
    docstring)."""
    return obs_select.make_agent_obs_buffers(spec, cfg, n_envs=1, device="cpu")


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


def _all_projectiles_on_screen(full, env_i=0) -> bool:
    """False in a frame where the sim path and the live path disagree by decision (see
    `_suppliers_from`): a live projectile the camera does not show."""
    prj = full["projectiles"]
    return bool((prj["in_view"][env_i] | ~prj["alive"][env_i]).all())


class _Track:
    """What `EntityTracker` hands `assemble`: a world position, a velocity, `seen_now`, and
    `misses`, the unseen decisions it has coasted through."""

    def __init__(self, pos, vel, seen_now, misses=0):
        self.pos, self.vel, self.seen_now, self.misses = pos, vel, seen_now, misses


def _rebuild(env, cfg):
    """The env's observation of its current state, rebuilt WITHOUT a decision of `slots.update`
    (which `env._build_observation` runs), so a slot table set by hand is read as set."""
    from brawl_sim.bots import perception
    from brawl_sim.core import observation
    los = perception.raw_los(env.state, env.bank, cfg) if cfg.obs_include_raw_los else None
    return observation.build_obs(env.state, env.bank, env._obs_vis, los, env.params, cfg,
                                 hero_view=env._obs_hero_view)


def _suppliers_from(full, cfg, spec, env_i=0, history=()):
    """The deployed suppliers' outputs, read off a sim observation instead of off a screen.

    Every line here is a claim about which real supplier produces which sim field, and they are
    the claims BRAWL_DEPLOYMENT_DESIGN.md 6.3--6.5 make: the shadow owns the hero's timers, the
    tracker owns enemy identity and velocity, `HealthTracker` owns both HP columns, `hud.py` owns
    the brawler count, the terrain map owns the two `in_bush` lookups, and `ZoneEstimator` owns
    the zone pair.

    `history` is the one supplier an observation cannot give back: it is `DeployLoop`'s record of
    the decisions BEFORE this one, so the caller keeps it, with `_snapshot_of`.

    The tracks are what the tracker holds for deploy6's `memory.*` rows: slot k's entity from the
    slot table, there while it holds the slot, coasting included, at its LAST SIGHTING, which is
    what a coasting track's `pos`, `vel` and committed HP are live.
    """
    at = lambda t: t[env_i]                                              # noqa: E731
    hero, prj, zone, meta = full["hero"], full["projectiles"], full["zone"], full["meta"]

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
        "gadget_ready": bool(at(hero["gadget_ready"])),
        "gadget_charge_frac": float(at(hero["gadget_charge_frac"])),
    }
    hero_pos = tuple(at(hero["pos"]).tolist())
    hero_vel = tuple(at(hero["vel"]).tolist())
    # The tracker's `hero_offset`: the player box against its nominal screen anchor, which in the
    # sim is the hero against the clamped camera centre (core/camera.camera_centre).
    cam = camera.camera_centre(hero["pos"][env_i:env_i + 1], cfg)[0]
    hero_offset = tuple((at(hero["pos"]) - cam).tolist())

    # Slot k of the tracker holds whichever entity the sim's slot table says (core/slots.py):
    # what a `slots: tracked` group reads there, and what the tracker's own slots are live. An
    # empty slot (entity 0) has no track.
    enemies, enemy_hp, enemy_bush = [], {}, {}
    mem = full["memory"]
    for slot, e in enumerate(at(full["slots"]["entity"]).tolist()):
        if e == 0:
            enemies.append(None)
            continue
        i = e - 1
        rel, rel_v = at(mem["rel_pos"])[i].tolist(), at(mem["rel_vel"])[i].tolist()
        enemies.append(_Track((rel[0] + hero_pos[0], rel[1] + hero_pos[1]),
                              (rel_v[0] + hero_vel[0], rel_v[1] + hero_vel[1]),
                              bool(at(mem["seen"])[i]),
                              misses=round(float(at(mem["age_s"])[i]) / cfg.agent_dt)))
        enemy_hp[slot] = float(at(mem["hp"])[i])
        enemy_bush[slot] = bool(at(mem["in_bush"])[i])

    # Only the projectiles the camera shows: the live `ProjectileTracker` has no track for an
    # off-screen one (assemble._put_projectiles marks every track `in_view`). The sim path still
    # RANKS off-screen projectiles into its 12 slots and blanks their rows (a quirk kept by
    # decision, BRAWL_SIM_DESIGN.md §9), so in a frame with one alive the two paths lay their rows
    # out differently; `_all_projectiles_on_screen` is how the parity tests skip those frames.
    alive_p = (at(prj["alive"]) & at(prj["in_view"])).nonzero().flatten().tolist()
    projectiles = [(tuple(at(prj["rel_pos"])[j].tolist()), tuple(at(prj["vel"])[j].tolist()),
                    float(at(prj["time_to_closest"])[j])) for j in alive_p]

    # The two keys `ZoneEstimator.estimate` returns, and nothing it does not.
    zone_kw = {"hero_margin_local": tuple(at(zone["hero_margin_local"]).tolist()),
               "active": bool(at(zone["active"]))}

    grid_group = next(g for g in spec.groups if g.view_channels is not None)
    grid = at(full["view"])[list(grid_group.channel_idx)].numpy().astype(np.uint8)

    return dict(
        hero_pos=hero_pos, hero_vel=hero_vel, shadow=shadow,
        hero_hp=float(at(hero["hp"])), n_enemies_alive=float(at(meta["n_enemies_alive"])),
        elapsed_s=float(at(meta["time_frac"])) * float(cfg.max_episode_steps) * float(cfg.dt),
        hero_in_bush=bool(at(hero["in_bush"])), hero_in_zone=bool(at(hero["in_zone"])),
        enemies=enemies, enemy_hp=enemy_hp, enemy_in_bush=enemy_bush,
        projectiles=projectiles, zone=zone_kw, history=tuple(history), grid=grid,
        hero_offset=hero_offset,
    )


def _without_near_edge_spec() -> str:
    """deploy6 with `hero.near_edge` dropped from its self group, written to a temp file: the
    shape every spec before deploy5 had, and the only one that needs no `hero_offset`."""
    import tempfile
    doc = yaml.safe_load(open(DEPLOY6).read())
    for g in doc["groups"]:
        if "hero.near_edge" in g.get("fields", ()):
            g["fields"].remove("hero.near_edge")
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w") as f:
        yaml.dump(doc, f)
    return path


def _snapshot_of(full, move, attack, env_i=0):
    """The `DecisionSnapshot` `DeployLoop` takes of a decision: the numbers its observation was
    built from, and the action that answered it. Read before `env.step`, since `full`'s tensors
    are views into the state the step is about to move. `enemies` feed the grid, which these
    tests take from the sim's own view, so they are left empty here."""
    hero = full["hero"]
    return DecisionSnapshot(move_bin=int(move), attack=int(attack),
                            hp=float(hero["hp"][env_i]),
                            ammo_frac=float(hero["ammo_frac"][env_i]),
                            pos=tuple(hero["pos"][env_i].tolist()), enemies=())


# ---- parity ---------------------------------------------------------------------------------

@pytest.mark.parametrize("hero_xy, want", [((3.0, 10.0), True), ((10.0, 10.0), False)])
def test_near_edge_assembles_to_the_sims_bit(hero_xy, want):
    """On the tiny 20-wide map the camera is pinned at x 9.65, so a hero at x 3 is 6.65 tiles
    off its anchor (past the 2-tile flag) and one at x 10 is 0.35. The whole `self` group must
    still match the sim's."""
    env, cfg = _env()
    asm = _assembler(DEPLOY6, cfg)
    spec = asm.spec
    env.reset()
    env.state.ent_pos[0, 0] = torch.tensor(hero_xy)
    full = env._build_observation()
    idx = obs_select.agent_obs_index_map(spec, cfg)["self"]["hero.near_edge"][0]
    sim = obs_select.build_agent_obs(full, spec, cfg, _sim_buffers(spec, cfg))["self"][0].numpy()
    got = asm.assemble(**_suppliers_from(full, cfg, spec))["self"]
    assert bool(sim[idx]) is want and got[idx] == sim[idx]
    np.testing.assert_allclose(got, sim, atol=1e-5)


def test_hero_offset_is_only_required_by_a_spec_that_reads_near_edge():
    env, cfg = _env()
    full = env.reset()
    for spec_path, needs in ((_without_near_edge_spec(), False), (DEPLOY6, True)):
        asm = _assembler(spec_path, cfg)
        kw = _suppliers_from(full, cfg, asm.spec)
        kw.pop("hero_offset")
        if needs:
            with pytest.raises(ValueError, match="hero_offset"):
                asm.assemble(**kw)
        else:
            asm.assemble(**kw)


def test_parity_survives_a_frame_with_live_projectiles():
    """The projectile group is the one place the two paths select rather than copy: `max_slots`
    takes the nearest 12 by `time_to_closest` out of `max_projectiles` slots, and deployment's
    live projectiles sit at slots 0..n-1 while the sim's sit wherever they were spawned. The
    selection has to be index-independent for that to be safe, which is why it is left to
    `obs_select` on both sides -- this is the test that says it stayed there.

    Frames with an off-screen live projectile are skipped: the sim ranks it into the 12 slots
    and blanks its row (a quirk kept by decision, BRAWL_SIM_DESIGN.md §9), while the tracker
    never holds it, so the two paths differ there by design. Until 2026-10-21 `want` here was
    read out of the assembler's own buffers, so `assemble` rewrote it to its own answer: a
    0.5-tile shift of every projectile passed all 28 compared frames."""
    env, cfg = _env(n_envs=1, seed=3)
    asm = _assembler(DEPLOY6, cfg)
    spec = asm.spec
    buffers = _sim_buffers(spec, cfg)

    full = env.reset()
    gen = torch.Generator().manual_seed(1)
    seen = 0
    for _ in range(40):
        n_live = int(full["projectiles"]["alive"][0].sum())
        if n_live and _all_projectiles_on_screen(full):
            seen = max(seen, n_live)
            want = obs_select.build_agent_obs(full, spec, cfg, buffers)["projectiles"][0].numpy()
            got = asm.assemble(**_suppliers_from(full, cfg, spec))["projectiles"]
            np.testing.assert_allclose(got, want, atol=1e-5)
        action = torch.stack([torch.randint(0, cfg.n_move_bins + 1, (1,), generator=gen),
                              torch.ones(1, dtype=torch.int64)], dim=1)   # attack every step
        full, *_ = env.step(action)
    assert seen > 0, "no projectile ever went live; this test asserted nothing"


# `hist.valid` over a match's first four decisions: nothing, then one slot more per decision until
# the ring is full. Checked here at the assembler.
VALID_BY_DECISION = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0])


def test_deploy6_assembles_whole():
    """deploy6 through `assemble` against a live env: the gadget pair, the `history`
    group from the snapshots `DeployLoop` would have taken, and the thirteen-plane grid.

    The first action throws the gadget. A charged gadget reads (True, 1.0), which a swapped pair
    reproduces exactly; after the throw the sim reads (False, 0.2 / 18), and only the right
    wiring reproduces that. The same throw is slot 0's attack one-hot a decision later. Six
    decisions, so the ring fills and then drops its oldest slot twice. The history is compared
    as bytes. The critic groups are left to the test below."""
    env, cfg = _env()
    asm = _assembler(DEPLOY6, cfg)
    spec = asm.spec
    buffers = _sim_buffers(spec, cfg)

    full = env.reset()
    gen = torch.Generator().manual_seed(2)
    history = deque(maxlen=3)
    for step in range(6):
        want = obs_select.build_agent_obs(full, spec, cfg, buffers)
        want = {k: v[0].clone().numpy() for k, v in want.items()}
        got = asm.assemble(**_suppliers_from(full, cfg, spec, history=history))
        assert got.keys() == want.keys()
        assert got["history"].tobytes() == want["history"].tobytes(), (
            f"history differs at step {step}\ngot  {got['history']}\nwant {want['history']}")
        assert np.array_equal(got["grid"], want["grid"]), f"grid differs at step {step}"
        for g in spec.groups:
            np.testing.assert_allclose(got[g.name], want[g.name], atol=1e-5,
                                       err_msg=f"{g.name} differs at step {step}")
        assert got["history"][:3].tolist() == VALID_BY_DECISION[min(step, 3)]
        if step == 1:        # the frame after the throw; the pair is self 23, 24 on deploy6
            assert got["self"][23] == 0.0
            assert got["self"][24] == pytest.approx(0.0111111, abs=1e-6)
            assert got["history"][57] == 1.0      # attack one-hots start at 54; slot 0 threw
        move = int(torch.randint(0, cfg.n_move_bins + 1, (1,), generator=gen))
        attack = 3 if step == 0 else int(torch.randint(0, 2, (1,), generator=gen))
        history.appendleft(_snapshot_of(full, move, attack))
        full, _, terminated, truncated, _ = env.step(torch.tensor([[move, attack]], dtype=torch.int64))
        assert not bool(terminated[0] or truncated[0]), "a reset would clear the sim's ring"


def test_deploy6_assembles_every_actor_group_and_leaves_the_critic_at_zero():
    """deploy6 against a live env at the run's 12-miss coast: every actor group equals the sim's
    at every decision, `enemies` included, read from memory -- a coasting slot's last sighting
    and its age -- and both critic groups stay zero, since the assembler builds with
    critic=False. The sim side gets its own buffers: its build fills the critic groups, and
    sharing the assembler's would hand those values back through `assemble`."""
    from brawl_sim.env import BrawlVecEnv

    cfg = load_config(CONFIGS, overrides={**TINY, "slots": {"max_misses": 12}})
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    asm = _assembler(DEPLOY6, cfg)
    spec = asm.spec
    sim_buffers = _sim_buffers(spec, cfg)
    assert [g.name for g in spec.critic_groups] == ["critic_enemies", "critic_world"]

    full = env.reset()
    gen = torch.Generator().manual_seed(4)
    history = deque(maxlen=3)
    coasting = projectile_frames = 0
    for step in range(40):
        want = obs_select.build_agent_obs(full, spec, cfg, sim_buffers)
        want = {k: v[0].clone().numpy() for k, v in want.items()}
        got = asm.assemble(**_suppliers_from(full, cfg, spec, history=history))
        assert got.keys() == want.keys()
        comparable = _all_projectiles_on_screen(full)
        for g in spec.groups:
            if g.name == "projectiles" and not comparable:
                continue  # the kept ranking quirk; see _suppliers_from
            np.testing.assert_allclose(got[g.name], want[g.name], atol=1e-5,
                                       err_msg=f"{g.name} differs at step {step}")
        for g in spec.critic_groups:
            assert not got[g.name].any(), f"{g.name} reached the live observation at step {step}"
        projectile_frames += int(comparable and bool(full["projectiles"]["alive"][0].any()))
        mem = full["memory"]
        coasting += int(bool((mem["tracked"][0] & ~mem["seen"][0]).any()))
        if step == 8:
            # The camera window covers nearly all of this 20-tile map, so nothing walks out of
            # view: kill entity 1 instead. Dead is unseen, so its slot coasts its 12 decisions.
            env.state.ent_alive[0, 1] = False
        move = int(torch.randint(0, cfg.n_move_bins + 1, (1,), generator=gen))
        attack = int(torch.randint(0, 2, (1,), generator=gen))
        history.appendleft(_snapshot_of(full, move, attack))
        full, _, terminated, truncated, _ = env.step(torch.tensor([[move, attack]], dtype=torch.int64))
        if bool(terminated[0] or truncated[0]):
            history.clear()                     # the reset cleared the sim's ring too
    assert coasting >= 10, f"a slot coasted in only {coasting} decisions of 40"
    assert projectile_frames > 0, "no frame with live projectiles was compared"


def test_a_coasting_memory_row_assembles_to_the_sims_row():
    """A slot table set by hand, so every kind of row is in one frame: slot 0 holds entity 2 on
    its 5th unseen decision, its last sighting nowhere near where it is now; slot 1 holds entity
    1, seen this decision; slot 2 is empty. The sim's deploy6 rows read exactly that, and the
    assembler, handed the tracks the tracker would hold, writes the same rows."""
    from brawl_sim.env import BrawlVecEnv

    cfg = load_config(CONFIGS, overrides={**TINY, "entities": {"n_enemies": 3},
                                          "slots": {"max_misses": 12}})
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    asm = _assembler(DEPLOY6, cfg)
    env.reset()
    st = env.state
    st.slot_ent[0] = torch.tensor([3, 2, 0])
    st.ent_slot[0] = torch.tensor([0, 2, 1, 0])
    st.ent_misses[0] = torch.tensor([0, 0, 5, 0], dtype=torch.int32)
    st.ent_seen_pos[0, 1], st.ent_seen_vel[0, 1] = st.ent_pos[0, 1], st.ent_vel[0, 1]
    st.ent_seen_hp[0, 1] = st.ent_hp[0, 1]
    st.ent_seen_pos[0, 2] = torch.tensor([4.5, 15.5])
    st.ent_seen_vel[0, 2] = torch.tensor([1.0, -0.5])
    st.ent_seen_hp[0, 2] = 1234.0
    assert float((st.ent_pos[0, 2] - st.ent_seen_pos[0, 2]).norm()) > 1.0
    full = _rebuild(env, cfg)

    spec = asm.spec
    want = obs_select.build_agent_obs(full, spec, cfg,
                                      _sim_buffers(spec, cfg))["enemies"][0].numpy()
    col = {f: slice(*c) for f, c in obs_select.agent_obs_index_map(spec, cfg)["enemies"].items()}
    scale = float(max(cfg.map_w, cfg.map_h))
    hero = st.ent_pos[0, 0]
    assert want[0, col["memory.tracked"]].tolist() == [1.0]
    assert want[0, col["memory.seen"]].tolist() == [0.0]
    assert want[0, col["memory.age_s"]].tolist() == [1.25]
    np.testing.assert_allclose(want[0, col["memory.rel_pos"]],
                               ((torch.tensor([4.5, 15.5]) - hero) / scale).numpy(), atol=1e-6)
    assert want[0, col["memory.hp"]] == pytest.approx(1234.0 / 20000.0)
    assert want[1, col["memory.seen"]].tolist() == [1.0] and want[1, col["memory.age_s"]] == 0.0
    assert not want[2].any()

    got = asm.assemble(**_suppliers_from(full, cfg, spec))["enemies"]
    np.testing.assert_allclose(got, want, atol=1e-5)


# Three decisions' worth of hand-picked numbers, newest first: (move_bin, attack, hp, ammo_frac,
# pos). Every column differs between slots, so a slot read from the wrong index cannot agree by
# accident, and every value is exact in float32, so "the same numbers" is literally true.
SYNTHETIC = ((5, 3, 7400.0, 0.25, (6.5, 4.25)),
             (16, 0, 3150.5, 1.0, (7.75, 3.0)),
             (0, 2, 9999.0, 0.5, (9.0, 11.5)))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_synthetic_snapshots_give_the_sims_history_vector_byte_for_byte(n):
    """The sim's rings are written by hand, `n` valid slots
    with in-range garbage behind them, and `build_obs` + `obs_select` make the `history` group
    of that. `n` snapshots of the same numbers go through `assemble`. Same bytes, or it fails."""
    env, cfg = _env()
    asm = _assembler(DEPLOY6, cfg)
    full = env.reset()
    assert float(full["hero"]["ammo_frac"][0]) == 1.0, "a hero spawns full; max_ammo reads off it"
    state = env.state
    max_ammo = float(state.ent_ammo[0, 0])
    state.ent_pos[0, 0] = torch.tensor([8.25, 7.5])             # the hero NOW
    state.hist_valid[0] = torch.tensor([i < n for i in range(3)])
    state.hist_action[0] = torch.tensor([16, 3])                 # garbage a valid slot could hold
    state.hist_hp[0] = 12345.0
    state.hist_ammo[0] = max_ammo
    state.hist_pos[0] = torch.tensor([99.0, -99.0])
    snapshots = []
    for i, (move, attack, hp, frac, pos) in enumerate(SYNTHETIC[:n]):
        state.hist_action[0, i] = torch.tensor([move, attack])
        state.hist_hp[0, i] = hp
        state.hist_ammo[0, i] = frac * max_ammo
        state.hist_pos[0, i] = torch.tensor(pos)
        snapshots.append(DecisionSnapshot(move_bin=move, attack=attack, hp=hp, ammo_frac=frac,
                                          pos=pos, enemies=()))
    full = env._build_observation()

    want = obs_select.build_agent_obs(full, asm.spec, cfg,
                                      _sim_buffers(asm.spec, cfg))["history"][0].numpy()
    got = asm.assemble(**_suppliers_from(full, cfg, asm.spec, history=snapshots))["history"]
    assert got.dtype == want.dtype == np.float32 and got.shape == want.shape == (78,)
    assert got.tobytes() == want.tobytes(), f"\ngot  {got}\nwant {want}"

    # And the column layout itself, so the two cannot agree on a wrong one. TINY's map
    # is 20 tiles, the displacement divisor; hp divides by 20000.
    assert got[:3].tolist() == VALID_BY_DECISION[n]
    if n == 0:
        assert not got.any()
    if n == 3:
        assert np.flatnonzero(got[3:66]).tolist() == [5, 33, 34, 54, 55, 61]
        np.testing.assert_allclose(got[66:], [0.37, 0.157525, 0.49995, 0.25, 1.0, 0.5,
                                              -0.0875, -0.1625, -0.025, -0.225, 0.0375, 0.2],
                                   rtol=1e-6)


def test_more_history_than_the_config_keeps_is_refused():
    """The loop's deque is sized from the same `history_frames`, so a fourth snapshot means the
    two were built from different configs. Truncating would hide that."""
    asm = _assembler(DEPLOY6)
    snap = DecisionSnapshot(move_bin=0, attack=0, hp=8000.0, ammo_frac=1.0, pos=(10.0, 10.0),
                            enemies=())
    with pytest.raises(ValueError, match="history holds 4 decisions and the config keeps 3"):
        asm.assemble(**_base_kwargs(asm, history=(snap,) * 4))


# ---- the decisions the sim has no opinion about ----------------------------------------------

def test_a_tracked_enemy_with_no_hp_yet_is_not_promoted_to_a_slot():
    """§6.5's rule, and the reason it is safe. 84% of enemy boxes yield an HP and every track that
    NEVER yielded one lived four frames or fewer, so a slot held back for want of an HP is one
    that would have been a glancing detection anyway -- and a track that will persist gets its
    number within 1 frame at p50, 3 at p90, inside one decision.

    The alternative is a made-up HP in a column the policy was trained to trust. An empty slot is
    a state the sim produces constantly; a fabricated HP is one it never produces at all."""
    asm = _assembler(DEPLOY6)
    tracks = [None] * (asm.cfg.n_entities - 1)
    tracks[0] = _Track((12.0, 9.0), (1.0, 0.0), True)
    tracks[1] = _Track((14.0, 9.0), (1.0, 0.0), True)

    obs = asm.assemble(**_base_kwargs(asm, enemies=tracks, enemy_hp={0: 5400}))
    col = obs_select.agent_obs_index_map(asm.spec, asm.cfg)["enemies"]["memory.tracked"][0]
    tracked = obs["enemies"][:, col]
    assert tracked[0] == 1.0, "slot 0 has a committed HP and must be visible"
    assert tracked[1] == 0.0, "slot 1 has no HP yet, so it is not an enemy the policy is told about"
    assert not obs["enemies"][1].any(), "an unpromoted slot must be entirely zero, not partly filled"


@pytest.mark.parametrize("missing", ["hero_hp", "n_enemies_alive", "hero_in_bush", "zone",
                                     "history", "hero_offset"])
def test_a_missing_supplier_raises_rather_than_becoming_a_zero(missing):
    """The rule a deploy spec exists to enforce, at the last point it can still be enforced. A
    field in the spec has a supplier; if that supplier produced nothing this tick, the loop's
    answer is to fail closed, not to substitute a plausible number."""
    asm = _assembler(DEPLOY6)
    kwargs = _base_kwargs(asm)
    kwargs[missing] = None
    with pytest.raises(ValueError, match="no value supplied"):
        asm.assemble(**kwargs)


def test_a_zone_field_the_spec_needs_cannot_be_silently_omitted():
    """`_put_zone` is driven by the loaded spec, because the zone group has changed under a
    retrain before (9.16): the first deploy spec read the unclamped `hero_margin`. Handing that
    spec's dict to a deploy6 assembler must raise, not quietly produce a shorter group."""
    asm = _assembler(DEPLOY6)
    kwargs = _base_kwargs(asm)
    kwargs["zone"] = {"hero_margin": (1.0, 1.0, 1.0, 1.0), "active": True}
    with pytest.raises(ValueError, match="zone.hero_margin_local"):
        asm.assemble(**kwargs)


def test_the_view_scatter_puts_each_plane_at_its_canonical_sim_channel():
    """The grid group is `index_select` over the sim's view, 12 base channels and then one
    enemy_hist plane per history slot, so a plane written to the wrong index is read as a
    different channel entirely -- walls arriving as bushes. Marking each plane with its own value
    makes a permutation visible; equal planes would not."""
    asm = _assembler(DEPLOY6)
    assert asm._view.shape[0] == 15
    grid = np.zeros((len(asm._grid_channels), asm.cfg.view_h, asm.cfg.view_w), np.uint8)
    for i in range(grid.shape[0]):
        grid[i, 0, 0] = i + 1
    obs = asm.assemble(**_base_kwargs(asm, grid=grid))
    assert list(obs["grid"][:, 0, 0]) == list(range(1, grid.shape[0] + 1))
    # and the channels nothing supplies were never touched
    unfilled = set(range(15)) - set(asm._grid_channels)
    assert unfilled == {5, 7}                                    # enemy_any, enemy_hidden
    assert not asm._view[sorted(unfilled)].any()


def test_time_frac_is_clamped_because_the_sim_truncates_at_one():
    """Running short is harmless -- the policy just never sees the top of the range, exactly as
    when a sim episode ends early. Running LONG is out-of-distribution: the column's entire
    training range is bounded by 1, and a 200-second match would feed it 1.33 (9.6)."""
    asm = _assembler(DEPLOY6)
    idx = obs_select.agent_obs_index_map(asm.spec, asm.cfg)["self"]["meta.time_frac"][0]
    for elapsed, want in ((0.0, 0.0), (75.0, 0.5), (150.0, 1.0), (400.0, 1.0)):
        obs = asm.assemble(**_base_kwargs(asm, elapsed_s=elapsed))
        assert obs["self"][idx] == pytest.approx(want, abs=1e-6)


def test_the_map_frame_is_the_only_thing_that_reads_absolute_position():
    """`hero.pos_norm` needs an origin and an extent the deployed loop does not have, and
    `MapFrame` is where that is quarantined. Moving it must change those two columns and NOTHING
    else -- if it ever moves another column, absolute position has leaked into a relative field
    and the leak is worth finding immediately."""
    asm = _assembler(DEPLOY6)
    a = asm.assemble(**_base_kwargs(asm))
    asm.map_frame = MapFrame(origin=(7.0, -3.0), extent=(80.0, 80.0))
    b = asm.assemble(**_base_kwargs(asm))

    assert not np.allclose(a["self"][:2], b["self"][:2])
    np.testing.assert_allclose(a["self"][2:], b["self"][2:], atol=0)
    for name in ("enemies", "projectiles", "zone", "history", "grid"):
        assert np.array_equal(a[name], b[name]), f"{name} moved with the map frame"


def test_the_default_map_frame_centres_odometry_and_keeps_the_sims_scale():
    """The measured choice (see `MapFrame`'s docstring). A constant origin error is free -- 0.734
    at +0.15 and 0.743 at +0.35 against a 0.734 baseline -- while zeroing the column costs 4.7 pp
    overall and 12 pp on elite. A SCALE error is a different perturbation and was not priced, so
    the extent must stay the sim's own: odometry is in tiles, `hero.pos` is in tiles, and dividing
    by the sim's map size makes one tile of real movement produce exactly the column movement it
    produced in training. This pins both halves of that."""
    asm = ObservationAssembler.from_paths(DEPLOY6, CONFIGS)
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
    field wants to see the other fifteen are present, since the thing under test is often that a
    missing one is fatal. `history` is a match's first decision: empty, and still supplied. The
    zone is the pair `ZoneEstimator.estimate` returns."""
    shadow = {"facing_vec": (1.0, 0.0), "ammo_frac": 0.83, "ammo_whole": 2.0, "attack_cd": 0.1,
              "can_attack": True, "dashing": False, "dash_t": 0.0, "dash_dir": (0.0, 0.0),
              "invuln": False, "long_dash_ready": True, "long_dash_frac": 1.0,
              "super_ready": False, "super_charge_frac": 0.4,
              "gadget_ready": False, "gadget_charge_frac": 0.3}
    kwargs = dict(
        hero_pos=(10.0, 10.0), hero_vel=(0.5, -0.5), shadow=shadow, hero_hp=8000.0,
        n_enemies_alive=4.0, elapsed_s=42.0, hero_in_bush=False, hero_in_zone=False,
        enemies=[None] * (asm.cfg.n_entities - 1), enemy_hp={}, enemy_in_bush={},
        projectiles=[],
        zone={"hero_margin_local": (5.0, 10.0, -1.0, 10.0), "active": True},
        history=(),
        grid=np.zeros((len(asm._grid_channels), asm.cfg.view_h, asm.cfg.view_w), np.uint8),
        hero_offset=(0.0, 0.0),
    )
    kwargs.update(overrides)
    return kwargs
