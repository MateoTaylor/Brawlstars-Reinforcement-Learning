"""Edgar: the two-hit forward combo and its lifesteal.

Edgar is the first brawler whose sub-swings all fire at ONE angle (`hitscan_sweep_rad: 0`) and
the first with `melee_lifesteal_fraction`, so these tests pin the two things that are genuinely
new rather than re-testing core/melee_sweep's schedule (tests/test_melee_sweep.py owns that).
"""
import copy

import torch
import yaml

from brawl_sim.config import build_params, load_config
from brawl_sim.constants import Kind
from brawl_sim.core import combat, melee_sweep
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
EDGAR = int(Kind.BOT_EDGAR)


def _load_spec():
    spec = {}
    for path in (CONFIGS_DEFAULT, "configs/brawlers.yaml"):
        with open(path) as f:
            spec.update(yaml.safe_load(f))
    return spec


# Parsed once at import rather than on every _params call; build_params gets a deep copy, so
# nothing it does to the spec can leak from one test into the next.
_SPEC = _load_spec()


def _params(n_envs=1):
    cfg = load_config(CONFIGS_DEFAULT, overrides={"world": {"map_h": 20, "map_w": 20}})
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    return cfg, build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=copy.deepcopy(_SPEC))


def _env_with_edgars(n_enemies=2):
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "entities": {"n_enemies": n_enemies, "randomize_enemy_types": False,
                     "fixed_enemy_types": ["edgar"] * n_enemies},
        "zone": {"enabled": False}, "regen": {"enabled": False},
        "observation": {"include_world_grid": False},
        # Per-tick mechanics: one step must be one tick, or the whole 0.5 s combo resolves
        # inside a single step() and the schedule under test is invisible.
        "sim": {"action_repeat": 1},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, verbose=False)
    env.reset()
    return env


def _stage_a_swing(env, victim_dist=1.2):
    """Puts the hero `victim_dist` tiles down Edgar's facing and hands Edgar a loaded clip.

    Anchored on Edgar's OWN spawn position rather than hand-picked coordinates: the shipped map
    is 60x60 with real terrain, and an arbitrary pair of coordinates can land a body in a wall or
    put one between the two -- `melee_hitscan` runs a physical LOS march per victim, so that
    silently produces a swing that connects with nothing and a test that passes for the wrong
    reason. Offsetting from a real spawn keeps both on legal, mutually visible floor."""
    st = env.state
    anchor = st.ent_pos[0, 1].clone()
    st.ent_facing[0, 1] = 0.0
    st.ent_pos[0, 0] = anchor + torch.tensor([victim_dist, 0.0])
    for e in range(2, env.cfg.n_entities):
        st.ent_pos[0, e] = anchor + torch.tensor([0.0, 12.0])
    st.ent_ammo[0, 1] = 3.0
    st.ent_attack_cd[0, 1] = 0.0
    return st


def _overrides(env):
    """(fire, hold) override tensors. `hold` pins every bot to "idle, do not fire" so the only
    attack in the episode is the one the test triggers."""
    none = torch.full((1, env.cfg.n_entities, 2), -1, dtype=torch.int64)
    fire = none.clone()
    fire[0, 1] = torch.tensor([0, 1])
    hold = none.clone()
    for e in range(1, env.cfg.n_entities):
        hold[0, e] = torch.tensor([0, 0])
    return fire, hold


def _run_combo(env, ticks=16, victim_dist=1.2, edgar_hp=3000.0):
    """Fires one combo and returns the per-tick (damage to hero, HP gained by Edgar) events.

    16 ticks: the 0.5 s combo is 10 ticks at action_repeat 1 (hits on ticks 0 and 5), and the
    6 spare ones would catch a third or a late hit. Nothing fires after it: `hold` keeps every
    bot's trigger up."""
    st = _stage_a_swing(env, victim_dist)
    st.ent_hp[0, 1] = edgar_hp
    fire, hold = _overrides(env)
    idle = torch.zeros(1, 2, dtype=torch.int64)

    events = []
    prev_v, prev_e = float(st.ent_hp[0, 0]), float(st.ent_hp[0, 1])
    for t in range(ticks):
        env.step(idle, override=fire if t == 0 else hold)
        v, e = float(st.ent_hp[0, 0]), float(st.ent_hp[0, 1])
        if prev_v - v > 1.0 or e - prev_e > 1.0:
            events.append({"tick": t, "t": (t + 1) * env.cfg.dt, "dmg": prev_v - v, "heal": e - prev_e})
        prev_v, prev_e = v, e
    return events


# ---- the combo ------------------------------------------------------------------

def test_edgar_lands_exactly_two_hits_a_quarter_second_apart():
    """The stated mechanic: "each ammo usage fires two attacks forward ... each of those attacks
    takes around 0.25s to fire (total attack animation is 0.5s)". Both halves are pinned -- the
    COUNT (exactly two, not one and not three) and the SPACING."""
    env = _env_with_edgars()
    events = _run_combo(env)

    assert len(events) == 2, f"expected a two-hit combo, got {len(events)} hits: {events}"
    gap = events[1]["t"] - events[0]["t"]
    assert abs(gap - 0.25) < 1e-6, f"hits were {gap:.4f}s apart, expected 0.25s"
    for ev in events:
        assert abs(ev["dmg"] - 1080.0) < 1e-3, f"hit dealt {ev['dmg']}, expected 1080"


def test_the_whole_combo_costs_one_ammo():
    """Two hits, one ammo -- the same rule Buzz's five-hitscan sweep follows. If this ever became
    one ammo per hit Edgar's sustained output would halve.

    EXACTLY one: Edgar starts at max ammo, and the combo's own 0.5 s cooldown pauses the reload for
    the 10 ticks read here, so nothing refills while it runs. Until 2026-10-20 this ran 12 ticks,
    long enough for the reload to resume, and so could only check `<= 1` -- which a combo that cost
    no ammo at all passed too."""
    env = _env_with_edgars()
    st = _stage_a_swing(env)
    fire, hold = _overrides(env)
    idle = torch.zeros(1, 2, dtype=torch.int64)

    before = float(st.ent_ammo[0, 1])
    for t in range(10):  # both hits, and the whole cooldown that pauses the reload
        env.step(idle, override=fire if t == 0 else hold)
    spent = before - float(st.ent_ammo[0, 1])
    assert abs(spent - 1.0) < 1e-4, f"the combo spent {spent:.4f} ammo, expected exactly 1"


def test_both_hits_fire_down_the_trigger_facing():
    """"His attacks do not follow a radial sequence, they just fire in whatever direction he was
    facing when he triggered the attack." With `hitscan_sweep_rad: 0` the sub-swing offset is
    identically zero, so both cones must sit exactly on the anchored facing -- and the facing must
    still be the trigger-time one at the end of the combo, which is what
    core/movement.apply_movement's sweep freeze guarantees."""
    cfg, params = _params()
    from brawl_sim.core.state import allocate
    st = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    st.ent_alive.fill_(True)
    st.ent_kind[0, 1] = EDGAR
    st.ent_facing[0, 1] = 0.7

    cooldown = float(params.attack_cooldown[0, EDGAR])
    seen = []
    for tick in range(int(round(cooldown / cfg.dt)) + 1):
        st.ent_attack_cd[0, 1] = max(cooldown - tick * cfg.dt, 0.0)
        fire, cone_dir = melee_sweep.sweep_cone_dir(st, params, cfg)
        if bool(fire[0, 1]):
            seen.append(float(cone_dir[0, 1]))

    assert len(seen) == 2, f"expected two sub-swings, saw {len(seen)}"
    for angle in seen:
        assert abs(angle - 0.7) < 1e-6, f"a sub-swing fired at {angle}, not the anchored 0.7"


def test_edgar_counts_as_swept_so_his_facing_is_frozen_mid_combo():
    """`is_swept` is `hitscan_count > 1`, which is also what gates the facing freeze in
    core/movement. A zero sweep angle must NOT accidentally exclude Edgar from it -- if it did,
    he would re-aim between his two hits and the second would follow his walk direction."""
    cfg, params = _params()
    from brawl_sim.core.state import allocate
    st = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    st.ent_alive.fill_(True)
    st.ent_kind[0, 1] = EDGAR
    assert bool(melee_sweep.is_swept(st, params)[0, 1])


# ---- lifesteal ------------------------------------------------------------------

def test_each_hit_heals_35_percent_of_its_damage():
    env = _env_with_edgars()
    events = _run_combo(env)

    assert len(events) == 2
    for ev in events:
        expected = 0.35 * ev["dmg"]
        assert abs(ev["heal"] - expected) < 1e-2, \
            f"hit for {ev['dmg']} healed {ev['heal']}, expected {expected}"


def test_a_half_connected_combo_heals_half_as_much():
    """Lifesteal is per HIT, not per attack: the two cones are re-tested 0.25 s apart, so a victim
    who leaves between them costs Edgar the second hit AND its healing. This is the property that
    makes the zero-angle combo different from one double-damage swing, and it is the reason
    config.validate no longer rejects `hitscan_sweep_rad: 0`."""
    env = _env_with_edgars()
    st = _stage_a_swing(env)
    st.ent_hp[0, 1] = 3000.0
    fire, hold = _overrides(env)
    idle = torch.zeros(1, 2, dtype=torch.int64)

    healed = 0.0
    prev_e = float(st.ent_hp[0, 1])
    for t in range(12):
        env.step(idle, override=fire if t == 0 else hold)
        # Teleport the victim out of reach right after the FIRST hit lands, before the second.
        if t == 0:
            st.ent_pos[0, 0] = st.ent_pos[0, 1] + torch.tensor([0.0, 10.0])
        e = float(st.ent_hp[0, 1])
        healed += max(e - prev_e, 0.0)
        prev_e = e

    assert abs(healed - 378.0) < 1e-2, \
        f"a combo that only landed its first hit healed {healed}, expected one hit's 378"


def test_lifesteal_ignores_damage_an_invulnerable_victim_never_took():
    """`combat.melee_lifesteal` re-applies the i-frame mask `apply_damage` uses, because `dmg_by`
    is raw cone output: if a victim is i-framed, the heal must follow the damage it actually took.

    This pins a DORMANT mask. Nothing in the sim seeds `ent_invuln_t` (the dash has no i-frames),
    so the i-frame is set by hand; the test keeps the heal consistent with `apply_damage` if
    anything ever grants one again."""
    cfg, params = _params()
    from brawl_sim.core.state import allocate
    st = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    st.ent_alive.fill_(True)
    st.ent_kind[0, 1] = EDGAR

    dmg_by = torch.zeros(1, cfg.n_entities, cfg.n_entities)
    dmg_by[0, 1, 0] = 1080.0  # Edgar hit entity 0

    vulnerable = combat.melee_lifesteal(st, dmg_by, params)
    assert abs(float(vulnerable[0, 1]) - 378.0) < 1e-3

    st.ent_invuln_t[0, 0] = 0.2  # entity 0 i-framed, by hand: nothing in the sim seeds this
    blocked = combat.melee_lifesteal(st, dmg_by, params)
    assert float(blocked[0, 1]) == 0.0, "healed off damage an invulnerable victim never took"


def test_lifesteal_cannot_take_edgar_over_max_hp():
    """`apply_heal` owns the clamp; this pins that the melee path actually goes through it rather
    than adding to ent_hp directly."""
    env = _env_with_edgars()
    st = env.state
    events = _run_combo(env, edgar_hp=float(st.ent_max_hp[0, 1]))  # start at full HP
    assert len(events) >= 1
    assert float(st.ent_hp[0, 1]) <= float(st.ent_max_hp[0, 1]) + 1e-3


def test_no_other_kind_lifesteals():
    """`melee_lifesteal_fraction` defaults to 0, so the field costs the rest of the roster a
    masked multiply and nothing else. Buzz in particular swings a cone and must NOT heal."""
    cfg, params = _params()
    for kind in Kind:
        value = float(params.melee_lifesteal_fraction[0, int(kind)])
        expected = 0.35 if kind is Kind.BOT_EDGAR else 0.0
        assert abs(value - expected) < 1e-6, f"{kind!r} has lifesteal {value}, expected {expected}"


def test_boxes_do_not_feed_lifesteal():
    """Edgar's combo into a loot box damages the box and heals him nothing. Healing off boxes
    would hand Edgar a free top-up on every map, from a target that cannot fight back.

    Driven through the real env tick, where the box damage and the lifesteal meet (the hero stands
    out of reach, so the box is the only thing hit). Until 2026-10-20 this called
    `melee_lifesteal` on an all-zero damage matrix, which returns zero whatever the env does with
    box damage."""
    env = _env_with_edgars()
    st = _stage_a_swing(env)
    anchor = st.ent_pos[0, 1].clone()
    st.ent_pos[0, 0] = anchor + torch.tensor([0.0, 12.0])
    st.box_alive[0, 0] = True
    st.box_pos[0, 0] = anchor + torch.tensor([1.2, 0.0])  # where the hero would stand: down the facing
    st.box_hp[0, 0] = 1e6
    st.box_max_hp[0, 0] = 1e6
    st.ent_hp[0, 1] = 3000.0  # well under max, so any heal would show
    fire, hold = _overrides(env)
    idle = torch.zeros(1, 2, dtype=torch.int64)

    for t in range(10):  # the whole combo: both hits
        env.step(idle, override=fire if t == 0 else hold)

    assert float(st.box_hp[0, 0]) < 1e6, "the combo never reached the box"
    assert float(st.ent_hp[0, 1]) == 3000.0, f"Edgar healed to {float(st.ent_hp[0, 1])} off a loot box"
