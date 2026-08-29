import torch

from tests.bot_fixtures import FakeBank, build_targeting
from brawl_sim.config import load_config, build_params
from brawl_sim.constants import Kind, Person, Tile
from brawl_sim.core import stats
from brawl_sim.core.state import allocate
from brawl_sim.bots import combat_rules, perception, personality, policy

CONFIGS_DEFAULT = "configs/default.yaml"


_FakeBank = FakeBank  # shared fixture; also supplies the bush waypoints hunt_waypoint reads


def _grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def _cfg_and_params(n_envs=1, n_enemies=4, map_h=20, map_w=20):
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"map_h": map_h, "map_w": map_w},
        "entities": {"n_enemies": n_enemies},
        "zone": {"enabled": False},
    })
    import yaml
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params, gen


def _fresh_state_one_of_each(cfg, params, n_envs=1):
    """Entity 0 = hero, 1 = sniper, 2 = artillery, 3 = melee, 4 = rifle (requires n_enemies=4)."""
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    kinds = [Kind.HERO_MORTIS, Kind.BOT_SNIPER, Kind.BOT_ARTILLERY, Kind.BOT_MELEE, Kind.BOT_RIFLE]
    for e, k in enumerate(kinds):
        state.ent_kind[:, e] = int(k)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    # RUSH for everyone: this file tests the KIND dispatch, so personality is held constant.
    state.ent_person.fill_(int(Person.RUSH))
    state.ent_hunt_t.fill_(cfg.bots_hunt_timeout_seconds)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    return state


def _scatter_entities_around(state, center=(10.0, 10.0), spread=3.0):
    n_e = state.ent_pos.shape[1]
    for e in range(n_e):
        angle = e * (6.28318 / n_e)
        state.ent_pos[0, e] = torch.tensor([
            center[0] + spread * torch.cos(torch.tensor(angle)),
            center[1] + spread * torch.sin(torch.tensor(angle)),
        ])


# ---- selection correctness -----------------------------------------------------------

def test_fire_and_aim_match_the_combat_rule_for_every_kind():
    """The dispatcher passes bots/combat_rules' output through unaltered except for the two
    gates it owns (decision period on fire, hero/dead zeroing).

    step_count=0 means only entities whose slot index is itself a multiple of their
    decision_period get to reconsider firing on this exact tick (staggering by design, see
    policy.all_bot_intents' docstring) -- so `fire` is only directly comparable to the raw rule
    for entities where that holds. aim_dir/aim_point never pass through decision-period gating or
    reaction-delay smoothing at all, so those match unconditionally for every living non-hero
    entity.

    Before Step E1 this compared against four separate archetype modules; there is one rule now,
    and the per-kind DIFFERENCES it used to prove live in tests/test_combat_rules.py."""
    cfg, params, gen = _cfg_and_params(n_enemies=4)
    state = _fresh_state_one_of_each(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    _scatter_entities_around(state)

    vis, tgt = build_targeting(state, bank, params, cfg)  # match all_bot_intents' own preamble
    ref_fire, ref_dir, ref_point = combat_rules.combat(state, tgt, bank, params, cfg, gen)

    # re-seed so the dispatcher's own aim noise draws line up with the reference call above
    gen.manual_seed(0)
    vis2 = perception.visibility(state, bank, params, cfg)
    intent = policy.all_bot_intents(state, vis2, bank, params, cfg, gen)

    decision_period = torch.clamp(
        stats.gather_kind(params.decision_period, state.ent_kind), min=1,
    )
    entity_idx = torch.arange(cfg.n_entities, dtype=torch.int64).view(1, -1)
    decision_tick = (entity_idx % decision_period) == 0  # step_count == 0
    # Slots 1..4 are one of each bot kind (see _fresh_state_one_of_each).
    for e in range(1, cfg.n_entities):
        assert torch.allclose(intent.aim_dir[0, e], ref_dir[0, e]), f"aim_dir entity {e}"
        assert torch.allclose(intent.aim_point[0, e], ref_point[0, e]), f"aim_point entity {e}"
        expected_fire = ref_fire[0, e] & decision_tick[0, e]
        assert bool(intent.fire[0, e]) == bool(expected_fire), f"fire entity {e}"

    # ...and the hero's row is zeroed on the way out no matter what the rule computed for it.
    assert not bool(intent.fire[0, 0])
    assert torch.all(intent.aim_dir[0, 0] == 0)
    assert torch.all(intent.aim_point[0, 0] == 0)


def test_move_dir_on_first_tick_is_rate_scaled_raw_intent():
    # ent_move_smooth starts at exactly 0 (allocate()'s zero-init), so the very first EMA
    # update is exactly rate * raw_move_dir -- an exact, not approximate, check. Movement comes
    # from the personality layer now, so the reference is personality.movement, not an archetype.
    cfg, params, gen = _cfg_and_params(n_enemies=4)
    state = _fresh_state_one_of_each(cfg, params)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)
    _scatter_entities_around(state)
    assert torch.allclose(state.ent_move_smooth, torch.zeros_like(state.ent_move_smooth))

    _vis, tgt = build_targeting(state, bank, params, cfg)
    gen.manual_seed(0)
    # movement() mutates wander/hunt state, so snapshot what it would produce from THIS state and
    # then restore, letting the dispatcher below start from the same place.
    saved = {name: getattr(state, name).clone() for name in
             ("ent_wander_dir", "ent_wander_t", "ent_hunt_seen", "ent_hunt_t")}
    ref_move, _ref_mode = personality.movement(state, tgt, bank, params, cfg, gen)
    ref_move = ref_move.clone()
    for name, value in saved.items():
        getattr(state, name).copy_(value)

    gen.manual_seed(0)
    vis2 = perception.visibility(state, bank, params, cfg)
    intent = policy.all_bot_intents(state, vis2, bank, params, cfg, gen)

    reaction_delay = params.reaction_delay[0, int(Kind.BOT_SNIPER)].item()
    rate = min(1.0, cfg.dt / reaction_delay)
    expected = rate * ref_move[0, 1]
    assert torch.allclose(intent.move_dir[0, 1], expected, atol=1e-5)
    assert torch.allclose(state.ent_move_smooth[0, 1], expected, atol=1e-5)


def test_move_smooth_converges_toward_raw_target_over_many_ticks():
    cfg, params, gen = _cfg_and_params(n_enemies=1)
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    state.ent_kind[:, 1] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)

    # Hero fixed due east so the sniper (a RUSH by default, since allocate() zero-inits
    # ent_person) always seeks the exact same direction, +x. 12 tiles apart, not the 17 this test
    # originally used: bots only acquire targets within cfg.bots_sight_tiles (14) now, and an
    # out-of-sight hero would leave the bot wandering at a random heading instead of converging on
    # anything -- see bots/perception.bot_visibility.
    state.ent_pos[0, 0] = torch.tensor([14.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([2.0, 10.0])

    for _ in range(200):
        vis = perception.visibility(state, bank, params, cfg)
        intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    # CLOSE blends seek (1.0) with strafe (0.3), so the converged unit heading is
    # 1/sqrt(1 + 0.3^2) ~= 0.958 along x, not exactly 1.
    assert intent.move_dir[0, 1, 0].item() > 0.9  # converged close to the raw unit direction


# ---- decision period (fire) ---------------------------------------------------------

def test_decision_period_gates_fire_to_staggered_ticks():
    cfg, params, gen = _cfg_and_params(n_enemies=1)
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    state.ent_kind[:, 1] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)

    decision_period = int(params.decision_period[0, int(Kind.BOT_SNIPER)].item())
    assert decision_period > 1  # sanity: bot_sniper.decision_period_ticks is 4 by default

    # well within range, ammo never runs out, LOS clear: sniper WANTS to fire every tick
    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])
    params.attack_cooldown[:, int(Kind.BOT_SNIPER)] = 0.0  # no cooldown gap between shots

    fired_on_step = []
    for step in range(decision_period * 3):
        state.ent_ammo[0, 1] = 3.0  # keep it topped up so ammo never gates fire
        vis = perception.visibility(state, bank, params, cfg)
        intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)
        if bool(intent.fire[0, 1]):
            fired_on_step.append(step)
        state.step_count += 1

    expected = [s for s in range(decision_period * 3) if (s + 1) % decision_period == 0]
    assert fired_on_step == expected


# ---- entity 0 / dead-entity zeroing --------------------------------------------------

def test_entity_zero_and_dead_entities_get_zeroed_intent():
    cfg, params, gen = _cfg_and_params(n_enemies=2)
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    state.ent_kind[:, 1] = int(Kind.BOT_SNIPER)
    state.ent_kind[:, 2] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([13.0, 10.0])
    state.ent_pos[0, 2] = torch.tensor([7.0, 10.0])
    state.ent_alive[0, 2] = False  # entity 2 dead

    vis = perception.visibility(state, bank, params, cfg)
    intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    for field in (intent.move_dir, intent.aim_dir, intent.aim_point):
        assert torch.allclose(field[0, 0], torch.zeros(2))  # hero
        assert torch.allclose(field[0, 2], torch.zeros(2))  # dead
    assert not bool(intent.fire[0, 0])
    assert not bool(intent.fire[0, 2])
    # entity 1 is alive, non-hero, in range with clear LOS -- should be free to act
    assert not torch.allclose(intent.aim_dir[0, 1], torch.zeros(2))


# ---- aim_noise_std_rad plumbing (acceptance) ------------------------------------------

def test_changing_aim_noise_changes_behavior_through_the_dispatcher():
    cfg, params, gen = _cfg_and_params(n_enemies=1)
    state = allocate(cfg, n_envs=1, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    state.ent_kind[:, 1] = int(Kind.BOT_SNIPER)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    tiles = _grid(20, 20)
    bank = _FakeBank(tiles)

    state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    state.ent_pos[0, 1] = torch.tensor([12.0, 10.0])

    params.aim_noise_std_rad[:, int(Kind.BOT_SNIPER)] = 0.0
    vis = perception.visibility(state, bank, params, cfg)
    gen.manual_seed(0)
    intent_zero = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    params.aim_noise_std_rad[:, int(Kind.BOT_SNIPER)] = 1.0  # huge, guarantees a visible change
    gen.manual_seed(0)
    intent_noisy = policy.all_bot_intents(state, vis, bank, params, cfg, gen)

    assert not torch.allclose(intent_zero.aim_dir[0, 1], intent_noisy.aim_dir[0, 1])


# ---- fuzz -------------------------------------------------------------------------------

def test_no_nan_batched_smoke():
    cfg, params, gen = _cfg_and_params(n_envs=8, n_enemies=4)
    state = _fresh_state_one_of_each(cfg, params, n_envs=8)
    tiles = _grid(20, 20)
    tiles[5, 5] = Tile.BUSH
    bank = _FakeBank(tiles)

    for _ in range(50):
        state.ent_pos.uniform_(1, 19)
        state.ent_vel.uniform_(-3, 3)
        state.ent_hp.uniform_(0, 1)
        state.ent_hp.mul_(state.ent_max_hp)
        state.ent_alive.copy_(torch.rand(state.ent_alive.shape) > 0.2)
        state.ent_ammo.uniform_(0, 5)

        vis = perception.visibility(state, bank, params, cfg)
        intent = policy.all_bot_intents(state, vis, bank, params, cfg, gen)
        state.step_count += 1

        for t in (intent.move_dir, intent.aim_dir, intent.aim_point):
            assert torch.all(torch.isfinite(t))
        assert intent.move_dir.shape == (8, cfg.n_entities, 2)
