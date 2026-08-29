from pathlib import Path

import torch
import yaml

from brawl_sim.config import load_config, build_params
from brawl_sim.constants import DeathCause, Kind
from brawl_sim.core import combat, events, stats
from brawl_sim.core.state import allocate

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _cfg_and_params(n_envs=1, n_enemies=3, max_episode_steps=300):
    overrides = {
        "world": {"map_h": 20, "map_w": 20},
        "entities": {"n_enemies": n_enemies},
        "sim": {"max_episode_steps": max_episode_steps},
    }
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)
    spec = {
        **yaml.safe_load((CONFIGS / "default.yaml").read_text()),
        **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params


def _fresh_state(cfg, params, n_envs=1):
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(Kind.BOT_MELEE)
    state.map_id.fill_(0)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_max_hp.copy_(max_hp)
    state.ent_hp.copy_(max_hp)
    state.ent_last_hit_by.fill_(-1)
    state.ent_death_step.fill_(-1)
    state.ent_target.fill_(-1)
    return state


def _zeros_by(cfg, n_envs=1):
    return torch.zeros(n_envs, cfg.n_entities, cfg.n_entities)


# ---- compute_done ---------------------------------------------------------------------

def test_terminated_on_hero_death():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_alive[0, 0] = False
    terminated, truncated = events.compute_done(state, cfg)
    assert bool(terminated[0])
    assert not bool(truncated[0])


def test_terminated_on_hero_last_alive():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.ent_alive[0, 1:] = False  # only the hero (slot 0) remains
    state.n_alive.fill_(1)
    terminated, truncated = events.compute_done(state, cfg)
    assert bool(terminated[0])


def test_not_terminated_multiple_alive():
    cfg, params = _cfg_and_params()
    state = _fresh_state(cfg, params)
    state.n_alive.fill_(cfg.n_entities)
    terminated, truncated = events.compute_done(state, cfg)
    assert not bool(terminated[0])
    assert not bool(truncated[0])


def test_truncated_at_step_limit():
    cfg, params = _cfg_and_params(max_episode_steps=300)
    state = _fresh_state(cfg, params)
    state.n_alive.fill_(cfg.n_entities)
    state.step_count.fill_(300)
    terminated, truncated = events.compute_done(state, cfg)
    assert bool(truncated[0])
    assert not bool(terminated[0])


def test_terminated_and_truncated_can_both_be_true_on_final_tick_death():
    cfg, params = _cfg_and_params(max_episode_steps=300)
    state = _fresh_state(cfg, params)
    state.ent_alive[0, 0] = False
    state.step_count.fill_(300)
    terminated, truncated = events.compute_done(state, cfg)
    assert bool(terminated[0]) and bool(truncated[0])


# ---- hero_rank --------------------------------------------------------------------------

def test_hero_rank_zero_when_hero_last_alive():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    state.ent_alive[0, 1:] = False
    state.n_alive.fill_(1)
    info = events.compute_info(state, _zeros_by(cfg), torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg)
    assert int(info["hero_rank"][0]) == 0


def test_hero_rank_positive_when_others_still_alive():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)  # everyone alive -> hero ties at n_alive-1 (0-indexed)
    info = events.compute_info(state, _zeros_by(cfg), torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg)
    assert int(info["hero_rank"][0]) == cfg.n_entities - 1


# ---- damage_matrix / damage_dealt_tick / damage_taken_tick (acceptance) -----------------

def test_damage_matrix_row_sums_equal_damage_dealt_tick():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    dmg_by = _zeros_by(cfg)
    dmg_by[0, 0, 1] = 100.0  # hero (0) hits enemy 1
    dmg_by[0, 0, 2] = 50.0   # hero (0) also hits enemy 2
    dmg_by[0, 2, 3] = 30.0   # enemy 2 hits enemy 3

    info = events.compute_info(state, dmg_by, torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg)

    assert torch.allclose(info["damage_matrix"], dmg_by)
    assert torch.allclose(info["damage_dealt_tick"], dmg_by.sum(dim=2))
    assert info["damage_dealt_tick"][0, 0].item() == 150.0
    assert torch.allclose(info["damage_taken_tick"], dmg_by.sum(dim=1))
    assert info["damage_taken_tick"][0, 1].item() == 100.0


# ---- kills_tick matches a manual count over a scripted rollout (acceptance) --------------

def test_kills_tick_matches_manual_count_over_scripted_rollout():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    n_envs = 1

    torch.manual_seed(0)
    total_kills_tick = torch.zeros(n_envs, cfg.n_entities, dtype=torch.int32)

    for step in range(500):
        state.step_count.fill_(step)
        # every ~40 ticks, entity (step//40 % 3)+1 "kills" a random other alive victim by
        # zeroing its hp and crediting last_hit_by, mimicking what apply_damage would have done
        if step % 40 == 0 and step > 0:
            alive_idx = torch.nonzero(state.ent_alive[0], as_tuple=False).squeeze(-1)
            if alive_idx.numel() >= 2:
                killer = alive_idx[0].item()
                victim = alive_idx[-1].item()
                if killer != victim:
                    state.ent_hp[0, victim] = 0.0
                    state.ent_last_hit_by[0, victim] = killer

        newly_dead = combat.resolve_deaths(state, cfg)
        info = events.compute_info(
            state, _zeros_by(cfg), newly_dead, torch.zeros_like(state.box_alive),
            torch.zeros_like(state.ent_cubes), cfg,
        )
        total_kills_tick += info["kills_tick"]

        if not bool(state.ent_alive[0, 0]) or int(state.n_alive[0]) <= 1:
            break

    assert torch.equal(total_kills_tick, state.ent_kills)
    assert int(total_kills_tick.sum()) > 0  # sanity: the scripted rollout actually killed someone


def test_deaths_tick_and_death_cause_tick():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    state.ent_hp[0, 1] = 0.0
    state.ent_last_hit_by[0, 1] = 0  # combat death
    state.ent_hp[0, 2] = 0.0
    state.ent_last_hit_by[0, 2] = -1  # zone death

    newly_dead = combat.resolve_deaths(state, cfg)
    info = events.compute_info(state, _zeros_by(cfg), newly_dead, torch.zeros_like(state.box_alive),
                                torch.zeros_like(state.ent_cubes), cfg)

    assert bool(info["deaths_tick"][0, 1]) and bool(info["deaths_tick"][0, 2])
    assert not bool(info["deaths_tick"][0, 0]) and not bool(info["deaths_tick"][0, 3])
    assert int(info["death_cause_tick"][0, 1]) == int(DeathCause.COMBAT)
    assert int(info["death_cause_tick"][0, 2]) == int(DeathCause.ZONE)
    assert int(info["death_cause_tick"][0, 3]) == int(DeathCause.ALIVE)  # never died -> masked to 0


# ---- boxes_broken_tick / cubes_gained_tick ------------------------------------------------

def test_boxes_broken_and_cubes_gained_tick_pass_through():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    newly_broken = torch.zeros_like(state.box_alive)
    newly_broken[0, 0] = True
    newly_broken[0, 1] = True
    cubes_gained = torch.zeros_like(state.ent_cubes)
    cubes_gained[0, 2] = 5

    info = events.compute_info(state, _zeros_by(cfg), torch.zeros_like(state.ent_alive),
                                newly_broken, cubes_gained, cfg)

    assert int(info["boxes_broken_tick"][0]) == 2
    assert torch.equal(info["cubes_gained_tick"], cubes_gained)


# ---- hp_healed_tick ------------------------------------------------------------------------

def test_hp_healed_tick_defaults_to_zeros_and_passes_through_what_env_supplies():
    """Healing has no attacker x victim matrix and no SimState counter to diff, so compute_info
    can only report what env.py hands it -- and every caller that doesn't (tests, tools) must
    still get a well-shaped zero rather than a missing key."""
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)

    info = events.compute_info(state, _zeros_by(cfg), torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg)
    assert info["hp_healed_tick"].shape == (1, cfg.n_entities)
    assert float(info["hp_healed_tick"].abs().sum()) == 0.0

    healed = torch.zeros((1, cfg.n_entities))
    healed[0, 0] = 260.0
    info = events.compute_info(state, _zeros_by(cfg), torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg,
                                hp_healed=healed)
    assert torch.equal(info["hp_healed_tick"], healed)


# ---- dash_hits_tick / shots_fired_tick -----------------------------------------------------

def test_dash_hits_tick_only_counts_dashing_attackers():
    cfg, params = _cfg_and_params(n_enemies=3)
    state = _fresh_state(cfg, params)
    dmg_by = _zeros_by(cfg)
    dmg_by[0, 0, 1] = 100.0  # entity 0 (dashing) hits 1
    dmg_by[0, 2, 3] = 50.0   # entity 2 (not dashing) hits 3
    state.ent_dash_t[0, 0] = 0.1
    state.ent_dash_t[0, 2] = 0.0

    info = events.compute_info(state, dmg_by, torch.zeros_like(state.ent_alive),
                                torch.zeros_like(state.box_alive), torch.zeros_like(state.ent_cubes), cfg)
    assert int(info["dash_hits_tick"][0]) == 1
    assert int(info["shots_fired_tick"][0, 0]) == 1
    assert int(info["shots_fired_tick"][0, 2]) == 1
    assert int(info["shots_fired_tick"][0, 1]) == 0


# ---- batched / no-NaN smoke ---------------------------------------------------------------

def test_batched_smoke():
    n_envs = 8
    cfg, params = _cfg_and_params(n_envs=n_envs, n_enemies=3)
    state = _fresh_state(cfg, params, n_envs)
    dmg_by = torch.rand(n_envs, cfg.n_entities, cfg.n_entities) * (torch.rand(n_envs, cfg.n_entities, cfg.n_entities) > 0.8)
    newly_dead = torch.zeros_like(state.ent_alive)
    newly_broken = torch.zeros_like(state.box_alive)
    cubes_gained = torch.zeros_like(state.ent_cubes)

    info = events.compute_info(state, dmg_by, newly_dead, newly_broken, cubes_gained, cfg)
    for key, value in info.items():
        assert value.shape[0] == n_envs, key
        if value.is_floating_point():
            assert not torch.any(torch.isnan(value)), key

    terminated, truncated = events.compute_done(state, cfg)
    assert terminated.shape == (n_envs,)
    assert truncated.shape == (n_envs,)
