"""BRAWL_SIM_DESIGN.md §10 (operator-approved 2026-09-21): the `attack_in_reach` shaping term.

One count per decision, paid by training/reward.py: the hero attacked (attack column 1) or used its
super (column 2) while an enemy it could SEE stood inside its uncharged dash reach. The radius is
scripts/audit_attack_cadence.py's own, so the term pays for exactly what A1's utilization measured.

The first half drives the flag through `BrawlVecEnv.step` and `_attack_phase`; the second prices
it through `ShapedReward`. Numbers are pinned as literals from the shipped Mortis block
(`dash_distance 2.67`, `dash_radius 0.70`, `unit_radius 0.40`, so a 3.77-tile reach;
`super_charge_hits 5`, `gadget_cooldown 18.0`) and from configs/train.yaml's `0.05`; a test that
derived them from `params` could not fail.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.constants import Kind
from brawl_sim.training.config import RewardConfig, load_train_config
from brawl_sim.training.reward import TERM_NAMES, ShapedReward

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

_ATTACK = 1      # attack-column values, as literals: importing them would hide a renumbering
_SUPER = 2
_GADGET = 3
_FAR = [3.0, 3.0]


def _env(action_repeat=1, max_episode_steps=2000, autoreset=True, n_envs=1, device="cpu",
         debug_checks=True, reward_fn=None, latency_seconds=None):
    from brawl_sim.env import BrawlVecEnv

    tiny = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())
    sim = {**tiny["sim"], "action_repeat": action_repeat, "max_episode_steps": max_episode_steps}
    if latency_seconds is not None:
        sim["action_latency_seconds"] = latency_seconds
    cfg = load_config(CONFIGS / "default.yaml", overrides={
        **tiny, "sim": sim, "engine": {"debug_checks": debug_checks, "compile": False},
    })
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=0, verbose=False,
                      autoreset=autoreset, reward_fn=reward_fn)
    env.reset()
    return env


def _duel(enemy_at, **env_kwargs):
    """Hero at (10,10) facing +x, bot 1 at `enemy_at`, bot 2 far away at (3,3), every crate dead,
    both bots frozen by the override. Returns `(env, override)`."""
    env = _env(**env_kwargs)
    st = env.state
    assert env.cfg.n_entities == 3
    st.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[0, 1] = torch.tensor(enemy_at)
    st.ent_pos[0, 2] = torch.tensor(_FAR)
    st.ent_facing.zero_()
    st.box_alive.fill_(False)
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[:, 0, 0] = -1
    return env, override


def _act(env, override, attack, move=0):
    return env.step(torch.tensor([[move, attack]], dtype=torch.int64), override)


# ---- the flag, through env.step ------------------------------------------------------------

def test_the_reach_is_the_audits_uncharged_dash_of_3_77_tiles():
    env = _env()
    mortis = int(Kind.HERO_MORTIS)
    assert int(env.state.ent_kind[0, 0]) == mortis
    assert float(env.params.dash_distance[0, mortis]) == pytest.approx(2.67)
    assert float(env.params.dash_radius[0, mortis]) == pytest.approx(0.70)
    assert float(env.params.unit_radius[0]) == pytest.approx(0.40)


@pytest.mark.parametrize("dx, expected", [(3.70, 1), (3.84, 0)])
def test_an_attack_counts_only_with_an_enemy_inside_3_77_tiles(dx, expected):
    env, override = _duel([10.0 + dx, 10.0])
    *_, info = _act(env, override, _ATTACK)
    assert int(env.state.ent_shots_fired[0, 0]) == 1        # the attack went out either way
    assert info["attack_in_reach_tick"].dtype == torch.int32
    assert info["attack_in_reach_tick"].tolist() == [expected]


def test_no_attack_counts_nothing_with_an_enemy_in_reach():
    env, override = _duel([11.5, 10.0])
    *_, info = _act(env, override, 0)
    assert info["attack_in_reach_tick"].tolist() == [0]


def test_an_attack_the_mask_refuses_counts_nothing():
    env, override = _duel([11.5, 10.0])
    env.state.ent_ammo[0, 0] = 0.0
    *_, info = _act(env, override, _ATTACK)
    assert int(env.state.ent_shots_fired[0, 0]) == 0
    assert info["attack_in_reach_tick"].tolist() == [0]


def test_a_super_in_reach_counts_like_an_attack():
    env, override = _duel([11.5, 10.0])
    env.state.ent_super_charge[0, 0] = 5                    # super_charge_hits 5: a full meter
    *_, info = _act(env, override, _SUPER)
    # It went out: the meter is spent to 0, then the idle super's bolt, aimed at bot 1 like the
    # game's tap-to-fire (user decision, 2026-09-30), hits it on this tick for the next charge.
    assert float(env.state.ent_super_charge[0, 0]) == 1.0
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert info["attack_in_reach_tick"].tolist() == [1]


def test_the_gadget_is_not_an_attack_for_the_term():
    """As in the audit, whose "attacked" is attack column 1 or 2: the gadget has its own button
    and its own cooldown, and this term exists to pay for spending ammo."""
    env, override = _duel([11.5, 10.0])
    *_, info = _act(env, override, _GADGET)
    assert float(env.state.ent_gadget_cd[0, 0]) == 18.0     # it was thrown
    assert info["attack_in_reach_tick"].tolist() == [0]


def test_attack_phase_flag_needs_the_heros_own_attack_and_a_live_enemy_it_can_see():
    """`_attack_phase` directly, so `vis` can be handed in: the fair visibility is the only thing
    that separates an enemy in a bush from one in the open at the same distance."""
    from brawl_sim.bots import perception

    env, _override = _duel([11.5, 10.0])
    st, cfg = env.state, env.cfg
    E = cfg.n_entities
    none = torch.zeros(1, E, dtype=torch.bool)
    hero_fires, bot_fires = none.clone(), none.clone()
    hero_fires[0, 0] = True
    bot_fires[0, 1] = True
    zeros2 = torch.zeros(1, E, 2)
    vis = perception.visibility(st, env.bank, env.params, cfg)
    assert bool(vis[0, 0, 1])

    def _flag(fire, vis):
        st.ent_attack_cd.zero_()            # re-arm: the previous call's attack started a dash
        st.ent_dash_t.zero_()
        st.ent_ammo.fill_(3.0)
        _dmg_by, _healed, flag = env._attack_phase(zeros2, fire, none, zeros2, st.ent_pos.clone(),
                                                   none, vis)
        return flag.tolist()

    assert _flag(hero_fires, vis) == [True]
    hidden = vis.clone()
    hidden[0, 0, 1] = False
    assert _flag(hero_fires, hidden) == [False]
    assert _flag(bot_fires, vis) == [False]                 # a bot's attack is never the hero's
    st.ent_alive[0, 1] = False
    assert _flag(hero_fires, vis) == [False]


def test_one_decision_counts_one_attack_under_action_repeat():
    """The tick hook re-arms the hero after every sub-tick, so if the held action still carried
    the fire bit, sub-ticks 2-5 would each attack and count again."""
    env, override = _duel([11.5, 10.0], action_repeat=5)
    st = env.state

    def _rearm(e):
        e.state.ent_attack_cd.zero_()
        e.state.ent_dash_t.zero_()

    env.tick_hook = _rearm
    *_, info = _act(env, override, _ATTACK)
    env.tick_hook = None

    assert int(st.ent_shots_fired[0, 0]) == 1
    assert int(info["n_ticks"][0]) == 5
    assert info["attack_in_reach_tick"].tolist() == [1]


def test_an_attack_that_lands_on_a_later_sub_tick_still_counts():
    """0.1 s of action latency is 2 ticks, so the decision's attack reaches `_attack_phase` on
    sub-tick 3 of 5. The count must be summed over the sub-ticks, not read off the first."""
    env, override = _duel([11.5, 10.0], action_repeat=5, latency_seconds=0.1, autoreset=False)
    assert env.cfg.action_latency_ticks == 2
    *_, info = _act(env, override, _ATTACK)
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert info["attack_in_reach_tick"].tolist() == [1]


def test_an_attack_that_lands_after_the_episode_ended_mid_decision_does_not_count():
    """Same latency, but the tick cap is crossed on sub-tick 1. The world keeps ticking to the end
    of the decision and the attack does go out on sub-tick 3, outside the episode; the same `live`
    gate that stops damage and kills from counting there stops this count too."""
    env, override = _duel([11.5, 10.0], action_repeat=5, latency_seconds=0.1, autoreset=False,
                          max_episode_steps=2000)
    env.state.step_count.fill_(1999)
    *_, info = _act(env, override, _ATTACK)
    assert bool(info["truncated"][0])
    assert int(info["n_ticks"][0]) == 1
    assert int(env.state.ent_shots_fired[0, 0]) == 1
    assert info["attack_in_reach_tick"].tolist() == [0]


# ---- the price, through ShapedReward ---------------------------------------------------------

_ONLY_REACH = dict(win_bonus=0.0, death_penalty=0.0, rank_bonus=0.0, damage_dealt=0.0,
                   damage_taken=0.0, hp_healed=0.0, kill=0.0, cube_pickup=0.0,
                   survive_per_step=0.0, in_zone_per_step=0.0, attack_in_reach=0.05)


def _info(counts):
    """The keys ShapedReward reads whatever its weights, plus this term's own."""
    n = len(counts)
    return {
        "terminated": torch.zeros(n, dtype=torch.bool),
        "truncated": torch.zeros(n, dtype=torch.bool),
        "hero_rank": torch.zeros(n, dtype=torch.int64),
        "attack_in_reach_tick": torch.tensor(counts, dtype=torch.int32),
    }


def test_term_names_append_new_terms_last():
    """Appended, not inserted: `term_means` reports in this order, so a TensorBoard run keeps its
    existing curves. `gadget_hit` (tests/test_gadget_hit.py) and `move_reversal`
    (tests/test_move_reversal.py) came after this file's term."""
    assert TERM_NAMES == (
        "damage_dealt", "damage_taken", "hp_healed", "kill", "cube_pickup",
        "survive_per_step", "in_zone_per_step", "win_bonus", "death_penalty", "rank_bonus",
        "attack_in_reach", "gadget_hit", "move_reversal",
    )


def test_the_term_pays_its_weight_per_counted_attack_and_reports_its_mean():
    fn = ShapedReward(RewardConfig(**_ONLY_REACH), track_terms=True)
    reward = fn({}, _info([1, 0, 1]), SimpleNamespace(n_entities=3))
    assert reward.tolist() == pytest.approx([0.05, 0.0, 0.05])
    assert fn.term_means() == pytest.approx({"attack_in_reach": 0.1 / 3})


def test_off_by_default_and_then_never_read():
    """A hand-built RewardConfig keeps its meaning, and an `info` without the key (any caller of
    compute_info that predates it) is not a KeyError."""
    assert RewardConfig().attack_in_reach == 0.0
    fn = ShapedReward(RewardConfig(**{**_ONLY_REACH, "attack_in_reach": 0.0}), track_terms=True)
    info = _info([1, 1, 1])
    del info["attack_in_reach_tick"]
    assert fn({}, info, SimpleNamespace(n_entities=3)).tolist() == [0.0, 0.0, 0.0]
    assert fn.term_means() == {}


def test_train_yaml_ships_it_at_0_05():
    assert load_train_config(CONFIGS / "train.yaml").reward.attack_in_reach == 0.05


def test_through_the_env_an_in_reach_attack_pays_0_05_and_the_idle_decision_after_it_0():
    fn = ShapedReward(RewardConfig(**_ONLY_REACH), track_terms=False)
    env, override = _duel([11.5, 10.0], reward_fn=fn)
    _obs, reward, *_ = _act(env, override, _ATTACK)
    assert reward.tolist() == pytest.approx([0.05])
    _obs, reward, *_ = _act(env, override, 0)
    assert reward.tolist() == [0.0]


def test_the_flag_and_its_sum_are_sync_free_on_cuda():
    """Every env attacks an enemy 1.5 tiles away on every decision under
    `set_sync_debug_mode("error")`, so a data-dependent branch anywhere on the new path (the
    reach, the `live` gate, the int32 sum) raises. The scene is re-armed between decisions with
    device-side copies; the count is accumulated ON DEVICE and read once, after the window."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    n_envs, decisions = 64, 30
    env = _env(action_repeat=5, n_envs=n_envs, device="cuda", debug_checks=False)  # check_invariants syncs
    st = env.state
    E = env.cfg.n_entities
    assert E == 3
    action = torch.tensor([[0, _ATTACK]], dtype=torch.int64, device="cuda").repeat(n_envs, 1)
    override = torch.zeros(n_envs, E, 2, dtype=torch.int64, device="cuda")
    override[:, 0, 0] = -1
    scene = torch.tensor([[10.0, 10.0], [11.5, 10.0], _FAR], device="cuda").expand(n_envs, E, 2)
    counted = torch.zeros((), dtype=torch.int64, device="cuda")

    def _arm():
        st.ent_pos.copy_(scene)
        st.ent_facing.zero_()
        st.ent_attack_cd.zero_()
        st.ent_dash_t.zero_()
        st.ent_ammo.fill_(3.0)
        st.ent_hp.copy_(st.ent_max_hp)

    for _ in range(3):                           # warm-up: lazy first-call allocations may sync
        _arm()
        env.step(action, override)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(decisions):
            _arm()
            *_, info = env.step(action, override)
            counted.add_(info["attack_in_reach_tick"].sum())
    finally:
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("default")

    assert int(counted) == 1920                  # 64 envs x 30 decisions, every one in reach
