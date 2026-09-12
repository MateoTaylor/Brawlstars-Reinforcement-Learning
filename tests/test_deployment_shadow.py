"""The shadow hero state. BRAWL_DEPLOYMENT_DESIGN.md 6.3.

The load-bearing test here is `test_the_shadow_matches_the_sim_*`: it drives a real `BrawlVecEnv`
and a `ShadowHero` off ONE scripted action stream and asserts they agree field for field, tick for
tick. Everything the module claims -- the phase order, the reload gate, the `_held` fire drop, the
long-dash stopwatch, float32 -- is a claim about matching the sim, so the sim is what checks it.
The bots are overridden to idle and given absurd HP so nothing but the hero's own actions can move
any of the compared fields.

The rest pin failure modes the sim cannot reach because they do not exist there: a CV reading that
disagrees, a reader that returns None, a super spend the next frame has not caught up with yet.
"""
import math
from pathlib import Path
from random import Random

import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.env import BrawlVecEnv
from brawl_deployment.control.buttons import ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER
from brawl_deployment.perception.shadow import (
    AMMO_TOLERANCE, AMMO_TOLERANCE_UNPAINTED, DESYNC, DESYNC_GRACE_SECONDS, GRACE, NO_READ,
    OK, SUSPECT, ShadowHero, ShadowParams,
)
from brawl_vision.object_detection.hp_detection.hero_bars import AmmoReading, SuperReading

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TINY = yaml.safe_load((CONFIGS / "presets" / "debug_tiny.yaml").read_text())


def _params():
    return ShadowParams.load()


def _shadow(**kw):
    return ShadowHero(_params(), **kw)


def _ammo(value, whole=None):
    """An `AmmoReading` carrying one number. The other fields are diagnostics `check_ammo` does
    not read, so they are filled plausibly rather than faithfully."""
    return AmmoReading(ammo=value, frac=value / 3.0, whole=whole if whole is not None else int(value),
                       slots=(1.0, 1.0, 1.0), row=100, pip_px=34)


def _super(charge, ready):
    return SuperReading(charge=charge, ready=ready,
                        state="ready" if ready else ("empty" if charge == 0 else "charging"),
                        track_px=118, row=120)


# ---------------------------------------------------------------------------
# params
# ---------------------------------------------------------------------------

def test_the_params_are_mortis_own_block_not_a_copy_of_it():
    p = _params()
    block = yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())["hero_mortis"]
    for name in ("max_ammo", "reload_seconds", "attack_cooldown", "dash_distance",
                 "dash_duration", "long_dash_seconds", "long_dash_multiplier",
                 "super_charge_hits", "move_speed"):
        assert getattr(p, name) == block[name], name


def test_a_randomized_range_refuses_to_load(tmp_path):
    """`config._resolve_value` samples a two-element list per environment. Dead reckoning against
    a value the sim randomizes is not dead reckoning, so this fails loudly instead of picking an
    end."""
    spec = {"hero_x": {**yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())["hero_mortis"]}}
    spec["hero_x"]["attack_cooldown"] = [0.3, 0.4]
    path = tmp_path / "brawlers.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="randomized range"):
        ShadowParams.load("hero_x", path)


def test_a_brawler_without_a_dash_names_the_field_it_is_missing(tmp_path):
    spec = {"bot_x": {"max_ammo": 3, "reload_seconds": 2.0, "move_speed": 2.4}}
    path = tmp_path / "brawlers.yaml"
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(KeyError, match="attack_cooldown"):
        ShadowParams.load("bot_x", path)


# ---------------------------------------------------------------------------
# parity with the sim
# ---------------------------------------------------------------------------

def _sim_env(action_repeat):
    """One env, tiny map, zone off, bots unkillable. `autoreset=False` so an episode that somehow
    ends is a visible assertion failure rather than a silent state reset mid-comparison."""
    overrides = {name: dict(section) if isinstance(section, dict) else section
                 for name, section in TINY.items()}
    # The preset's 300-tick episode truncates at decision 60 when action_repeat is 5, which is one
    # decision short of the script -- the cap is about keeping training rollouts small, not about
    # anything this test exercises.
    overrides["sim"] = {**overrides.get("sim", {}), "action_repeat": action_repeat,
                        "max_episode_steps": 2000}
    cfg = load_config(CONFIGS / "default.yaml", overrides=overrides)

    spec = {**yaml.safe_load((CONFIGS / "default.yaml").read_text()),
            **yaml.safe_load((CONFIGS / "brawlers.yaml").read_text())}
    for kind, block in spec.items():
        if isinstance(block, dict) and "base_hp" in block:
            block["base_hp"] = 1e7
    return BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0, spec=spec,
                       autoreset=False, verbose=False)


def _script(n_decisions, seed=0):
    """Random bins and fire, plus one deliberate 20-decision idle stretch (5.0 s, past
    `long_dash_seconds: 4.5`) followed by a dash, so the long dash is actually exercised."""
    rng = Random(seed)
    plan = [(rng.randrange(0, 17), ATTACK_FIRE if rng.random() < 0.3 else ATTACK_NONE)
            for _ in range(n_decisions - 21)]
    plan += [(3, ATTACK_NONE)] * 20 + [(3, ATTACK_FIRE)]
    return plan


@pytest.mark.parametrize("action_repeat", [1, 5])
def test_the_shadow_matches_the_sim_field_for_field(action_repeat):
    env = _sim_env(action_repeat)
    env.reset()
    shadow = _shadow()
    shadow.reset(facing=float(env.state.ent_facing[0, 0]))

    # -1 in column 0 is `_override_phase`'s "no override for this slot", so the hero keeps its own
    # action while every bot is pinned to idle-and-not-firing.
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[0, 0, 0] = -1

    for decision, (move, attack) in enumerate(_script(60)):
        # The deployed policy is masked before it acts, so the shadow's mask is what gates the
        # stream here too -- see the module docstring on why it is a strict subset of the sim's.
        if attack == ATTACK_FIRE and not shadow.attack_mask()[1]:
            attack = ATTACK_NONE
        shadow.act(move, attack)
        obs, _, terminated, truncated, _ = env.step(
            torch.tensor([[move, attack]], dtype=torch.int64), override)
        assert not bool(terminated[0]) and not bool(truncated[0]), f"episode ended at {decision}"
        shadow.advance(action_repeat * env.cfg.dt)

        h = obs["hero"]
        s = shadow.observe()
        where = f"decision {decision} (move={move}, attack={attack})"
        # Exact, not approximate: these are the same float32 operations in the same order, and a
        # tolerance here would hide precisely the one-tick precision bug the module documents.
        assert s["ammo"] == float(h["ammo"][0]), where
        assert s["attack_cd"] == float(h["attack_cd"][0]), where
        assert s["dash_t"] == float(h["dash_t"][0]), where
        assert s["attack_idle_t"] == float(h["attack_idle_t"][0]), where
        assert s["ammo_whole"] == int(h["ammo_whole"][0]), where
        assert s["can_attack"] == bool(h["can_attack"][0]), where
        assert s["dashing"] == bool(h["dashing"][0]), where
        assert s["invuln"] == bool(h["invuln"][0]), where
        assert s["long_dash_ready"] == bool(h["long_dash_ready"][0]), where
        assert s["long_dash_frac"] == pytest.approx(float(h["long_dash_frac"][0]), abs=1e-6), where
        # Direction vectors are cos/sin, so torch's and libm's last ulp may differ.
        for i in (0, 1):
            assert s["facing_vec"][i] == pytest.approx(float(h["facing_vec"][0, i]), abs=1e-6), where
            assert s["dash_dir"][i] == pytest.approx(float(h["dash_dir"][0, i]), abs=1e-6), where


def test_a_super_costs_the_sim_and_the_shadow_the_same_thing():
    """The super branch is the one attack the parity script cannot reach -- charge comes from
    landing hits, which is not in the action stream. Seeded by hand on both sides instead: it must
    take the cooldown and the long-dash reset while leaving the clip alone."""
    env = _sim_env(5)
    env.reset()
    shadow = _shadow()
    shadow.reset(facing=float(env.state.ent_facing[0, 0]))
    override = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    override[0, 0, 0] = -1

    env.state.ent_super_charge[0, 0] = env.params.super_charge_hits[0, 0]
    shadow.set_super(_super(1.0, True))
    assert shadow.attack_mask()[2]

    shadow.act(4, ATTACK_SUPER)
    obs, _, _, _, _ = env.step(torch.tensor([[4, ATTACK_SUPER]], dtype=torch.int64), override)
    shadow.advance(5 * env.cfg.dt)

    h, s = obs["hero"], shadow.observe()
    assert float(h["super_charge"][0]) == 0.0
    assert s["super_charge_frac"] == 0.0
    for field in ("ammo", "attack_cd", "dash_t", "attack_idle_t"):
        assert s[field] == float(h[field][0]), field
    assert s["ammo"] == 3.0


def test_the_scripted_run_actually_exercises_a_dash_and_a_long_dash():
    """Guards the parity test above from passing vacuously: a stream that never fires would agree
    with anything."""
    shadow = _shadow()
    dashes = long_dashes = 0
    for move, attack in _script(60):
        if attack == ATTACK_FIRE and shadow.attack_mask()[1]:
            if shadow.long_dash_ready:
                long_dashes += 1
            dashes += 1
            shadow.act(move, attack)
        else:
            shadow.act(move, ATTACK_NONE)
        shadow.advance(5 * 0.05)
    assert dashes >= 5
    assert long_dashes == 1


def test_one_decision_is_at_most_one_dash_however_many_sub_ticks_it_spans():
    """`env._held` zeroes the fire column on sub-ticks 2..K. Here the same rule falls out of a
    single queued attack, and the check is that five sub-ticks spend one ammo, not five."""
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(5 * 0.05)
    assert shadow.observe()["ammo"] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# the tick, in isolation
# ---------------------------------------------------------------------------

def test_the_cooldown_blocks_reload_for_exactly_seven_ticks():
    """`attack_cooldown: 0.35` at `dt: 0.05`. Seven is what float32 gives and what the sim does;
    float64 gives eight, because 0.35 - 7 x 0.05 leaves a positive residue there. If this ever
    reads eight, the timers stopped being float32."""
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    assert shadow.observe()["ammo"] == pytest.approx(2.0, abs=1e-6)

    blocked = 0
    while shadow.observe()["attack_cd"] > 0:
        before = shadow.observe()["ammo"]
        shadow.advance(0.05)
        if shadow.observe()["ammo"] == before:
            blocked += 1
    assert blocked == 7
    assert shadow.observe()["attack_cd"] == 0.0


def test_ammo_regen_resumes_the_tick_the_cooldown_clears():
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.35 + 0.05)          # the firing tick plus the six it blocks, then one more
    assert shadow.observe()["attack_cd"] == 0.0
    before = shadow.observe()["ammo"]
    shadow.advance(0.05)
    assert shadow.observe()["ammo"] > before


def test_the_i_frames_outlast_the_dash_by_exactly_one_sub_tick():
    """Both are seeded from `dash_duration: 0.30` on the same tick, and they do not end together.
    `dash_t` is decremented in phase 8 of the tick that SET it; `invuln_t` waits for phase 2 of
    the next one. So `dashing` reads true for six sub-ticks (0.30 s, its duration) and `invuln`
    for seven. Verified against the sim by the parity test; pinned here because it is the kind of
    off-by-one a rewrite would quietly round away."""
    shadow = _shadow()
    shadow.act(5, ATTACK_FIRE)
    dashing, invuln = [], []
    for _ in range(10):
        shadow.advance(0.05)
        dashing.append(shadow.observe()["dashing"])
        invuln.append(shadow.observe()["invuln"])
    assert dashing == [True] * 6 + [False] * 4
    assert invuln == [True] * 7 + [False] * 3
    assert shadow.observe()["dash_dir"] == (0.0, 0.0)


def test_the_long_dash_is_unavailable_for_the_first_long_dash_seconds_of_a_match():
    """`attack_idle_t` starts at zero at spawn, so the ability is genuinely cold at the gate --
    the same as in training. A shadow that seeded it high would hand the policy a 5.34-tile dash
    on the first decision of every match."""
    shadow = _shadow()
    assert not shadow.long_dash_ready
    shadow.advance(4.45)
    assert not shadow.long_dash_ready
    assert shadow.long_dash_frac == pytest.approx(4.45 / 4.5, abs=1e-3)
    shadow.advance(0.10)
    assert shadow.long_dash_ready
    assert shadow.long_dash_frac == 1.0


def test_attacking_is_the_only_thing_that_resets_the_long_dash_stopwatch():
    shadow = _shadow()
    shadow.advance(5.0)
    assert shadow.long_dash_ready
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    assert not shadow.long_dash_ready
    assert shadow.observe()["attack_idle_t"] == 0.0


def test_a_charged_dash_covers_twice_the_ground_in_the_travel_budget():
    """The budget is the only place `long_dash_multiplier` shows up -- every timer field is
    identical between a normal and a charged dash."""
    def budget(idle_seconds):
        shadow = _shadow()
        shadow.advance(idle_seconds)
        shadow.reset_travel()
        shadow.act(0, ATTACK_FIRE)      # idle bin: dashes along facing, and walks no distance
        shadow.advance(0.35)
        return shadow.travel_budget

    short, long = budget(1.0), budget(5.0)
    assert short == pytest.approx(2.67, abs=0.02)
    assert long == pytest.approx(5.34, abs=0.02)


def test_the_travel_budget_counts_a_walk_at_move_speed():
    shadow = _shadow()
    shadow.act(1, ATTACK_NONE)
    shadow.advance(1.0)
    assert shadow.travel_budget == pytest.approx(2.73, abs=0.01)
    shadow.reset_travel()
    assert shadow.travel_budget == 0.0


def test_an_idle_bin_covers_no_ground():
    shadow = _shadow()
    shadow.act(0, ATTACK_NONE)
    shadow.advance(1.0)
    assert shadow.travel_budget == 0.0


def test_advance_banks_the_remainder_instead_of_dropping_it():
    """A loop that never lands exactly on `dt` must not lose a tick per iteration."""
    shadow = _shadow()
    assert shadow.advance(0.02) == 0
    assert shadow.advance(0.02) == 0
    assert shadow.advance(0.02) == 1
    assert shadow.ticks == 1
    assert shadow.advance(0.0) == 0
    assert shadow.advance(-1.0) == 0


def test_a_long_stall_still_expires_the_cooldown_it_really_outlasted():
    """Real time passed, so the game's timers really did run. Capping the advance would leave the
    shadow claiming a cooldown the game has long since finished."""
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    assert shadow.advance(30.0) == 600
    assert shadow.observe()["attack_cd"] == 0.0
    assert shadow.observe()["ammo"] == 3.0


def test_facing_tracks_the_move_bin_and_survives_the_idle_bin():
    shadow = _shadow()
    shadow.act(5, ATTACK_NONE)            # bin 5 -> (5-1) * 2pi/16 = pi/2
    shadow.advance(0.05)
    assert shadow.observe()["facing"] == pytest.approx(math.pi / 2, abs=1e-6)
    shadow.act(0, ATTACK_NONE)
    shadow.advance(0.05)
    assert shadow.observe()["facing"] == pytest.approx(math.pi / 2, abs=1e-6)


def test_a_dead_hero_cannot_attack_but_its_timers_keep_running():
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    shadow.set_alive(False)
    assert shadow.attack_mask() == (True, False, False)
    assert not shadow.observe()["can_attack"]
    assert shadow.act(1, ATTACK_FIRE) == ATTACK_NONE
    shadow.advance(5.0)
    assert shadow.observe()["ammo"] == pytest.approx(3.0, abs=1e-6)   # reloaded while dead
    assert shadow.observe()["attack_cd"] == 0.0


# ---------------------------------------------------------------------------
# act / the mask
# ---------------------------------------------------------------------------

def test_act_refuses_what_the_mask_refuses_and_says_so():
    shadow = _shadow()
    assert shadow.act(1, ATTACK_FIRE) == ATTACK_FIRE
    shadow.advance(0.05)
    assert shadow.act(1, ATTACK_FIRE) == ATTACK_NONE      # mid-dash, mid-cooldown
    assert shadow.act(1, ATTACK_SUPER) == ATTACK_NONE     # never charged


def test_a_second_act_before_a_sub_tick_cannot_queue_a_second_attack():
    """The device would swallow the second tap -- `Buttons.tap` releases a still-held contact
    before re-pressing -- so modelling it would spend ammo the game kept."""
    shadow = _shadow()
    assert shadow.act(1, ATTACK_FIRE) == ATTACK_FIRE
    assert shadow.act(2, ATTACK_FIRE) == ATTACK_NONE
    shadow.advance(0.05)
    assert shadow.observe()["ammo"] == pytest.approx(2.0, abs=1e-6)


def test_an_empty_clip_masks_the_attack_but_not_the_super():
    """`action_mask`'s super term deliberately omits the ammo test: a super costs charge."""
    shadow = _shadow()
    for _ in range(3):
        shadow.act(1, ATTACK_FIRE)
        shadow.advance(0.40)
    assert shadow.observe()["ammo"] < 1.0
    shadow.set_super(_super(1.0, True))
    no_fire, fire, super_ok = shadow.attack_mask()
    assert not fire and super_ok


def test_the_mask_is_never_wider_than_the_sims_post_timer_one():
    """The deployed mask is read before phase 2's timers and the sim's after them. Every term
    moves one way only, so this can be stale but never permissive."""
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    for _ in range(20):
        before = shadow.attack_mask()
        shadow.advance(0.05)
        after = shadow.attack_mask()
        assert not (before[1] and not after[1])


# ---------------------------------------------------------------------------
# the super
# ---------------------------------------------------------------------------

def test_the_super_is_only_offered_once_cv_says_it_is_charged():
    shadow = _shadow()
    assert not shadow.super_ready
    shadow.set_super(_super(0.6, False))
    assert not shadow.super_ready
    assert shadow.observe()["super_charge_frac"] == pytest.approx(0.6)
    shadow.set_super(_super(1.0, True))
    assert shadow.super_ready
    assert shadow.attack_mask()[2]


def test_firing_the_super_spends_the_meter_immediately_and_costs_no_ammo():
    shadow = _shadow()
    shadow.set_super(_super(1.0, True))
    assert shadow.act(1, ATTACK_SUPER) == ATTACK_SUPER
    shadow.advance(0.05)
    s = shadow.observe()
    assert s["ammo"] == pytest.approx(3.0, abs=1e-6)
    assert s["super_charge_frac"] == 0.0
    assert not s["super_ready"]
    assert s["attack_cd"] > 0


def test_a_spent_super_is_not_re_armed_by_a_stale_ready_read():
    """The frame in flight when we tapped still shows magenta. Believing it taps a super we no
    longer have -- and because a super costs charge rather than ammo, the canary cannot see it."""
    shadow = _shadow()
    shadow.set_super(_super(1.0, True))
    shadow.act(1, ATTACK_SUPER)
    shadow.advance(0.05)
    shadow.set_super(_super(1.0, True))          # the stale frame
    assert not shadow.super_ready
    assert not shadow.attack_mask()[2]


def test_the_spend_latch_clears_on_the_first_read_that_agrees():
    shadow = _shadow()
    shadow.set_super(_super(1.0, True))
    shadow.act(1, ATTACK_SUPER)
    shadow.advance(0.05)
    shadow.set_super(_super(0.0, False))         # the frame that caught up
    assert not shadow.super_ready
    shadow.set_super(_super(1.0, True))          # charged again, legitimately
    assert shadow.super_ready


def test_a_stale_super_read_fails_closed_while_holding_its_fraction():
    shadow = _shadow()
    shadow.set_super(_super(1.0, True))
    shadow.advance(0.95)
    assert shadow.super_ready
    shadow.advance(0.10)
    assert not shadow.super_ready
    assert shadow.observe()["super_charge_frac"] == pytest.approx(1.0)


def test_a_missing_super_read_changes_nothing():
    shadow = _shadow()
    shadow.set_super(_super(0.8, False))
    shadow.set_super(None)
    assert shadow.observe()["super_charge_frac"] == pytest.approx(0.8)


def test_the_super_mask_still_obeys_the_cooldown_and_the_dash():
    """`hero.super_ready` in the observation is the charge test alone; the mask adds the terms
    only the shadow knows about."""
    shadow = _shadow()
    shadow.set_super(_super(1.0, True))
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    assert shadow.observe()["super_ready"]
    assert not shadow.attack_mask()[2]


# ---------------------------------------------------------------------------
# the ammo canary
# ---------------------------------------------------------------------------

def test_reader_noise_does_not_trip_the_canary():
    """A full pip reads 0.94-1.03 and 98.3% of reads land within 0.15 of an integer, against a
    0.5 tolerance."""
    shadow = _shadow()
    shadow.advance(1.0)
    for value in (3.0, 2.94, 3.03, 2.86, 3.15):
        verdict = shadow.check_ammo(_ammo(value))
        assert verdict.status == OK, value
        assert not verdict.tripped


def test_a_dropped_dash_trips_only_after_three_consecutive_strikes():
    """One bad read must not force a resync: the resync reseeds every unobservable timer."""
    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.4)
    shadow.act(1, ATTACK_FIRE)                          # TWO -- see the one-shot test below
    shadow.advance(DESYNC_GRACE_SECONDS + 0.1)          # past the grace window
    assert shadow.observe()["ammo"] < 2.5

    statuses = [shadow.check_ammo(_ammo(3.0)).status for _ in range(3)]
    assert statuses == [SUSPECT, SUSPECT, DESYNC]


def test_the_error_says_which_way_the_shadow_is_wrong():
    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(1.0)
    verdict = shadow.check_ammo(_ammo(3.0))
    assert verdict.error > 0                  # the game has ammo we thought we spent


def test_one_good_read_clears_the_strikes():
    shadow = _shadow()
    shadow.advance(1.0)
    assert shadow.check_ammo(_ammo(1.0)).status == SUSPECT
    assert shadow.check_ammo(_ammo(3.0)).status == OK
    assert shadow.strikes == 0


def test_a_missing_read_neither_strikes_nor_clears():
    """12% of frames return None. That is not evidence in either direction."""
    shadow = _shadow()
    shadow.advance(1.0)
    assert shadow.check_ammo(_ammo(1.0)).status == SUSPECT
    assert shadow.check_ammo(None).status == NO_READ
    assert shadow.strikes == 1


def test_a_shadow_ABOVE_the_read_is_forgiven_up_to_a_whole_pip():
    """`read_ammo` sums painted fill, and the second live deployment showed it detects a
    RECHARGING pip only intermittently -- 2.00 -> 2.56 -> 2.00 inside half a second while the true
    value climbed monotonically. Unpainted pixels can only lose ammo, never invent it, so a shadow
    sitting up to one pip above the read is exactly what a missed partial looks like.

    A symmetric 0.5 sits below that sensor's own noise floor (p90 of |error| was 0.64) and
    manufactured all three of that run's resyncs."""
    assert AMMO_TOLERANCE_UNPAINTED > AMMO_TOLERANCE

    shadow = _shadow()
    shadow.advance(10.0)                       # full clip, no attack, well outside grace
    shadow.ammo = 2.6
    for _ in range(5):
        assert shadow.check_ammo(_ammo(2.0)).status == OK      # the read missed the partial
    assert shadow.strikes == 0


def test_a_shadow_BELOW_the_read_keeps_the_TIGHT_bound_because_that_is_the_real_failure():
    """The asymmetry must not cost sensitivity in the direction the canary exists for. A read
    ABOVE the shadow means the game is holding ammo we already spent -- our taps are not landing --
    and no amount of missing paint can produce that, because missing paint reads LOW."""
    shadow = _shadow()
    shadow.advance(10.0)
    shadow.ammo = 2.0
    statuses = [shadow.check_ammo(_ammo(2.6)).status for _ in range(3)]
    assert statuses == [SUSPECT, SUSPECT, DESYNC]


def test_the_grace_window_covers_the_frame_that_was_captured_before_the_tap():
    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)
    verdict = shadow.check_ammo(_ammo(3.0))    # the pre-tap frame, a full pip out
    assert verdict.status == GRACE
    assert shadow.strikes == 0


def test_the_grace_window_outlasts_the_measured_tap_to_pip_delay():
    """The first live deployment measured 0.50-1.25 s from a modelled tap to the bar showing the
    spend, and the window was 0.30 -- so every burst manufactured a desync, and all four of that
    run's resyncs were false. A window shorter than the delay does not merely waste the
    fail-closed budget: the resync ADOPTS the CV reading, so a trip inside the delay writes a
    stale ammo value into the shadow and causes the next trip. Resync 2 of run1 was resync 1's
    doing.

    1.25 s is the worst clean observation from that run. Pinning it here means shortening the
    window has to argue with the measurement rather than with a comment."""
    assert DESYNC_GRACE_SECONDS > 1.25

    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)                 # shadow spends immediately; the game has not yet
    shadow.advance(1.25)
    assert shadow.check_ammo(_ammo(3.0)).status == GRACE
    assert shadow.strikes == 0


def test_the_canary_is_live_again_through_the_reload():
    """Blind during the delay, awake for the rest. Mortis's reload is 2.50 s per pip (measured,
    6.15), so the window has to expire well inside it or the canary never sees the reload it exists
    to check."""
    assert DESYNC_GRACE_SECONDS < _params().reload_seconds, (
        "a window longer than one reload would mute it forever")
    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.4)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(DESYNC_GRACE_SECONDS + 0.05)
    assert shadow.check_ammo(_ammo(3.0)).status == SUSPECT


def test_a_SINGLE_unlanded_shot_is_below_the_canary_and_that_is_the_price_of_the_window():
    """**A limitation, asserted so it is a known one.** Widening the grace to the measured 1.5 s
    tap-to-pip delay costs single-shot sensitivity: by the time the window expires, a lone missing
    spend has regrown to within about half a pip of agreement, and it never escalates.

    Pinned as "never reaches DESYNC", not as "reads OK at the first check", because the first
    check is a coin-flip on the kit constants. At the old `reload_seconds: 2.25` the gap was 0.47
    at that check, just inside `AMMO_TOLERANCE`. At the measured 2.50 (6.15) it is 0.54, just
    outside: one SUSPECT, then back inside the tolerance a decision later, two strikes short of
    `DESYNC_STRIKES`.

    That is the right trade rather than a regression. At 0.30 s the canary was not detecting
    dropped shots either -- it was firing on TRUE readings the game had not drawn yet, four times
    in the 41 s of run1, and each false trip wrote a stale value into the shadow. And the failure
    the canary actually exists for -- input stopping altogether -- is never one shot; two are
    already loud, which the test above pins.

    If single-shot detection is ever wanted, the fix is not a shorter window: it is comparing CV
    against the shadow as it was one delay ago, which needs a history this class does not keep."""
    shadow = _shadow()
    shadow.advance(1.0)
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(DESYNC_GRACE_SECONDS + 0.05)

    checks = []
    for _ in range(8):                         # two seconds of 4 Hz checks, CV still full
        checks.append(shadow.check_ammo(_ammo(3.0)))
        shadow.advance(0.25)
    assert all(c.status != DESYNC for c in checks)
    assert max(c.strikes for c in checks) < 2
    assert checks[1].status == OK and shadow.observe()["ammo"] > 2.5


# ---------------------------------------------------------------------------
# resync
# ---------------------------------------------------------------------------

def test_resync_takes_ammo_from_cv_and_reseeds_everything_else_to_not_ready():
    shadow = _shadow()
    shadow.advance(5.0)                        # long dash charged
    shadow.act(1, ATTACK_FIRE)
    shadow.advance(0.05)                       # mid-dash, invulnerable, on cooldown
    shadow.set_super(_super(1.0, True))
    assert shadow.observe()["dashing"] and shadow.observe()["invuln"]

    shadow.resync(_ammo(3.0))
    s = shadow.observe()
    assert s["ammo"] == pytest.approx(3.0)
    assert not s["dashing"] and s["dash_dir"] == (0.0, 0.0)
    assert not s["invuln"]
    assert not s["long_dash_ready"] and s["long_dash_frac"] == 0.0
    assert not s["super_ready"] and s["super_charge_frac"] == 0.0
    assert not s["can_attack"]                 # the full cooldown, not zero
    assert shadow.attack_mask() == (True, False, False)
    assert shadow.strikes == 0 and shadow.desyncs == 1


def test_resync_never_takes_more_ammo_than_the_clip_holds():
    shadow = _shadow()
    shadow.resync(_ammo(3.7))                  # a reader over-count, measured at up to 3.09
    assert shadow.observe()["ammo"] == pytest.approx(3.0)


def test_resync_without_a_reading_still_reseeds_the_timers():
    shadow = _shadow()
    shadow.advance(5.0)
    shadow.resync()
    assert not shadow.long_dash_ready
    assert shadow.observe()["attack_cd"] > 0


def test_a_queued_attack_does_not_survive_a_resync():
    shadow = _shadow()
    shadow.act(1, ATTACK_FIRE)
    shadow.resync()
    shadow.advance(0.05)
    assert shadow.observe()["ammo"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# the contract with the observation spec
# ---------------------------------------------------------------------------

# The `self` group fields that come from CV or the wall clock rather than from here -- 6.3's table.
_NOT_SHADOWED = {"hero.pos_norm", "hero.vel", "hero.hp", "hero.in_bush", "hero.in_zone",
                 "meta.time_frac", "meta.n_enemies_alive"}


def test_observe_covers_every_self_field_the_deploy_spec_asks_of_it():
    """If a field is added to `agent_obs_deploy.yaml`'s `self` group, it either gets a CV supplier
    (and joins the set above) or it has to come from here. Silently returning neither is how a
    column of zeros reaches a policy trained to trust it."""
    spec = yaml.safe_load((CONFIGS / "agent_obs_deploy.yaml").read_text())
    group = next(g for g in spec["groups"] if g["name"] == "self")
    keys = set(_shadow().observe())
    for field in group["fields"]:
        if field in _NOT_SHADOWED:
            continue
        assert field.startswith("hero."), field
        assert field.split(".", 1)[1] in keys, field
