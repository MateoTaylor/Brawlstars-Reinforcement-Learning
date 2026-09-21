from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never open a GUI window/event loop during tests

import torch
import yaml
from matplotlib.colors import to_rgba

from brawl_sim.config import load_config
from brawl_sim.core import geometry as geo
from brawl_sim.env import BrawlVecEnv
from brawl_sim.maps.loader import build_map_bank
from scripts.play_manual import ManualPlaySession

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


class _FakeKeyEvent:
    def __init__(self, key):
        self.key = key


class _FakeButtonEvent:
    def __init__(self, button):
        self.button = button


def _session(overrides=None, seed=0):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=seed)
    bank = build_map_bank(cfg, device="cpu")
    return ManualPlaySession(env, bank, cfg), env, cfg


# ---- held-key state -> action_tensor -----------------------------------------------------------

def test_no_keys_held_gives_the_idle_action():
    session, env, cfg = _session()
    action = session.action_tensor()
    assert action.shape == (1, 2)
    assert int(action[0, 0]) == 0
    assert int(action[0, 1]) == 0


def test_holding_d_selects_the_bin_nearest_positive_x():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("d"))
    action = session.action_tensor()
    expected_bin = int(geo.bin_from_dir(torch.tensor([[1.0, 0.0]]), cfg.n_move_bins)[0]) + 1
    assert int(action[0, 0]) == expected_bin
    assert int(action[0, 0]) != 0


def test_holding_w_selects_the_bin_nearest_up_on_screen():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("w"))
    action = session.action_tensor()
    # "up" on screen is -y (origin="upper" / y-down convention, same as render/viewer.py)
    expected_bin = int(geo.bin_from_dir(torch.tensor([[0.0, -1.0]]), cfg.n_move_bins)[0]) + 1
    assert int(action[0, 0]) == expected_bin


def test_arrow_keys_are_equivalent_to_wasd():
    session_wasd, _, _ = _session()
    session_wasd.on_key_press(_FakeKeyEvent("a"))
    session_arrows, _, _ = _session()
    session_arrows.on_key_press(_FakeKeyEvent("left"))
    assert int(session_wasd.action_tensor()[0, 0]) == int(session_arrows.action_tensor()[0, 0])


def test_diagonal_wd_combines_to_the_up_right_bin():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("w"))
    session.on_key_press(_FakeKeyEvent("d"))
    action = session.action_tensor()
    expected_bin = int(geo.bin_from_dir(torch.tensor([[1.0, -1.0]]), cfg.n_move_bins)[0]) + 1
    assert int(action[0, 0]) == expected_bin


def test_releasing_a_key_removes_it_from_held_state():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("d"))
    assert int(session.action_tensor()[0, 0]) != 0
    session.on_key_release(_FakeKeyEvent("d"))
    assert int(session.action_tensor()[0, 0]) == 0


def test_uppercase_key_names_are_treated_case_insensitively():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("D"))
    assert int(session.action_tensor()[0, 0]) != 0


# ---- fire: space and left-click, key/button symmetry -------------------------------------------

def test_space_sets_fire_and_release_clears_it():
    session, env, cfg = _session()
    assert int(session.action_tensor()[0, 1]) == 0
    session.on_key_press(_FakeKeyEvent(" "))
    assert int(session.action_tensor()[0, 1]) == 1
    session.on_key_release(_FakeKeyEvent(" "))
    assert int(session.action_tensor()[0, 1]) == 0


def test_left_click_sets_fire_and_release_clears_it():
    session, env, cfg = _session()
    session.on_button_press(_FakeButtonEvent(1))
    assert int(session.action_tensor()[0, 1]) == 1
    session.on_button_release(_FakeButtonEvent(1))
    assert int(session.action_tensor()[0, 1]) == 0


def test_right_click_does_not_fire():
    session, env, cfg = _session()
    session.on_button_press(_FakeButtonEvent(3))
    assert int(session.action_tensor()[0, 1]) == 0


# ---- gadget: `g` is attack-column value 3 (SIM_OVERHAUL Step G3) ---------------------------------

def test_g_sets_the_gadget_value_and_release_clears_it():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("g"))
    assert session.action_tensor().tolist() == [[0, 3]]
    session.on_key_release(_FakeKeyEvent("G"))          # case-insensitive, like every other key
    assert session.action_tensor().tolist() == [[0, 0]]


def test_g_alone_is_sent_raw_even_while_the_gadget_is_on_cooldown():
    """Not pre-filtered: a masked 3 is a silent no-op inside `env.step`, same contract as fire."""
    session, env, cfg = _session()
    env.state.ent_gadget_cd[0, 0] = 9.0
    session.on_key_press(_FakeKeyEvent("g"))
    assert int(session.action_tensor()[0, 1]) == 3


def test_g_with_space_takes_the_column_only_while_the_gadget_is_legal():
    """One column, one value. A tap of `g` spans several 20 Hz ticks; after the throw the gadget
    is masked for 18 s, and those ticks must go back to the dash the player is also holding."""
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent(" "))
    session.on_key_press(_FakeKeyEvent("g"))
    assert int(session.action_tensor()[0, 1]) == 3

    env.state.ent_gadget_cd[0, 0] = 17.0
    assert int(session.action_tensor()[0, 1]) == 1


def test_pressing_g_throws_a_spinner_and_the_title_counts_the_cooldown_down():
    session, env, cfg = _session()
    assert "gadget=READY" in session.viewer.title.get_text()

    # The spinner's 4 ticks of flight fit inside the one 5-tick decision, so it is already gone
    # when `tick()` returns: count it per SUB-tick instead. Projectile kind 7 is
    # `Proj.GADGET_SPINNER`, as a literal. Without this the test passed on an env that started
    # the cooldown and threw nothing (Step G3 review).
    spinners, owners = [], set()

    def _count(e):
        live = e.state.prj_alive[0] & (e.state.prj_kind[0] == 7)
        spinners.append(int(live.sum()))
        owners.update(e.state.prj_owner[0][live].tolist())

    env.tick_hook = _count
    session.on_key_press(_FakeKeyEvent("g"))
    session.tick()
    env.tick_hook = None

    assert spinners == [1, 1, 1, 0, 0]          # in flight after sub-ticks 1-3, detonated on the 4th
    assert owners == {0}                         # the hero's, not a bot's
    state = env.state
    # 18.0 s set on the first sub-tick, four more sub-ticks (action_repeat 5) of 0.05 s after it.
    assert abs(float(state.ent_gadget_cd[0, 0]) - 17.8) < 1e-4
    assert "gadget=17.8s" in session.viewer.title.get_text()


def test_the_game_window_takes_its_keys_away_from_matplotlibs_default_keymap():
    """Every pyplot figure is born with matplotlib's own key handler connected, and `g` is its
    grid toggle (two `ax.grid` calls and a full redraw while the cursor is over the axes -- which
    it is, since left-click fires) and `s` its save dialog. `connect_input` disconnects it. Real
    `KeyEvent`s through the canvas, not `on_key_press` called by hand: the collision lives in
    what ELSE is connected to the figure, which a direct call cannot see (Step G3 review)."""
    from matplotlib.backend_bases import KeyEvent

    session, env, cfg = _session()
    fig, ax = session.viewer.fig, session.viewer.ax
    fig.canvas.draw()
    x, y = ax.transAxes.transform((0.5, 0.5))           # a cursor in the middle of the map

    grid_calls, saves = [], []
    ax.grid = lambda *args, **kwargs: grid_calls.append(kwargs)

    class _Toolbar:                                      # Agg has none; every GUI backend does
        def save_figure(self, *args):
            saves.append(1)

    def press(key, toolbar=None):
        fig.canvas.toolbar = toolbar
        KeyEvent("key_press_event", fig.canvas, key, x, y)._process()
        fig.canvas.toolbar = None

    # Control: before the session takes the keyboard matplotlib's defaults DO fire, so the
    # unchanged counts below are the disconnect's doing and not a dead observable.
    press("g")
    press("s", toolbar=_Toolbar())
    assert [call["axis"] for call in grid_calls] == ["x", "y"]
    assert saves == [1]
    assert session._held_keys == set()

    session.connect_input(fig)
    press("g")
    press("s", toolbar=_Toolbar())
    assert len(grid_calls) == 2 and saves == [1], "matplotlib still handles the game's keys"
    assert session._held_keys == {"g", "s"}
    assert session.action_tensor().tolist() == [[5, 3]]  # bin 5 is straight down (+y); 3 the gadget


# ---- quit ----------------------------------------------------------------------------------------

def test_q_key_requests_quit():
    session, env, cfg = _session()
    assert not session._quit_requested
    session.on_key_press(_FakeKeyEvent("q"))
    assert session._quit_requested
    # a quit key must not also get treated as a held movement/fire key
    assert not session._held_keys


def test_escape_key_requests_quit():
    session, env, cfg = _session()
    session.on_key_press(_FakeKeyEvent("escape"))
    assert session._quit_requested


# ---- tick: steps the env and refreshes the overlay ----------------------------------------------

def test_tick_steps_the_env_and_updates_the_title_with_mask_and_dash_info():
    session, env, cfg = _session()
    before_step_count = int(env.state.step_count[0])
    session.tick()
    # One session.tick() is one env.step(), i.e. one DECISION -- cfg.action_repeat sim ticks.
    assert int(env.state.step_count[0]) == before_step_count + cfg.action_repeat
    title = session.viewer.title.get_text()
    assert "fire=" in title
    assert "dash=" in title


def test_indicator_is_ready_immediately_after_reset():
    from scripts.play_manual import _INDICATOR_READY
    session, env, cfg = _session()
    assert session.indicator.get_facecolor() == to_rgba(_INDICATOR_READY)


def test_indicator_greys_out_when_out_of_ammo():
    from scripts.play_manual import _INDICATOR_BLOCKED
    session, env, cfg = _session()
    env.state.ent_ammo[0, 0] = 0.0
    session._update_overlay()
    assert session.indicator.get_facecolor() == to_rgba(_INDICATOR_BLOCKED)


def test_indicator_greys_out_while_dashing():
    from scripts.play_manual import _INDICATOR_BLOCKED
    session, env, cfg = _session()
    env.state.ent_dash_t[0, 0] = 0.1
    session._update_overlay()
    assert session.indicator.get_facecolor() == to_rgba(_INDICATOR_BLOCKED)


# ---- feel check: the real Mortis numbers from configs/brawlers.yaml, driven through the exact ---
# ---- input -> action_tensor -> env.step pipeline a human playtester would actually exercise -----

def test_dash_covers_dash_distance_in_dash_duration_and_cancels_the_walk():
    # action_repeat=1: this walks the dash forward ONE SIM TICK AT A TIME to check it covers
    # exactly dash_distance over dash_duration. At the shipped action_repeat=5 a single
    # session.tick() would advance five ticks and the whole ~6-tick dash would resolve inside
    # one or two calls, measuring nothing. The dash physics under test are per-tick and are
    # unaffected by the decision rate.
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"maps": ["open"], "map_selection": "fixed", "fixed_map": "open"},
        "entities": {"n_enemies": 1},
        "sim": {"action_repeat": 1},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    bank = build_map_bank(cfg, device="cpu")
    session = ManualPlaySession(env, bank, cfg)

    # place the hero at the map center, far from any wall, so the dash is never clipped --
    # isolating exactly the number this feel-check cares about (dash_distance/dash_duration).
    center = torch.tensor([cfg.map_w / 2.0, cfg.map_h / 2.0])
    env.state.ent_pos[0, 0] = center
    env.state.ent_ammo[0, 0] = 3.0
    env.state.ent_attack_cd[0, 0] = 0.0

    dash_distance = float(env.params.dash_distance[0, 0])
    dash_duration = float(env.params.dash_duration[0, 0])
    n_ticks = round(dash_duration / cfg.dt)
    # Cross-check that SimParams really did pick these up from configs/brawlers.yaml, by reading
    # the file rather than restating its numbers. The literal `5.0` this used to assert became
    # wrong when hero_mortis.dash_distance was corrected to the real game's 2.67 -- and the rest
    # of this test already derives everything from dash_distance, so that one line was the only
    # thing standing between a config fix and a green suite.
    hero_spec = yaml.safe_load((Path("configs") / "brawlers.yaml").read_text())["hero_mortis"]
    assert abs(dash_distance - float(hero_spec["dash_distance"])) < 1e-4
    assert abs(dash_duration - float(hero_spec["dash_duration"])) < 1e-4
    assert n_ticks >= 2  # the feel-check needs a dash that spans more than a single tick

    session.on_key_press(_FakeKeyEvent("d"))  # hold "move right" for the whole dash
    session.on_key_press(_FakeKeyEvent(" "))  # hold fire -> triggers the dash this tick

    start_pos = env.state.ent_pos[0, 0].clone()
    assert float(env.state.ent_dash_t[0, 0]) == 0.0  # not dashing yet
    session.tick()
    assert float(env.state.ent_dash_t[0, 0]) > 0.0  # the dash started this very tick
    for _ in range(n_ticks - 1):
        session.tick()
    assert float(env.state.ent_dash_t[0, 0]) < 1e-6  # dash has fully resolved

    displacement = float(torch.linalg.norm(env.state.ent_pos[0, 0] - start_pos))
    assert abs(displacement - dash_distance) < 0.05, (
        f"dash moved {displacement:.3f} tiles over {n_ticks} ticks, expected ~{dash_distance} "
        "(dash_distance) -- if this drifts, brawlers.yaml's dash numbers or movement.py's "
        "walk-during-dash exclusion changed"
    )
