import torch
import yaml

from brawl_sim.bots import perception
from brawl_sim.config import load_config
from brawl_sim.core import stats
from brawl_sim.core.state import snapshot as state_snapshot
from brawl_sim.env import BrawlVecEnv
from brawl_sim.maps.loader import build_map_bank
from brawl_sim.render.ascii import _BOT_CHAR, render_ascii

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _env_bank_snap(overrides=None, seed=0):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=seed)
    env.reset()
    bank = build_map_bank(cfg, device="cpu")
    snap = state_snapshot(env.state, 0)
    return env, bank, cfg, snap


def _with_extras(env, snap):
    vis = perception.visibility(env.state, env.bank, env.params, env.cfg)
    snap = dict(snap)
    snap["revealed_to_hero"] = vis[0, 0].numpy()
    snap["max_ammo"] = stats.gather_kind(env.params.max_ammo, env.state.ent_kind)[0].numpy()
    return snap


# ---- frame 0 shows a sane map with 1 + n_enemies units --------------------------------------

def test_frame_0_shows_a_sane_map_with_all_units():
    env, bank, cfg, snap = _env_bank_snap(overrides={"entities": {"n_enemies": 3}})
    out = render_ascii(snap, bank, cfg, mode="world")
    map_lines = out.split("\n\n")[0].split("\n")
    assert len(map_lines) == cfg.map_h
    assert all(len(line) == cfg.map_w for line in map_lines)

    joined = "\n".join(map_lines)
    assert joined.count("H") == 1  # hero
    # Taken from _BOT_CHAR rather than spelled out. The literal this replaced ("saymrSAMR") had
    # silently stopped covering the roster two brawlers ago: it never listed Edgar's "e", so the
    # test only passed while a 3-enemy draw happened to miss him, and it then failed on an
    # unrelated change to the archetype weights rather than on the change that actually broke it.
    revealed = "".join(c.upper() for c in _BOT_CHAR.values())
    hidden = "".join(c.lower() for c in _BOT_CHAR.values())
    # Every non-hero entity is alive and (freshly spawned) revealed -> uppercase, exactly once each
    assert sum(joined.count(c) for c in revealed) == cfg.n_enemies
    # ... and none of them double-counted via its lowercase form.
    assert sum(joined.count(c) for c in revealed + hidden) == cfg.n_enemies


# ---- view mode outputs exactly view_h lines of view_w characters -----------------------------

def test_view_mode_outputs_exactly_view_h_lines_of_view_w_chars():
    env, bank, cfg, snap = _env_bank_snap()
    out = render_ascii(snap, bank, cfg, mode="view")
    map_lines = out.split("\n\n")[0].split("\n")
    assert len(map_lines) == cfg.view_h
    for line in map_lines:
        assert len(line) == cfg.view_w


def test_agent_mode_also_outputs_exactly_view_h_lines_of_view_w_chars():
    env, bank, cfg, snap = _env_bank_snap()
    snap = _with_extras(env, snap)
    out = render_ascii(snap, bank, cfg, mode="agent")
    map_lines = out.split("\n\n")[0].split("\n")
    assert len(map_lines) == cfg.view_h
    for line in map_lines:
        assert len(line) == cfg.view_w


# ---- agent mode never shows a bush-hidden bot -------------------------------------------------

def test_agent_mode_never_shows_a_bush_hidden_bot():
    cfg = load_config(CONFIGS_DEFAULT, overrides={
        "world": {"maps": ["bushy"], "map_selection": "fixed", "fixed_map": "bushy"},
        "entities": {"n_enemies": 1},
    })
    env = BrawlVecEnv(cfg, n_envs=1, device="cpu", seed=0)
    env.reset()
    bank = build_map_bank(cfg, device="cpu")

    bush_mask = env.bank.is_bush[0]
    ys, xs = torch.nonzero(bush_mask, as_tuple=True)
    bush_pos = torch.stack([xs[0].float() + 0.5, ys[0].float() + 0.5])
    env.state.ent_pos[0, 1] = bush_pos
    env.state.ent_pos[0, 0] = bush_pos + torch.tensor([20.0, 0.0])  # far from the bush
    env.state.ent_reveal_t[0, 1] = 0.0

    vis = perception.visibility(env.state, env.bank, env.params, env.cfg)
    assert not bool(vis[0, 0, 1])  # sanity: bot really is hidden from the hero

    snap = state_snapshot(env.state, 0)
    snap["revealed_to_hero"] = vis[0, 0].numpy()
    snap["max_ammo"] = stats.gather_kind(env.params.max_ammo, env.state.ent_kind)[0].numpy()

    world_out = render_ascii(snap, bank, cfg, mode="world")
    world_map = world_out.split("\n\n")[0]
    assert "s" in world_map or "a" in world_map or "m" in world_map or "r" in world_map  # lowercase, shown

    agent_out = render_ascii(snap, bank, cfg, mode="agent")
    agent_map = agent_out.split("\n\n")[0]
    assert "S" not in agent_map and "A" not in agent_map and "M" not in agent_map and "R" not in agent_map
    assert "s" not in agent_map and "a" not in agent_map and "m" not in agent_map and "r" not in agent_map


def test_agent_mode_shows_a_revealed_bot():
    env, bank, cfg, snap = _env_bank_snap(overrides={"entities": {"n_enemies": 1}})
    # place the bot close to the hero, well within the view crop window -- "revealed" (vis) and
    # "inside the egocentric view crop" are different things; a revealed bot far from the hero
    # still wouldn't show in a spatially-cropped mode, same as the real obs["view"] grid.
    env.state.ent_pos[0, 0] = torch.tensor([10.0, 10.0])
    env.state.ent_pos[0, 1] = torch.tensor([11.0, 10.0])
    env.state.ent_reveal_t[0, 1] = 1.0
    snap = state_snapshot(env.state, 0)
    snap = _with_extras(env, snap)
    assert bool(snap["revealed_to_hero"][1])
    out = render_ascii(snap, bank, cfg, mode="agent")
    map_part = out.split("\n\n")[0]
    assert any(c in map_part for c in "SAMR")


# ---- mode='agent' requires revealed_to_hero ---------------------------------------------------

def test_agent_mode_raises_a_clear_error_without_revealed_to_hero():
    env, bank, cfg, snap = _env_bank_snap()
    try:
        render_ascii(snap, bank, cfg, mode="agent")
        assert False, "expected a ValueError"
    except ValueError as e:
        assert "revealed_to_hero" in str(e)


def test_unknown_mode_raises():
    env, bank, cfg, snap = _env_bank_snap()
    try:
        render_ascii(snap, bank, cfg, mode="bogus")
        assert False, "expected a ValueError"
    except ValueError:
        pass


# ---- status line contents ----------------------------------------------------------------------

def test_status_line_contains_expected_fields():
    env, bank, cfg, snap = _env_bank_snap()
    snap = _with_extras(env, snap)
    out = render_ascii(snap, bank, cfg, mode="world")
    status = out.split("\n\n")[1]
    assert "t=" in status
    assert "HP=" in status
    assert "ammo=" in status
    assert "cubes=" in status
    assert "alive=" in status
    assert "zone_step=" in status
    assert "dash=" in status


def test_ammo_pips_fall_back_to_plain_number_without_max_ammo():
    env, bank, cfg, snap = _env_bank_snap()  # no max_ammo extra added
    out = render_ascii(snap, bank, cfg, mode="world")
    status = out.split("\n\n")[1]
    assert "ammo=" in status
    assert "[" not in status.split("ammo=")[1].split(" ")[0]


# ---- obstacles show through the zone marker; zone only overlays floor -------------------------

def test_zone_marker_never_overwrites_a_wall_char():
    env, bank, cfg, snap = _env_bank_snap()
    # shrink the zone to a tiny rect near the map center so most of the map (including the
    # border walls) is "outside" and would-be zone-marked.
    snap = dict(snap)
    center = (cfg.map_w / 2.0, cfg.map_h / 2.0)
    snap["zone_lo"] = snap["zone_lo"].copy()
    snap["zone_hi"] = snap["zone_hi"].copy()
    snap["zone_lo"][:] = [center[0] - 1, center[1] - 1]
    snap["zone_hi"][:] = [center[0] + 1, center[1] + 1]

    out = render_ascii(snap, bank, cfg, mode="world")
    map_lines = out.split("\n\n")[0].split("\n")
    # border row must still be all '#', not ':'
    assert set(map_lines[0]) == {"#"}
    assert set(map_lines[-1]) == {"#"}
    # but the interior, now outside the shrunk zone, should show ':' somewhere
    assert any(":" in line for line in map_lines[1:-1])
