"""render_ascii: a text-mode renderer for one CPU snapshot (core.state.snapshot). See
BRAWL_SIM_BUILD_PLAN.md Step 37.

**Signature stays exactly `(snapshot, bank, cfg, mode)` -- no `params`/`state`/`bots` import.**
Two things the plan's char legend and `mode="agent"` need aren't part of raw `SimState` at all:
`revealed_to_hero` (a `bots/perception.visibility` OUTPUT, not stored state) and `max_ammo` (a
`SimParams` field, not a `SimState` one). Rather than take extra parameters or import `bots/`
into `render/` (this package has never needed a `core/`-vs-`bots/` layering rule before now, and
introducing one late for a debug tool felt like the wrong tradeoff), both are OPTIONAL EXTRA
KEYS a caller may add to the plain `core.state.snapshot()` dict before calling this function --
`scripts/record_rollout.py` (this same step) computes and adds them every frame. A bare
`state.snapshot(state, i)` dict still renders correctly in `"world"`/`"view"` mode (every bot
just reads as revealed/uppercase, ammo pips fall back to a plain number); `mode="agent"` raises
a clear `ValueError` if `revealed_to_hero` is missing, since there's no sane default for "which
enemies is the hero allowed to see."

**`mode="agent"` is a deliberately scoped-down reading of "exactly what `AgentObsSpec(fair=True)`
exposes."** It reuses the `"view"` mode's egocentric crop and omits bush-hidden bots (the one
behavior Step 37's own acceptance criterion tests), but does NOT reproduce `AgentObsSpec`'s
finer-grained gating -- e.g. projectiles are still shown even when they'd be gated by `in_view`
under the real fair spec. Reproducing that exactly would mean rebuilding the full `obs_select`
pipeline (which needs live GPU tensors, not a numpy snapshot) inside a text debug renderer;
out of scope for what this step actually needs. Noted here as a forward pointer, not silently
glossed over.

**Overlay priority (highest wins a cell, ties keep whatever painted first):** hero > bot > box >
projectile > pickup > obstacle terrain (wall/bush/water/fence) > zone marker > floor terrain.
Obstacles always show through the zone marker (walls don't move because of the zone); the zone
marker only ever replaces plain floor, never an obstacle or an occupant.

**"revealed" and "inside the crop window" are two different gates in `"view"`/`"agent"` mode,**
found while writing this module's own tests: a bot can be `revealed_to_hero` (bush/targeting
visibility) while still standing outside the `view_h x view_w` egocentric crop -- it simply
won't be drawn either way, same as the real `obs["view"]` grid, which is also spatially cropped
independent of bush-hiding. `mode="world"` has no crop (the whole map is in range), so this
distinction only matters for `"view"`/`"agent"`.
"""
import numpy as np

from ..constants import Kind, Tile

_TERRAIN_CHAR = {
    int(Tile.FLOOR): ".", int(Tile.WALL): "#", int(Tile.BUSH): ",",
    int(Tile.WATER): "~", int(Tile.FENCE): "=", int(Tile.SPAWN): ".", int(Tile.BOX): ".",
}
_OBSTACLE_TILES = {int(Tile.WALL), int(Tile.BUSH), int(Tile.WATER), int(Tile.FENCE)}
_OUT_OF_MAP_CHAR = "#"  # matches maps/loader.build_map_bank's own pad_tiles(..., value=WALL)

# One lowercase letter per bot kind; painted uppercase when revealed, lowercase when hidden (see
# the render loop), so the glyph must be a letter that HAS a case. A kind missing from this table
# renders as "?" rather than raising, which is survivable but unreadable -- add a row when a kind
# is added to constants.Kind.
_BOT_CHAR = {
    int(Kind.BOT_SNIPER): "s", int(Kind.BOT_ARTILLERY): "a",
    int(Kind.BOT_MELEE): "m", int(Kind.BOT_RIFLE): "r",
    int(Kind.BOT_EDGAR): "e",
    # "k", not "s": BOT_SNIPER already owns "s", and a glyph collision here is invisible in the
    # output -- two different bots would just both read as the same letter.
    int(Kind.BOT_SPIKE): "k",
    int(Kind.BOT_BULL): "b",
}

# low -> high; a later paint() only overwrites an earlier one at >= priority.
_P_FLOOR, _P_ZONE, _P_OBSTACLE, _P_PICKUP, _P_PROJECTILE, _P_BOX, _P_BOT, _P_HERO = range(8)

_HERO = 0
_MODES = ("world", "view", "agent")


def status_line(snapshot: dict) -> str:
    """Shared with render/viewer.py (Step 38) so both renderers report identical HP/ammo/cubes/
    dash text for the same frame -- one formatting rule, not two copies that can drift."""
    hp = float(snapshot["ent_hp"][_HERO])
    max_hp = float(snapshot["ent_max_hp"][_HERO])
    ammo = float(snapshot["ent_ammo"][_HERO])
    cubes = int(snapshot["ent_cubes"][_HERO])
    t = float(snapshot["time"])
    n_alive = int(snapshot["n_alive"])
    zone_step = int(snapshot["zone_step"])
    dashing = bool(snapshot["ent_dash_t"][_HERO] > 0)

    max_ammo = snapshot.get("max_ammo")
    if max_ammo is None or float(max_ammo[_HERO]) <= 0:
        ammo_str = f"ammo={ammo:.1f}"
    else:
        total = int(round(float(max_ammo[_HERO])))
        filled = max(0, min(int(np.floor(ammo)), total))
        ammo_str = "ammo=[" + "*" * filled + "." * (total - filled) + "]"

    return (
        f"t={t:.2f}s HP={hp:.0f}/{max_hp:.0f} {ammo_str} cubes={cubes} "
        f"alive={n_alive} zone_step={zone_step} dash={'DASH' if dashing else '-'}"
    )


def render_ascii(snapshot: dict, bank, cfg, mode: str = "world") -> str:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    revealed = snapshot.get("revealed_to_hero")
    if mode == "agent" and revealed is None:
        raise ValueError(
            "mode='agent' requires snapshot['revealed_to_hero'] (bots.perception.visibility's "
            "hero row) -- a bare core.state.snapshot() doesn't include it, since it's a "
            "perception OUTPUT, not stored SimState. scripts/record_rollout.py adds it."
        )

    map_id = int(snapshot["map_id"])
    tiles = bank.tiles[map_id].detach().cpu().numpy() if hasattr(bank.tiles, "detach") else bank.tiles[map_id]
    H, W = tiles.shape

    hero_pos = snapshot["ent_pos"][_HERO]
    hero_ix, hero_iy = int(np.floor(hero_pos[0])), int(np.floor(hero_pos[1]))

    if mode == "world":
        ox, oy, out_w, out_h = 0, 0, W, H
    else:
        out_w, out_h = cfg.view_w, cfg.view_h
        ox, oy = hero_ix - out_w // 2, hero_iy - out_h // 2

    chars = np.full((out_h, out_w), _OUT_OF_MAP_CHAR, dtype="<U1")
    # Sentinel LOWER than every real priority (including _P_FLOOR=0) -- out-of-map cells stay
    # "#" because the terrain loop below never calls paint() for them (bounded by the real map
    # intersection), not because this sentinel outranks anything. A sentinel >= _P_FLOOR would
    # (and, before this fix, did) block ordinary floor tiles from ever painting over it.
    prio = np.full((out_h, out_w), _P_FLOOR - 1, dtype=np.int64)

    def paint(ix: int, iy: int, ch: str, p: int) -> None:
        gx, gy = ix - ox, iy - oy
        if 0 <= gx < out_w and 0 <= gy < out_h and p >= prio[gy, gx]:
            chars[gy, gx] = ch
            prio[gy, gx] = p

    y0, y1 = max(oy, 0), min(oy + out_h, H)
    x0, x1 = max(ox, 0), min(ox + out_w, W)
    for iy in range(y0, y1):
        row = tiles[iy]
        for ix in range(x0, x1):
            t = int(row[ix])
            paint(ix, iy, _TERRAIN_CHAR.get(t, "."), _P_OBSTACLE if t in _OBSTACLE_TILES else _P_FLOOR)

    zone_lo, zone_hi = snapshot["zone_lo"], snapshot["zone_hi"]
    for iy in range(y0, y1):
        cy = iy + 0.5
        if not (zone_lo[1] <= cy <= zone_hi[1]):
            for ix in range(x0, x1):
                paint(ix, iy, ":", _P_ZONE)
        else:
            for ix in range(x0, x1):
                cx = ix + 0.5
                if not (zone_lo[0] <= cx <= zone_hi[0]):
                    paint(ix, iy, ":", _P_ZONE)

    pku_alive, pku_pos = snapshot["pku_alive"], snapshot["pku_pos"]
    for i in range(len(pku_alive)):
        if pku_alive[i]:
            paint(int(np.floor(pku_pos[i, 0])), int(np.floor(pku_pos[i, 1])), "*", _P_PICKUP)

    prj_alive, prj_pos = snapshot["prj_alive"], snapshot["prj_pos"]
    for i in range(len(prj_alive)):
        if prj_alive[i]:
            paint(int(np.floor(prj_pos[i, 0])), int(np.floor(prj_pos[i, 1])), "'", _P_PROJECTILE)

    box_alive, box_pos = snapshot["box_alive"], snapshot["box_pos"]
    for i in range(len(box_alive)):
        if box_alive[i]:
            paint(int(np.floor(box_pos[i, 0])), int(np.floor(box_pos[i, 1])), "o", _P_BOX)

    ent_alive, ent_pos, ent_kind = snapshot["ent_alive"], snapshot["ent_pos"], snapshot["ent_kind"]
    for e in range(len(ent_alive)):
        if not ent_alive[e]:
            continue
        ix, iy = int(np.floor(ent_pos[e, 0])), int(np.floor(ent_pos[e, 1]))
        if e == _HERO:
            paint(ix, iy, "H", _P_HERO)
            continue
        is_revealed = bool(revealed[e]) if revealed is not None else True
        if mode == "agent" and not is_revealed:
            continue  # hidden enemies simply absent
        ch = _BOT_CHAR.get(int(ent_kind[e]), "?")
        paint(ix, iy, ch.upper() if is_revealed else ch.lower(), _P_BOT)

    lines = ["".join(chars[y]) for y in range(out_h)]
    return "\n".join(lines) + "\n\n" + status_line(snapshot)
