"""ReplayViewer: an interactive matplotlib replay of one scripts/record_rollout.py rollout.npz.
See BRAWL_SIM_BUILD_PLAN.md Step 38 / Notice 3 -- matplotlib, not pygame (pygame ships no
Windows wheel past cp313, this project's only interpreter is cp314; render/ascii.py, Step 37,
was never affected by that constraint).

**Same optional-extra-keys contract as render/ascii.py (Step 37).** `frames` (a dict of
`(steps, ...)` numpy arrays, one array per `core.state.snapshot()` field, exactly what
`scripts/record_rollout.py.record_rollout()` returns / `np.load(...)` gives back) only strictly
needs the required `SimState` fields to construct and play. `revealed_to_hero`/`los_to_hero`
missing just means the reveal overlay never lights up (the 'v' toggle still exists, it has
nothing to show); `unit_radius` missing falls back to `_DEFAULT_RADIUS`. The status-line title
text is produced by `render.ascii.status_line` -- the SAME function `render_ascii` itself calls
-- so the two renderers report identical HP/ammo/cubes/dash text for a given frame by
construction, not by keeping two copies in sync by hand.

**Reveal overlay semantics ('v' toggle):** a bright ring appears on entity `e` iff
`los_to_hero[e]` (no WALL between the hero and `e`, `bots.perception.raw_los`) is True AND
`revealed_to_hero[e]` (`bots.perception.visibility`) is False -- i.e. bush-hiding is the ONLY
reason the hero can't currently target/see it. An entity that's simply not `los_to_hero` (wall
in the way) is never ringed: per Notice 4, walls never hide anything from this game's fixed
bird's-eye camera, so there is nothing to visually call out there. Ground truth is always drawn
regardless of the ring -- this viewer shows full information, unlike `render_ascii`'s
`mode="agent"`; the ring only ANNOTATES why the hero's own obs would consider an entity hidden.

**Entities get dedicated per-slot artists (Circle body, HP-bar Rectangle pair, ammo Text,
reveal-ring Circle), one fixed set per `n_entities` slot, updated in place every frame.** Boxes,
pickups, and projectiles instead share one (or one-per-kind) `Axes.scatter`/PathCollection,
per the plan. The split follows directly from what each needs: entity annotations (HP/ammo/
dash/reveal) are inherently per-instance and `n_entities` is small (`1 + n_enemies`, no
perf concern going through individual artists); boxes/pickups/projectiles are larger,
visually-homogeneous populations where the plan explicitly calls for shared scatter collections.

**PNG sprites are optional, resolved once at construction, and never mixed with their fallback
on the same artist.** See `render/assets/README.md` for exact filenames/sizing. Every entity
slot, and the box/pickup pools, get BOTH a fallback artist (Circle / scatter, as above) and a
sprite artist (`Axes.imshow`, extent set to a world-space square every frame so it zooms with
the 'g' pan toggle like everything else here); per frame, whichever one currently has image data
for that Kind/category is shown and the other is hidden. A missing PNG is the expected steady
state, not a degraded one -- nothing needs the assets directory populated to run correctly.

**Blitting and the 'g' world/view pan toggle don't mix cleanly, and this is a deliberate,
documented tradeoff rather than a bug.** `FuncAnimation(blit=True)` caches ONE static background
snapshot and only re-draws the artists an `_advance` call returns on top of it -- exactly the
plan's "update artist data in place... do not call ax.clear()/re-plot every frame" performance
note, and it's what gets 20 fps in the default "world" camera (static full-map limits). But 'g'
mode pans the camera to a hero-centered `view_w x view_h` crop every frame, which changes what
the cached background itself should look like -- blit alone can't express that. So `_draw_frame`
calls `self.fig.canvas.draw()` (a full, non-blit redraw, which also happens to refresh
FuncAnimation's blit background cache via the `draw_event` it emits) whenever `crop_to_view` is
on. Net effect: the default world camera hits the 20 fps blit target from the acceptance bar;
panned "view" mode is correct but slower. `render_ascii`'s `"view"` mode has no such tradeoff
since it re-renders text unconditionally every call.

Key bindings (bound in `show()`; also reachable directly via `on_key(event)` for tests that
don't want to drive a real event loop): space=pause/resume, left/right=scrub one frame,
+/-=playback speed (halves/doubles), v=reveal overlay, c=view-rect outline, g=world/view pan.
"""
import argparse
import itertools
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

from ..config import load_config
from ..constants import Kind, Proj, ProjClass, Tile
from ..core.projectiles import BOX_RADIUS
from ..maps.loader import build_map_bank
from .ascii import status_line as ascii_status_line

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

_HERO = 0
_DEFAULT_RADIUS = 0.4  # fallback when a bare snapshot (no `unit_radius` extra) is played

# ---- optional PNG sprites (see render/assets/README.md for what to drop in and where) --------
# Every lookup here is by-name against a Kind/literal, so adding a PNG with the right filename
# is the entire integration step -- no code change needed. A missing file just means that slot
# keeps drawing its existing Circle/scatter fallback; sprites and fallbacks are never mixed
# within the same artist, so there's no per-frame cost to leaving assets out.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ENTITY_SPRITE_PATHS = {k: _ASSETS_DIR / "entities" / f"{k.name.lower()}.png" for k in Kind}
_BOX_SPRITE_PATH = _ASSETS_DIR / "items" / "box.png"
_PICKUP_SPRITE_PATH = _ASSETS_DIR / "items" / "pickup.png"

# Sprites are drawn larger than the collision circle they replace (Brawl Stars character art
# extends well past the hitbox -- weapon, shadow, wind-up frames) but boxes/pickups are simple
# props whose art roughly fills their physical footprint, hence the smaller item scale.
_ENTITY_SPRITE_SCALE = 1.8
_ITEM_SPRITE_SCALE = 1.3
_PICKUP_RADIUS = 0.3  # no collision-radius field for cubes exists (pickup_radius is the much
# larger *collection* range, e.g. 0.6 -- see configs/default.yaml); this is a visual-only size
# chosen to look like a small cube next to a BOX_RADIUS=0.5 crate.


def _load_sprite(path: Path):
    """Returns an (H, W, 4) RGBA float array via plt.imread, or None if `path` doesn't exist.

    PNG only (matplotlib decodes those natively, no Pillow dependency needed). Missing files
    are the expected steady state until the user drops assets in -- not an error.
    """
    if not path.exists():
        return None
    return plt.imread(str(path))

# Ordered by Tile's int value (0..N_TILES-1) so the ListedColormap index matches the tile-id
# grid directly -- same convention as render/ascii.py's _TERRAIN_CHAR table.
#
# PUBLIC because brawl_vision's terrain overlay renders its perceived occupancy grid in these
# exact colours: a perceived grid and a simulated one drawn from the same palette are comparable
# by eye, which is most of the point of building the overlay. Promoted on second use, the same
# move already made on core/zone.current_dps and core/observation.compute_rank.
TILE_COLORS = {
    Tile.FLOOR: "#d9d2b8", Tile.WALL: "#3b3b3b", Tile.BUSH: "#3f7a34",
    Tile.WATER: "#3a78c2", Tile.FENCE: "#8a6a3a", Tile.SPAWN: "#d9d2b8", Tile.BOX: "#d9d2b8",
}
TILE_CMAP = ListedColormap([TILE_COLORS[t] for t in Tile])

# IntEnum members hash/compare as their int value, so these dicts are also indexable by a
# plain int (e.g. `_KIND_COLOR.get(int(ent_kind[e]), ...)`) with no explicit enum conversion.
_KIND_COLOR = {
    Kind.HERO_MORTIS: "#e8342a",
    Kind.BOT_SNIPER: "#2a6ae8",
    Kind.BOT_ARTILLERY: "#e8a52a",
    Kind.BOT_MELEE: "#8a2ae8",
    Kind.BOT_RIFLE: "#2ae8b0",
    Kind.BOT_EDGAR: "#e82a8a",
    Kind.BOT_SPIKE: "#7ae82a",
    Kind.BOT_BULL: "#a86a2a",
}
_PROJ_COLOR = {
    Proj.SNIPER_BOLT: "#2a6ae8",
    Proj.ARTILLERY_SHELL: "#e8a52a",
    Proj.RIFLE_ARROW: "#2ae8b0",
    Proj.SUPER_BOLT: "#ff3ce0",   # Mortis's super; drawn via super_bolt_circles, not this scatter
    Proj.SPIKE_SHELL: "#7ae82a",  # shell AND its six shards -- same weapon, same Proj (see _spawn_splits)
    Proj.BULL_SLUG: "#a86a2a",
}
# Brock's lingering sphere: a translucent patch of denied ground, not a projectile dot.
_HAZARD_COLOR = "#2a6ae8"
_HAZARD_EDGE = "#7fb0ff"
_HAZARD_ALPHA = 0.28
_SUPER_BOLT_ALPHA = 0.55
_DASH_RING_COLOR = "#ffffff"
_REVEAL_RING_COLOR = "#ffe040"
_ZONE_COLOR = "#2ae82a"
_ZONE_ALPHA = 0.30


class ReplayViewer:
    def __init__(self, frames: dict, bank, cfg, fps: int = 20):
        self.frames = frames
        self.bank = bank
        self.cfg = cfg
        self.fps = fps

        self.n_frames = frames["ent_pos"].shape[0]
        self.n_entities = frames["ent_pos"].shape[1]
        self.frame_idx = 0
        self.paused = False
        self.speed = 1.0
        self.show_reveal = True
        self.show_view_rect = False
        self.crop_to_view = False

        tiles_all = bank.tiles
        self._tiles_all = tiles_all.detach().cpu().numpy() if hasattr(tiles_all, "detach") else tiles_all
        self.H, self.W = self._tiles_all.shape[1:]

        self.anim = None
        self._build_figure()

    # ---- figure/artist construction (once) -----------------------------------------------

    def _build_figure(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(min(self.W / 4, 12), min(self.H / 4, 12)))
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(self.H, 0)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.im = self.ax.imshow(
            self._tiles_all[0], cmap=TILE_CMAP, origin="upper", extent=(0, self.W, self.H, 0),
            vmin=0, vmax=len(Tile) - 1, interpolation="nearest", zorder=0,
        )

        # 4 bands framing the safe rect (top/bottom/left/right) -- an overlap-free partition of
        # "outside the safe zone", same logic as render/ascii.py's per-cell zone marker.
        self.zone_rects = [
            Rectangle((0, 0), 0, 0, facecolor=_ZONE_COLOR, alpha=_ZONE_ALPHA, linewidth=0, zorder=1)
            for _ in range(4)
        ]
        for r in self.zone_rects:
            self.ax.add_patch(r)

        self.view_rect = Rectangle(
            (0, 0), self.cfg.view_w, self.cfg.view_h, fill=False, edgecolor="white",
            linewidth=1.5, zorder=5, visible=self.show_view_rect,
        )
        self.ax.add_patch(self.view_rect)

        # Sprite lookups resolved once at construction (a rollout never changes what PNGs exist
        # on disk mid-playback), so the per-frame path is just "is this entry None or not".
        self._entity_sprite_imgs = {k: _load_sprite(p) for k, p in _ENTITY_SPRITE_PATHS.items()}
        self._box_sprite_img = _load_sprite(_BOX_SPRITE_PATH)
        self._pickup_sprite_img = _load_sprite(_PICKUP_SPRITE_PATH)

        self.box_scatter = self.ax.scatter([], [], marker="s", s=60, c="#c98a3a", zorder=2)
        self.pickup_scatter = self.ax.scatter([], [], marker="D", s=40, c="#f0d020", zorder=2)
        # SUPER_BOLT (Mortis's super) gets its own circle-patch pool below, sized to its real
        # `prj_radius`, so it's excluded from the fixed-size scatter every other Proj kind uses.
        self.proj_scatter = {
            kind: self.ax.scatter([], [], s=15, c=color, zorder=3)
            for kind, color in _PROJ_COLOR.items() if kind != Proj.SUPER_BOLT
        }

        # HAZARD-class projectiles (Brock's lingering sphere, Step C3b) are drawn as translucent
        # circles at their real `prj_aoe` radius rather than as scatter dots. A dot would be
        # actively misleading: the thing that matters when watching a rollout is the GROUND it
        # denies, and a 0.75-tile sphere rendered as a 15-point marker looks like a stray bullet
        # sitting still. One patch per projectile slot, preallocated and hidden, matching how
        # every other per-slot artist pool here works.
        self.hazard_circles = [
            plt.Circle((0.0, 0.0), 0.0, facecolor=_HAZARD_COLOR, edgecolor=_HAZARD_EDGE,
                        alpha=_HAZARD_ALPHA, zorder=2, visible=False)
            for _ in range(self.frames["prj_pos"].shape[1])
        ]
        for circle in self.hazard_circles:
            self.ax.add_patch(circle)

        # Mortis's super (Proj.SUPER_BOLT) is a piercing bolt with a real hit-width (`prj_radius`
        # -- 0.70 tiles for Mortis, configs/brawlers.yaml `super_radius`), not a point. The fixed
        # 15-point scatter dot every other Proj kind uses makes it look tiny; drawn at true scale
        # it's nearly as wide as the hero. Same preallocated/hidden circle-per-slot pattern as
        # the hazard circles above, keyed off `prj_radius` instead of `prj_aoe` since a piercing
        # bolt has no area-of-effect (`prj_aoe` is 0 for it -- see projectiles.spawn_supers).
        self.super_bolt_circles = [
            plt.Circle((0.0, 0.0), 0.0, facecolor=_PROJ_COLOR[Proj.SUPER_BOLT],
                        edgecolor=_PROJ_COLOR[Proj.SUPER_BOLT], alpha=_SUPER_BOLT_ALPHA,
                        zorder=3, visible=False)
            for _ in range(self.frames["prj_pos"].shape[1])
        ]
        for circle in self.super_bolt_circles:
            self.ax.add_patch(circle)

        # Boxes/pickups only get a sprite artist pool when their one shared PNG actually exists
        # -- otherwise the scatter collections above are the whole story, unchanged from before.
        self.n_boxes = self.frames["box_pos"].shape[1]
        self.box_sprites = []
        if self._box_sprite_img is not None:
            self.box_scatter.set_visible(False)
            for _ in range(self.n_boxes):
                im = self.ax.imshow(np.zeros((1, 1, 4)), extent=(0, 0, 0, 0), origin="upper",
                                     zorder=2, visible=False)
                self.box_sprites.append(im)

        self.n_pickups = self.frames["pku_pos"].shape[1]
        self.pickup_sprites = []
        if self._pickup_sprite_img is not None:
            self.pickup_scatter.set_visible(False)
            for _ in range(self.n_pickups):
                im = self.ax.imshow(np.zeros((1, 1, 4)), extent=(0, 0, 0, 0), origin="upper",
                                     zorder=2, visible=False)
                self.pickup_sprites.append(im)

        self.body, self.sprite, self.dash_ring = [], [], []
        self.hp_bg, self.hp_fill, self.ammo_text, self.reveal_ring = [], [], [], []
        for e in range(self.n_entities):
            body = Circle((0, 0), _DEFAULT_RADIUS, facecolor="#888888", edgecolor="none", zorder=4)
            self.ax.add_patch(body)
            self.body.append(body)

            # Same slot's sprite artist, only ever shown for `e` if `_entity_sprite_imgs` has a
            # PNG for whatever Kind that slot currently is (checked fresh every frame -- a slot
            # could in principle host a different Kind across episodes within one recording).
            sprite = self.ax.imshow(np.zeros((1, 1, 4)), extent=(0, 0, 0, 0), origin="upper",
                                     zorder=4, visible=False)
            self.sprite.append(sprite)

            hp_bg = Rectangle((0, 0), 0, 0, facecolor="#202020", zorder=4)
            self.ax.add_patch(hp_bg)
            self.hp_bg.append(hp_bg)
            hp_fill = Rectangle((0, 0), 0, 0, facecolor="#2ecc40", zorder=5)
            self.ax.add_patch(hp_fill)
            self.hp_fill.append(hp_fill)

            text = self.ax.text(0, 0, "", fontsize=6, ha="center", va="center", color="white", zorder=6)
            self.ammo_text.append(text)

            # Dashing used to be shown as the body Circle's own edge, but that cue needs to
            # render on top of a sprite too now -- pulled out into its own ring so it composes
            # with either visual instead of only working in the circle-fallback path.
            dash = Circle((0, 0), _DEFAULT_RADIUS, fill=False, edgecolor=_DASH_RING_COLOR,
                           linewidth=2.0, zorder=4, visible=False)
            self.ax.add_patch(dash)
            self.dash_ring.append(dash)

            ring = Circle((0, 0), _DEFAULT_RADIUS, fill=False, edgecolor=_REVEAL_RING_COLOR,
                           linewidth=1.5, zorder=4, visible=False)
            self.ax.add_patch(ring)
            self.reveal_ring.append(ring)

        self.facing_lines = LineCollection(
            [[(0, 0), (0, 0)]] * self.n_entities, colors="black", linewidths=1.5, zorder=6,
        )
        self.ax.add_collection(self.facing_lines)

        self.title = self.ax.set_title("", fontsize=9, family="monospace")

        self._draw_frame(0)

    def _artists(self) -> list:
        return (
            [self.im] + list(self.zone_rects) + [self.view_rect, self.box_scatter, self.pickup_scatter]
            + list(self.proj_scatter.values()) + self.hazard_circles + self.super_bolt_circles
            + self.box_sprites + self.pickup_sprites
            + self.body + self.sprite + self.dash_ring + self.hp_bg + self.hp_fill
            + self.ammo_text + self.reveal_ring + [self.facing_lines, self.title]
        )

    # ---- per-frame update -----------------------------------------------------------------

    def _draw_frame(self, idx: int) -> list:
        self.frame_idx = idx
        f = {name: arr[idx] for name, arr in self.frames.items()}

        self.im.set_data(self._tiles_all[int(f["map_id"])])

        zx0 = float(np.clip(f["zone_lo"][0], 0, self.W))
        zy0 = float(np.clip(f["zone_lo"][1], 0, self.H))
        zx1 = float(np.clip(f["zone_hi"][0], 0, self.W))
        zy1 = float(np.clip(f["zone_hi"][1], 0, self.H))
        top, bottom, left, right = self.zone_rects
        top.set_bounds(0.0, 0.0, self.W, zy0)
        bottom.set_bounds(0.0, zy1, self.W, max(0.0, self.H - zy1))
        left.set_bounds(0.0, zy0, zx0, max(0.0, zy1 - zy0))
        right.set_bounds(zx1, zy0, max(0.0, self.W - zx1), max(0.0, zy1 - zy0))

        box_alive = f["box_alive"].astype(bool)
        if self._box_sprite_img is not None:
            half = BOX_RADIUS * _ITEM_SPRITE_SCALE
            for i, im in enumerate(self.box_sprites):
                alive = bool(box_alive[i])
                im.set_visible(alive)
                if alive:
                    bx, by = (float(v) for v in f["box_pos"][i])
                    im.set_data(self._box_sprite_img)
                    im.set_extent((bx - half, bx + half, by + half, by - half))
        else:
            self.box_scatter.set_offsets(f["box_pos"][box_alive] if box_alive.any() else np.empty((0, 2)))

        pku_alive = f["pku_alive"].astype(bool)
        if self._pickup_sprite_img is not None:
            half = _PICKUP_RADIUS * _ITEM_SPRITE_SCALE
            for i, im in enumerate(self.pickup_sprites):
                alive = bool(pku_alive[i])
                im.set_visible(alive)
                if alive:
                    px_, py_ = (float(v) for v in f["pku_pos"][i])
                    im.set_data(self._pickup_sprite_img)
                    im.set_extent((px_ - half, px_ + half, py_ + half, py_ - half))
        else:
            self.pickup_scatter.set_offsets(f["pku_pos"][pku_alive] if pku_alive.any() else np.empty((0, 2)))

        prj_alive = f["prj_alive"].astype(bool)
        prj_kind = f["prj_kind"]
        # A hazard keeps the Proj kind of the weapon that left it, so it would otherwise be drawn
        # as that weapon's bullet. Exclude it from the scatter and give it a circle below.
        is_hazard = f["prj_class"] == int(ProjClass.HAZARD)
        for kind, scat in self.proj_scatter.items():
            mask = prj_alive & (prj_kind == int(kind)) & ~is_hazard
            scat.set_offsets(f["prj_pos"][mask] if mask.any() else np.empty((0, 2)))

        for p, circle in enumerate(self.hazard_circles):
            if prj_alive[p] and is_hazard[p]:
                circle.set_center(tuple(f["prj_pos"][p]))
                circle.set_radius(float(f["prj_aoe"][p]))
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        is_super_bolt = prj_kind == int(Proj.SUPER_BOLT)
        for p, circle in enumerate(self.super_bolt_circles):
            if prj_alive[p] and is_super_bolt[p]:
                circle.set_center(tuple(f["prj_pos"][p]))
                circle.set_radius(float(f["prj_radius"][p]))
                circle.set_visible(True)
            else:
                circle.set_visible(False)

        revealed = f.get("revealed_to_hero")
        los = f.get("los_to_hero")
        radius = float(f["unit_radius"]) if "unit_radius" in f else _DEFAULT_RADIUS

        facing_segments = []
        hero_pos = f["ent_pos"][_HERO]
        for e in range(self.n_entities):
            body, sprite, dash, ring = self.body[e], self.sprite[e], self.dash_ring[e], self.reveal_ring[e]
            hp_bg, hp_fill, text = self.hp_bg[e], self.hp_fill[e], self.ammo_text[e]
            alive = bool(f["ent_alive"][e])

            if not alive:
                for artist in (body, sprite, dash, hp_bg, hp_fill, text, ring):
                    artist.set_visible(False)
                facing_segments.append(((-1000.0, -1000.0), (-1000.0, -1000.0)))
                continue

            pos = f["ent_pos"][e]
            px, py = float(pos[0]), float(pos[1])
            kind = int(f["ent_kind"][e])
            dashing = bool(f["ent_dash_t"][e] > 0)

            sprite_img = self._entity_sprite_imgs.get(kind)
            if sprite_img is not None:
                body.set_visible(False)
                sprite.set_visible(True)
                sprite.set_data(sprite_img)
                half = radius * _ENTITY_SPRITE_SCALE
                sprite.set_extent((px - half, px + half, py + half, py - half))
            else:
                sprite.set_visible(False)
                body.set_visible(True)
                body.center = (px, py)
                body.set_radius(radius)
                body.set_facecolor(_KIND_COLOR.get(kind, "#aaaaaa"))

            dash.set_visible(dashing)
            if dashing:
                dash.center = (px, py)
                dash.set_radius(radius * 1.2)

            barw, barh = radius * 2.2, radius * 0.3
            bx0, by0 = px - barw / 2.0, py - radius - barh - 0.05
            hp_bg.set_visible(True)
            hp_bg.set_bounds(bx0, by0, barw, barh)
            hp_frac = float(np.clip(f["ent_hp"][e] / max(float(f["ent_max_hp"][e]), 1e-6), 0.0, 1.0))
            hp_fill.set_visible(True)
            hp_fill.set_bounds(bx0, by0, barw * hp_frac, barh)

            text.set_visible(True)
            text.set_position((px, py + radius + 0.2))
            text.set_text(str(int(np.floor(f["ent_ammo"][e]))))

            bush_hidden_from_hero = (
                e != _HERO and self.show_reveal and revealed is not None and los is not None
                and bool(los[e]) and not bool(revealed[e])
            )
            ring.set_visible(bush_hidden_from_hero)
            if bush_hidden_from_hero:
                ring.center = (px, py)
                ring.set_radius(radius * 1.4)

            facing = float(f["ent_facing"][e])
            dx, dy = radius * 1.3 * np.cos(facing), radius * 1.3 * np.sin(facing)
            facing_segments.append(((px, py), (px + dx, py + dy)))

        self.facing_lines.set_segments(facing_segments)

        self.view_rect.set_bounds(
            float(hero_pos[0]) - self.cfg.view_w / 2.0, float(hero_pos[1]) - self.cfg.view_h / 2.0,
            self.cfg.view_w, self.cfg.view_h,
        )

        if self.crop_to_view:
            self.ax.set_xlim(float(hero_pos[0]) - self.cfg.view_w / 2.0, float(hero_pos[0]) + self.cfg.view_w / 2.0)
            self.ax.set_ylim(float(hero_pos[1]) + self.cfg.view_h / 2.0, float(hero_pos[1]) - self.cfg.view_h / 2.0)
            self.fig.canvas.draw()  # forces blit's cached background to resync -- see module docstring

        self.title.set_text(ascii_status_line(f))

        return self._artists()

    # ---- interaction ------------------------------------------------------------------------

    def _advance(self, _unused_frame):
        if not self.paused:
            self.frame_idx = (self.frame_idx + 1) % self.n_frames
        return self._draw_frame(self.frame_idx)

    def _apply_speed(self) -> None:
        if self.anim is not None:
            self.anim.event_source.interval = max(1.0, 1000.0 / self.fps / self.speed)

    def on_key(self, event) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.paused = not self.paused
        elif key == "right":
            self.paused = True
            self._draw_frame(min(self.frame_idx + 1, self.n_frames - 1))
            self.fig.canvas.draw_idle()
        elif key == "left":
            self.paused = True
            self._draw_frame(max(self.frame_idx - 1, 0))
            self.fig.canvas.draw_idle()
        elif key in ("+", "="):
            self.speed = min(self.speed * 2.0, 16.0)
            self._apply_speed()
        elif key == "-":
            self.speed = max(self.speed / 2.0, 1.0 / 16.0)
            self._apply_speed()
        elif key == "v":
            self.show_reveal = not self.show_reveal
            self._draw_frame(self.frame_idx)
            self.fig.canvas.draw_idle()
        elif key == "c":
            self.show_view_rect = not self.show_view_rect
            self.view_rect.set_visible(self.show_view_rect)
            self.fig.canvas.draw_idle()
        elif key == "g":
            self.crop_to_view = not self.crop_to_view
            if not self.crop_to_view:
                self.ax.set_xlim(0, self.W)
                self.ax.set_ylim(self.H, 0)
            self._draw_frame(self.frame_idx)
            self.fig.canvas.draw()

    def show(self, blit: bool = True) -> None:
        self.anim = FuncAnimation(
            self.fig, self._advance, frames=itertools.count(), interval=1000.0 / self.fps,
            blit=blit, cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        plt.show()

    def save(self, path: str, fps: int | None = None, dpi: int = 150) -> None:
        """Render every frame with no live window and write the whole episode to `path`.

        Extension picks the codec: `.mp4` needs an `ffmpeg` binary, found either on PATH or
        via the optional `imageio-ffmpeg` pip package (`pip install imageio-ffmpeg` -- it
        bundles a static binary, no system-wide install/PATH edit required). `.gif` always
        works out of the box (Pillow, already a matplotlib dependency) at the cost of a much
        larger file with no inter-frame delta compression -- fine for a quick share, worse for
        a long match.
        """
        suffix = Path(path).suffix.lower()
        fps = fps or self.fps
        if suffix == ".mp4":
            ffmpeg_path = locate_ffmpeg()
            if ffmpeg_path is None:
                raise RuntimeError(
                    "no ffmpeg binary found for .mp4 export. Either install ffmpeg and put it "
                    "on PATH, or run `pip install imageio-ffmpeg` (bundles one, no system "
                    "install needed) -- or save as .gif instead, which needs neither."
                )
            mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
            writer = FFMpegWriter(fps=fps, bitrate=-1)
        elif suffix == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError(f"unsupported video extension {suffix!r} -- use .mp4 or .gif")

        anim = FuncAnimation(self.fig, self._draw_frame, frames=self.n_frames, blit=False)
        anim.save(path, writer=writer, dpi=dpi)
        # save() leaves self.frame_idx at the last rendered frame and the figure showing it;
        # an interactive show() called afterward should still start from the beginning.
        self._draw_frame(0)


def locate_ffmpeg() -> str | None:
    """PATH first (respects an existing `animation.ffmpeg_path` override too), then the
    optional `imageio-ffmpeg` package's bundled binary. Returns None if neither is available."""
    configured = mpl.rcParams.get("animation.ffmpeg_path", "ffmpeg")
    found = shutil.which(configured)
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", help="path to a rollout.npz written by scripts/record_rollout.py")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument(
        "--preset", default=None,
        help="optional configs/presets/*.yaml overrides -- MUST match the preset used to "
             "record the npz, since map_id indexes cfg.map_names",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--video", default=None,
                   help="write the replay to this .mp4 or .gif instead of opening the "
                        "interactive viewer (see ReplayViewer.save for what .mp4 needs)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = None
    if args.preset:
        import yaml
        overrides = yaml.safe_load(Path(args.preset).read_text())
    cfg = load_config(args.config, overrides=overrides)
    bank = build_map_bank(cfg, device="cpu")

    with np.load(args.npz) as data:
        frames = {name: data[name] for name in data.files}

    viewer = ReplayViewer(frames, bank, cfg, fps=args.fps)
    if args.video:
        viewer.save(args.video, fps=args.fps)
        print(f"[viewer] wrote {args.video}")
    else:
        viewer.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
