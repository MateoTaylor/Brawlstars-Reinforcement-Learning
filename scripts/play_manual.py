"""Live keyboard play of BrawlVecEnv(n_envs=1) at 20 Hz -- the human feel-check for Mortis's
dash. See BRAWL_SIM_BUILD_PLAN.md Step 39.

**Matplotlib** (Notice 3 -- no pygame wheel for this machine's cp314 interpreter), same as
`render/viewer.py` (Step 38). This module reuses `ReplayViewer` PURELY as an artist/drawing
toolkit -- it never calls `ReplayViewer.show()`, whose own `FuncAnimation` and key bindings
(space=pause, left/right=scrub, ...) are built for scrubbing a pre-recorded rollout, not driving
a live one, and would collide with this module's WASD/space controls (space means "fire" here,
not "pause"). `ManualPlaySession` instead drives `ReplayViewer._draw_frame` directly: each tick
it replaces `viewer.frames` with a fresh one-frame dict (`{name: arr[None] for ...}`, the same
shape trick `tests/test_viewer.py` uses to feed `ReplayViewer` hand-built scenarios) built from
`scripts.record_rollout.extra_fields(env)` + `core.state.snapshot(env.state, 0)` -- the exact
same per-tick data a recorded rollout would have produced, so this is honestly the same renderer
Step 38 already shipped and tested, not a fork of it.

**Held-key state, not discrete key events, drives movement/fire** -- `key_press_event` adds a
key to `_held_keys`, `key_release_event` discards it, and every 20 Hz tick reads whatever's
currently held. This is what makes "hold W to walk" or "hold space to keep dashing once ammo/
cooldown allow it again" work the way an actual human playtester expects; it also matches
`core/hero.py.decode_action`'s own documented contract that illegal fire is a silent no-op, not
an error -- holding space through a reload is expected and harmless, action_mask just keeps
gating it every tick until ammo/cooldown allow it. One caveat, stated plainly rather than
assumed away: key-repeat/release fidelity is a property of the matplotlib GUI backend in use
(Qt/Tk/etc.), not of this module -- if a backend ever drops a release event, run() prints the
backend name at startup so that's the first thing to check.

**WASD/arrows select "the nearest of the 16 move bins"** via `core.geometry.bin_from_dir` (the
existing inverse of `dir_from_bin`, previously unused outside its own module/tests) on the held
direction's (x, y) unit-ish vector -- not a hand-rolled angle table. `action[:, 0]` follows
`core/hero.py.decode_action`'s own encoding exactly: 0 = idle, `bin_from_dir(...) + 1` otherwise
(bin k's direction lives at move_bin k+1, since move_bin 0 is reserved for idle).

**The attack indicator is `hero.action_mask(...)["attack"][:, 1]`** (the exact same gate
`decode_action` applies to `action[:, 1]` before it ever reaches the sim) drawn as a small
`Rectangle` in axes-fraction coordinates (`transform=ax.transAxes`, top-left corner) so it
doesn't scale or move with the map -- green when firing is legal, grey otherwise. Since that one
boolean is `alive & ammo>=1 & cooldown<=0 & dash_t<=0`, it already IS "greys out during dash and
while reloading" in one gate, per the plan's acceptance bullet -- no separate dash/reload cases
needed.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from brawl_sim.config import load_config
from brawl_sim.core import geometry as geo
from brawl_sim.core import hero
from brawl_sim.core.state import snapshot as state_snapshot
from brawl_sim.env import BrawlVecEnv
from brawl_sim.maps.loader import build_map_bank
from brawl_sim.render.viewer import ReplayViewer
try:
    from scripts.record_rollout import extra_fields
except ModuleNotFoundError:
    # `python scripts/play_manual.py` (the documented invocation) puts scripts/ -- not the repo
    # root -- on sys.path, so the `scripts.` namespace package isn't importable that way; under
    # pytest the repo root IS on the path and the qualified form is the one that resolves.
    # Without this fallback the documented CLI raised ModuleNotFoundError while the tests passed.
    from record_rollout import extra_fields

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

_MOVE_KEYS = {
    "w": (0.0, -1.0), "up": (0.0, -1.0),
    "s": (0.0, 1.0), "down": (0.0, 1.0),
    "a": (-1.0, 0.0), "left": (-1.0, 0.0),
    "d": (1.0, 0.0), "right": (1.0, 0.0),
}
_FIRE_KEYS = {" ", "space"}
_QUIT_KEYS = {"q", "escape"}

_INDICATOR_READY = "#2ecc40"
_INDICATOR_BLOCKED = "#888888"


class ManualPlaySession:
    def __init__(self, env: BrawlVecEnv, bank, cfg, fps: int = 20):
        self.env = env
        self.bank = bank
        self.cfg = cfg
        self.fps = fps

        self._held_keys: set[str] = set()
        self._quit_requested = False
        self.anim = None

        env.reset()
        self.viewer = ReplayViewer(self._one_frame(), bank, cfg, fps=fps)
        self.indicator = Rectangle(
            (0.02, 0.94), 0.05, 0.04, transform=self.viewer.ax.transAxes,
            facecolor=_INDICATOR_READY, edgecolor="black", linewidth=0.5, zorder=10,
        )
        self.viewer.ax.add_patch(self.indicator)
        self._update_overlay()

    # ---- state -> renderer plumbing --------------------------------------------------------

    def _one_frame(self) -> dict:
        frame = state_snapshot(self.env.state, 0)
        frame.update(extra_fields(self.env))
        return {name: arr[None] for name, arr in frame.items()}

    def _update_overlay(self) -> None:
        """Redraws the current state and appends the action-mask/dash line the plan's
        acceptance bar asks for ('Display the action mask and current dash state')."""
        self.viewer.frames = self._one_frame()
        artists = self.viewer._draw_frame(0)

        mask = hero.action_mask(self.env.state, self.env.params, self.env.cfg)
        fire_ok = bool(mask["attack"][0, 1])
        dash_t = float(self.env.state.ent_dash_t[0, 0])
        dash_str = f"ACTIVE({dash_t:.2f}s)" if dash_t > 0 else "-"
        extra = f"  |  fire={'READY' if fire_ok else 'blocked'}  dash={dash_str}"
        self.viewer.title.set_text(self.viewer.title.get_text() + extra)

        self.indicator.set_facecolor(_INDICATOR_READY if fire_ok else _INDICATOR_BLOCKED)
        return artists + [self.indicator]

    # ---- input --------------------------------------------------------------------------------

    def _held_direction(self) -> tuple:
        dx = sum(v[0] for k, v in _MOVE_KEYS.items() if k in self._held_keys)
        dy = sum(v[1] for k, v in _MOVE_KEYS.items() if k in self._held_keys)
        return dx, dy

    def action_tensor(self) -> torch.Tensor:
        """(1,2) i64, exactly `core.hero.decode_action`'s own encoding: action[:,0]=0 is idle,
        `bin_from_dir(dir, n_move_bins) + 1` otherwise; action[:,1] is raw fire intent (still
        subject to `action_mask`'s legality gate once it reaches `env.step`, same as any other
        action source -- this module never pre-filters it, matching decode_action's own "illegal
        fire is a silent no-op" contract)."""
        dx, dy = self._held_direction()
        if dx == 0.0 and dy == 0.0:
            move_bin = torch.zeros(1, dtype=torch.int64, device=self.env.device)
        else:
            direction = torch.tensor([[dx, dy]], dtype=torch.float32, device=self.env.device)
            move_bin = geo.bin_from_dir(direction, self.cfg.n_move_bins) + 1  # (1,), already squeezed

        fire = torch.tensor(
            [1 if self._held_keys & _FIRE_KEYS else 0], dtype=torch.int64, device=self.env.device,
        )
        return torch.stack([move_bin, fire], dim=1)

    def on_key_press(self, event) -> None:
        key = (getattr(event, "key", None) or "").lower()
        if key in _QUIT_KEYS:
            self._quit_requested = True
            return
        self._held_keys.add(key)

    def on_key_release(self, event) -> None:
        self._held_keys.discard((getattr(event, "key", None) or "").lower())

    def on_button_press(self, event) -> None:
        if getattr(event, "button", None) == 1:
            self._held_keys.add(" ")

    def on_button_release(self, event) -> None:
        if getattr(event, "button", None) == 1:
            self._held_keys.discard(" ")

    # ---- loop -----------------------------------------------------------------------------

    def tick(self, _unused_frame=None) -> list:
        self.env.step(self.action_tensor())
        artists = self._update_overlay()
        if self._quit_requested and self.anim is not None:
            self.anim.event_source.stop()
            plt.close(self.viewer.fig)
        return artists

    def run(self) -> None:
        print(f"matplotlib backend: {matplotlib.get_backend()}")
        print("WASD/arrows move, space or left-click fires, q/Esc quits.")
        fig = self.viewer.fig
        self.anim = FuncAnimation(
            fig, self.tick, interval=1000.0 / self.fps, blit=True, cache_frame_data=False,
        )
        fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        plt.show()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--preset", default=None, help="optional configs/presets/*.yaml overrides file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--fps", type=int, default=20)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = None
    if args.preset:
        import yaml
        overrides = yaml.safe_load(Path(args.preset).read_text())
    cfg = load_config(args.config, overrides=overrides)

    env = BrawlVecEnv(cfg, n_envs=1, device=args.device, seed=args.seed)
    bank = build_map_bank(cfg, device="cpu")
    session = ManualPlaySession(env, bank, cfg, fps=args.fps)
    session.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
