"""The camera as the sim models it: what the screen shows and where it sits. Three facts measured
from the deployed capture path (BRAWL_SIM_DESIGN.md §9):

- the screen shows a trapezoid of ground (`cfg.camera_quad`), not the 21x13 rectangle the grid
  crops, so `in_camera` is four half-planes and not a box test -- the box would discard 23.6 % of
  on-screen sightings;
- the game camera follows the hero until it is `cfg.camera_clamp_onset` tiles from a map edge and
  then stops, so near an edge the hero drifts off-centre on screen (`camera_centre`);
- the deployed detector reports exactly the enemies inside that trapezoid, so the hero's reveal is
  concealment (`bots/perception.visibility`) AND the window (`hero_view`), never the whole map.

`cfg.camera_quad` is in tiles relative to the hero's nominal screen anchor, the viewport centre
plus `brawl_vision.camera.HERO_ANCHOR_TILES`; tests/test_sim_camera.py pins it to the shipped
homography through that constant. The quad derives from HERO_ANCHOR_TILES and must never diverge
from it (user's rule).

Everything here is branch-free and the constants are built once per (device, dtype, config), so
the per-decision path constructs no tensors (see core/geometry.vec2 on why that matters on CUDA).
"""
import torch

_cache: dict = {}


def _constants(cfg, device, dtype):
    key = (str(device), dtype, cfg.camera_quad, cfg.camera_clamp_onset, cfg.map_w, cfg.map_h)
    c = _cache.get(key)
    if c is None:
        quad = torch.tensor(cfg.camera_quad, dtype=dtype, device=device)          # (4, 2) TL TR BR BL
        edge = torch.roll(quad, -1, 0) - quad                                     # (4, 2) a -> b
        west, east, north, south = cfg.camera_clamp_onset
        lo = torch.tensor((west, north), dtype=dtype, device=device)
        hi = torch.tensor((float(cfg.map_w) - east, float(cfg.map_h) - south), dtype=dtype,
                          device=device)
        # A map narrower than the two onsets on an axis has no tracking range there at all: the
        # camera sits at the midpoint. Folding that into lo/hi makes camera_centre a plain clamp.
        mid = (lo + hi) / 2
        narrow = hi < lo
        c = (quad, edge, torch.where(narrow, mid, lo), torch.where(narrow, mid, hi),
             (quad.amin(0), quad.amax(0)))
        _cache[key] = c
    return c


def camera_centre(hero_pos: torch.Tensor, cfg) -> torch.Tensor:
    """(N, 2) hero position -> (N, 2) the point the viewport is centred on: the hero, clamped to
    stay `clamp_onset` tiles inside each map edge (west/north are the lower bounds, east/south
    the upper). Equals the hero wherever the camera is tracking."""
    _, _, lo, hi, _ = _constants(cfg, hero_pos.device, hero_pos.dtype)
    return torch.minimum(torch.maximum(hero_pos, lo), hi)


def in_camera(rel: torch.Tensor, cfg) -> torch.Tensor:
    """(..., 2) camera-relative tiles -> (...) bool, inside the trapezoid the screen shows.

    Four half-planes, boundary inclusive, corners wound TL, TR, BR, BL with y down (config.validate
    refuses any other winding). The same predicate scripts/probes/obs_quad_measure.py uses, so its
    numbers stay comparable."""
    quad, edge, _, _, _ = _constants(cfg, rel.device, rel.dtype)
    d = rel.unsqueeze(-2) - quad                                    # (..., 4, 2)
    cross = edge[:, 0] * d[..., 1] - edge[:, 1] * d[..., 0]         # (..., 4)
    return (cross >= 0).all(-1)


def hero_view(state, vis: torch.Tensor, cfg) -> torch.Tensor:
    """(N, E) bool: what the hero's observation may show. `vis`'s hero row (bush concealment, and
    nothing for a dead hero) AND inside the camera window around the clamped camera centre.
    Column 0 is the hero itself, left as `vis` has it; consumers that must drop it already do."""
    cam = camera_centre(state.ent_pos[:, 0], cfg)
    return vis[:, 0, :] & in_camera(state.ent_pos - cam.unsqueeze(1), cfg)


def window_bbox(cfg, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis (min, max) of the quad, two (2,) tensors: the axis-aligned box around the window,
    for a cheap first test before the half-planes (the zone latch uses it)."""
    return _constants(cfg, device, dtype)[4]
