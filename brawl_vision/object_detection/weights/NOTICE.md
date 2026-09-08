# Third-party detector weights

Nothing in this folder was trained here, and the `.onnx` files are **not tracked in git** —
`weights.py` fetches them on demand and this file records what they are. Two reasons they are
fetched rather than committed: 10.6 MB against a repo whose largest tracked file is 753 KB, and
they are someone else's AGPL-licensed artifact that this repo is not in a position to
redistribute.

```
python scripts/vision_fetch_detector.py
```

## PylaEntityDetectorV2

| | |
|---|---|
| File | `PylaEntityDetectorV2.onnx` |
| SHA-256 | `e9e2394ab3b65c92b88334a4ecc6675c054ee717c5fea9f35ad8f111ba13a9a4` |
| Size | 10,585,057 bytes |
| Source | [AngelFireLA/BrawlStarsBotMaking](https://github.com/AngelFireLA/BrawlStarsBotMaking), `models/entity detection/.onnx/` |
| Architecture | Ultralytics YOLOv11n, exported by ultralytics 8.3.73 on 2025-05-26 |
| Input | `images`, `(1, 3, 640, 640)` float32 RGB in [0, 1] |
| Output | `output0`, `(1, 7, 8400)` — `cxcywh` + 3 class scores, **no NMS baked in** |
| Classes | `0: enemy, 1: teammate, 2: player` |
| Licence | AGPL-3.0 (stamped into the file's own metadata by the ultralytics exporter) |
| Author | listed as REDACTED in the upstream table; uploaded 16/06/2025 |

**The class ORDER differs between V1 and V2.** V1 is `enemy, player, teammate`; V2 is
`enemy, teammate, player`. `detector.py` reads the names out of the file's metadata for exactly
this reason — hardcoding either list silently swaps player and teammate on the other model.

**Upstream has no releases and no tags.** The URL points at `main`, so the file there can change
under us. That is why `weights.py` pins the SHA-256 and checks it on every fetch: a replaced model
would otherwise show up as mysteriously different boxes rather than as an error.

### Licence, honestly

Ultralytics stamps AGPL-3.0 into every model it exports, and the upstream repo that publishes this
one carries no licence file at all. Fetching it for local experiments is one thing; shipping a
product built on it is a question to settle before that happens, not after. Nothing in this repo
depends on these weights — `object_detection/` is optional, and the terrain pipeline never touches
it.

## Not fetched, but available upstream

Same repo, same `models/` tree, should any of them become useful. Adding one is a row in
`weights.MODELS` (name, URL, digest) — the detector reads its classes and input size from the file.

| Model | Classes |
|---|---|
| `PylaEntityDetectorV1` | `enemy, player, teammate` (YOLOv8) |
| `PylaWallDetectorV2` | `wall, bush, close_bush` |
| `PylaWallDetectorV1` | 15 specific obstacle types (`cactus`, `wooden_barrel`, `blue_post`, …) |
| `PylaSpecificBrawlerDetectorV1` | ~100: per-brawler identity, health bars, ammo, gadget, super, wall, bush |

`PylaWallDetectorV2` is the interesting one for this project: walls are the terrain classifier's
outstanding failure, and a second, independent opinion on where they are is worth having. It would
be a **comparison** against the terrain map, though, not a replacement for it — this detector sees
one frame in screen space, and the occupancy grid accumulates a world in tile space.
