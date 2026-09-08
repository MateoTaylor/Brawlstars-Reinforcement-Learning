"""Download the third-party detector weights that `brawl_vision.object_detection` runs.

    python scripts/vision_fetch_detector.py               # the configured model
    python scripts/vision_fetch_detector.py --all
    python scripts/vision_fetch_detector.py --force       # re-download and re-verify

They are fetched rather than committed: 10.6 MB of somebody else's AGPL-licensed artifact, from a
repo with no releases and no licence file. `brawl_vision/object_detection/weights/NOTICE.md`
records the provenance, the digest and the licence question; read it before building anything
that ships.
"""
import argparse
import sys

from brawl_vision.config import load_vision_config
from brawl_vision.object_detection import weights


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default=None,
                   help=f"one of: {', '.join(sorted(weights.MODELS))} "
                        f"(default: whatever configs/vision.yaml's detector.model names)")
    p.add_argument("--all", action="store_true", help="fetch every known model")
    p.add_argument("--force", action="store_true", help="re-download even if the file is here")
    args = p.parse_args(argv)

    if args.all:
        names = sorted(weights.MODELS)
    elif args.model:
        names = [args.model]
    else:
        names = [load_vision_config().detector_model]

    for name in names:
        try:
            path = weights.weights_path(name)
        except KeyError as exc:
            print(exc.args[0], file=sys.stderr)
            return 2
        had = path.exists()
        path = weights.fetch(name, force=args.force)
        size = path.stat().st_size
        verb = "already present" if had and not args.force else "fetched"
        print(f"{name}: {verb} -- {path} ({size / 1e6:.1f} MB)")
    print("\nSee brawl_vision/object_detection/weights/NOTICE.md for provenance and licence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
