"""Generate candidate Solo Showdown maps.

    .venv/Scripts/python.exe scripts/gen_maps.py --family standard --seed 1 --count 3 --out-dir runs/maps --preview
    .venv/Scripts/python.exe scripts/gen_maps.py --family all --seed 100 --out-dir runs/maps

Writes `<out-dir>/<family>_<seed>.csv` per map (the CSV the loader reads; copy it to
`brawl_sim/maps/csv/<name>.csv` to adopt it) and prints one line per map with the seed that
passed and its shares, plus the seeds `generate` rejected on the way and why. `--family all`
produces each family's planned count (6 standard, 1 open, 1 dense, 2 water_border, and 3 each
of the five screenshot families: maze, lake_ring, branches, crescent, vines) unless `--count`
is given.
Seeds walk forward: the map after the one that passed at seed s starts from s + 1, so every
output is reproducible from its own seed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from brawl_sim.maps import generate as g  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="standard", choices=list(g.FAMILIES) + ["all"])
    ap.add_argument("--seed", type=int, default=1, help="first seed to try")
    ap.add_argument("--count", type=int, default=None,
                    help="maps per family (default 1, or the family's planned count with --family all)")
    ap.add_argument("--symmetry", default=None, choices=[g.POINT, g.MIRROR],
                    help="override the family's symmetry")
    ap.add_argument("--out-dir", default="runs/maps")
    ap.add_argument("--preview", action="store_true", help="print the ASCII grid of each map")
    args = ap.parse_args(argv)

    families = list(g.FAMILIES) if args.family == "all" else [args.family]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    for name in families:
        fam = g.FAMILIES[name]
        count = args.count if args.count is not None else (fam.count if args.family == "all" else 1)
        for _ in range(count):
            grid, stats = g.generate(seed, fam, symmetry=args.symmetry)
            for rejected_seed, reasons in stats["rejected"]:
                print(f"[gen_maps] {name} seed {rejected_seed} rejected: {'; '.join(reasons)}")
            path = out_dir / f"{name}_{stats['seed']}.csv"
            path.write_text(g.to_csv(grid))
            print(f"[gen_maps] {name} seed={stats['seed']} sym={stats['symmetry']} "
                  f"wall={stats['wall']:.3f} bush={stats['bush']:.3f} water={stats['water']:.3f} "
                  f"boxes={stats['boxes']} waypoints={stats['bush_waypoints']} "
                  f"spread={stats['bush_spread']:.2f} pocket_min={stats['pocket_min']:.2f} "
                  f"attempts={stats['attempts']} -> {path}")
            if args.preview:
                print(g.render(grid))
                print()
            seed = stats["seed"] + 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
