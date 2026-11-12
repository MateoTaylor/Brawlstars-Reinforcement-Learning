#!/bin/bash
# Replays every regression clip with each model given as tag=terrain_name (label checks + seen maps).
#   regress.sh shipped=default cand=truth_candidate plus=truth_plus
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../../.."
SCR="$HERE"
OUT=runs/map_checks/truth/regression
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=-1 EXTRA_TERRAIN="truth_candidate=runs/terrain_truth/candidate.pt;truth_plus=runs/terrain_truth/candidate_plus.pt"
one() {  # clip map terrain tag
  SEEN_OUT=$OUT/$4.npz .venv/Scripts/python.exe "$SCR/replay_seen.py" brawl_vision/data/training_videos/$1.mp4 \
    --map $2 --terrain $3 --mask on --device cpu --quiet --csv $OUT/$4.csv > $OUT/$4.log 2>&1
}
for spec in "$@"; do
  tag=${spec%%=*}; model=${spec#*=}
  one bluestacks-dark-passage-bottom-left dark_passage $model bottom-left-$tag &
  one bluestacks-dark-passage-bottom-right dark_passage $model bottom-right-$tag &
  one bluestacks-dark-passage-top-right dark_passage $model top-right-$tag &
  one bluestacks-starfruit-supernova-west-upper starfruit_supernova $model starfruit-$tag &
  wait
done
for spec in "$@"; do tag=${spec%%=*}; for f in $OUT/*-$tag.log; do echo "== $f"; grep -E "map: |label check" $f | cut -c1-200; done; done
