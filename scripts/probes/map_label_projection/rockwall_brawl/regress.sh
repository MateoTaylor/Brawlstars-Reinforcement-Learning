#!/bin/bash
# Replays every regression clip with each model given as tag=terrain_name (label checks + seen maps):
# Dark Passage's three tuning clips, Starfruit, Truth and both Doom Shroom recordings, seven at a time.
# A name that is not in TERRAIN_MODELS comes from $EXTRA_TERRAIN ("name=path;name=path"), registered
# by ../truth/replay_seen.py. Doom Shroom's copy (../doom_shroom/regress.sh) replayed the first five.
#   EXTRA_TERRAIN=rock_cand=runs/terrain_rockwall_brawl/candidate.pt regress.sh shipped=default cand=rock_cand
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../../.."
OUT=runs/map_checks/rockwall_brawl/regression
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=-1 EXTRA_TERRAIN="${EXTRA_TERRAIN:-}"
one() {  # clip map terrain tag
  SEEN_OUT=$OUT/$4.npz .venv/Scripts/python.exe "$HERE/../truth/replay_seen.py" brawl_vision/data/training_videos/$1.mp4 \
    --map $2 --terrain $3 --mask on --device cpu --quiet --csv $OUT/$4.csv > $OUT/$4.log 2>&1
}
for spec in "$@"; do
  tag=${spec%%=*}; model=${spec#*=}
  one bluestacks-dark-passage-bottom-left dark_passage $model bottom-left-$tag &
  one bluestacks-dark-passage-bottom-right dark_passage $model bottom-right-$tag &
  one bluestacks-dark-passage-top-right dark_passage $model top-right-$tag &
  one bluestacks-starfruit-supernova-west-upper starfruit_supernova $model starfruit-$tag &
  one bluestacks-truth-west-upper truth $model truth-$tag &
  one bluestacks-doom-shroom-south-center doom_shroom $model doom-south-$tag &
  one bluestacks-doom-shroom-north-center doom_shroom $model doom-north-$tag &
  wait
done
for spec in "$@"; do tag=${spec%%=*}; for f in $OUT/*-$tag.log; do echo "== $f"; grep -E "map: |label check" $f | cut -c1-200; done; done
