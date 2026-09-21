# Terrain relabel — `2maps_dataset.mp4`

One BlueStacks session, 12.8 minutes, **13 back-to-back Showdown matches on 13 different maps** — 6 purple graveyard, 7 tan desert. This is the whole training set for the next terrain classifier; none of the older labels are used.

**100 frames to label.** They were picked by running odometry over each match and greedily taking the frame that shows the most world no earlier pick shows, then topping up with frames that carry water or fence. Every one was looked at; damage-flash frames were thrown out. Skip any that still look bad — the list has slack.

```bash
python scripts/vision_label.py 2maps_dataset <frame>
```

## The 13 maps

| match | map | style | frames to label | gameplay | tell |
|---|---|---|---|---|---|
| m01 | Dark Passage (train) | graveyard | 3 | 12 s | lantern-and-pumpkin ledge, crystal-capped walls, NO water |
| m02 | Shadow Spirits (train) | graveyard | 9 | 46 s | one long 12x3 acid pool, narrow vertical bush columns |
| m03 | Cavern Churn (**test**) | graveyard | 3 | 14 s | one small square acid pool inside the bush field, rectangular bush blocks |
| m04 | Twisting Vines (train) | graveyard | 11 | 36 s | wide winding acid rivers, pink plank crates, crypt skyline |
| m05 | Hard Limits (train) | desert | 7 | 38 s | blue-white glyph crates, tiny brown-rimmed pools, NO fence |
| m06 | Clash Colosseum (train) | desert | 11 | 63 s | brick-red rock blocks, large L-shaped pools, wooden rail fences |
| m07 | Eggshell (train) | desert | 9 | 48 s | canyon lakes incl. the clover pool, green leafy bush column east |
| m08 | Stocky Stockades (**test**) | desert | 8 | 29 s | small square/plus puddles, a long log-rail fence in every frame |
| m09 | Outrageous Outback (train) | desert | 10 | 56 s | narrow stair-step stream, picket fence, green bush band north |
| m10 | Island Invasion (**test**) | desert | 9 | 30 s | badlands: bones, enormous hay fields, big west lake, NO fence |
| m11 | Final Four (train) | desert | 8 | 30 s | three blue pools, white cross-braced fence column east, hay bales |
| m12 | Mystical Thirty Three (**test**) | graveyard | 7 | 32 s | gravestone walls, small lime pools, white iron-spike fence, heavy late gas |
| m13 | Gated Community (train) | graveyard | 5 | 16 s | bone-white zig-zag blocks east, green-tipped spike fence, two acid pools |

The names are read off the map-pick overlay in the queue footage, not guessed from the art, and
the art agrees with them everywhere it can be checked. Nothing repeats: 13 matches, 13 maps. Any
match-level split is therefore map-disjoint, but there is also no second recording of any map to
fall back on if one turns out to be unusable.

## The frames

Frames tagged `water/fence` carry one of the two rare classes. The note says what is in the way, if anything.

### m01 — Dark Passage, graveyard — 3 frames

Lantern-and-pumpkin ledge, crystal-capped walls, no water.

```bash
python scripts/vision_label.py 2maps_dataset 120   # 323 new cells
python scripts/vision_label.py 2maps_dataset 270   # 161 new cells
python scripts/vision_label.py 2maps_dataset 420   # 319 new cells
```

- **120** — clean mid-match view with full fresh coverage; only the player, two loot barrels and scattered stumps.
- **270** — cleanest frame of the match — player plus one bot, wide open floor, large bush mass and lavender walls all crisply lit.
- **420** — colours normal (floor red channel 70 vs 113 on the rejected f450); an opaque kill card and two bot portraits cover ~10 cells in the north-west and two brawlers stand mid-patch.

### m02 — Shadow Spirits, graveyard — 9 frames

One long 12x3 acid pool, narrow vertical bush columns.

```bash
python scripts/vision_label.py 2maps_dataset 1245  # 323 new cells
python scripts/vision_label.py 2maps_dataset 1455  # 119 new cells
python scripts/vision_label.py 2maps_dataset 1605  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 1815  #  44 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 1875  # 321 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 1935  # 106 new cells
python scripts/vision_label.py 2maps_dataset 2175  # 154 new cells
python scripts/vision_label.py 2maps_dataset 2295  # 232 new cells
python scripts/vision_label.py 2maps_dataset 2505  # 317 new cells
```

- **1245** — match-start drop frame: whole patch uniformly darkened and the spawn bubble is on the player, so it is a weaker colour reference despite 323 fresh cells.
- **1455** — clean; good example of gravestone-capped wall rows on both sides plus four well-spread loot barrels.
- **1605** — best frame of the match — 323 fresh cells, almost no sprites, and the entire green acid pool (~30 WATER cells) in clear view.
- **1815** — acid pool fills the top third at high contrast; only two kill cards over ~6 cells in the top-left corner intrude.
- **1875** — near-full fresh coverage and clean; kill cards on ~6 top-left cells, one enemy mid-left, and the acid pool clipped along the top edge.
- **1935** — a pink Super beam streaks across the upper right with an orange flash on the hit enemy, and poison blobs line the bottom edge; centre and left still readable. 18 cells are gassed and will be refused.
- **2175** — clean, only the player; poison blobs stay on the left margin and a sliver of acid pool shows at the right edge. 11 cells are gassed and will be refused.
- **2295** — poison cloud band over the left ~20-25% (green blobs and skulls over bushes and floor); the remaining 232 new cells are clean and readable. 23 cells are gassed and will be refused.
- **2505** — clean — player only — with 317 fresh cells, strong gravestone-wall and long bush-column examples; joystick ring dims ~6 cells at left. 4 cells are gassed and will be refused.

### m03 — Cavern Churn, graveyard — 3 frames  (held out for test)

One small square acid pool inside the bush field, rectangular bush blocks.

```bash
python scripts/vision_label.py 2maps_dataset 3495  # 323 new cells
python scripts/vision_label.py 2maps_dataset 3645  # 103 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 3825  # 309 new cells
```

- **3495** — full-patch fresh view, terrain clean; only a small floating-joystick ring on the player and one emote bubble.
- **3645** — cleanest m03 frame, the whole green acid pool visible, one brawler and two crates.
- **3825** — kill card top-left (~6 cells) plus five brawlers, health bars and a super effect mid-right; roughly 10% hidden, rest reads fine.

### m04 — Twisting Vines, graveyard — 11 frames

Wide winding acid rivers, pink plank crates, crypt skyline.

```bash
python scripts/vision_label.py 2maps_dataset 4590  #  69 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 4620  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 4650  #  41 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 4740  # 232 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 4830  #  68 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 5010  # 186 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 5100  # 322 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 5220  # 119 new cells
python scripts/vision_label.py 2maps_dataset 5370  # 250 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 5550  # 109 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 5610  # 320 new cells, water/fence
```

- **4590** — clean, wide acid ribbon left of centre; the white floating-joystick ring sits on the player over ~4 cells only. 2 cells are gassed and will be refused.
- **4620** — clean full-patch view with two wide acid ribbons; only the player and a small joystick/super ring.
- **4650** — very clean, acid ribbons on both left and right, no overlays.
- **4740** — clean, several acid pools right and left; only two loot crates and one brawler.
- **4830** — clean, acid channel through the middle; only loot crates and a small orange glow in the top-right corner.
- **5010** — clean central acid stream; a couple of brawlers and attack splashes confined to the right edge.
- **5100** — kill card top-left covers ~6 cells, everything else readable; big bush field and an acid pool bottom-centre. 1 cells are gassed and will be refused.
- **5220** — kill card top-left plus poison gas band recolouring the top two rows; play area below is readable. 42 cells are gassed and will be refused.
- **5370** — pale-green poison gas washes the right one-to-two columns; the remaining ~85% is clean and readable. 20 cells are gassed and will be refused.
- **5550** — poison gas over the right two-to-three columns; centre acid stream and bushes still clean. 37 cells are gassed and will be refused.
- **5610** — large acid stream down the right side; one enemy with a yellow muzzle flash over ~4 cells. 3 cells are gassed and will be refused.

### m05 — Hard Limits, desert — 7 frames

Blue-white glyph crates, tiny brown-rimmed pools, no fence.

```bash
python scripts/vision_label.py 2maps_dataset 6450  # 323 new cells
python scripts/vision_label.py 2maps_dataset 6630  # 149 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 6750  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 6900  # 166 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 7020  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 7110  # 145 new cells
python scripts/vision_label.py 2maps_dataset 7320  # 128 new cells
```

- **6450** — opening full-map view, no overlay or wash, whole trapezoid readable.
- **6630** — clean wide view, water at left, nothing occluding.
- **6750** — full-map view, clean; a few loot boxes only, water pool top-right.
- **6900** — wide clean view; only a 1-cell projectile beam top-centre, water pool bottom-centre.
- **7020** — full-map view, clean, two water pools plus both wall types and long bush bars.
- **7110** — clean; water at right edge, sprites confined to the centre column. 2 cells are gassed and will be refused.
- **7320** — damage numbers and one translucent bubble over ~2 cells; terrain elsewhere fully readable. 1 cells are gassed and will be refused.

### m06 — Clash Colosseum, desert — 11 frames

Brick-red rock blocks, large l-shaped pools, wooden rail fences.

```bash
python scripts/vision_label.py 2maps_dataset 8265  # 131 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 8355  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 8385  #  25 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 8805  # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 8955  # 143 new cells
python scripts/vision_label.py 2maps_dataset 9165  # 248 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 9315  # 124 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 9465  # 318 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 9585  #  48 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 9735  # 100 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 9945  # 298 new cells, water/fence
```

- **8265** — clean opening view; vertical fence at centre, rail top-right, small pool bottom-right. 1 cells are gassed and will be refused.
- **8355** — opening full-map view, clean; one vertical and two horizontal fences, water at the right edge.
- **8385** — clean; long fence rail top-right and a pool on the right edge.
- **8805** — full map, clean apart from a small kill card top-left; L-shaped pool centre-left and a vertical fence.
- **8955** — clean except a poison-gas band down the right ~10% of the patch; water at left edge. 24 cells are gassed and will be refused.
- **9165** — wide view, only a ~3x2 purple translucent patch at the far left edge; big centre pool and a fence. 1 cells are gassed and will be refused.
- **9315** — green poison gas recolours the top third; lower half is clean and holds a large pool plus a bottom fence. 92 cells are gassed and will be refused.
- **9465** — wide and clean, only a 1-cell projectile trail; large pool top-right plus two fences.
- **9585** — clean; fence and pool at the right, sprites confined to the centre. 1 cells are gassed and will be refused.
- **9735** — cleanest rare-class frame of the set - four wooden rail fences and four water pools, no overlay at all.
- **9945** — wide and clean apart from a small kill card top-left; two fences and two pools. 11 cells are gassed and will be refused.

### m07 — Eggshell, desert — 9 frames

Canyon lakes incl. the clover pool, green leafy bush column east.

```bash
python scripts/vision_label.py 2maps_dataset 10815 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 10875 # 109 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11025 # 218 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11175 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11415 # 251 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11715 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11775 #  98 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11895 #  48 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 11955 # 219 new cells, water/fence
```

- **10815** — match-start frame (10 brawlers), no effects, water top-left plus the clover lake east of centre; only a small white spawn glow on own brawler.
- **10875** — clean, big cloud-shaped lake right of centre and a water strip on the west edge; kill-feed portraits sit inside the dimmed HUD strip.
- **11025** — clean, water at top-left and a second pool on the right; only a tiny pink projectile trail at the top-left corner.
- **11175** — huge lake across the top-right plus a strip bottom-left; the five loot-box health bars each cover only a cell or two.
- **11415** — clean patch, water band across the north and the green leafy bush column along the east edge. 25 cells are gassed and will be refused.
- **11715** — wide river down the north-west, second pool at the bottom, and a clear log-rail fence run at the north edge; joystick ring is faint.
- **11775** — clean, river running down the middle plus a second pool bottom-right; sprites are sparse. 8 cells are gassed and will be refused.
- **11895** — very large lake across the whole north plus a fence run beside the barrels; the yellow arrow button covers only ~4 cells.
- **11955** — huge north lake, two clear wooden rail runs and crates; a kill card sits in the north-west and a 780 damage number over the player, but the floor is normal salmon, not the wash that rejected f11985. 2 cells are gassed and will be refused.

### m08 — Stocky Stockades, desert — 8 frames  (held out for test)

Small square/plus puddles, a long log-rail fence in every frame.

```bash
python scripts/vision_label.py 2maps_dataset 12960 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13020 #  85 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13110 #  57 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13200 # 221 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13350 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13470 # 122 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13620 # 300 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 13650 #  40 new cells, water/fence
```

- **12960** — match-start frame, very clean, plus-shaped pool top-left and a long fence run only partly hidden behind own health bar.
- **13020** — clean, plus-shaped pool in the centre and two fence runs; kill-feed portraits stay inside the dimmed HUD strip.
- **13110** — clean, sizeable pool centre-left and three separate fence runs.
- **13200** — clean, two separate fence runs; the bottom pool is partly under the dimmed button zone but the rest is fully readable.
- **13350** — clean, fence run bottom-left and a water pool at the bottom; health bars are small and isolated.
- **13470** — best m08 frame - large square pool centre-right plus a long unobstructed fence run, almost no sprite clutter.
- **13620** — clean, long fence run beside the cactus, green bush field on the west edge, water strip at the top. 14 cells are gassed and will be refused.
- **13650** — water pool top-right, the green bush field on the west edge and a fence run - good class mix despite only 40 new cells. 14 cells are gassed and will be refused.

### m09 — Outrageous Outback, desert — 10 frames

Narrow stair-step stream, picket fence, green bush band north.

```bash
python scripts/vision_label.py 2maps_dataset 14520 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15030 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15120 #  76 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15210 # 183 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15390 # 313 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15450 #  62 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15480 #  91 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15720 # 161 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 15930 # 118 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 16080 # 312 new cells, water/fence
```

- **14520** — clean patch, only the player's small white super ring; stream, hay bushes and a picket fence south of centre all readable.
- **15030** — clean full-frame view, one brawler; long water channel, crates and a fence at the top right.
- **15120** — clean; two water channels and hay columns, joystick ring parked in the already-dimmed lower-left.
- **15210** — one of the best frames in the set - two long water channels, a picket fence at top centre, green leafy bushes west and hay east, with only the player and one bot on screen. 10 cells are gassed and will be refused.
- **15390** — clean; water pool and channel plus a fence on the upper right.
- **15450** — clean; the north band of green leafy bushes, a rounded pool top-right and a rail structure mid-right, kill card confined to the dimmed north-west corner. 69 cells are gassed and will be refused.
- **15480** — north-edge green bush band, small pool and a fence; the kill card sits inside the already-dimmed top-left corner. 96 cells are gassed and will be refused.
- **15720** — clean; water channels and two fences, the one purple effect is in the dimmed bottom-right.
- **15930** — pink smoke puff over roughly 3x3 cells left of centre and a translucent ghost sprite, but the two water channels and green bushes still read. 52 cells are gassed and will be refused.
- **16080** — pink Super burst covers about 3x4 cells top centre plus two brawlers, the remaining ~90% of the patch is clean. 3 cells are gassed and will be refused.

### m10 — Island Invasion, desert — 9 frames  (held out for test)

Badlands: bones, enormous hay fields, big west lake, no fence.

```bash
python scripts/vision_label.py 2maps_dataset 16890 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 16920 #  36 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 16980 #  97 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 17070 # 323 new cells
python scripts/vision_label.py 2maps_dataset 17310 # 299 new cells
python scripts/vision_label.py 2maps_dataset 17460 # 103 new cells
python scripts/vision_label.py 2maps_dataset 17520 # 323 new cells
python scripts/vision_label.py 2maps_dataset 17580 #  38 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 17670 # 200 new cells, water/fence
```

- **16890** — clean, only the player's super ring; huge west lake and south-east water, best water coverage of the match.
- **16920** — very clean single-brawler frame with the big west lake and crate cluster.
- **16980** — very clean; west lake, hay field and crates, nothing occluding.
- **17070** — narrow cyan projectile beam over ~3 cells and a fading kill card in the dim corner; full 323-cell view otherwise clean.
- **17310** — kill card in the dim corner plus a bright white Super flash over ~2x2 cells and a faint smoke haze low centre; terrain colours still read. 4 cells are gassed and will be refused.
- **17460** — two brawlers side by side with health bars in the centre, everything else clean; crates and bones readable. 2 cells are gassed and will be refused.
- **17520** — four brawlers with stacked health bars down the centre column and a small pink effect at the left, but roughly 80% of this full-view frame is clean.
- **17580** — clean; wide north water band, barrels and crates, the only effect is in the dimmed bottom strip. 2 cells are gassed and will be refused.
- **17670** — wide north water band and huge hay field; kill card in the dim corner and a thin pink projectile trail near the bottom. 1 cells are gassed and will be refused.

### m11 — Final Four, desert — 8 frames

Three blue pools, white cross-braced fence column east, hay bales.

```bash
python scripts/vision_label.py 2maps_dataset 18855 # 323 new cells
python scripts/vision_label.py 2maps_dataset 18885 #  32 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19005 # 136 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19185 #  33 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19215 # 321 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19275 #  67 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19635 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 19695 # 108 new cells, water/fence
```

- **18855** — clean match-start full patch; only the player, crates and a ~2-cell joystick knob, with water/bush/wall all legible.
- **18885** — clean match-start; single brawler, clear water pool in the south, only a faint translucent joystick ring.
- **19005** — opaque kill card over the north-west corner (~12 cells) and a red-flash fight in the north-east corner; the fence column and the rest read fine.
- **19185** — clean; fence column on the east, two enemies in the north-east corner, small joystick knob at the west. 2 cells are gassed and will be refused.
- **19215** — very clean; long white cross-brace fence column on the east, hay bushes, two distant enemies. 2 cells are gassed and will be refused.
- **19275** — cleanest frame on the sheet - one brawler, big water pools, fence column north-east, no overlays.
- **19635** — full patch with three large blue pools plus both fence arts, only a small skirmish in the south-centre.
- **19695** — three-way brawl with damage numbers, loot and purple/pink effects covering roughly a fifth of the patch centre-left; north half and east side stay clean.

### m12 — Mystical Thirty Three, graveyard — 7 frames  (held out for test)

Gravestone walls, small lime pools, white iron-spike fence, heavy late gas.

```bash
python scripts/vision_label.py 2maps_dataset 20535 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 20805 # 251 new cells
python scripts/vision_label.py 2maps_dataset 20925 #  65 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 20985 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 21195 # 149 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 21285 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 21465 # 226 new cells
```

- **20535** — match-start: opaque kill card over the north-west corner, white spawn bubble around the player, flame/pumpkin effects in the north-east; acid pool and most terrain still readable.
- **20805** — kill card in the north-west plus a wide scatter of green gas-warning flames over the west half and south (~1/8 of cells); the rest reads fine.
- **20925** — clean; a clear white spike-fence run across the north-centre, gravestone rows and teal bushes, only uneven map lighting.
- **20985** — clean full patch showing two clear runs of white spike fence; clutter is limited to four bots with health bars along the north-centre.
- **21195** — clean single-brawler frame with a large acid pool centre-left and smaller ones at the west edge. 4 cells are gassed and will be refused.
- **21285** — clean full patch with a bright acid pool in the south-west, both wall arts and plenty of teal bush.
- **21465** — clean and evenly dim; gas flames confined to the north-east diagonal, long gravestone wall run and bushes clearly readable.

### m13 — Gated Community, graveyard — 5 frames

Bone-white zig-zag blocks east, green-tipped spike fence, two acid pools.

```bash
python scripts/vision_label.py 2maps_dataset 22320 # 323 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 22410 #  37 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 22500 # 224 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 22620 #  77 new cells, water/fence
python scripts/vision_label.py 2maps_dataset 22740 # 323 new cells, water/fence
```

- **22320** — 323 new cells, clean early-game read; only two brawlers with small white spawn bubbles, walls/bush/fence rows all legible.
- **22410** — clean, well-lit, the sharpest read of the green-tipped fence rows; only 37 new cells but nothing obscured.
- **22500** — 224 new cells, opens the east side with the bone-white wall blocks and the second acid pool; one small emote bubble over the pool.
- **22620** — 77 new cells but the clearest large water pool together with a fence line; small kill card top-left is the only obstruction.
- **22740** — 323 new cells and the widest acid-pool coverage in the match; only a small kill card in the top-left corner and a few projectile sprites.

## Labelling notes for this footage

- **The joystick floats.** It is drawn wherever you last touched, so the emulator HUD mask does not cover it. Leave those cells `?` as you planned; the notes above say which frames have it sitting over the play area.
- **Green acid is WATER.** The graveyard maps' bright lime pools are cell-aligned terrain with a glowing rim, the same role the desert maps' cyan pools play. Label them `4 ~`. The dark navy oval decals painted on the floor and on bushes are decoration, not water.
- **Fence is not as rare as you thought.** Eight of the 13 maps have real fences: wooden rails (m06, m07, m08), a picket fence (m09), a white cross-brace column (m11), iron spike railing (m12) and green-tipped spikes (m13). Judge by the art — you can see floor between the posts. Gravestone-capped blocks are flush and solid, so they are WALL.
- **Crates, barrels and cacti are not their own class.** Crates and barrels block movement and shots, so they go in with WALL; cacti and flowers sit on FLOOR.
- **Two bush arts coexist in the desert maps**: yellow straw and the bright green leafy clumps along one map edge (m07, m08, m09, m10). Both are BUSH.
- **Gas barely appears.** Most frames have none; the handful that do are noted above and the tool refuses those cells anyway.
- **Kill cards land in the top-left**, which the emulator mask already dims, so they usually cost nothing. Loot boxes and brawlers do not — leave what they cover `?`.

## Train and test

Every label carries the same clip name, so `--hold-out` cannot split this set — it matches on clip name and would take all 100 or none. Split by directory instead, whole matches on one side only:

```bash
mkdir -p tests/fixtures/vision/labels_2maps/train tests/fixtures/vision/labels_2maps/test
# then move each 2maps_dataset_f<frame>.json into train/ or test/ by the table above
```

Held out for test: **m03, m08, m10, m12** — 27 frames, two graveyard-style and two desert-style maps that between them carry all five classes. The other nine maps, 73 frames, train.

```bash
python scripts/vision_train_terrain.py --labels tests/fixtures/vision/labels_2maps/train --out runs/terrain_2maps.pt
python scripts/vision_score_map.py --labels tests/fixtures/vision/labels_2maps/test --model runs/terrain_2maps.pt
```

The trainer will say nothing measures cross-map generalization — that is the clip-name check firing, and the scorer run above is what actually answers it.

One caveat on the scorer: it builds each label's map from views ±10 s, and for a label near a match boundary that window reaches into the menu before or after the match. Measured on the first and last pick of every match, 24 of 26 let through 2–93 out-of-match views, worst at m13 f22320. If a boundary frame scores oddly, that is why.

## Sidecars written for this recording

- `2maps_dataset.mp4.bounds.json` — usable range re-measured with `tail_fraction=0.98`. The 0.85 default cuts at frame 19781, a match transition, which would have hidden matches 12 and 13 from the labeller and the scorer.
- `2maps_dataset.mp4.gameplay.json` — the button scan at `DEPLOY_RADIUS_PX`, with all 13 match runs. A default rescan at the wider radius reports no gameplay at all on this footage.
- `labeling.EMULATOR_CLIP_PREFIXES` now includes `2maps_dataset`, so new labels take the emulator HUD mask rather than the phone one.

