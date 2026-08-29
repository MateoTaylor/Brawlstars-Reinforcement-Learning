# Sprite assets for `render/viewer.py`

Drop PNGs in here with the exact filenames below and the replay viewer picks them up
automatically on next launch -- no code changes needed. Anything missing just keeps drawing
its existing circle/diamond/square fallback, so you can fill this in incrementally.

## Where each file goes

```
entities/hero_mortis.png
entities/bot_sniper.png
entities/bot_artillery.png
entities/bot_melee.png
entities/bot_rifle.png
entities/bot_edgar.png
entities/bot_spike.png
entities/bot_bull.png
items/box.png
items/pickup.png
```

The `entities/` filenames are `Kind.<NAME>.name.lower() + ".png"` (see `constants.Kind`) --
if a new brawler/bot kind is ever added to that enum, its sprite slot is
`entities/<new_name_lowercased>.png`, same pattern, still no code change.

Projectiles aren't covered by this -- they stay small colored dots regardless (there can be
dozens on screen at once, and they're too small for character art to read anyway).

## File format

- PNG, with an alpha channel (transparent background). Anything opaque behind the character
  will render as a visible square/box behind it in the sim.
- Square canvas. The viewer stretches whatever you give it to fit a square extent in world
  space, so a non-square source will look squashed or stretched.
- Crop tight: leave only ~5-10% padding around the character/object silhouette. The viewer
  sizes the sprite off the character's collision radius, not the PNG's own whitespace, so a
  loosely-cropped image just renders the character smaller than it should.
- Resolution doesn't have to be exact -- matplotlib rescales -- but 256x256 for `entities/`
  and 128x128 for `items/` is a reasonable target; keep it consistent across files so nothing
  looks softer/blurrier than its neighbors.

## Sizing in the sim

Each sprite is centered on the same world-space point its fallback shape used, sized off that
entity/item's existing collision radius (`unit_radius` for entities, `BOX_RADIUS = 0.5` for
crates) times a scale factor:

- entities (hero + bots): `radius * 1.8` -- bigger than the hitbox on purpose. In-game character
  art (weapon, wind-up frames, shadow) extends well past the collision circle; sizing 1:1 to the
  hitbox made every fallback circle look tiny once swapped for real art in testing.
- items (box, pickup cube): `radius * 1.3` -- these are simple props whose art roughly fills
  their physical footprint, so they need less headroom than a character.

If your sprites consistently look too big or small once dropped in, adjust
`_ENTITY_SPRITE_SCALE` / `_ITEM_SPRITE_SCALE` in `render/viewer.py` rather than re-cropping
every file.

## Not committed to git

These PNGs are extracted game assets, not this project's own IP, so `.gitignore` excludes
`*.png` under this directory (the folder structure and this README are still tracked). Keep
your local copies wherever you gathered them from as the actual backup.
