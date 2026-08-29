"""Terrain scanning: everything that turns pixels into a tile grid.

Sibling chunks (entities, HUD state) land next to this package and reuse `brawl_vision`'s root
modules -- `capture`, `clips`, `camera`, `config` -- unchanged.
"""
