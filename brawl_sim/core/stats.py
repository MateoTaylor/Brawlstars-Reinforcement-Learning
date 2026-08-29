"""Per-entity stat computation: gather archetype stats by kind, then apply cube bonuses and
the enemy_*_mult randomization knobs -- which, per their names and D05/D06, never touch the
hero. See BRAWL_SIM_BUILD_PLAN.md Step 10.
"""
import torch

from ..constants import Kind

_HERO_KIND = int(Kind.HERO_MORTIS)


def gather_kind(param_nk: torch.Tensor, kind_ne: torch.Tensor) -> torch.Tensor:
    return param_nk.gather(1, kind_ne)


def is_hero(kind: torch.Tensor) -> torch.Tensor:
    return kind == _HERO_KIND


def effective_max_hp(kind: torch.Tensor, cubes: torch.Tensor, params) -> torch.Tensor:
    """A power cube adds a FLAT `cubes.hp_per_cube` to max HP -- 400 at the shipped config, the
    real game's number -- not a percentage of the brawler's own base.

    That distinction is the whole shape of the mechanic and it is not a small one. As a
    percentage, cubes scaled with HP and made the tankiest brawler tankier still: at the previous
    0.15/cube and `max_cubes: 16`, a fully-stacked 10000 HP Buzz reached 34000 while a 6000 HP
    Brock reached 20400, so cubes WIDENED the gap. Flat, they narrow it -- +6400 for everyone --
    which is why the real game can hand out the same cube to every brawler.

    `enemy_hp_mult` multiplies the TOTAL, matching `effective_damage`'s treatment of the cube
    damage bonus. Leaving cube HP outside it would let a cube-stacked enemy escape the one knob
    that says how tanky enemies are, and at 16 cubes that escape is larger than most base HPs.
    """
    base = gather_kind(params.base_hp, kind)
    with_cubes = base + params.cube_hp_flat.unsqueeze(-1) * cubes.to(base.dtype)
    mult = torch.where(is_hero(kind), torch.ones_like(base), params.enemy_hp_mult.unsqueeze(-1))
    return with_cubes * mult


def effective_damage(kind: torch.Tensor, cubes: torch.Tensor, params) -> torch.Tensor:
    base = gather_kind(params.base_damage, kind)
    bonus = 1.0 + params.cube_damage_bonus.unsqueeze(-1) * cubes.to(base.dtype)
    mult = torch.where(is_hero(kind), torch.ones_like(base), params.enemy_damage_mult.unsqueeze(-1))
    return base * bonus * mult


def effective_speed(kind: torch.Tensor, params) -> torch.Tensor:
    return gather_kind(params.move_speed, kind)


def apply_cube_gain(
    hp: torch.Tensor, kind: torch.Tensor, old_cubes: torch.Tensor, new_cubes: torch.Tensor, params,
) -> torch.Tensor:
    """Current HP rises by the same absolute amount as max HP does -- a cube pickup heals by
    the delta, it doesn't refill to full."""
    old_max = effective_max_hp(kind, old_cubes, params)
    new_max = effective_max_hp(kind, new_cubes, params)
    return hp + (new_max - old_max)
