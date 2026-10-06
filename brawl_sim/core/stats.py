"""Per-entity stat computation: gather archetype stats by kind, then apply cube bonuses and
the enemy_*_mult randomization knobs, which never touch the hero.
"""
import torch

from ..constants import Kind

_HERO_KIND = int(Kind.HERO_MORTIS)


def gather_kind(param_nk: torch.Tensor, kind_ne: torch.Tensor) -> torch.Tensor:
    return param_nk.gather(1, kind_ne)


def aggression_of(kind: torch.Tensor, params) -> torch.Tensor:
    """(N,E) f32 `params.aggression` gathered per entity, with 0 read as 1.0.

    The ONE place that read happens. A missing `aggression` key resolves to 0 so hand-built
    partial specs still validate, and every consumer DIVIDES by this value (the HUNTER/KITE
    retreat threshold, the KITE hold distance) or compares it against 1.25 (the CAMPER fire
    veto), so 0 has to mean "the bot as authored", never "divide by zero" or "maximally timid".
    validate() already rejects negatives, hence `> 0` not `!= 0`."""
    a = gather_kind(params.aggression, kind)
    return torch.where(a > 0, a, torch.ones_like(a))


def is_hero(kind: torch.Tensor) -> torch.Tensor:
    return kind == _HERO_KIND


def effective_max_hp(kind: torch.Tensor, cubes: torch.Tensor, params) -> torch.Tensor:
    """A power cube adds a FLAT `cubes.hp_per_cube` to max HP (400 at the shipped config, the
    real game's number), not a percentage of the brawler's base: flat, cubes narrow the gap
    between tanky and fragile brawlers (+6400 for everyone at 16 cubes), where a percentage
    would widen it.

    `enemy_hp_mult` multiplies the TOTAL, matching `effective_damage`'s treatment of the cube
    damage bonus, so a cube-stacked enemy cannot escape the one knob that says how tanky enemies
    are (16 cubes add more than a light bot's whole 6000 base).
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


def effective_gadget_damage(kind: torch.Tensor, cubes: torch.Tensor, params) -> torch.Tensor:
    """`effective_damage` on `gadget_damage` instead of `base_damage`: the SAME cube bonus and
    enemy multiplier, so a cube-stacked Mortis's spinner grows at exactly the rate his attack
    does (2000 -> 2600 at 3 cubes). Its own function because scaling `effective_damage` by
    `gadget_damage / base_damage` divides by zero for any kind whose `base_damage` is 0. A kind
    with no gadget (`gadget_damage: 0`, every bot today) resolves to 0."""
    base = gather_kind(params.gadget_damage, kind)
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
