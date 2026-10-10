"""Every bot kind's FIRE and AIM decision, as one data-driven rule read off per-kind SimParams.
A new brawler costs a configs/brawlers.yaml block, not code.

The fields, and the reason behind each:

  - **fire_needs_los.** `vis` is bush-only concealment, so it hands a shooter targets on the far
    side of a wall that a straight shot would hit. A lobbed shell that arcs over walls (Grom)
    does not need a line. Cone swingers (`attack_arc_rad > 0`) do not either:
    `combat.melee_hitscan` runs its own per-victim LOS march in tick phase 6.
  - **fire_range_fraction** (0 means 1.0). Holds fire in the outer band of attack_range, for a
    spread that diverges too much at its rim to land reliably.
  - **fire_lateral_speed_limit / fire_lateral_hold_range_fraction.** Holds fire on a target
    crossing faster than the limit, but only past that fraction of range: a fan aimed at a fast
    lateral mover misses either side of it, while close in it lands anyway. A box never triggers
    it, since its `tgt.vel` is zero (bots/policy.targeting).
  - **The cone gate.** No separate field: "swings a cone" is `attack_arc_rad > 0`, as in
    `config.cone_ray_tiles` and `combat.melee_hitscan`, so the gate MUST be conditional on it
    (a zero arc never satisfies `geo.point_in_cone`; an unconditional test would stop every
    ranged kind firing). `fire` is True only when this tick's swing would land, tested with
    `geo.point_in_cone` against `ent_facing` before this tick's movement (this runs in phase 4,
    attacks resolve in phase 6, movement is phase 7), so turning to face costs a tick or two.
    The half-angle is `(hitscan_sweep_rad + attack_arc_rad) / 2`, the arc the WHOLE swept attack
    reaches: `arc / 2` alone would stop Buzz starting a sweep whose later sub-swings would land.
    For an unswept kind it is the plain `arc / 2`.
  - **aim_model.** See constants.AimModel. LOB scatters the landing POINT (`aim_noise_tiles`)
    where LEAD scatters the bearing (`aim_noise_std_rad`): the payload is an area landing on a
    point, so a scattered point is what a miss physically means.
  - **cfg.bots_tap_aim.** Every LEAD and LOB shot aims at where the target IS, lead fraction 0,
    as the game's bots do: they attack with a tap, whose auto-aim fires at the target's current
    position and never leads (user decision, 2026-10-06). The per-tier aim noise still applies,
    so the curriculum's tiers still differ in accuracy.

RNG: each call draws twice from `gen` for all (N,E) entities regardless of kind, first an (N,E)
angular tensor (LEAD) and then an (N,E,2) positional one (LOB).

Everything is computed for ALL (N,E) entities, the hero and dead entities included:
bots/policy.all_bot_intents zeroes entity 0 and every dead entity on the way out, and the hero's
own attacks come from `hero.decode_action`, never from here.
"""
import torch

from ..constants import AimModel
from ..core import geometry as geo
from ..core import stats
from . import policy as shared


def combat(state, tgt, bank, params, cfg, gen):
    """(fire (N,E) bool, aim_dir (N,E,2) unit, aim_point (N,E,2)) for every entity, by kind.

    `bank` is unused; it stays in the signature for a future rule that needs a terrain query,
    and every caller already passes it.
    """
    pos = state.ent_pos
    kind = state.ent_kind

    attack_range = stats.gather_kind(params.attack_range, kind)

    # --- FIRE -------------------------------------------------------------------------------
    needs_los = stats.gather_kind(params.fire_needs_los, kind) > 0
    range_fraction = stats.gather_kind(params.fire_range_fraction, kind)
    lateral_limit = stats.gather_kind(params.fire_lateral_speed_limit, kind)
    lateral_range_fraction = stats.gather_kind(params.fire_lateral_hold_range_fraction, kind)
    attack_arc = stats.gather_kind(params.attack_arc_rad, kind)
    sweep = stats.gather_kind(params.hitscan_sweep_rad, kind)

    los_ok = tgt.los | ~needs_los

    # 0 means "no restriction beyond attack_range". An exact 1.0, not a clamp, keeps the neutral
    # case bit-identical to fire_gate's own `dist <= attack_range` (1.0 * x == x for every float).
    effective_fraction = torch.where(range_fraction > 0, range_fraction,
                                     torch.ones_like(range_fraction))
    range_ok = tgt.dist <= effective_fraction * attack_range

    half_angle = (sweep + attack_arc) / 2.0
    in_cone = geo.point_in_cone(tgt.pos, pos, state.ent_facing, attack_range, half_angle)
    cone_ok = in_cone | (attack_arc <= 0)

    to_target = geo.normalize(tgt.pos - pos)
    lateral_speed = (tgt.vel * geo.perp(to_target)).sum(dim=-1).abs()
    # The `limit > 0` term is load-bearing, not defensive: with the field unset a bare
    # `lateral_speed > 0` would hold fire against ANY moving target, for every kind.
    holds_fire = (
        (lateral_limit > 0)
        & (lateral_speed > lateral_limit)
        & (tgt.dist > lateral_range_fraction * attack_range)
    )

    fire = (
        shared.fire_gate(state, tgt.dist, tgt.has_target, attack_range)
        & los_ok & range_ok & cone_ok & ~holds_fire
    )

    # --- AIM --------------------------------------------------------------------------------
    proj_speed = stats.gather_kind(params.proj_speed, kind)
    lead_fraction = stats.gather_kind(params.lead_target_fraction, kind)
    if cfg.bots_tap_aim:
        lead_fraction = torch.zeros_like(lead_fraction)
    flight_seconds = stats.gather_kind(params.proj_flight_seconds, kind)
    noise_rad = stats.gather_kind(params.aim_noise_std_rad, kind)
    noise_tiles = stats.gather_kind(params.aim_noise_tiles, kind)
    model = stats.gather_kind(params.aim_model, kind)

    # One solve shared by LEAD, and by LOB kinds with no fixed flight time.
    predicted = geo.lead_target(pos, tgt.pos, tgt.vel, proj_speed, lead_fraction)

    # LEAD: angular noise ~ N(0, aim_noise_std_rad) about the leaded bearing.
    angle_noise = torch.randn(kind.shape, generator=gen, device=pos.device) * noise_rad
    lead_dir = geo.rotate(geo.normalize(predicted - pos), angle_noise)
    lead_point = pos + lead_dir * tgt.dist.unsqueeze(-1)

    # LOB: a shell with a fixed `proj_flight_seconds` lands that long after firing whatever the
    # distance, so the intercept is closed form; geo.lead_target's fixed point only approximates a
    # distance-dependent flight time and would add error here. A constant-speed lob
    # (flight_seconds == 0) keeps the lead_target solve.
    timed = tgt.pos + tgt.vel * (lead_fraction * flight_seconds).unsqueeze(-1)
    landing = torch.where((flight_seconds > 0).unsqueeze(-1), timed, predicted)
    point_noise = torch.randn(pos.shape, generator=gen, device=pos.device)
    lob_point = landing + point_noise * noise_tiles.unsqueeze(-1)
    lob_dir = geo.normalize(lob_point - pos)

    # DIRECT is `to_target` (computed above for the lateral hold). A hitscan cone has nothing in
    # flight to lead; its aim outputs exist only for shape (combat.melee_hitscan reads
    # ent_facing, not these).
    is_lead = (model == int(AimModel.LEAD)).unsqueeze(-1)
    is_lob = (model == int(AimModel.LOB)).unsqueeze(-1)
    aim_dir = torch.where(is_lead, lead_dir, torch.where(is_lob, lob_dir, to_target))
    aim_point = torch.where(is_lead, lead_point, torch.where(is_lob, lob_point, tgt.pos))
    return fire, aim_dir, aim_point
