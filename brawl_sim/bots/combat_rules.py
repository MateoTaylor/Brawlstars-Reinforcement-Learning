"""Every kind's FIRE and AIM decision, as one data-driven rule read off SimParams. Step E1 of
bot_overhaul.md; replaces bots/sniper.py, bots/artillery.py, bots/melee.py and bots/rifle.py,
which are deleted.

WHY THIS EXISTS
Those four modules were ~15 lines of logic each and differed in exactly five ways, all of which
are values rather than code:

    kind        fire_needs_los  fire_range_fraction  lateral hold  cone gate      aim_model
    ----------  --------------  -------------------  ------------  -------------  ---------
    sniper      yes             1.0 (unset)          --            -- (arc 0)     LEAD
    artillery   no              1.0 (unset)          --            -- (arc 0)     LOB
    melee       no              1.0 (unset)          --            arc 0.65 > 0   DIRECT
    rifle       yes             0.9                  2.0 / 0.6     -- (arc 0)     LEAD

So a sixth brawler used to cost a fifth module, an entry in `policy._archetype_modules`, a fifth
pass of the dispatcher's torch.where chain, and a row in `policy.RANGE_FRACTION_BY_KIND` (whose
`assert len(...) == N_KINDS` would have failed on the spot). It now costs a `configs/brawlers.yaml`
block and nothing else -- which is the point: the roster is meant to keep growing.

It should also be cheaper, for a reason worth stating plainly. The old dispatcher ran ALL FOUR
`combat()` functions over ALL (N,E) entities every tick and threw three of the four results away --
a deliberate, documented choice (see the deleted sniper.py's docstring: batched code cannot branch
per row, so computing garbage for rows you discard is the normal cost of doing business). Folding
the four into one rule does not remove a branch; it removes redundant copies of shared work. By
kernel count: 16 `gather_kind` calls become 13 (more distinct fields, each fetched once instead of
`attack_range` four times and `lead_target_fraction` three times), three `geo.lead_target` solves
become one, three noise draws become two, and a 12-term selection chain becomes 4.

**That is a count, not a measurement.** The end-to-end `_bot_phase` saving is deliberately NOT
quoted here: E1 landed while the GPU was 63% busy with an unrelated process, and this repo has
already had to retract two speedup figures measured under exactly that condition. Step E2 re-runs
`scripts/benchmark.py` on an idle machine and records the real number.

THE ONE AXIS THAT IS NOT A FIELD
"Does this kind need its target inside a swing cone" is `attack_arc_rad > 0`, not a separate flag.
That is already what `config.cone_ray_tiles` and `combat.melee_hitscan` use to mean "this kind
swings a cone", and a second field saying the same thing is a second field that can disagree with
the first. A kind with `attack_arc_rad == 0` can never satisfy `geo.point_in_cone` at all, so the
gate MUST be conditional on it -- an unconditional cone test would silently stop every ranged kind
from ever firing.

WHAT EACH RULE MEANT, preserved from the four modules this replaces:

  - **LOS (sniper, rifle).** `vis` is bush-only targeting (Notice 4): it will happily hand a
    shooter a target on the far side of a wall, and a straight-line bolt would hit the wall.
    Artillery deliberately does NOT require it -- an arcing shell does not need a physical line,
    which is the whole shape of that archetype. Melee does not require it either, but for a
    different reason: `combat.melee_hitscan` runs its own per-victim LOS march in tick phase 6, so
    gating here would just be paying for the same answer twice.

  - **Range fraction (rifle 0.9).** Shelly's fan diverges too much at the rim of her range to
    reliably land, so she holds until the target is inside 0.9 of it.

  - **Lateral hold (rifle 2.0 tiles/s past 0.6 range).** Same reason, applied to crossing targets:
    a fan aimed at a fast lateral mover misses either side of it. Close in, the fan is tight
    enough that it lands anyway, hence the range term. A loot box can never trigger this --
    `tgt.vel` is exactly zero for a box (bots/policy.targeting), which falls out of the arithmetic
    without needing its own branch.

  - **Cone gate (melee).** `fire` is only True when this tick's actual swing would land, tested
    with `geo.point_in_cone` -- the identical primitive `melee_hitscan` itself uses, against the
    same pre-movement `ent_facing` (attack resolution is phase 6, movement is phase 7, and this
    runs in phase 4, so both read the value last tick's movement left). Turning to face a target
    costs an extra tick or two, exactly like a real melee attacker having to turn before swinging.
    The half-angle is `(hitscan_sweep_rad + attack_arc_rad) / 2` -- the arc the WHOLE attack can
    reach, not one sub-swing's: gating on `arc/2` alone would mean Buzz refuses to start a sweep
    whose 2nd through 5th sub-swings would have landed squarely (Step C2). For an unswept kind
    `hitscan_sweep_rad` is 0 and this is exactly the plain `arc/2` test.

  - **Aim models.** See constants.AimModel for what the three are and why DIRECT is the default.
    The LOB model's positional noise is not interchangeable with LEAD's angular noise: artillery's
    payload is an area landing on a POINT, so a scattered landing point is what a miss physically
    means, and it matters more since Step B1 made the blast one tile wide rather than a 1.5-tile
    disc -- a shell that lands off-target is a genuine miss the split cross may or may not
    clean up.

RNG STREAM (the one intentional behavioral difference from the four modules)
The old dispatcher drew noise three times per tick -- sniper's angular draw, artillery's positional
draw, then rifle's angular draw -- because each module drew its own. This draws twice: one (N,E)
angular tensor shared by every LEAD kind and one (N,E,2) positional tensor for every LOB kind. Two
entities never alias (they read different cells of the same tensor), so nothing correlates that did
not before, but the generator advances differently, so a fixed seed produces a different --
statistically identical -- rollout than it did before E1. `tests/test_combat_rules.py` pins the
equivalence that does survive: `fire` is bit-identical for all four kinds (it consumes no
randomness at all), and so is `aim` with the noise parameters zeroed.

Everything here is computed for ALL (N,E) entities regardless of kind, including the hero and
dead entities. That is unchanged from the four modules and is safe for the same reason it was
then: bots/policy.all_bot_intents zeroes entity 0 and every dead entity on the way out, and the
hero's own attacks come from `decode_action`, never from here.
"""
import torch

from ..constants import AimModel
from ..core import geometry as geo
from ..core import stats
from . import policy as shared


def combat(state, tgt, bank, params, cfg, gen):
    """(fire (N,E) bool, aim_dir (N,E,2) unit, aim_point (N,E,2)) for every entity, by kind.

    `bank` and `cfg` are unread today. They stay in the signature because they are the two things
    a future rule most plausibly needs (a terrain query, a feature toggle) and because every
    caller and test already passes them -- see the module docstring on this being the extension
    point for new brawlers.
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

    # 0 means "no restriction beyond attack_range". Substituting an exact 1.0 rather than, say,
    # clamping keeps the neutral case bit-identical to `fire_gate`'s own `dist <= attack_range`
    # term (1.0 * x == x exactly, for every float x), so a kind that does not set this field
    # gets literally the old gate rather than an arithmetically-similar one.
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
    flight_seconds = stats.gather_kind(params.proj_flight_seconds, kind)
    noise_rad = stats.gather_kind(params.aim_noise_std_rad, kind)
    noise_tiles = stats.gather_kind(params.aim_noise_tiles, kind)
    model = stats.gather_kind(params.aim_model, kind)

    # Computed ONCE and shared by LEAD and LOB, which is most of E1's saving -- each of the three
    # ranged modules used to solve this separately for the same target.
    predicted = geo.lead_target(pos, tgt.pos, tgt.vel, proj_speed, lead_fraction)

    # LEAD: angular noise ~ N(0, aim_noise_std_rad) about the leaded bearing.
    angle_noise = torch.randn(kind.shape, generator=gen, device=pos.device) * noise_rad
    lead_dir = geo.rotate(geo.normalize(predicted - pos), angle_noise)
    lead_point = pos + lead_dir * tgt.dist.unsqueeze(-1)

    # LOB: with a fixed-flight-time shell the intercept is CLOSED FORM -- time of flight is
    # `proj_flight_seconds` no matter where the target is, so the lead is just "where will it be
    # that many seconds from now". geo.lead_target's two-step fixed point exists to approximate a
    # time of flight that DEPENDS on the (unknown) intercept distance, which is exactly the
    # dependency proj_flight_seconds removes; running it here would re-introduce the error it was
    # built to reduce. A lobbed kind still on a constant-speed arc (flight_seconds == 0) keeps it.
    timed = tgt.pos + tgt.vel * (lead_fraction * flight_seconds).unsqueeze(-1)
    landing = torch.where((flight_seconds > 0).unsqueeze(-1), timed, predicted)
    point_noise = torch.randn(pos.shape, generator=gen, device=pos.device)
    lob_point = landing + point_noise * noise_tiles.unsqueeze(-1)
    lob_dir = geo.normalize(lob_point - pos)

    # DIRECT is `to_target`, already computed above for the lateral-hold test -- a hitscan cone
    # has nothing in flight to lead, and its aim outputs are kept only for shape consistency
    # (combat.melee_hitscan reads ent_facing, not these).
    is_lead = (model == int(AimModel.LEAD)).unsqueeze(-1)
    is_lob = (model == int(AimModel.LOB)).unsqueeze(-1)
    aim_dir = torch.where(is_lead, lead_dir, torch.where(is_lob, lob_dir, to_target))
    aim_point = torch.where(is_lead, lead_point, torch.where(is_lob, lob_point, tgt.pos))
    return fire, aim_dir, aim_point
