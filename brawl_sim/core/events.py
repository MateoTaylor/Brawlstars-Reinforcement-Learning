"""Termination/truncation and the per-tick event package (`info`). See
BRAWL_SIM_BUILD_PLAN.md Step 27.

`info` holds per-tick EVENTS; `obs` (Step 25) holds per-tick STATE. Episode-cumulative counters
(ent_kills, ent_damage_dealt, ent_damage_taken, ent_shots_fired, boxes_broken) live in
`SimState`, are zeroed on reset (state.zero_, Step 9), and are already mirrored into `obs` --
this module's job is only the THIS-TICK deltas, several of which aren't stored anywhere and
have to be reconstructed from what a single call gets handed.

`compute_info`'s `dmg_by` contract: the caller (the future `env.py` step loop, Step 29) is
expected to pass the SUM of every combat damage source's own (N,E,E) dmg_by this tick --
`hero.advance_dash` + `combat.melee_hitscan` + `projectiles.step_projectiles` -- since
`combat.apply_damage` is invoked once per source (its own docstring: "never mixing causes
within one call") but `compute_info` only takes one combined matrix. Zone damage has no
attacker (`combat.py`'s `_NO_ATTACKER = -1` sentinel) and so structurally cannot appear in an
attacker x victim matrix; it's absent from `damage_matrix`/`damage_dealt_tick`/
`damage_taken_tick` entirely -- `entities.damage_taken` in `obs` (the cumulative counter) is
the complete record across all causes including zone, this module's per-tick fields are not.

`hp_healed_tick` has the same "caller must supply it" shape as `dmg_by`, for the opposite
reason: healing (out-of-combat regen in `combat.apply_regen`, super lifesteal in
`combat.apply_heal`) leaves no attacker x victim matrix and no cumulative SimState counter to
diff, so `env.py` collects what those two calls report applying and passes it down. Absent, it
is zeros.

`shots_fired_tick` / `dash_hits_tick` are best-effort, NOT exact, given the signature this step
specifies -- see their own comments below for exactly what they miss and why. Flagged here as a
forward pointer: if reward shaping in a later phase needs a true miss-inclusive fire-attempt
count, `env.py` (Step 29) is the natural place to widen `compute_info`'s signature with an
explicit `fired_mask` from `all_bot_intents`/`decode_action`, since compute_info as specified
here has no visibility into an attack that fires and hits nothing.

**Action repeat (`cfg.action_repeat`, `env.py`'s decision loop).** One agent decision now spans
`action_repeat` SIM TICKS, so "this tick's events" and "this decision's events" are no longer
the same thing. The per-tick DELTAS above (damage, kills, cubes, boxes) are summed by `env.py`
across the sub-ticks and handed to `compute_info` pre-accumulated -- nothing in this module has
to change for those. The per-tick STATE flags cannot be summed that way, so they get their own
small accumulator here: `new_decision_tally` / `advance_decision_tally` latch the outcome fields
(`terminated`, `truncated`, `hero_alive`, `hero_rank`) at the sub-tick each env's episode
actually ended, and count the two flags reward shaping integrates over time (`alive_ticks`,
`in_zone_ticks`). Latching is not cosmetic: without it, a hero that becomes last-alive on
sub-tick 2 of 5 keeps taking zone damage for sub-ticks 3-5 and a win silently becomes a death.
At `action_repeat=1` the tally is exactly one call to `new_decision_tally`, and every field it
produces equals what this module computed before it existed.
"""
import torch

from . import observation, zone

_HERO = 0


def compute_done(state, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """(terminated (N,) bool, truncated (N,) bool). D16: terminate on hero death or
    hero-last-alive; truncate on hitting the episode step limit. Independent conditions -- both
    can be True together only on the exact tick that coincides with both (e.g. the hero wins or
    dies on the very last allowed step)."""
    hero_alive = state.ent_alive[:, _HERO]
    terminated = (~hero_alive) | ((state.n_alive == 1) & hero_alive)
    truncated = state.step_count >= cfg.max_episode_steps
    return terminated, truncated


def hero_tick_state(state, cfg) -> dict:
    """The five (N,) HERO quantities that describe a single sim tick and cannot be recovered by
    summing deltas -- what the action-repeat tally below latches or counts. `hero_in_zone` is
    True when the hero is OUTSIDE the safe rect (i.e. taking zone damage), matching both
    `obs["hero"]["in_zone"]`'s naming and the identical `zone._outside_rect` call
    `core/observation.build_obs` makes; `hero_rank` is 0-INDEXED (0 = won), matching
    `info["hero_rank"]` rather than `observation.compute_rank`'s own 1-indexed return."""
    terminated, truncated = compute_done(state, cfg)
    # `.clone()` is load-bearing, not defensive habit: `state.ent_alive[:, _HERO]` is a VIEW into
    # storage that `combat.resolve_deaths` rewrites in place on every later sub-tick, so latching
    # the view would latch nothing at all -- `advance_decision_tally` would faithfully carry
    # forward a tensor whose contents had already changed underneath it, and a hero that won and
    # was then killed by the zone would still read as dead. Every other field here is the result
    # of an operation and is therefore already a fresh tensor.
    alive = state.ent_alive[:, _HERO].clone()
    outside = zone._outside_rect(state.ent_pos[:, _HERO], state.zone_lo, state.zone_hi) & alive
    rank = observation.compute_rank(state.ent_alive, state.ent_death_step)[:, _HERO] - 1
    return {
        "terminated": terminated, "truncated": truncated,
        "hero_alive": alive, "hero_in_zone": outside, "hero_rank": rank,
    }


def new_decision_tally(state, cfg) -> dict:
    """Seeds the action-repeat accumulator from the first sub-tick of a decision. Sync-free,
    allocation-light ((N,) tensors only). See module docstring."""
    tick = hero_tick_state(state, cfg)
    return {
        "terminated": tick["terminated"],
        "truncated": tick["truncated"],
        "hero_alive": tick["hero_alive"],
        "hero_rank": tick["hero_rank"],
        "alive_ticks": tick["hero_alive"].to(torch.int32),
        "in_zone_ticks": tick["hero_in_zone"].to(torch.int32),
        "n_ticks": torch.ones_like(tick["hero_alive"], dtype=torch.int32),
        "done": tick["terminated"] | tick["truncated"],
    }


def advance_decision_tally(tally: dict, state, cfg) -> dict:
    """Folds one more sub-tick into `tally`, returning a new dict (`tally` is not mutated).

    **Envs whose episode already ended earlier in this same decision are frozen out.** Their
    four latched outcome fields stop moving and their three counters stop growing, so the
    leftover sub-ticks of a decision can neither rewrite an outcome that is already settled (a
    hero that became last-alive on sub-tick 2 must not be killed by the zone on sub-tick 4 and
    reported as a death) nor keep accruing per-tick reward for an episode that is over. The
    world itself keeps ticking for those envs -- freezing SimState per-env would mean masking
    every write in every phase -- so their `final_observation` can be up to `action_repeat - 1`
    ticks staler than the moment they finished. That staleness is confined to observation
    fields; every field any outcome, reward, or win-rate signal reads comes from here."""
    tick = hero_tick_state(state, cfg)
    done = tally["done"]
    live = ~done
    return {
        "terminated": torch.where(done, tally["terminated"], tick["terminated"]),
        "truncated": torch.where(done, tally["truncated"], tick["truncated"]),
        "hero_alive": torch.where(done, tally["hero_alive"], tick["hero_alive"]),
        "hero_rank": torch.where(done, tally["hero_rank"], tick["hero_rank"]),
        "alive_ticks": tally["alive_ticks"] + (tick["hero_alive"] & live).to(torch.int32),
        "in_zone_ticks": tally["in_zone_ticks"] + (tick["hero_in_zone"] & live).to(torch.int32),
        "n_ticks": tally["n_ticks"] + live.to(torch.int32),
        "done": done | tick["terminated"] | tick["truncated"],
    }


def compute_info(
    state, dmg_by: torch.Tensor, newly_dead: torch.Tensor, newly_broken: torch.Tensor,
    cubes_gained: torch.Tensor, cfg, decision: dict | None = None,
    hp_healed: torch.Tensor | None = None,
) -> dict:
    """(N,)/(N,E)/(N,E,E) device tensors, one dict of THIS DECISION's events -- see module
    docstring for the `dmg_by` contract and the shots_fired_tick/dash_hits_tick caveat.

    `decision` is the tally `env.py` accumulated over the decision's `cfg.action_repeat`
    sub-ticks; the four delta arguments must already be summed over those same sub-ticks. Left
    at `None` (every call site outside `env.py`, including every test written before action
    repeat existed) it is derived from the current state as a single tick, which is exactly what
    this function did before the argument existed.

    `hp_healed` is the (N,E) HP restored over the same sub-ticks -- regen plus super lifesteal,
    as actually applied (see `combat.apply_heal`/`apply_regen`). Unlike damage there is no
    matrix to project it out of and nothing in `SimState` records it, so it can only come from
    the caller; left at `None` it is zeros, which is what every pre-existing call site means."""
    if decision is None:
        decision = new_decision_tally(state, cfg)
    if hp_healed is None:
        hp_healed = torch.zeros_like(dmg_by[:, :, 0])
    terminated, truncated = decision["terminated"], decision["truncated"]

    damage_dealt_tick = dmg_by.sum(dim=2)  # (N,E): per attacker, this tick
    damage_taken_tick = dmg_by.sum(dim=1)  # (N,E): per victim, this tick

    # kills_tick: replays combat.resolve_deaths' own credit rule (newly_dead & last_hit_by>=0)
    # against THIS tick's newly_dead, rather than reading the cumulative ent_kills counter
    # resolve_deaths already mutated -- there's no "before this tick" snapshot of that counter
    # to diff against.
    E = state.ent_kind.shape[1]
    credit = newly_dead & (state.ent_last_hit_by >= 0)
    killer_idx = torch.clamp(state.ent_last_hit_by, min=0)
    kills_tick = torch.zeros((state.ent_kind.shape[0], E), dtype=torch.int32, device=state.ent_pos.device)
    kills_tick.scatter_add_(1, killer_idx, credit.to(torch.int32))

    # death_cause_tick: ent_death_cause masked to newly_dead -- the raw state field holds a
    # PERMANENT historical value once an entity has ever died, meaningless as a "this tick"
    # signal on its own.
    death_cause_tick = torch.where(
        newly_dead, state.ent_death_cause, torch.zeros_like(state.ent_death_cause),
    )

    # shots_fired_tick: a LOWER BOUND, not an exact fire-attempt count -- a shot/dash that
    # fires and hits nothing is invisible here (dmg_by only records connected hits). See
    # module docstring's forward pointer.
    shots_fired_tick = (damage_dealt_tick > 0).to(torch.int32)

    # dash_hits_tick: dmg_by is a same-tick SUM across dash + melee + projectile damage, so a
    # hit can't be attributed to its source by value alone -- but dash and other attacks are
    # mutually exclusive for the same entity on the same tick (N02/N03/R02: dash "fully
    # replaces" the walk/attack that tick), so any hit dealt by a CURRENTLY DASHING attacker
    # this tick is unambiguously a dash hit.
    is_dashing = (state.ent_dash_t > 0).unsqueeze(2)  # (N,E,1), attacker axis
    dash_hits_tick = ((dmg_by > 0) & is_dashing).sum(dim=(1, 2)).to(torch.int32)

    boxes_broken_tick = newly_broken.sum(dim=1).to(torch.int32)

    return {
        "damage_matrix": dmg_by,
        "damage_dealt_tick": damage_dealt_tick,
        "damage_taken_tick": damage_taken_tick,
        # hp_healed_tick is the mirror image of damage_taken_tick and covers ALL heal sources
        # (regen + super lifesteal), where damage_taken_tick is COMBAT damage only -- zone damage
        # has no attacker and cannot appear in dmg_by. So over an episode the two are NOT a
        # matched pair: healing back a zone burn shows up here with no damage_taken_tick entry
        # against it. training/reward.py prices that gap deliberately; see its docstring.
        "hp_healed_tick": hp_healed,
        "kills_tick": kills_tick,
        "deaths_tick": newly_dead,
        "death_cause_tick": death_cause_tick,
        "cubes_gained_tick": cubes_gained,
        "boxes_broken_tick": boxes_broken_tick,
        "shots_fired_tick": shots_fired_tick,
        "dash_hits_tick": dash_hits_tick,
        "hero_rank": decision["hero_rank"],
        "terminated": terminated,
        "truncated": truncated,
        # --- action-repeat fields (see module docstring). At action_repeat=1 these are exactly
        # the current tick's own hero state, so nothing reading them behaves differently there.
        # `hero_alive` is LATCHED at the sub-tick the episode ended; `obs["hero"]["alive"]` is
        # the live post-decision value, and the two can disagree for an env that finished early.
        # Reward shaping must use this one -- see training/reward.py.
        "hero_alive": decision["hero_alive"],
        "alive_ticks": decision["alive_ticks"],
        "in_zone_ticks": decision["in_zone_ticks"],
        "n_ticks": decision["n_ticks"],
        "time": state.time,
        "step_count": state.step_count,
    }
