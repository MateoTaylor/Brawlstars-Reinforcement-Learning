"""Termination/truncation and the per-decision event package (`info`).

`info` holds EVENTS; `obs` holds STATE. The episode-cumulative counters (ent_kills,
ent_damage_dealt, ent_damage_taken, ent_shots_fired, boxes_broken) live in `SimState` and are
zeroed on reset (state.zero_); this module produces only the deltas, several of which are stored
nowhere and must be handed in by the caller (env.py, accumulated over the decision's sub-ticks):

- `dmg_by`: the SUM of every combat source's (N,E,E) attacker x victim matrix (dash, melee,
  projectiles), since `combat.apply_damage` runs once per source. Zone damage has no attacker
  (`combat._NO_ATTACKER`), so it is absent from `damage_matrix`/`damage_dealt_tick`/
  `damage_taken_tick`; only the cumulative `entities.damage_taken` covers every cause.
- `hp_healed`: HP restored by regen, super lifesteal and melee lifesteal, as applied. Healing
  leaves no matrix and no counter to diff, so only the caller knows it.
- `attacks_in_reach`: the hero's attacks and supers made with a visible enemy in reach, known
  only inside env.py's attack phase; training/reward.py's `attack_in_reach` term reads it.
- `gadget_hits`: landings of the hero's gadget spinner that hurt at least one player (crates
  never count), known only inside env.py's projectile phase; training/reward.py's `gadget_hit`
  term reads it.
- `move_reversals`: whether this decision's move reversed the last one's (`move_reversals`
  below), readable only before `history.push` overwrites the last one; training/reward.py's
  `move_reversal` term reads it.

`shots_fired_tick` and `dash_hits_tick` are proxies, not exact counts (see their comments).

**Action repeat.** One decision spans `cfg.action_repeat` sim ticks. The deltas above accumulate
across the sub-ticks; STATE flags cannot, so `new_decision_tally`/`advance_decision_tally` latch
the outcome fields (`terminated`, `truncated`, `hero_alive`, `hero_rank`) at the sub-tick each
env's episode ended and count `alive_ticks`/`in_zone_ticks`/`n_ticks`. Without the latch, a hero
that becomes last-alive on sub-tick 2 of 5 keeps taking zone damage and a win becomes a death.
At `action_repeat=1` the tally is the current tick's own state.
"""
import torch

from . import observation, zone

_HERO = 0


def compute_done(state, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """(terminated (N,) bool, truncated (N,) bool). Terminate on hero death or hero-last-alive;
    truncate at the episode step limit. Independent: both are True only on a tick that meets
    both (the hero wins or dies on the last allowed step)."""
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
    # `.clone()` is load-bearing: `state.ent_alive[:, _HERO]` is a VIEW that
    # `combat.resolve_deaths` rewrites on every later sub-tick, so the tally would latch nothing
    # (a hero that won and was then killed by the zone would read as dead). The other fields are
    # fresh tensors already.
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

    **Envs whose episode already ended earlier in this decision are frozen out**: their four
    latched outcome fields stop moving and their three counters stop growing, so leftover
    sub-ticks can neither rewrite a settled outcome nor accrue reward for a finished episode. The
    world keeps ticking for those envs (freezing SimState per env would mean masking every write
    in every phase), so their `final_observation` can be up to `action_repeat - 1` ticks staler
    than the moment they finished; every outcome, reward and win-rate signal reads this tally."""
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


def move_reversals(state, action: torch.Tensor, cfg) -> torch.Tensor:
    """(N,) int32, 1 where this decision's move bin reverses the previous decision's: both
    non-idle (bin 0 is idle) and at least 135 degrees apart around the circle, 6 of 16 bins
    (8 x gap >= 3 x n_move_bins in general). Needs a previous decision on record:
    `hist_valid[:, 0]` is False after a reset, so the first decision of an episode never counts.

    Reads ring slot 0, so env.step calls it BEFORE `history.push` writes this action there.
    Per DECISION, like the ring: training/reward.py's `move_reversal` term reads it
    (SIM_ISSUES_PLAN.md §5.2; user decision, 2026-10-07)."""
    n = cfg.n_move_bins
    prev = state.hist_action[:, 0, 0]
    cur = action[:, 0]
    ahead = torch.remainder(cur - prev, n)
    gap = torch.minimum(ahead, n - ahead)
    reversal = state.hist_valid[:, 0] & (prev > 0) & (cur > 0) & (8 * gap >= 3 * n)
    return reversal.to(torch.int32)


def compute_info(
    state, dmg_by: torch.Tensor, newly_dead: torch.Tensor, newly_broken: torch.Tensor,
    cubes_gained: torch.Tensor, cfg, decision: dict | None = None,
    hp_healed: torch.Tensor | None = None, attacks_in_reach: torch.Tensor | None = None,
    gadget_hits: torch.Tensor | None = None, move_reversals: torch.Tensor | None = None,
) -> dict:
    """(N,)/(N,E)/(N,E,E) device tensors, one dict of THIS DECISION's events; the module
    docstring gives the contracts of the caller-supplied inputs.

    `decision` is the tally `env.py` accumulated over the decision's `cfg.action_repeat`
    sub-ticks, and the four delta arguments must already be accumulated over those same
    sub-ticks. None derives it from the current state as a single tick.

    `hp_healed`: (N,E) HP restored over the same sub-ticks by regen, super lifesteal and melee
    lifesteal, as applied (`combat.apply_regen`/`apply_heal`). None is zeros.

    `attacks_in_reach`: (N,) int32 count of the hero's attacks and supers made with a visible
    enemy inside its uncharged dash reach over the same sub-ticks (env.py's attack phase). None
    is zeros.

    `gadget_hits`: (N,) int32 count of the hero's gadget-spinner landings that hurt at least one
    player over the same sub-ticks (env.py's projectile phase). None is zeros.

    `move_reversals`: (N,) int32, 1 where this decision's move reversed the last one's
    (`move_reversals` above, taken before env.step's `history.push`). None is zeros."""
    if decision is None:
        decision = new_decision_tally(state, cfg)
    if hp_healed is None:
        hp_healed = torch.zeros_like(dmg_by[:, :, 0])
    if attacks_in_reach is None:
        attacks_in_reach = torch.zeros_like(decision["n_ticks"])
    if gadget_hits is None:
        gadget_hits = torch.zeros_like(decision["n_ticks"])
    if move_reversals is None:
        move_reversals = torch.zeros_like(decision["n_ticks"])
    terminated, truncated = decision["terminated"], decision["truncated"]

    damage_dealt_tick = dmg_by.sum(dim=2)  # (N,E): per attacker, this tick
    damage_taken_tick = dmg_by.sum(dim=1)  # (N,E): per victim, this tick

    # kills_tick replays combat.resolve_deaths' credit rule (newly_dead & last_hit_by >= 0): the
    # cumulative ent_kills has no "before this decision" snapshot to diff against.
    E = state.ent_kind.shape[1]
    credit = newly_dead & (state.ent_last_hit_by >= 0)
    killer_idx = torch.clamp(state.ent_last_hit_by, min=0)
    kills_tick = torch.zeros((state.ent_kind.shape[0], E), dtype=torch.int32, device=state.ent_pos.device)
    kills_tick.scatter_add_(1, killer_idx, credit.to(torch.int32))

    # death_cause_tick: masked to newly_dead, since ent_death_cause keeps its value once set.
    death_cause_tick = torch.where(
        newly_dead, state.ent_death_cause, torch.zeros_like(state.ent_death_cause),
    )

    # shots_fired_tick is a proxy, not a fire count: 1 where the entity dealt combat damage this
    # decision. A miss never shows, and a projectile fired earlier that lands now does.
    shots_fired_tick = (damage_dealt_tick > 0).to(torch.int32)

    # dash_hits_tick: (N,) count of (attacker, victim) pairs with damage this decision whose
    # attacker is still dashing when it ends. dmg_by sums every source, so a projectile of the
    # dasher's landing meanwhile (a Super bolt, or a gadget spinner thrown mid-dash) counts too,
    # and a dash that ended within the decision does not.
    is_dashing = (state.ent_dash_t > 0).unsqueeze(2)  # (N,E,1), attacker axis
    dash_hits_tick = ((dmg_by > 0) & is_dashing).sum(dim=(1, 2)).to(torch.int32)

    boxes_broken_tick = newly_broken.sum(dim=1).to(torch.int32)

    return {
        "damage_matrix": dmg_by,
        "damage_dealt_tick": damage_dealt_tick,
        "damage_taken_tick": damage_taken_tick,
        # Covers every heal source (regen, super and melee lifesteal), while damage_taken_tick
        # is combat damage only, so healing back a zone burn shows here with no matching
        # damage entry. training/reward.py prices that gap deliberately; see its docstring.
        "hp_healed_tick": hp_healed,
        "kills_tick": kills_tick,
        "deaths_tick": newly_dead,
        "death_cause_tick": death_cause_tick,
        "cubes_gained_tick": cubes_gained,
        "boxes_broken_tick": boxes_broken_tick,
        "shots_fired_tick": shots_fired_tick,
        "dash_hits_tick": dash_hits_tick,
        # (N,) int32, the HERO's alone -- what training/reward.py's `attack_in_reach` term pays.
        "attack_in_reach_tick": attacks_in_reach,
        # (N,) int32, the HERO's alone -- what training/reward.py's `gadget_hit` term pays.
        "gadget_hit_tick": gadget_hits,
        # (N,) int32, 0 or 1 per DECISION -- what training/reward.py's `move_reversal` term pays.
        "move_reversal_tick": move_reversals,
        "hero_rank": decision["hero_rank"],
        "terminated": terminated,
        "truncated": truncated,
        # --- action-repeat fields (see module docstring). `hero_alive` is LATCHED at the
        # sub-tick the episode ended, while `obs["hero"]["alive"]` is the post-decision value;
        # they can disagree for an env that finished early, and reward shaping must use this one.
        "hero_alive": decision["hero_alive"],
        "alive_ticks": decision["alive_ticks"],
        "in_zone_ticks": decision["in_zone_ticks"],
        "n_ticks": decision["n_ticks"],
        "time": state.time,
        "step_count": state.step_count,
    }
