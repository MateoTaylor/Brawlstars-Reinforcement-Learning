"""ShapedReward: the dense reward PPO actually trains on.

Satisfies `core.reward.RewardFn` structurally (`__call__(obs, info, cfg) -> (N,) f32`), so it
drops into `BrawlVecEnv(reward_fn=...)` / `BrawlSB3VecEnv(..., reward_fn=...)` with no
inheritance. Device-resident and sync-free like every hot-path callable; called from `env.py`'s
phase 16 (`_observe`), inside `step()`.

**Why shaped.** An episode is hundreds of decisions ending in a 1-of-`n_entities` placement, and
one win/lose bit per episode does not get a policy off the ground in a reasonable budget. Every
weight lives in `configs/train.yaml`, so the shaping can be dialled toward sparse without
touching this file.

**Every weight is priced per SIM TICK, not per decision, and stays that way when `action_repeat`
changes.** The delta terms (damage, kills, cubes, attacks in reach, gadget hits) read `info`
fields `env.py` already summed over the decision's sub-ticks, and the two rate terms read
`info["alive_ticks"]` / `info["in_zone_ticks"]`, which count sub-ticks. So the episode return is
invariant to `action_repeat`; `gamma` is the one knob that is not, since it discounts per
decision.

**Reward-hacking notes.** `damage_dealt` is capped by the hero's ammo and cooldown.
`survive_per_step` must stay small against `win_bonus + rank_bonus * (n_entities - 1)`, or the
agent correctly learns to stall. `in_zone_per_step` is flat per tick rather than proportional to
zone damage, so it means the same at the zone's opening rate as late. `hp_healed` nets out at
worst to zero against `damage_taken` for COMBAT damage (no healing above max), but zone damage
never enters `damage_taken_tick` (no attacker), so burning in the gas and regenerating scores
`hp_healed` with no matching debit. At the shipped weights a second in the zone costs far more
than regen can return, and regen is slower than the burn; if `hp_healed` is raised a lot, price
zone damage proportionally instead.

**The returned tensor is a shared, preallocated buffer** (same contract as
`core.reward.ZeroReward`): callers must treat it as read-only, since the next `step()`
overwrites it in place. `EpisodeStats` and `BrawlSB3VecEnv` copy out of it.
"""
import torch

from .config import RewardConfig

_HERO = 0

# (RewardConfig weight name, label used in logs). Order is the order they're summed in and the
# order `term_means()` reports; keep it stable so a TensorBoard run stays comparable across edits.
TERM_NAMES = (
    "damage_dealt", "damage_taken", "hp_healed", "kill", "cube_pickup",
    "survive_per_step", "in_zone_per_step", "win_bonus", "death_penalty", "rank_bonus",
    "attack_in_reach", "gadget_hit",
)


class ShapedReward:
    """See module docstring. `track_terms=True` also accumulates each term's batch sum on device,
    so `training/callbacks.py` can log the per-term breakdown once per rollout (one host sync per
    rollout, not per step), at the cost of one small reduction per enabled term per tick."""

    def __init__(self, cfg: RewardConfig, track_terms: bool = True) -> None:
        self.cfg = cfg
        self.track_terms = track_terms
        self._cache_key: tuple | None = None
        self._buf: torch.Tensor | None = None
        self._term_sums: dict[str, torch.Tensor] = {}
        self._ticks: int = 0
        # Only terms with a non-zero weight are computed at all -- a plain Python truth test on a
        # config float, resolved once here rather than per tick.
        self._active = tuple(name for name in TERM_NAMES if getattr(cfg, name) != 0.0)

    # ---- RewardFn -------------------------------------------------------------------------

    def __call__(self, obs: dict, info: dict, cfg) -> torch.Tensor:
        w = self.cfg
        # `obs` is deliberately unread: `info` is aggregated and latched over the whole decision,
        # while `obs` describes only its final sim tick. The argument stays because
        # `core.reward.RewardFn` defines the signature.
        terminated, truncated = info["terminated"], info["truncated"]
        done = terminated | truncated

        reward = self._buffer(terminated.shape[0], terminated.device)
        reward.zero_()

        # ---- per tick ----
        if w.damage_dealt != 0.0:
            self._add(reward, "damage_dealt", info["damage_dealt_tick"][:, _HERO], w.damage_dealt)
        if w.damage_taken != 0.0:
            # damage_taken_tick is COMBAT damage only (a projection of the attacker x victim
            # matrix; zone damage has no attacker). Zone pressure is priced by in_zone_per_step,
            # a flat per-tick cost that reads the same at the zone's opening rate as late.
            self._add(reward, "damage_taken", info["damage_taken_tick"][:, _HERO], w.damage_taken)
        if w.hp_healed != 0.0:
            # damage_taken's counterpart (train.yaml prices it as the mirror image), so a wound
            # the hero out-regens or lifesteals back is a wash. hp_healed_tick is HP ACTUALLY
            # restored (core/combat.py's clamps zero overheal and a dead hero's pending lifesteal),
            # so the term is bounded per episode by the HP the hero has lost.
            self._add(reward, "hp_healed", info["hp_healed_tick"][:, _HERO], w.hp_healed)
        if w.kill != 0.0:
            self._add(reward, "kill", info["kills_tick"][:, _HERO], w.kill)
        if w.cube_pickup != 0.0:
            self._add(reward, "cube_pickup", info["cubes_gained_tick"][:, _HERO], w.cube_pickup)
        if w.survive_per_step != 0.0:
            self._add(reward, "survive_per_step", info["alive_ticks"], w.survive_per_step)
        if w.in_zone_per_step != 0.0:
            # `in_zone` is True when the hero is OUTSIDE the shrinking safe rect (core/zone.py's
            # `_outside_rect`) -- i.e. taking zone damage. Named for the death-zone, not the safe
            # zone; the sign here depends on reading it that way.
            self._add(reward, "in_zone_per_step", info["in_zone_ticks"], w.in_zone_per_step)

        # ---- terminal (only meaningful on the tick the episode actually ends) ----
        # info["hero_rank"] is 0-INDEXED: 0 == last one standing. A timeout that catches the hero
        # alive but not alone is rank >= 1, so it is correctly neither a win nor a death.
        rank = info["hero_rank"]
        if w.win_bonus != 0.0:
            self._add(reward, "win_bonus", done & (rank == 0), w.win_bonus)
        if w.death_penalty != 0.0:
            # `info["hero_alive"]`, NOT `obs["hero"]["alive"]`: with action_repeat > 1 the two
            # disagree for an env that finished mid-decision. info's is latched at the sub-tick
            # the episode ended; obs' is the live value after the remaining sub-ticks ran, so a
            # hero that won and was then finished off by the zone would be charged the death
            # penalty for an episode it had already won. See core/events.advance_decision_tally.
            self._add(reward, "death_penalty", done & ~info["hero_alive"], w.death_penalty)
        if w.rank_bonus != 0.0:
            places_above_last = torch.clamp(cfg.n_entities - 1 - rank, min=0)
            self._add(reward, "rank_bonus", done * places_above_last, w.rank_bonus)

        # ---- attack shaping (user decision, 2026-09-21; BRAWL_SIM_DESIGN.md §10) ----
        if w.attack_in_reach != 0.0:
            # A count, like `kill`: attacks and supers the hero made while an enemy it could see
            # stood inside its uncharged dash reach (core/events.py, env.py's attack phase), at
            # most one per decision. Blunt on purpose: a dash AWAY from that enemy and a miss are
            # paid too. Ammo and cooldown bound it at one per legal attack, so at the shipped 0.05
            # a whole episode of in-reach attacks is worth a few units, against 10 for a win.
            self._add(reward, "attack_in_reach", info["attack_in_reach_tick"], w.attack_in_reach)

        # ---- gadget shaping (user decision, 2026-09-30; BRAWL_SIM_DESIGN.md §10) ----
        if w.gadget_hit != 0.0:
            # A count: landings of the hero's gadget spinner that hurt at least one player
            # (core/events.py, env.py's projectile phase). One per landing however many players
            # it catches, and none for a spinner that lands only on crates or on nothing. Paid on
            # top of what `damage_dealt` pays for the same blast. The 18 s cooldown bounds it: at
            # the shipped 0.3 a match of landed spinners is worth about 3, against 10 for a win.
            self._add(reward, "gadget_hit", info["gadget_hit_tick"], w.gadget_hit)

        if w.scale != 1.0:
            reward.mul_(w.scale)
        self._ticks += 1
        return reward

    # ---- term tracking ----------------------------------------------------------------------

    def term_means(self) -> dict:
        """Mean per-env-per-DECISION contribution of each active term since the last call, then
        resets the accumulators. **This is the one host sync in this module** -- call it once per
        rollout from a callback, never inside the step loop. Returns `{}` when `track_terms` is
        off or no decisions have elapsed.

        Per decision, not per sim tick: `__call__` runs once per `step()`, so with
        `action_repeat > 1` each of these numbers covers `action_repeat` ticks of simulated time.
        They stay comparable across runs with the same `action_repeat` and are rescaled by it
        otherwise -- which is the point, since the summed episode return is what's invariant."""
        if not self.track_terms or self._ticks == 0 or not self._term_sums:
            return {}
        denom = float(self._ticks) * float(self._buf.shape[0])
        stacked = torch.stack([self._term_sums[name] for name in self._active])
        values = (stacked / denom).tolist()
        out = dict(zip(self._active, values))
        for name in self._active:
            self._term_sums[name].zero_()
        self._ticks = 0
        return out

    # ---- internals ---------------------------------------------------------------------------

    def _add(self, reward: torch.Tensor, name: str, term: torch.Tensor, weight: float) -> None:
        """`reward += weight * term`, in place. `term` may be bool/int32/int64 -- `add_` promotes
        it against the float32 accumulator, which is what keeps every call site above free of
        explicit `.to(torch.float32)` noise."""
        reward.add_(term, alpha=weight)
        if self.track_terms:
            self._term_sums[name] += term.sum() * weight

    def _buffer(self, n_envs: int, device) -> torch.Tensor:
        key = (n_envs, device)
        if key != self._cache_key:
            self._buf = torch.zeros(n_envs, dtype=torch.float32, device=device)
            self._term_sums = {
                name: torch.zeros((), dtype=torch.float32, device=device) for name in self._active
            }
            self._ticks = 0
            self._cache_key = key
        return self._buf
