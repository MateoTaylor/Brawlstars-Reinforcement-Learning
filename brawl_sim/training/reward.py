"""ShapedReward: the dense reward PPO actually trains on.

Satisfies `core.reward.RewardFn` structurally (`__call__(obs, info, cfg) -> (N,) f32`), so it
drops straight into `BrawlVecEnv(reward_fn=...)` / `BrawlSB3VecEnv(..., reward_fn=...)` with no
inheritance. Like every other hot-path callable in this codebase it is **device-resident and
sync-free**: no `.item()`, no `.cpu()`, no boolean indexing, no Python loop over envs. It is
called from `env.py`'s phase 16 (`_observe`), inside `step()`.

**Why shaped and not terminal-only.** An episode runs up to `max_episode_steps` sim ticks (3000
by default, i.e. `max_agent_steps` = 600 decisions at `action_repeat=5`) and ends in a 1-of-10
placement; a pure win/lose signal gives PPO one bit per episode, which in practice does not get
a policy off the ground in any reasonable wall-clock budget on this hardware. Every per-tick
term here exists to make the credit assignment tractable, and every weight lives in
`configs/train.yaml` so the shaping can be dialled back toward sparse (set the per-tick weights
to 0.0) once a policy is competent, without touching this file.

**Every weight below is priced per SIM TICK, not per decision, and stays that way when
`action_repeat` changes.** The delta terms (damage, kills, cubes) read `info` fields `env.py`
already summed over the decision's sub-ticks; the two rate terms read `info["alive_ticks"]` /
`info["in_zone_ticks"]`, which count sub-ticks rather than firing once per call. So total
episode return -- the thing PPO actually optimizes -- is invariant to `action_repeat`, and
retuning the decision rate does not silently rescale the reward out from under a tuned config.
`gamma` is the one knob that does NOT come along for free: it discounts per decision.

**Reward-hacking notes, since shaping invites it.** `damage_dealt` is capped in practice by the
hero's own ammo/cooldown, so it can't be farmed indefinitely; `survive_per_step` is deliberately
small relative to `win_bonus`, so hiding in a bush for 3000 ticks scores strictly worse than
winning; `in_zone_per_step` is a flat per-tick penalty rather than proportional to zone damage,
so it stays meaningful even at the low DPS the zone opens with. If you raise `survive_per_step`,
re-check it against `win_bonus + rank_bonus * (n_entities - 1)` -- the terminal payoff must stay
the dominant term or the agent will correctly learn to stall.

`hp_healed` is the one term that can in principle pay for taking damage, so it is worth spelling
out why it does not. Total HP healed in an episode cannot exceed total HP lost (you cannot heal
above max), so at `hp_healed = -damage_taken` the pair nets out at worst to zero -- for COMBAT
damage. Zone damage is the gap: it never enters `damage_taken_tick` (no attacker, see
core/events.py), so burning in the gas and regenerating it back scores `hp_healed` with no
matching debit. At the shipped numbers that trade is heavily negative anyway -- a second in the
zone costs `20 ticks * in_zone_per_step` (-1.0) and returns at most `0.20 * max_hp * hp_healed`
(+0.12 at 6000 max HP and 1e-4), and the regen it takes to earn that back runs at 0.13/s against
the zone's 0.20/s, i.e. slower than the burn. Raising `hp_healed` past roughly `-5 *
in_zone_per_step / (zone_fraction_per_second * max_hp)` flips the sign of that arithmetic; if you
do, price zone damage proportionally instead of with a flat per-tick penalty.

**The returned tensor is a shared, preallocated buffer** (same contract as `core.reward
.ZeroReward`): callers must treat it as read-only, since the next `step()` overwrites it in
place. `EpisodeStats` and `BrawlSB3VecEnv` both already do (they copy out of it, never into it).
"""
import torch

from .config import RewardConfig

_HERO = 0

# (RewardConfig weight name, label used in logs). Order is the order they're summed in and the
# order `term_means()` reports; keep it stable so a TensorBoard run stays comparable across edits.
TERM_NAMES = (
    "damage_dealt", "damage_taken", "hp_healed", "kill", "cube_pickup",
    "survive_per_step", "in_zone_per_step", "win_bonus", "death_penalty", "rank_bonus",
)


class ShapedReward:
    """See module docstring. `track_terms=True` additionally accumulates each term's batch sum
    into a device-side scalar, so `training/callbacks.py` can log the per-term contribution
    breakdown once per rollout (one host sync per rollout, not per step). That costs one extra
    small reduction kernel per enabled term per tick -- a few percent on a launch-bound sim, and
    worth it while you're still tuning weights. Set it False for a production run with settled
    weights."""

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
        # `obs` is deliberately unread: every field this used to take from it (hero alive/in_zone)
        # now comes from `info`, which -- unlike `obs` -- is aggregated and latched over the whole
        # decision rather than describing only its final sim tick. The argument stays because
        # `core.reward.RewardFn` defines the signature and other reward functions do use it.
        terminated, truncated = info["terminated"], info["truncated"]
        done = terminated | truncated

        reward = self._buffer(terminated.shape[0], terminated.device)
        reward.zero_()

        # ---- per tick ----
        if w.damage_dealt != 0.0:
            self._add(reward, "damage_dealt", info["damage_dealt_tick"][:, _HERO], w.damage_dealt)
        if w.damage_taken != 0.0:
            # damage_taken_tick is COMBAT damage only (it's a projection of the attacker x victim
            # matrix, and zone damage has no attacker). Zone pressure is priced by
            # in_zone_per_step instead, deliberately: a flat per-tick cost reads the same at the
            # zone's opening DPS as at its escalated late-game DPS.
            self._add(reward, "damage_taken", info["damage_taken_tick"][:, _HERO], w.damage_taken)
        if w.hp_healed != 0.0:
            # The counterpart to damage_taken, and priced at its mirror image (+1e-4 against
            # -1e-4) so a wound the hero out-regens or lifesteals back is a wash rather than a
            # permanent debt. hp_healed_tick is HP ACTUALLY restored -- healing at full HP, or a
            # dead hero's pending lifesteal, is already zeroed by core/combat.py's clamps -- so
            # this term is bounded per episode by the HP the hero has actually lost.
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
