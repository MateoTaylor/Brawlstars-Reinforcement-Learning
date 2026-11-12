"""BrawlVecEnv: the batched, torch-native Solo Showdown environment. `core/` never imports
`bots/`; this module is where the two meet (`_bot_phase` and `_build_observation` call
`bots/perception` and `bots/policy`).

**Tick order.** One `step()` is one AGENT DECISION: `history.push` records the pre-step state and
the action, `_run_decision` runs `cfg.action_repeat` sim ticks of phases 1-15 with that action
held, then phases 16-17 run once.

     1  action latency buffer       `_pop_action_buffer`
     2  timers, regen               `_tick_timers`
     3  decode the hero's action    `_decode`
     4  bot intents                 `_bot_phase` (its visibility is reused by phase 6)
     5  external overrides          `_override_phase`
    5b  stomps                      `_stomp_phase`: armed Stompers whose wind-up ran out land
     6  attacks                     `_attack_phase`: dashes start, supers, gadgets, volleys, melee
     7  movement                    `_movement_phase`, non-dashing entities only
     8  dash advance                `_dash_phase`
     9  projectiles                 `_projectile_phase`
    10  zone damage                 `_zone_phase`
    11  broken boxes -> pickups     `_box_phase`
    12  pickup collection           `_pickup_phase`
    13  deaths                      `_death_phase`
    14  zone shrink schedule        `_zone_schedule`
    15  bookkeeping                 `_bookkeeping`: damage dealt, super charge, time, step_count
    16  observation, info, reward   `_observe`, once per decision
    17  autoreset                   `_autoreset`, once per decision

Attacks (6) resolve before movement (7), so a dash replaces that tick's walk. Stomps (5b) land
before the attacks, so a stomp armed in phase 6 lands exactly its wind-up later, and a stun it
applies stops that same tick's attack and walk. Regen (2) runs before any damage, so it cannot
undo a lethal hit. Deaths (13) resolve after every damage source
(6-10), so simultaneous kills work.

**Autoreset happens inside `step()`.** A finished env is reset in place within the same call, so
the returned `obs` is the FIRST observation of the new episode for every env that finished, while
`reward`/`terminated`/`truncated`/`info` describe the decision that just ended. The finished
episode's own last obs/info are in `info["final_observation"]`/`info["final_info"]` (dense over
all N envs, meaningful only where `terminated | truncated`). Mixing the two up silently
misaligns reward/observation pairs.

**`obs`/`info` are valid only until the next `step()`/`reset()`.** Their fields are views into
`SimState`'s tensors, which the next call overwrites in place; a caller that keeps one must
`core.observation.clone_obs` it. `step()` clones `final_observation`/`final_info` itself, before
autoreset mutates the state.

**action_repeat.** default.yaml's `dt=0.05` with `action_repeat=5` simulates at 20 Hz and decides
at 4 Hz.

  *Units.* The world's clock (`cfg.max_episode_steps`, `state.step_count`, `state.time`, the zone
  timings, `decision_period_ticks`) is in sim ticks or sim seconds. Anything counted in `step()`
  calls (SB3 timesteps, `n_steps`, `info["episode"]["l"]`) is in decisions, `cfg.max_agent_steps`
  per episode. `gamma` is per decision: a per-tick discount becomes `gamma ** action_repeat`.

  *Movement is held; fire is not.* The move bin applies on every sub-tick, the attack column on
  the first only (`_held`), so one decision is at most one attack attempt. That is what
  `hero.action_mask` promises, since MaskablePPO evaluates it once per decision, and it holds for
  any weapon that could fire faster than the decision rate.

  *Events are summed, outcomes are latched.* Per-tick deltas (damage, kills, cubes, boxes, heals)
  sum over the sub-ticks; `terminated`, `truncated`, `hero_rank` and `hero_alive` latch at the
  sub-tick the episode ended, and later sub-ticks add nothing. See
  `core/events.advance_decision_tally`, including why a finished env's `final_observation` can be
  up to `action_repeat - 1` ticks stale.

**Visibility runs in three places, and none can stand in for another:** once per sub-tick for
the bots (phase 4, pre-movement), then once per observation build, for the finished decision
(phase 16, post-combat) and after autoreset (phase 17, post-reset).

**`cfg.compile`** (default False) wraps only `_run_tick`, phases 1-15, in
`torch.compile(dynamic=False, fullgraph=False)`; phases 16-17 build Python dicts and call the
arbitrary `reward_fn`. On this Windows machine neither inductor backend builds (CPU needs MSVC
`cl.exe`, CUDA needs Triton), so `_make_tick_fn` re-raises those failures as a `RuntimeError`.
"""
from pathlib import Path

import torch
import yaml

from .bots import perception, policy
from .config import (
    EnvConfig, apply_randomization, body_radius_tiles, build_params, load_randomization, spec_hero,
    validate,
)
from .constants import DeathCause, GadgetType, SuperType
from .core import boxes, camera, combat, events, geometry as geo, hero, history, melee_sweep, movement
from .core import observation, projectiles, slots, spawn, stats, status, zone
from .core.reward import ZeroReward
from .core.state import allocate, check_invariants, snapshot
from .maps.loader import build_map_bank

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_base_spec() -> dict:
    return {
        **yaml.safe_load((_CONFIGS_DIR / "default.yaml").read_text()),
        **yaml.safe_load((_CONFIGS_DIR / "brawlers.yaml").read_text()),
    }


class BrawlVecEnv:
    """`cfg` is a resolved `EnvConfig` (`config.load_config`), which carries no brawler stats, so
    the env loads `configs/default.yaml` + `configs/brawlers.yaml` itself to build the `spec`
    that `build_params`/`resample_params` read. **Trap:** a `load_config(..., overrides=...)`
    entry that touched a SimParams-only field (`base_hp`, `zone.step_seconds`,
    `entities.enemy_hp_mult` -- anything outside `_ENV_CONFIG_FIELDS`) is invisible to
    `EnvConfig` and would be lost here; pass the merged dict as `spec`.

    `reward_fn=None` builds a fresh `ZeroReward` per env rather than sharing one stateful default
    instance (it caches its zeros per `(n_envs, device)`).

    `autoreset=False` is for `wrappers/gym_single.py`, since a single-env `gymnasium.Env` must not
    reset inside `step()`: phase 17 is skipped, `state` stays as phase 16 left it, and `obs` is
    that decision's own (possibly terminal) observation. `final_observation`/`final_info` are
    still set, equal to `obs`/`info`.

    `verbose` is passed to `core.state.allocate`, whose memory report tools that build many envs
    (scripts/benchmark.py) turn off.
    """

    def __init__(
        self, cfg: EnvConfig, n_envs: int, device=None, seed: int = 0,
        reward_fn=None, randomization=None, spec: dict | None = None, autoreset: bool = True,
        verbose: bool = True, params_hook=None, tick_hook=None,
    ) -> None:
        self.cfg = cfg
        self.n_envs = n_envs
        self.device = torch.device(device if device is not None else cfg.device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.reward_fn = reward_fn if reward_fn is not None else ZeroReward()
        self.autoreset = autoreset
        # A plain attribute: training/builder.py installs the curriculum manager here after
        # construction (it needs `self.gen`), and every reset re-reads it. Contract:
        # spawn.reset_envs.
        self.params_hook = params_hook
        # `tick_hook(env)` runs after every SUB-TICK's phases 1-15, before phases 16/17, so
        # scripts/watch.py and scripts/record_rollout.py can sample the world at the sim rate.
        # Debug tooling only: what it does (`state.snapshot`, `.cpu()`) are host syncs, and
        # nothing in the training path installs one.
        self.tick_hook = tick_hook

        base_spec = spec if spec is not None else _load_base_spec()
        if randomization is None:
            randomization_spec = {}
        elif isinstance(randomization, dict):
            randomization_spec = randomization
        else:
            randomization_spec = load_randomization(randomization)
        # Applied once: the spec keeps {low, high} ranges, which resample_params redraws on
        # every reset.
        self.spec = apply_randomization(base_spec, randomization_spec) if randomization_spec else base_spec
        # The class docstring's trap, for the one key both halves read: `cfg.hero` names the hero
        # (viewer, validate's messages) while the spec's copy decides whose stats fill column 0.
        if spec_hero(self.spec) != cfg.hero:
            raise ValueError(
                f"cfg.hero is {cfg.hero!r} but the spec's entities.hero is {spec_hero(self.spec)!r}; "
                f"pass the spec merged with the same overrides as the cfg"
            )

        self.bank =build_map_bank(cfg, device=self.device, body_radius=body_radius_tiles(self.spec))
        self.params = build_params(cfg, n_envs=n_envs, device=self.device, gen=self.gen, spec=self.spec)
        validate(cfg, self.params)
        self.state = allocate(cfg, n_envs=n_envs, device=self.device, verbose=verbose)
        # The (N,E,E) visibility behind the most recent observation, stashed by
        # `_build_observation` so `step` can hand it to `history.push` without a second pass.
        # `_obs_hero_view` is its hero row cut to the camera window (`core/camera.hero_view`):
        # what the observation revealed, and so what the history may remember. None until the
        # first observation; a test that teleports entities sets it to None so the next step()
        # recomputes both.
        self._obs_vis: torch.Tensor | None = None
        self._obs_hero_view: torch.Tensor | None = None

        # Built once. Under cfg.compile, compilation happens lazily on the first call, specialized
        # to its shapes (dynamic=False); see _make_tick_fn for how a failure surfaces.
        self._tick_fn = self._make_tick_fn()

    # ---- public API -----------------------------------------------------------------

    def reset(self, reset_mask: torch.Tensor | None = None) -> dict:
        if reset_mask is None:
            reset_mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        spawn.reset_envs(self.state, reset_mask, self.bank, self.params, self.cfg, self.gen,
                         self.spec, self.params_hook)
        obs = self._build_observation()
        if self.cfg.debug_checks:
            check_invariants(self.state, self.cfg, self.params)
        return obs

    def step(self, action: torch.Tensor, override: torch.Tensor | None = None):
        """`action` (N,2) and `override` (N,E,2) are cast to int64 and moved onto `self.device`
        here, so callers need not pre-place them (`_pop_action_buffer`'s `scatter_` needs
        matching devices)."""
        assert action.shape == (self.n_envs, 2), f"action shape {tuple(action.shape)} != {(self.n_envs, 2)}"
        action = action.to(device=self.device, dtype=torch.int64)
        if override is not None:
            assert override.shape == (self.n_envs, self.cfg.n_entities, 2)
            override = override.to(device=self.device, dtype=torch.int64)

        # Record (pre-step state, this action) as the newest history slot BEFORE the world moves
        # -- once per decision, outside `_run_tick`, so action_repeat can't touch the ring
        # cadence. `_obs_hero_view` is the reveal of the observation this action answers; a
        # caller that steps without observing first gets a fresh pass instead of a None.
        if self._obs_hero_view is None:
            self._obs_vis = perception.visibility(self.state, self.bank, self.params, self.cfg)
            self._obs_hero_view = camera.hero_view(self.state, self._obs_vis, self.cfg)
        # Read slot 0 before the push overwrites it with this action.
        move_reversals = events.move_reversals(self.state, action, self.cfg)
        history.push(self.state, action, self._obs_hero_view)

        (dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, attacks_in_reach,
         gadget_hits, attack_misses, gadget_misses, decision) = self._run_decision(action, override)

        obs_before_reset, info, reward = self._observe(
            dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, attacks_in_reach,
            gadget_hits, attack_misses, gadget_misses, decision, move_reversals,
        )
        # Cloned here, and the local dropped before `_autoreset` rebuilds the observation: this
        # reference would otherwise keep the pre-reset grids (large at thousands of envs) alive
        # while the post-reset build allocates a same-sized copy, raising peak VRAM.
        final_observation = observation.clone_obs(obs_before_reset)
        final_info = observation.clone_obs(dict(info))

        if self.autoreset:
            del obs_before_reset
            obs, terminated, truncated = self._autoreset(info)
        else:
            # No reset (see class docstring): `obs` is this decision's own, possibly terminal,
            # observation.
            obs = obs_before_reset
            terminated, truncated = info["terminated"], info["truncated"]
        info["final_observation"] = final_observation
        info["final_info"] = final_info

        if self.cfg.debug_checks:
            check_invariants(self.state, self.cfg, self.params)

        return obs, reward, terminated, truncated, info

    def snapshot(self, env_index: int) -> dict:
        """CPU, rendering only -- see core.state.snapshot."""
        return snapshot(self.state, env_index)

    # ---- internals --------------------------------------------------------------------

    def _build_observation(self, slot_envs: torch.Tensor | None = None) -> dict:
        vis = perception.visibility(self.state, self.bank, self.params, self.cfg)
        hero_view = camera.hero_view(self.state, vis, self.cfg)  # concealment AND the camera window
        self._obs_vis, self._obs_hero_view = vis, hero_view       # read by step() -> history.push
        # Tracker-style enemy slots, promoted on this decision's sighting as the live tracker
        # does; `build_obs` reads them after the update. `slot_envs`: see `_autoreset`.
        slots.update(self.state, hero_view, self.cfg, envs=slot_envs)
        # zone.active is a latch: gas has been on screen at least once.
        zone.mark_seen(self.state, camera.camera_centre(self.state.ent_pos[:, 0], self.cfg), self.cfg)
        # Only the two obs fields it feeds read it, and train.yaml turns those off.
        los = perception.raw_los(self.state, self.bank, self.cfg) if self.cfg.obs_include_raw_los else None
        return observation.build_obs(self.state, self.bank, vis, los, self.params, self.cfg,
                                     hero_view=hero_view)

    def _run_tick(self, action: torch.Tensor, override: torch.Tensor | None):
        """Phases 1-15: one sim tick, mutating SimState in place. Its own method so
        `torch.compile` gets a single callable (`self._tick_fn`); phase 16 (`_observe`: Python
        dicts and the arbitrary `reward_fn`) and phase 17 (`_autoreset`, which reads that dict)
        stay outside."""
        effective_action = self._pop_action_buffer(action)
        regen_healed = self._tick_timers()
        hero_move_dir, hero_fire, hero_super, hero_gadget, hero_auto = self._decode(effective_action)
        move_dir, fire, super_fire, gadget_fire, aim_dir, aim_point, vis = self._bot_phase(
            hero_move_dir, hero_fire, hero_super, hero_gadget)
        move_dir, fire = self._override_phase(move_dir, fire, override)
        stomp = self._stomp_phase()

        dmg_by_melee, melee_healed, attack_in_reach = self._attack_phase(
            move_dir, fire, super_fire, aim_dir, aim_point, gadget_fire, vis, hero_auto=hero_auto)
        self._movement_phase(move_dir)
        dmg_by_dash, charging = self._dash_phase()
        policy.track_seen_velocity(self.state, self.cfg)  # ent_vel is final once the dash moved
        dmg_by_proj, super_healed, proj_charge_hits, gadget_hit, attack_miss = self._projectile_phase()
        # The reward's `gadget_hit` pays a spinner that hurt a player and a stomp that slowed or
        # stunned one alike (user decision 2026-10-23, plan D9). A spec with no STOMP kind has no
        # stomp phase, so its flag stays the spinner's alone. Its `gadget_miss` is a stomp that
        # landed on nobody (user decision 2026-11-02, BULL_HERO_PLAN.md §5.8); a spinner's miss is
        # not scored, so without a stomp phase the flag stays False.
        gadget_miss = torch.zeros_like(gadget_hit)
        if stomp is not None:
            gadget_hit = gadget_hit | stomp[0]
            gadget_miss = stomp[1]
        self._zone_phase()
        newly_broken = self._box_phase()
        cubes_gained = self._pickup_phase()
        newly_dead = self._death_phase()
        self._zone_schedule()

        dmg_by_total = dmg_by_melee + dmg_by_dash + dmg_by_proj
        # Every heal source, summed into the one (N,E) delta `compute_info` takes: phase 2's
        # regen, phase 6's melee lifesteal (Edgar) and phase 9's super lifesteal (Mortis). A power
        # cube's max-HP bump also raises current HP (core/stats.py) but is not counted: it is a
        # pickup, priced by the reward's `cube_pickup` term, not HP won back from a wound.
        hp_healed = regen_healed + super_healed + melee_healed
        # Super charge, per (attacker, victim) pair: a melee or dash hit counts 1, and the
        # projectile phase hands up how many of the attacker's projectiles hit, the gadget
        # spinner's left out (the gadget charges no super). So each of Bull's pellets counts
        # (user decision 2026-10-23: 9 hits, two full volleys). The larger count stands on a pair
        # that both hit this tick, which keeps Mortis's dash-plus-bolt tick at 1, as the OR it
        # replaced gave. A CHARGE super's own hits charge nothing: the user counts the meter in
        # projectile hits.
        dash_hit = (dmg_by_dash > 0) & ~charging.unsqueeze(-1)
        charge_hits = torch.maximum(((dmg_by_melee > 0) | dash_hit).to(proj_charge_hits.dtype), proj_charge_hits)
        self._bookkeeping(dmg_by_total, charge_hits)
        return (dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, attack_in_reach,
                gadget_hit, attack_miss, gadget_miss)

    def _run_decision(self, action: torch.Tensor, override: torch.Tensor | None):
        """One AGENT DECISION: `cfg.action_repeat` `_run_tick` calls holding the same action.
        Returns `_run_tick`'s nine values summed over the sub-ticks, plus the `core/events`
        decision tally. Its three (N,) bool flags, attack-in-reach, gadget-hit and gadget-miss,
        become int32 counts, like the attack-miss count beside them.

        The loop is a static `range` over a config int, so it stays sync-free (no `.any()` check
        for "every env finished") and `torch.compile` only ever sees `_run_tick`.

        **`live` is sampled BEFORE each sub-tick:** an env that finished on an earlier sub-tick
        adds nothing further to any delta, which would otherwise be credited to the finished
        episode's reward. The tally applies the same gate; see `events.advance_decision_tally`.

        **Sub-ticks 2..K hold the move bin but drop the attack column** (`_held`), so one
        decision is at most one attack attempt (see the module docstring)."""
        (dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed, in_reach,
         gadget_hit, attack_miss, gadget_miss) = self._tick_fn(action, override)
        # COUNTS from here on, int32 like the tally's counters. At most 1 attack per decision,
        # since `_held` drops the fire bit; with action latency the attack can land on a later
        # sub-tick, so the loop below adds it under the same `live` gate as every other delta. A
        # spinner lands `gadget_flight_seconds` after its throw, so its hit can fall on a later
        # sub-tick or decision than the throw. A stomp lands `stomp_windup_seconds` after its tap:
        # Bull's 0.30 s is 6 ticks, so at action_repeat 5 it lands on the next decision's second
        # sub-tick (both shifted alike by any action latency). A volley's miss is scored when its
        # last pellet ends, up to 12 ticks after the shot, so it too falls on a later decision.
        attacks_in_reach = in_reach.to(torch.int32)
        gadget_hits = gadget_hit.to(torch.int32)
        attack_misses = attack_miss
        gadget_misses = gadget_miss.to(torch.int32)
        decision = events.new_decision_tally(self.state, self.cfg)
        self._run_tick_hook()
        if self.cfg.action_repeat == 1:
            # Keeps the `_held` clones below off the action_repeat=1 path.
            return (dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed, attacks_in_reach,
                    gadget_hits, attack_misses, gadget_misses, decision)

        held_action, held_override = self._held(action), self._held(override)
        for _ in range(self.cfg.action_repeat - 1):
            live = ~decision["done"]
            (tick_dmg, tick_dead, tick_broken, tick_cubes, tick_healed, tick_in_reach,
             tick_gadget_hit, tick_attack_miss, tick_gadget_miss) = self._tick_fn(held_action, held_override)
            live_e = live.unsqueeze(-1)
            dmg_by = dmg_by + tick_dmg * live.view(-1, 1, 1)
            newly_dead = newly_dead | (tick_dead & live_e)
            newly_broken = newly_broken | (tick_broken & live_e)
            cubes_gained = cubes_gained + tick_cubes * live_e
            hp_healed = hp_healed + tick_healed * live_e
            attacks_in_reach = attacks_in_reach + (tick_in_reach & live).to(torch.int32)
            gadget_hits = gadget_hits + (tick_gadget_hit & live).to(torch.int32)
            attack_misses = attack_misses + tick_attack_miss * live.to(torch.int32)
            gadget_misses = gadget_misses + (tick_gadget_miss & live).to(torch.int32)
            decision = events.advance_decision_tally(decision, self.state, self.cfg)
            self._run_tick_hook()

        return (dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed, attacks_in_reach,
                gadget_hits, attack_misses, gadget_misses, decision)

    @staticmethod
    def _held(action: torch.Tensor | None) -> torch.Tensor | None:
        """The same action with its attack column zeroed, driving sub-ticks 2..K of a decision.
        Works for `action` (N,2) and `override` (N,E,2): the attack column is the last of either,
        and `override`'s `-1` "no override" sentinel in column 0 is left alone.

        Zeroing the whole column covers every value it can carry (1 attack, 2 super, 3 gadget,
        4 auto-aimed attack), so one decision is at most one of them (tests/test_gadget.py pins
        the gadget's). Built once per decision."""
        if action is None:
            return None
        held = action.clone()
        held[..., 1] = 0
        return held

    def _run_tick_hook(self) -> None:
        """Calls `self.tick_hook(self)` after a sub-tick's phases 1-15, if one is installed (see
        `__init__`)."""
        if self.tick_hook is not None:
            self.tick_hook(self)

    def _make_tick_fn(self):
        """`cfg.compile=False` (the default) returns `self._run_tick` unwrapped. `cfg.compile=True`
        wraps `torch.compile(self._run_tick, dynamic=False, fullgraph=False)` in a guard that
        re-raises a compile-machinery failure -- an exception whose type lives in
        `torch._dynamo`/`torch._inductor`, e.g. `TritonMissing`, or inductor's "cl is not found"
        on Windows CPU -- as a `RuntimeError` chained `from e`. Everything else (a real bug in
        `_run_tick`) propagates with its own type and message, so a domain bug is never
        reported as "compile unsupported".
        """
        if not self.cfg.compile:
            return self._run_tick

        compiled = torch.compile(self._run_tick, dynamic=False, fullgraph=False)

        def _guarded_tick(action, override):
            try:
                return compiled(action, override)
            except Exception as e:
                mod = type(e).__module__ or ""
                if not (mod.startswith("torch._dynamo") or mod.startswith("torch._inductor")):
                    raise
                raise RuntimeError(
                    "torch.compile failed while compiling BrawlVecEnv's inner tick "
                    "(cfg.compile=True). This is a known failure mode on some platforms "
                    "(BRAWL_SIM_DESIGN.md §12): the CPU inductor backend needs an MSVC `cl.exe` "
                    "on PATH, and the CUDA inductor backend needs a working Triton install -- "
                    "neither is guaranteed on Windows. `compile: false` (the default) is the "
                    "supported path and is unaffected by this. "
                    f"Original error -- {type(e).__name__}: {e}"
                ) from e

        return _guarded_tick

    # -- phase 1 --
    def _pop_action_buffer(self, action: torch.Tensor) -> torch.Tensor:
        """Writes `action` into act_buf[:, act_head], reads back act_buf[:, act_head -
        action_latency_ticks (mod L)], then advances act_head. At 0 ticks (L=1) this is a
        pass-through: write and read hit the same (only) slot.

        Runs once per SUB-TICK, so latency is in sim ticks whatever `cfg.action_repeat` is: each
        sub-tick's action comes back out `action_latency_ticks` later, so a decision's first
        `action_latency_ticks` sub-ticks still run the actions written before it."""
        state, cfg = self.state, self.cfg
        L = state.act_buf.shape[1]

        write_idx = state.act_head.view(-1, 1, 1).expand(-1, 1, 2)
        state.act_buf.scatter_(1, write_idx, action.unsqueeze(1))

        read_idx = ((state.act_head - cfg.action_latency_ticks) % L).view(-1, 1, 1).expand(-1, 1, 2)
        effective_action = state.act_buf.gather(1, read_idx).squeeze(1)

        state.act_head.copy_((state.act_head + 1) % L)
        return effective_action

    # -- phase 2: timers and regen --
    def _tick_timers(self) -> torch.Tensor:
        """Timers, then regen. Regen runs BEFORE this tick's combat, so it can never undo a
        lethal hit dealt later in the same tick. Returns the (N,E) HP regen actually restored,
        for `info["hp_healed_tick"]`."""
        hero.tick_timers(self.state, self.params, self.cfg)
        return combat.apply_regen(self.state, self.params, self.cfg)

    # -- phase 3 --
    def _decode(self, effective_action: torch.Tensor):
        return hero.decode_action(effective_action, self.state, self.params, self.cfg)

    # -- phase 4 --
    def _bot_phase(self, hero_move_dir: torch.Tensor, hero_fire: torch.Tensor,
                   hero_super: torch.Tensor, hero_gadget: torch.Tensor):
        """Returns `(move_dir, fire, super_fire, gadget_fire, aim_dir, aim_point, vis)`. `vis` is
        this tick's FAIR `(N,E,E)` visibility, handed on so `_attack_phase` can aim the gadget
        spinner at what each thrower can see without a second `perception.visibility` pass:
        nothing between here and phase 6 moves an entity or touches a reveal timer
        (`_override_phase` is pure, and `_stomp_phase` writes status timers only), so it is exactly
        what phase 6 would compute."""
        state, cfg = self.state, self.cfg
        vis = perception.visibility(state, self.bank, self.params, cfg)
        intent = policy.all_bot_intents(state, vis, self.bank, self.params, cfg, self.gen)

        move_dir = intent.move_dir.clone()
        move_dir[:, 0] = hero_move_dir
        fire = intent.fire.clone()
        fire[:, 0] = hero_fire
        # `BotIntent.super_fire` is all-False: no bot kind configures a super. Giving a bot one
        # takes its combat rule plus a brawlers.yaml block, with no plumbing change here.
        super_fire = intent.super_fire.clone()
        super_fire[:, 0] = hero_super
        # `BotIntent` has no gadget bit: every bot kind resolves `gadget_cooldown` to 0, so a bot
        # row could never pass `_attack_phase`'s `hero.gadget_ready` gate anyway. Only the hero's
        # slot can be True.
        gadget_fire = torch.zeros_like(fire)
        gadget_fire[:, 0] = hero_gadget
        # Placeholders for the hero's row. `_attack_phase` replaces aim_dir's with his volley aim
        # (Bull's shotgun; Mortis's dash follows move_dir and his proj_count is 0). aim_point is
        # read only by a lobbed shot, and no hero lobs.
        aim_dir = intent.aim_dir.clone()
        aim_dir[:, 0] = geo.from_angle(state.ent_facing[:, 0])
        aim_point = intent.aim_point.clone()
        aim_point[:, 0] = state.ent_pos[:, 0]
        return move_dir, fire, super_fire, gadget_fire, aim_dir, aim_point, vis

    # -- phase 5 --
    def _override_phase(self, move_dir: torch.Tensor, fire: torch.Tensor, override: torch.Tensor | None):
        """override[..., 0] == -1 means "no override for this entity slot" (valid move bins are
        0..n_move_bins); otherwise override[n,e] REPLACES that entity's move_dir/fire for this
        tick. aim_dir/aim_point are untouched: an overridden bot still aims by its own logic.

        **An override drives movement and the ORDINARY attack, nothing else.** Its attack column
        is read as a bool (`override[..., 1] != 0`), not decoded like the hero's: any nonzero
        value, 2, 3 and 4 included, is an ordinary attack, so an override can throw neither a
        super nor a gadget. `super_fire`/`gadget_fire` never pass through here, so an override on
        the hero's slot does not cancel a super or gadget his action asked for, and an override
        that FIRES under an action asking for the gadget yields a dash AND a gadget on one tick.
        tests/test_gadget.py pins all three cases. No caller overrides slot 0 outside tests; the
        narrow decode is deliberate."""
        if override is None:
            return move_dir, fire
        cfg = self.cfg
        has_override = override[..., 0] >= 0
        move_bin = torch.clamp(override[..., 0], min=0)
        is_idle = override[..., 0] == 0
        dirs = geo.dir_from_bin(torch.clamp(move_bin - 1, min=0), cfg.n_move_bins)
        override_dir = torch.where(is_idle.unsqueeze(-1), torch.zeros_like(dirs), dirs)
        override_fire = override[..., 1].to(torch.bool)

        move_dir = torch.where(has_override.unsqueeze(-1), override_dir, move_dir)
        fire = torch.where(has_override, override_fire, fire)
        return move_dir, fire

    # -- phase 5b --
    def _stomp_phase(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Lands every armed Stomper whose wind-up runs out this tick (`hero.advance_stomp`), on
        the positions this tick's attacks are decided on. Running before `_attack_phase`, where a
        stomp is armed, makes a k-tick wind-up land exactly k ticks after its tap, and lets a stun
        it applies stop this same tick's attack and walk. Returns two (N,) bools for the hero's
        stomp: "slowed or stunned at least one player", which `_run_tick` ORs into the reward's
        `gadget_hit`, and "landed on nobody", its `gadget_miss`. None in a spec with no STOMP
        kind (`params.stomp_radius_tiles` 0): every Mortis run skips the phase in Python."""
        if self.params.stomp_radius_tiles <= 0:
            return None
        caught, lands = hero.advance_stomp(self.state, self.params, self.cfg)
        hit = caught[:, 0].any(dim=-1)  # row 0: the hero
        return hit, lands[:, 0] & ~hit

    # -- phase 6 --
    def _attack_phase(self, move_dir: torch.Tensor, fire: torch.Tensor,
                      super_fire: torch.Tensor, aim_dir: torch.Tensor, aim_point: torch.Tensor,
                      gadget_fire: torch.Tensor, vis: torch.Tensor,
                      hero_auto: torch.Tensor | None = None):
        """Gates every attack source, starts dashes (clipping their paths now), spends ammo,
        fires supers, throws gadgets or arms stomps, spawns volleys and resolves melee.

        `hero_auto` is the (N,) bool `hero.decode_action` returns for attack value 4 (the
        auto-aimed attack, `cfg.auto_aim`); `None`, which tests pass by omission, means no row
        asked for it. It only changes the HERO's dash direction (see the swap before
        `hero.start_dash`); `fire` already carries the attack itself.

        `gadget_fire` is (N,E) bool and `vis` the fair (N,E,E) visibility `_bot_phase` computed
        this tick: the gadget spinner homes on the nearest enemy its thrower can SEE. A STOMP
        kind's gadget fire arms its stomp instead (`hero.start_stomp`), which `_stomp_phase` lands.

        Returns (dmg_by (N,E,E), healed (N,E), attack_in_reach (N,) bool). `healed` is melee
        lifesteal as ACTUALLY applied (`apply_heal` drops it for dead entities and clips it at max
        HP), the same contract as `_projectile_phase`'s, so it is what `info["hp_healed_tick"]` is
        paid on. `attack_in_reach` is the hero's flag for the reward (see `enemy_in_reach`).

        Ammo, cooldown and shots_fired for ranged and melee attacks are booked HERE: neither
        projectiles.spawn_volley nor combat.melee_hitscan writes ent_ammo/ent_attack_cd/
        ent_shots_fired (hero.start_dash does, for dashes)."""
        state, bank, params, cfg = self.state, self.bank, self.params, self.cfg

        # Safety net: bot-sourced and hero-sourced fire are already gated on this (fire_gate /
        # action_mask), but override-sourced fire has no such gate -- this makes it impossible
        # for ANY source to force an illegal attack through. A stunned or airborne entity
        # (core/status.py) is gated here and below like any other: a bot's intent does not know.
        stopped = status.disabled(state)
        can_attack = (
            state.ent_alive & (state.ent_ammo >= 1.0) & (state.ent_attack_cd <= 0) & (state.ent_dash_t <= 0)
            & ~stopped
        )
        fire = fire & can_attack

        # A SUPER costs charge, not ammo, so its gate omits the ammo term (an empty clip must not
        # block it) but keeps the cooldown, no-dashing and stun terms. Same safety-net role as
        # `can_attack`: `action_mask` already gates the hero's, this gates ANY source.
        can_super = (
            state.ent_alive & (state.ent_attack_cd <= 0) & (state.ent_dash_t <= 0)
            & hero.super_ready(state, params) & ~stopped
        )
        super_fire = super_fire & can_super

        # The GADGET's gate is its own timer and the stun, nothing else: no ammo, no `attack_cd`,
        # no `dash_t` (the game lets a gadget go mid-dash). `hero.gadget_ready` is the predicate
        # `action_mask` publishes, so a gadget requested while masked (cooldown running, dead, or
        # a kind with no gadget, i.e. every bot) is a silent no-op from ANY source.
        gadget_fire = gadget_fire & hero.gadget_ready(state, params) & ~stopped

        dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)
        is_dash_attack = fire & (dash_distance > 0)
        is_other_attack = fire & ~is_dash_attack

        # Attack in reach (user decision, 2026-09-21; BRAWL_SIM_DESIGN.md §10), the one input of
        # the reward's `attack_in_reach` term: is an enemy the hero can SEE (`vis` row 0) inside
        # his uncharged dash reach? The radius is scripts/audit_attack_cadence.py's (dash_distance
        # + dash_radius + unit_radius), so the term pays for exactly what the audit measures. A
        # hero with no dash (Bull) uses the distance where most of his volley can land instead
        # (`hero.in_reach_radius`; user decision 2026-10-23). Read before `hero.start_dash`, off
        # the positions the decision was made on, and ANDed with `attacked` below. An airborne
        # enemy (core/status.py) is not in reach: nothing the hero throws can land on it.
        hero_reach = hero.in_reach_radius(state, params)
        enemy_dist = geo.safe_norm(state.ent_pos[:, 1:] - state.ent_pos[:, :1], dim=-1)  # (N,E-1)
        enemy_in_reach = (
            state.ent_alive[:, 1:] & ~status.airborne(state)[:, 1:] & vis[:, 0, 1:]
            & (enemy_dist <= hero_reach.unsqueeze(-1))
        ).any(dim=-1)

        # The auto-aimed attack (attack value 4, `cfg.auto_aim`; user decision, 2026-09-26): on the
        # rows that asked for it AND have a target in reach, the hero's dash goes straight at
        # `hero.auto_aim_target`'s nearest enemy-or-crate instead of along the move bin; with
        # nothing in reach the row keeps the ordinary rule below (move bin, or facing when
        # idle), like a bare tap in the game with nothing near. A SEPARATE tensor for the dash:
        # `move_dir` is read again by the super's aim below and by `_movement_phase`, and the
        # hero's walk must not bend toward the target.
        dash_dir = move_dir
        auto_swap = None
        if hero_auto is not None and cfg.auto_aim:
            auto_dir, has_target = hero.auto_aim_target(state, params, cfg)
            auto_swap = (hero_auto & has_target).unsqueeze(-1)                  # (N,1)
            hero_dir = torch.where(auto_swap, auto_dir, move_dir[:, 0])
            dash_dir = torch.cat([hero_dir.unsqueeze(1), move_dir[:, 1:]], dim=1)

        # The SUPER's aim, its own tensor for the same reason. It follows the move bin, but an idle
        # bin's `move_dir` is (0, 0), which `spawn_supers` would launch as a bolt that never moves,
        # and the mask cannot tie the super to the move column. So an idle row falls back to
        # facing, as `start_dash` does for the dash, and the HERO's idle super aims like the game's
        # tap-to-fire (user decision, 2026-09-30): at `hero.super_aim_target`'s nearest enemy in
        # the bolt's reach, along facing with none. Read here, before `start_dash` turns a dasher's
        # facing, off the state the decision was made on.
        idle = (move_dir[..., 0] == 0) & (move_dir[..., 1] == 0)                   # (N,E)
        super_dir = torch.where(idle.unsqueeze(-1), geo.from_angle(state.ent_facing), move_dir)

        # The hero's VOLLEY aim, for a hero whose attack is a volley rather than a dash (Bull;
        # BULL_HERO_PLAN.md D5): the dash's own rule, applied to the shot. Value 1 fires along the
        # move bin, or along facing when the bin is idle (`super_dir`'s row before the tap swap
        # below), and value 4 at `auto_aim_target`'s nearest enemy or crate in his volley's reach.
        # Live, value 1 is already a drag along the bin and value 4 a bare tap. Mortis's row is
        # never read: his proj_count is 0, so spawn_volley spawns nothing for him.
        hero_aim = super_dir[:, 0]
        if auto_swap is not None:
            hero_aim = torch.where(auto_swap, auto_dir, hero_aim)
        aim_dir = torch.cat([hero_aim.unsqueeze(1), aim_dir[:, 1:]], dim=1)

        tap_dir, _has_target = hero.super_aim_target(state, params, cfg)          # (N,2)
        hero_super_dir = torch.where(idle[:, :1], tap_dir, super_dir[:, 0])
        super_dir = torch.cat([hero_super_dir.unsqueeze(1), super_dir[:, 1:]], dim=1)

        hero.start_dash(state, fire, dash_dir, bank, params, cfg)

        # Attacking breaks concealment (`perception.reveal_after_attack`), the hero's included, so
        # nobody attacks out of a bush unseen. torch.maximum, not assignment: a fresh shot must
        # never SHORTEN a longer reveal already running. tick_timers decrements at phase 2 and
        # this is phase 6, so a reveal set here lasts its full duration.
        # `attacked`, not `fire`, for everything that means "I just took an offensive action": a
        # super breaks concealment, breaks out-of-combat for regen and spends the long-dash charge
        # like an ordinary attack, or it would be a free way to shoot from a bush, out-heal a
        # fight or keep a charged long dash banked.
        attacked = fire | super_fire
        # The gadget is offensive for concealment and regen but NOT for the long dash:
        # `ent_attack_idle_t` below reads `attacked`, so a gadget thrown while the long dash
        # charges does not spend it.
        offensive = attacked | gadget_fire
        # The reward's flag reads `attacked` too: the gadget is no attack for it, as in
        # scripts/audit_attack_cadence.py, whose "attacked" is attack column 1, 2 or 4.
        attack_in_reach = attacked[:, 0] & enemy_in_reach
        reveal = params.reveal_after_attack.unsqueeze(-1).expand_as(state.ent_reveal_t)
        state.ent_reveal_t.copy_(
            torch.where(offensive, torch.maximum(state.ent_reveal_t, reveal), state.ent_reveal_t)
        )

        # Attacking also breaks you OUT OF COMBAT for regen, as being hit does (combat.apply_damage
        # owns that side): regen waits for a spell of neither taking damage nor attacking, and
        # this is the only phase that knows an attack happened.
        state.ent_out_of_combat_t.copy_(
            torch.where(offensive, torch.zeros_like(state.ent_out_of_combat_t), state.ent_out_of_combat_t)
        )

        # Long-dash charge: reset by attacking and by NOTHING ELSE -- not by taking damage, unlike
        # ent_out_of_combat_t above, and not by the gadget (`attacked`, not `offensive`). Must stay
        # after `hero.start_dash`, which reads the charge to decide whether THIS dash is a long
        # one; resetting before it would mean the long dash never fires.
        state.ent_attack_idle_t.copy_(
            torch.where(attacked, torch.zeros_like(state.ent_attack_idle_t), state.ent_attack_idle_t)
        )

        attack_cooldown = stats.gather_kind(params.attack_cooldown, state.ent_kind)
        state.ent_ammo.copy_(torch.where(is_other_attack, state.ent_ammo - 1.0, state.ent_ammo))
        state.ent_attack_cd.copy_(torch.where(is_other_attack, attack_cooldown, state.ent_attack_cd))
        state.ent_shots_fired.copy_(torch.where(is_other_attack, state.ent_shots_fired + 1, state.ent_shots_fired))

        # --- super: spend the whole charge, take the cooldown, launch the bolt or start the charge ---
        # Charge is zeroed rather than decremented by `super_charge_hits`: `add_super_charge` caps
        # it there, so the two agree, but "firing spends the meter" stays true if a brawler ever
        # banks more than one.
        state.ent_super_charge.copy_(
            torch.where(super_fire, torch.zeros_like(state.ent_super_charge), state.ent_super_charge)
        )
        state.ent_attack_cd.copy_(torch.where(super_fire, attack_cooldown, state.ent_attack_cd))
        state.ent_shots_fired.copy_(torch.where(super_fire, state.ent_shots_fired + 1, state.ent_shots_fired))
        # A CHARGE kind's super (Bull) is his body's rush, on the dash's state, not a bolt; both
        # follow `super_dir`. `params.charge_ray_tiles` is 0 in a spec with no CHARGE kind, so a
        # Mortis run skips the split and launches every super as a bolt, as it always did.
        bolt_fire = super_fire
        if params.charge_ray_tiles > 0:
            charge_kind = stats.gather_kind(params.super_type, state.ent_kind) == int(SuperType.CHARGE)
            bolt_fire = super_fire & ~charge_kind
            hero.start_charge(state, super_fire & charge_kind, super_dir, bank, params, cfg)
        projectiles.spawn_supers(state, bolt_fire, state.ent_pos, super_dir, params, cfg)

        # --- gadget: start the cooldown, throw the spinner or arm the stomp ---
        # The ONLY writer of `ent_gadget_cd` besides `tick_timers`' countdown. It touches neither
        # ammo, `attack_cd` nor `ent_shots_fired`: the gadget is a separate button, not an attack,
        # so it neither pauses the reload nor counts as a shot. `gadget_target` runs for every
        # entity every tick, since masking it to the firing rows would need a data-dependent shape.
        # Overflow: the cooldown is written from `gadget_fire`, not from `alloc_slots`' `ok`, so a
        # FULL projectile buffer spends the cooldown and throws nothing -- the same silent drop as
        # the super's charge above and every thinned volley. `config.peak_projectile_demand`
        # counts neither the spinner nor the hero's super bolt (his `proj_count` 0 zeroes his
        # term), so `limits.max_projectiles` must keep headroom above that bound for both.
        gadget_cooldown = stats.gather_kind(params.gadget_cooldown, state.ent_kind)
        state.ent_gadget_cd.copy_(torch.where(gadget_fire, gadget_cooldown, state.ent_gadget_cd))
        # A STOMP kind's gadget (Bull's Stomper) is a pulse around his own body once a wind-up
        # runs out, armed here (hero.start_stomp) and landed by `_stomp_phase`; it takes no
        # projectile slot, so no overflow drops it. `params.stomp_radius_tiles` is 0 in a spec with
        # no STOMP kind, so a Mortis run skips the split and throws every gadget as a spinner, as
        # it always did.
        spinner_fire = gadget_fire
        if params.stomp_radius_tiles > 0:
            stomp_kind = stats.gather_kind(params.gadget_type, state.ent_kind) == int(GadgetType.STOMP)
            spinner_fire = gadget_fire & ~stomp_kind
            hero.start_stomp(state, gadget_fire & stomp_kind, params)
        gadget_dir, gadget_travel = hero.gadget_target(state, vis, params, bank, cfg)
        gadget_damage = stats.effective_gadget_damage(state.ent_kind, state.ent_cubes, params)
        projectiles.spawn_gadget(state, spinner_fire, state.ent_pos, gadget_dir, gadget_travel,
                                 gadget_damage, params, cfg)

        damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
        projectiles.spawn_volley(state, is_other_attack, state.ent_pos, aim_dir, aim_point, state.ent_kind, damage, params, cfg)

        # --- melee, single-cone and SWEPT, resolved in ONE hitscan call ---
        # Buzz's attack is five cones fanned across his attack_cooldown rather than one at trigger
        # time. `sweep_cone_dir` reads the schedule off `ent_attack_cd`, which the block above just
        # set for anyone firing this tick -- so sub-swing 0 lands on the trigger tick and the other
        # four on later ticks of the same cooldown.
        #
        # A swept kind is masked OUT of the ordinary trigger-time cone (`& ~swept`), because
        # sub-swing 0 already covers that tick; letting both through would double the first swing.
        #
        # ONE melee_hitscan call with a per-entity cone direction, not two: it is a dense (N,E,E)
        # cone + LOS march, the most expensive thing in this phase, and the two populations are
        # disjoint per entity.
        #
        # A stun drops the sub-swings that fall inside it (`is_other_attack` is already gated, and
        # `sweep_fire` is ANDed here). Any left once it ends still land on schedule: the clock is
        # `ent_attack_cd`, which a stun does not pause.
        #
        # Only a kind that swings a cone (`attack_arc_rad > 0`, the rule config.cone_ray_tiles and
        # bots/combat_rules read) runs the hitscan. A ranged kind's arc is 0, but
        # `geo.point_in_cone` passes `diff <= 0`, so until 2026-10-23 every ranged attack also
        # landed its full damage on a unit or crate EXACTLY on its facing ray, in practice a crate
        # (measured in BULL_HERO_PLAN.md §5.3). The gate is also what keeps melee_hitscan's
        # shortened LOS march sound: its budget covers cone reach only.
        sweep_fire, sweep_dir = melee_sweep.sweep_cone_dir(state, params, cfg)
        sweep_fire = sweep_fire & ~stopped
        swept = melee_sweep.is_swept(state, params)
        swings_cone = stats.gather_kind(params.attack_arc_rad, state.ent_kind) > 0
        melee_fire = (is_other_attack & swings_cone & ~swept) | sweep_fire
        cone_dir = torch.where(sweep_fire, sweep_dir, state.ent_facing)

        dmg_ent, dmg_by, dmg_box = combat.melee_hitscan(state, melee_fire, bank, params, cfg, cone_dir=cone_dir)
        combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), params, cfg)
        # MELEE LIFESTEAL (Edgar), after apply_damage like _projectile_phase's heal: a trade that
        # kills the victim resolves the kill first. Every other kind has
        # `melee_lifesteal_fraction: 0`, so this is a masked multiply, not a branch.
        healed = combat.apply_heal(state, combat.melee_lifesteal(state, dmg_by, params))
        boxes.damage_boxes(state, dmg_box)
        return dmg_by, healed, attack_in_reach

    # -- phase 7 --
    def _movement_phase(self, move_dir: torch.Tensor) -> None:
        """ONLY non-dashing entities: apply_movement's `active = alive & dash_t<=0` gate excludes
        anyone who just started a dash in _attack_phase, which is what makes the dash replace
        the walk this same tick. The same gate holds a stunned or airborne entity still."""
        movement.apply_movement(self.state, move_dir, self.bank, self.params, self.cfg)

    # -- phase 8 --
    def _dash_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(dmg_by (N,E,E), charging (N,E) bool). The second marks the dashers whose dash is a
        CHARGE super, read BEFORE advance_dash clears the flag on a charge's last tick: their hits
        charge no super (`_run_tick`)."""
        charging = self.state.ent_dash_super & (self.state.ent_dash_t > 0)
        dmg_ent, dmg_by, dmg_box = hero.advance_dash(self.state, self.params, self.cfg, self.bank)
        combat.apply_damage(self.state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), self.params, self.cfg)
        boxes.damage_boxes(self.state, dmg_box)
        return dmg_by, charging

    # -- phase 9 --
    def _projectile_phase(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                         torch.Tensor]:
        """(dmg_by (N,E,E), healed (N,E), charge_hits (N,E,E) int32, gadget_hit (N,), attack_miss
        (N,) int32) -- the second is the lifesteal ACTUALLY applied, which is not `heal_ent`:
        apply_heal drops it for dead owners and clips it at max HP. The third is step_projectiles'
        super-charging hit count (the projectiles behind `dmg_by`, the gadget spinner's left out),
        passed up untouched for `_run_tick`. The fourth is "the hero's gadget spinner hurt at
        least one player this tick", which the reward's `gadget_hit` term pays. The fifth counts
        the hero's attack volleys that ended this tick with no pellet on a player or a crate
        (`projectiles._settle_hero_volleys`), which its `attack_miss` term charges."""
        dmg_ent, dmg_by, dmg_box, heal_ent, charge_hits, attack_miss = projectiles.step_projectiles(
            self.state, self.bank, self.params, self.cfg, volleys=True)
        combat.apply_damage(self.state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), self.params, self.cfg)
        # Lifesteal AFTER damage, so a super that trades with something lethal still resolves
        # the death first -- healing a corpse is rejected by apply_heal's own alive gate.
        healed = combat.apply_heal(self.state, heal_ent)
        boxes.damage_boxes(self.state, dmg_box)
        # The spinner is the one projectile whose hits charge no super, so the hero's hits in
        # `dmg_by` that `charge_hits` leaves out are its hits. Box damage never enters `dmg_by`,
        # so a spinner that lands only on crates is no hit. Blind spot: a super bolt that hits the
        # same player on the landing tick hides the spinner's hit on that player.
        gadget_hit = ((dmg_by[:, 0] > 0) & (charge_hits[:, 0] == 0)).any(dim=-1)  # row 0: the hero
        return dmg_by, healed, charge_hits, gadget_hit, attack_miss

    # -- phase 10 --
    def _zone_phase(self) -> None:
        state, params, cfg = self.state, self.params, self.cfg
        dmg = zone.zone_damage(state, params, cfg)
        no_attacker = torch.full_like(state.ent_last_hit_by, -1)
        combat.apply_damage(state, dmg, int(DeathCause.ZONE), no_attacker, params, cfg)

    # -- phase 11 (box HP already decremented per-source in phases 6/8/9) --
    def _box_phase(self) -> torch.Tensor:
        return boxes.resolve_broken_boxes(self.state, self.bank, self.params, self.cfg, self.gen)

    # -- phase 12 --
    def _pickup_phase(self) -> torch.Tensor:
        return combat.collect_pickups(self.state, self.params, self.cfg)

    # -- phase 13 --
    def _death_phase(self) -> torch.Tensor:
        newly_dead = combat.resolve_deaths(self.state, self.cfg)
        combat.drop_cubes_on_death(self.state, newly_dead, self.params, self.cfg)
        return newly_dead

    # -- phase 14 --
    def _zone_schedule(self) -> None:
        zone.step_zone(self.state, self.params, self.cfg)

    # -- phase 15 --
    def _bookkeeping(self, dmg_by_total: torch.Tensor, charge_hits: torch.Tensor) -> None:
        """Credits damage dealt and super charge, then advances time and step_count. The other
        cumulative counters are already current, each written as its event happened
        (resolve_deaths: n_alive/kills; resolve_broken_boxes: boxes_broken; _attack_phase:
        shots_fired; apply_damage: damage_taken). ent_damage_dealt is the exception: apply_damage
        writes only the victim's side, so the attacker's is summed here from this tick's
        attacker x victim matrix -- combat damage only, matching core/events.py's
        damage_dealt_tick (zone damage has no attacker to credit).

        `charge_hits` (N,E,E) int32 counts the super-charging hits behind `dmg_by_total > 0`
        (`_run_tick` builds it): 1 for a melee or dash hit, one per projectile that hit, and
        nothing for the gadget spinner or a CHARGE super's own rush. Damage dealt still counts
        both; only the charge does not."""
        state, cfg = self.state, self.cfg
        state.ent_damage_dealt.copy_(state.ent_damage_dealt + dmg_by_total.sum(dim=2))

        # SUPER CHARGE, from the attacker x victim matrix because it is the only place that
        # separates hits on PLAYERS from hits on boxes (box damage never enters `dmg_by`), so a
        # hero cannot farm his super off crates. A swing or a blast that catches several players
        # charges once per player, and a volley once per pellet that lands (Bull's 5 pellets
        # into one player charge 5 of his 9).
        hits = charge_hits.sum(dim=2).to(torch.int32)
        hero.add_super_charge(state, hits, self.params)

        state.time += cfg.dt
        state.step_count += 1

    # -- phase 16 --
    def _observe(self, dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed,
                 attacks_in_reach, gadget_hits, attack_misses, gadget_misses, decision, move_reversals):
        """Runs ONCE per decision: `reward_fn` sees the summed deltas and latched outcomes of the
        whole decision, once per `step()` whatever `action_repeat` is. Per-tick reward terms
        integrate over `info["alive_ticks"]`/`info["in_zone_ticks"]` rather than firing once
        per call, so the decision rate can change without re-tuning every reward weight."""
        obs = self._build_observation()
        info = events.compute_info(
            self.state, dmg_by_total, newly_dead, newly_broken, cubes_gained, self.cfg,
            decision=decision, hp_healed=hp_healed, attacks_in_reach=attacks_in_reach,
            gadget_hits=gadget_hits, attack_misses=attack_misses, gadget_misses=gadget_misses,
            move_reversals=move_reversals,
        )
        reward = self.reward_fn(obs, info, self.cfg)
        return obs, info, reward

    # -- phase 17 --
    def _autoreset(self, info: dict):
        """Resets done envs in place via a boolean mask. reset_envs is torch.where-masked
        internally, so calling it every decision, even when nobody is done, is sync-free and
        needs no `.any()` host check first.

        Does NOT take obs_before_reset: `step()` already cloned it and dropped its own reference,
        so those large buffers can be freed before the `_build_observation()` below allocates
        the post-reset copy (see step()'s comment)."""
        terminated, truncated = info["terminated"], info["truncated"]
        done = terminated | truncated
        spawn.reset_envs(self.state, done, self.bank, self.params, self.cfg, self.gen,
                         self.spec, self.params_hook)
        # Only the reset envs start a decision in this build: `_observe` already moved the rest
        # on, and `slots.update` counts a sighting or a miss every time it runs.
        obs = self._build_observation(slot_envs=done)
        return obs, terminated, truncated
