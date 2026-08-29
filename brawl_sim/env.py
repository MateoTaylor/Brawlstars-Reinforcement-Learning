"""BrawlVecEnv: the native, torch-native vectorized environment tying every core/ and bots/
module built in Steps 1-28 into the Section 4 tick order. See BRAWL_SIM_BUILD_PLAN.md Step 29.

**This is the first module allowed to import both `core/` and `bots/`.** Every `core/` module
through Step 27 deliberately avoided importing `bots/` (`core/zone.py`, `core/spawn.py`,
`core/observation.py` each duplicate a tiny predicate rather than cross that boundary) and
said so explicitly, promising "the future env.py step loop" as the place the two finally meet.
This is that place: `_bot_phase`/`_observe`/`reset`/`_autoreset` call `bots/perception.py` and
`bots/policy.py` directly.

**Autoreset happens inside `step()`.** A finished env is reset in place, within the SAME call,
so the very next `step()` call can act on it immediately -- no separate "reset this env" round
trip. This means the returned `obs` is the FIRST OBSERVATION OF THE NEW EPISODE for any env that
finished this tick, NOT a view of the tick that just ended. **The finished episode's own final
obs/info live in `info["final_observation"]` / `info["final_info"]`** (dense over all N envs,
like everything else in this sim -- only meaningful where `terminated | truncated` was True this
tick). `reward`/`terminated`/`truncated`/`info` (everything except `obs` itself) describe what
just happened; `obs` describes what comes next. Get this backwards and reward/observation pairs
silently misalign for every finished episode -- read this twice.

**Every `obs`/`info` returned by `step()`/`reset()` is a zero-copy view into `SimState`'s own
tensors** (Step 25/27's own established style -- nothing here allocates a fresh copy per field),
which means **it is only valid until the NEXT `step()`/`reset()` call**: that next call mutates
the exact same underlying storage in place, and anything still holding the old dict will
silently see the NEW values, not an error. Callers that need to retain an observation across
multiple steps (logging, replay buffers, ...) must `core.observation.clone_obs(...)` it
themselves. The one place this module can't get away with that convention is
`info["final_observation"]`/`info["final_info"]`: those specifically must survive the
autoreset that happens later in the SAME `step()` call, so `step()` clones them explicitly
(and drops its own reference to the pre-clone copy, to keep peak memory down -- see its
own comment) before `_autoreset` ever calls `reset_envs`.

**One `step()` is one AGENT DECISION, which is `cfg.action_repeat` SIM TICKS** (Section 4 phases
1-15, run back to back with the same held action -- `_run_decision`). At `action_repeat=1` this
is exactly the pre-existing one-tick-per-step behavior, down to the tensors allocated. Above 1
it decouples the agent's decision rate from the simulation rate: `dt=0.05` with
`action_repeat=5` simulates at 20 Hz and decides at 4 Hz, the way a human holds a direction for
a beat rather than re-aiming every 50 ms.

Three consequences worth reading before touching this:

  *Units.* `cfg.max_episode_steps`, `state.step_count`, `state.time`, `zone.step_seconds`, and
  `decision_period_ticks` all stay in SIM TICKS -- the world's clock is unchanged. Everything
  measured in `step()` CALLS (SB3 timesteps, `n_steps`, `info["episode"]["l"]`, any loop bound
  over an episode) is now in DECISIONS, which is `cfg.max_agent_steps` per episode, not
  `cfg.max_episode_steps`. `gamma` is per decision too, so a discount tuned at 20 Hz must be
  raised to `gamma ** action_repeat` to keep the same real-time horizon.

  *Movement is held; fire is not.* The move bin is re-applied on every sub-tick, so the hero
  keeps walking (or keeps dashing) for the whole window. The FIRE BIT is applied only on the
  FIRST sub-tick and forced to 0 for the rest -- **one decision means at most one attack
  attempt.** Two reasons, one of which survives any rebalance. (1) It lines the sim up exactly
  with `hero.action_mask`, which MaskablePPO evaluates once per decision: the mask promises "you
  may fire NOW", and the fire happens now, once -- "fire" therefore means the same thing at
  `action_repeat=5` as at 1. (2) It bounds what a held bit can do if a weapon ever fires faster
  than the decision rate. Every `attack_cooldown` in configs/brawlers.yaml is currently 0.30-0.50s
  and so already exceeds one 0.25s window, which makes this guard unreachable with the shipped
  numbers -- but those cooldowns were 0.05-0.20s until they were deliberately widened (see that
  file's own DELIBERATE DIVERGENCE note), and at those values a held bit emptied Mortis's entire
  3-shot clip inside a single decision. The guard is structural precisely so balance changes
  cannot silently reintroduce that.

  *Events are aggregated, outcomes are latched.* Per-tick deltas (damage, kills, cubes, boxes)
  are SUMMED across the sub-ticks; the hero's outcome fields (`terminated`, `truncated`,
  `hero_rank`, `hero_alive`) are LATCHED at the sub-tick that env's episode actually ended, and
  its per-tick reward counters stop there too. See `core/events.advance_decision_tally` for why
  latching is load-bearing and for the one thing it deliberately does not fix (a finished env's
  `final_observation` can be up to `action_repeat - 1` ticks stale).

**Visibility/observation are (re)computed THREE separate times per DECISION**, not once, and the
split is the main reason action repeat is faster than its tick count suggests:
`bots/perception.visibility` for bot targeting (phase 4, pre-movement) runs once per SUB-TICK
since bots re-target on the world's clock, but both `build_obs` calls are per DECISION --
for the finished decision's own state (phase 16, post-movement/combat -- this becomes
`info["final_observation"]` for any env that's done), and again after autoreset (phase 17) for
the actually-returned `obs`. Each reflects genuinely different, non-interchangeable state
(pre- vs post-movement vs post-reset), so none of the three can be skipped or cached across the
others -- documented here since it's the least obvious performance/design characteristic of this
whole module.

**`torch.compile` support (Step 31), off by default (`cfg.compile=False`).** When enabled,
`_tick_fn` (built once in `__init__`, see `_make_tick_fn`) wraps ONLY `_run_tick` -- Section 4
phases 1-15, the pure-tensor simulation update -- with `torch.compile(dynamic=False,
fullgraph=False)`. Phases 16-17 (`_observe`/`_autoreset`) are deliberately excluded: they build
nested python `dict`s and phase 16 calls `self.reward_fn`, an arbitrary user-supplied callable
`torch.compile` has no business tracing through. **On this machine (Windows, RTX 5070 Ti /
Blackwell / sm_120), `torch.compile` cannot actually reach a running state**: the CPU inductor
backend requires an MSVC `cl.exe` on PATH (not installed), and the CUDA inductor backend
requires a working Triton install (not installable here as of this writing, confirming the
plan's own upfront warning). This was root-caused by hand, not just observed to fail: one real
graph break was found and documented (`torch.randn(..., generator=gen, ...)` in the bot layer's
aim noise, now `bots/combat_rules.py` -- dynamo has no `as_proxy()` for a bare `torch.Generator`
argument), and
both backends fail during actual codegen with clear, typed exceptions (`InductorError`,
`TritonMissing`), not silent corruption or a hang. `_make_tick_fn`'s `_guarded_tick` catches
exactly those two failure modes (by checking the raised exception's `__module__`, not by
message-sniffing) and re-raises a `RuntimeError` naming the cause and pointing back at
`compile: false` as the fully-supported default -- anything else (a real bug in `_run_tick`
itself) propagates with its own original type untouched. See BRAWL_SIM_BUILD_PLAN.md Step 31
for the full investigation and confirmed CUDA-scale (`n_envs=4096`) behavior.
"""
from pathlib import Path

import torch
import yaml

from .bots import perception, policy
from .config import EnvConfig, apply_randomization, build_params, load_randomization, validate
from .constants import DeathCause
from .core import boxes, combat, events, geometry as geo, hero, melee_sweep, movement, observation
from .core import obs_schema, projectiles, spawn, stats, zone
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
    """See BRAWL_SIM_BUILD_PLAN.md Step 29 for the full design. `cfg` is a resolved
    `EnvConfig` (from `config.load_config`); it has no brawler stats on it at all (Step 3:
    "EnvConfig holds static Python values ... SimParams holds the numeric stats"), so this
    class separately loads `configs/default.yaml` + `configs/brawlers.yaml` (fixed,
    package-relative paths, same pattern as `maps/loader.py`'s `CSV_DIR`) to build the `spec`
    dict `build_params`/`resample_params` need. **If your `cfg` came from `load_config(...,
    overrides=...)` and any override touched a SimParams-only field** (e.g. `base_hp`,
    `zone.dps`, `entities.enemy_hp_mult` -- anything not in `_ENV_CONFIG_FIELDS`), that
    override is invisible to `EnvConfig` and would silently be lost here: pass the matching
    merged dict explicitly via the `spec` argument. This is a real, documented sharp edge, not
    an oversight -- there is no way to recover SimParams-only values from an `EnvConfig` alone.

    `reward_fn` defaults to `None` (constructing a fresh `ZeroReward()` inside `__init__`), NOT
    the plan text's literal `reward_fn=ZeroReward()` -- a mutable default argument would be
    shared across every `BrawlVecEnv` built without an explicit `reward_fn`, and `ZeroReward`
    is stateful (it caches a tensor keyed by `(n_envs, device)`). Standard Python correctness
    fix for a well-known footgun, not a design change.

    `autoreset` (Step 34) defaults to `True`, preserving every behavior described above and
    tested since Step 29 exactly as-is. Set `False` only for `wrappers/gym_single.py`'s use
    case -- standard single-env `gymnasium.Env`s are contractually forbidden from silently
    resetting inside `step()`; the caller must see `terminated`/`truncated` and call `reset()`
    itself. With `autoreset=False`, `step()`'s phase 17 (`_autoreset`) becomes a no-op: `state`
    is left exactly as phase 16 finished it (a "frozen", possibly-terminal state) and the
    returned `obs` IS that finished tick's own observation -- `info["final_observation"]`/
    `info["final_info"]` still get set (for API-shape consistency with the `autoreset=True`
    case) but are then simply equal to `obs`/`info`, not a separate pre-reset snapshot.

    `verbose` (Step 40) defaults to `True`, preserving `core.state.allocate`'s own long-standing
    default and every existing caller's behavior exactly. Threaded through as a passthrough
    kwarg (not a new default) because `scripts/benchmark.py` constructs a fresh `BrawlVecEnv`
    per `(path, n_envs)` sweep point -- a dozen-plus constructions per run -- and had no way to
    suppress `allocate`'s per-construction memory report before this, burying the actual
    benchmark results in the printed report. `verbose=False` there; everywhere else is unaffected.
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
        # Plain attribute, not just a constructor argument: `training/curriculum.py` swaps it in
        # AFTER construction (the manager needs `self.params`/`self.gen`, which don't exist until
        # __init__ finishes) and re-reads it on every reset. See spawn.reset_envs for the contract.
        self.params_hook = params_hook
        # `tick_hook(env)` runs after EVERY SUB-TICK's phases 1-15, i.e. `action_repeat` times per
        # `step()`. It exists so replay/recording tools can sample the world at the SIM rate
        # (20 Hz) rather than the decision rate (4 Hz) -- without it, `scripts/watch.py` and
        # `scripts/record_rollout.py` would only ever see every action_repeat-th tick and replays
        # would visibly stutter, with projectiles teleporting between frames.
        #
        # **Debug tooling only.** It is a python callback in the middle of the hot loop and the
        # things worth doing in it (`state.snapshot`, `.cpu()`) are host syncs, which is exactly
        # what the rest of this file exists to avoid. Nothing in the training path installs one,
        # and `None` (the default) costs a single `is not None` test per sub-tick. Also note that
        # phases 16/17 have NOT run when it fires: `state` is mid-decision, pre-observation and
        # pre-autoreset, which is precisely what makes it useful here.
        self.tick_hook = tick_hook

        base_spec = spec if spec is not None else _load_base_spec()
        if randomization is None:
            randomization_spec = {}
        elif isinstance(randomization, dict):
            randomization_spec = randomization
        else:
            randomization_spec = load_randomization(randomization)
        # Applied once here, not per-reset: the resulting spec still carries {low, high} RANGES
        # for any randomized field, and build_params/resample_params re-sample fresh values
        # from those ranges every reset (N08) -- baking randomization in once is what makes
        # that resampling behavior correct, not a shortcut that skips it.
        self.spec = apply_randomization(base_spec, randomization_spec) if randomization_spec else base_spec

        self.bank = build_map_bank(cfg, device=self.device)
        self.params = build_params(cfg, n_envs=n_envs, device=self.device, gen=self.gen, spec=self.spec)
        validate(cfg, self.params)
        self.state = allocate(cfg, n_envs=n_envs, device=self.device, verbose=verbose)

        # `_tick_fn` is built once here, not per-step -- torch.compile(fn) itself is cheap (it
        # doesn't compile anything yet, just wraps `fn` in a guard/dispatch shim); the actual
        # compilation happens lazily on the FIRST call, specialized to that call's actual
        # argument shapes/types (dynamic=False bakes those in as guards). See Step 31 for why
        # this can legitimately fail on this machine (Windows CPU inductor needs MSVC `cl.exe`,
        # not installed here; Windows+Blackwell CUDA inductor needs Triton, also not installed
        # here) and how that failure is surfaced via `_make_tick_fn`.
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
        """`action`/`override` are moved onto `self.device` here (in addition to the dtype
        cast already done before Step 31) if they aren't already -- callers are NOT required
        to pre-place them. Found by Step 31's own torch.compile diagnosis: dynamo's fake-tensor
        device propagation turned what used to be a same-machine-happens-to-match-device
        assumption into a hard, correctly-worded error the moment a CPU-authored action tensor
        met a CUDA env; eager mode has the exact same bug (`scatter_` in `_pop_action_buffer`
        requires matching devices), just with a less informative RuntimeError -- confirmed by
        reproducing the failure with `cfg.compile=False` before fixing it here."""
        assert action.shape == (self.n_envs, 2), f"action shape {tuple(action.shape)} != {(self.n_envs, 2)}"
        action = action.to(device=self.device, dtype=torch.int64)
        if override is not None:
            assert override.shape == (self.n_envs, self.cfg.n_entities, 2)
            override = override.to(device=self.device, dtype=torch.int64)

        dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, decision = self._run_decision(action, override)

        obs_before_reset, info, reward = self._observe(
            dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, decision
        )
        # Cloned and dropped HERE, before autoreset's own post-reset build_obs call, not inside
        # _autoreset itself -- obs_before_reset's world/view grids are large (obs["world"] alone
        # is ~177 MB at n_envs=4096), and a del() inside _autoreset wouldn't help: step()'s own
        # local variable is a second, independent reference to the same tensors that keeps them
        # alive for the rest of this call regardless of what _autoreset does internally. Freeing
        # it here, before the post-reset rebuild allocates a same-sized second copy, measurably
        # lowers peak VRAM (this was the difference between passing and failing Step 29's
        # "peak VRAM under ~2.5 GB at 4096 envs" acceptance check).
        final_observation = observation.clone_obs(obs_before_reset)
        final_info = observation.clone_obs(dict(info))

        if self.autoreset:
            del obs_before_reset
            obs, terminated, truncated = self._autoreset(info)
        else:
            # Step 34: gymnasium single envs must not silently reset inside step() -- state is
            # left exactly as this tick finished it (see class docstring), and `obs` IS this
            # tick's own (possibly terminal) observation, not a fresh episode's first one.
            obs = obs_before_reset
            terminated, truncated = info["terminated"], info["truncated"]
        info["final_observation"] = final_observation
        info["final_info"] = final_info

        if self.cfg.debug_checks:
            check_invariants(self.state, self.cfg, self.params)

        return obs, reward, terminated, truncated, info

    @property
    def observation_spec(self) -> dict:
        return obs_schema.obs_spec(self.cfg)

    @property
    def action_spec(self) -> dict:
        return {"nvec": self.cfg.action_nvec}

    def snapshot(self, env_index: int) -> dict:
        """CPU, rendering only -- see core.state.snapshot."""
        return snapshot(self.state, env_index)

    # ---- internals --------------------------------------------------------------------

    def _build_observation(self) -> dict:
        vis = perception.visibility(self.state, self.bank, self.params, self.cfg)
        los = perception.raw_los(self.state, self.bank, self.cfg)
        return observation.build_obs(self.state, self.bank, vis, los, self.params, self.cfg)

    def _run_tick(self, action: torch.Tensor, override: torch.Tensor | None):
        """Section 4 phases 1-15, exactly the sequence `step()` ran inline before Step 31 --
        pulled out into its own method ONLY so it has a stable, single callable to hand to
        `torch.compile` (`self._tick_fn`, built once in `__init__`). Deliberately excludes
        phase 16 (`_observe`, which builds nested python dicts and calls `self.reward_fn` --
        an arbitrary user callable `torch.compile` has no business tracing through) and phase
        17 (`_autoreset`, which reads the python `dict` `info` phase 16 produced). Every other
        SimState mutation for this tick happens here, in place, exactly as before -- `step()`
        just no longer has these 15 calls written out inline."""
        effective_action = self._pop_action_buffer(action)
        regen_healed = self._tick_timers()
        hero_move_dir, hero_fire, hero_super = self._decode(effective_action)
        move_dir, fire, super_fire, aim_dir, aim_point = self._bot_phase(
            hero_move_dir, hero_fire, hero_super)
        move_dir, fire = self._override_phase(move_dir, fire, override)

        dmg_by_melee, melee_healed = self._attack_phase(move_dir, fire, super_fire, aim_dir, aim_point)
        self._movement_phase(move_dir)
        dmg_by_dash = self._dash_phase()
        dmg_by_proj, super_healed = self._projectile_phase()
        self._zone_phase()
        newly_broken = self._box_phase()
        cubes_gained = self._pickup_phase()
        newly_dead = self._death_phase()
        self._zone_schedule()

        dmg_by_total = dmg_by_melee + dmg_by_dash + dmg_by_proj
        # Every heal source this sim has, summed into the one (N,E) delta `compute_info` takes:
        # phase 2's out-of-combat regen, phase 6's melee lifesteal (Edgar), and phase 9's super
        # lifesteal (Mortis). A power cube's max-HP bump
        # raises current HP by the same amount (core/stats.py) but is deliberately NOT counted --
        # it is a pickup, already priced by the reward's own `cube_pickup` term, not HP won back
        # from a wound.
        hp_healed = regen_healed + super_healed + melee_healed
        self._bookkeeping(dmg_by_total)
        return dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed

    def _run_decision(self, action: torch.Tensor, override: torch.Tensor | None):
        """One AGENT DECISION: `cfg.action_repeat` back-to-back `_run_tick` calls holding the
        same action, aggregated into the single set of per-decision quantities `_observe`
        expects. Returns `_run_tick`'s own five values (summed over the sub-ticks) plus the
        `core/events` decision tally.

        `action_repeat=1` returns after the first tick having allocated nothing extra beyond the
        tally itself, so the default configuration keeps its exact previous cost and behavior.

        The loop is a plain Python `range` over a config int, not a data-dependent `while`: the
        trip count is static, so this stays sync-free (no `.any()`/`.item()` host check to
        discover that every env has finished) and `torch.compile` still only ever sees
        `_run_tick`, one unchanged tick at a time.

        **`live` is sampled BEFORE each sub-tick, not after.** An env that finished on an
        earlier sub-tick contributes nothing further to any delta -- damage it deals or takes,
        boxes it breaks, cubes it collects after its own episode ended would otherwise be
        credited to that finished episode's reward. The tally applies the same gate to its own
        counters; see `events.advance_decision_tally`.

        **Sub-ticks 2..K hold the MOVE bin but drop the FIRE bit** (`_held`), so one decision is
        at most one attack attempt -- see the class docstring for why that guard is required
        rather than merely tidy."""
        dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed = self._tick_fn(action, override)
        decision = events.new_decision_tally(self.state, self.cfg)
        self._run_tick_hook()
        if self.cfg.action_repeat == 1:
            # Explicit early return, not just a zero-trip loop: it keeps the `_held` clones below
            # off the per-step path entirely, which is what makes action_repeat=1 cost exactly
            # what it cost before this method existed.
            return dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed, decision

        held_action, held_override = self._held(action), self._held(override)
        for _ in range(self.cfg.action_repeat - 1):
            live = ~decision["done"]
            tick_dmg, tick_dead, tick_broken, tick_cubes, tick_healed = self._tick_fn(held_action, held_override)
            live_e = live.unsqueeze(-1)
            dmg_by = dmg_by + tick_dmg * live.view(-1, 1, 1)
            newly_dead = newly_dead | (tick_dead & live_e)
            newly_broken = newly_broken | (tick_broken & live_e)
            cubes_gained = cubes_gained + tick_cubes * live_e
            hp_healed = hp_healed + tick_healed * live_e
            decision = events.advance_decision_tally(decision, self.state, self.cfg)
            self._run_tick_hook()

        return dmg_by, newly_dead, newly_broken, cubes_gained, hp_healed, decision

    @staticmethod
    def _held(action: torch.Tensor | None) -> torch.Tensor | None:
        """The same action with its FIRE column zeroed -- what sub-ticks 2..K of a decision are
        driven with. Works for both `action` (N,2) and `override` (N,E,2): the fire bit is the
        last column of either, and for `override` the `-1` "no override for this slot" sentinel
        lives in column 0 and is left untouched, so an unoverridden slot stays unoverridden.

        Built ONCE per decision, not per sub-tick -- it is the same tensor every time."""
        if action is None:
            return None
        held = action.clone()
        held[..., 1] = 0
        return held

    def _run_tick_hook(self) -> None:
        """Calls `self.tick_hook(self)` if one is installed, after a sub-tick's phases 1-15.
        `None` by default, which costs one `is not None` test per sub-tick and nothing else --
        see the constructor's own note on what it is for and why it is not free to use."""
        if self.tick_hook is not None:
            self.tick_hook(self)

    def _make_tick_fn(self):
        """`cfg.compile=False` (the default): returns `self._run_tick` completely unwrapped --
        zero overhead, zero behavior change from pre-Step-31 `env.py`. `cfg.compile=True`:
        wraps `torch.compile(self._run_tick, dynamic=False, fullgraph=False)` in a thin guard
        that turns a compile-MACHINERY failure (inductor/dynamo internals -- e.g. `TritonMissing`
        on this Windows+Blackwell box, or `InductorError: ... cl is not found` on Windows CPU,
        both confirmed by hand, see Step 31's plan notes) into a `RuntimeError` with a clear,
        actionable message and the original exception chained via `from e`. Deliberately does
        NOT catch exceptions from `torch._dynamo`/`torch._inductor` blindly rewords them and
        lets everything else (a real bug in `_run_tick` itself, which `fullgraph=False` would
        surface exactly as it would in eager, just interleaved with compiled segments) propagate
        with its own original type/message untouched -- a masked domain bug reported as "torch.
        compile isn't supported here" would be far worse than a clearly-labeled compile failure.
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
                    "(cfg.compile=True). This is a KNOWN, ACCEPTABLE failure mode on some "
                    "platforms (see BRAWL_SIM_BUILD_PLAN.md Step 31): the CPU inductor backend "
                    "needs an MSVC `cl.exe` on PATH, and the CUDA inductor backend needs a "
                    "working Triton install -- neither is guaranteed on Windows, and Triton on "
                    "Windows + Blackwell (sm_120) has been especially fragile as of this "
                    "writing. `compile: false` (the default) is the fully-supported path and "
                    f"is unaffected by this. Original error -- {type(e).__name__}: {e}"
                ) from e

        return _guarded_tick

    # -- Section 4 phase 1 --
    def _pop_action_buffer(self, action: torch.Tensor) -> torch.Tensor:
        """Writes `action` into act_buf[:, act_head], reads back act_buf[:, act_head -
        action_latency_ticks (mod L)], then advances act_head. At 0 ticks (L=1) this is a
        pass-through: write and read hit the same (only) slot.

        Runs once per SUB-TICK, so latency stays denominated in sim ticks no matter what
        `cfg.action_repeat` is: a decision's held action is written into the buffer on each of
        its sub-ticks and comes back out `action_latency_ticks` later, exactly as it would at
        `action_repeat=1`. Correct, but worth knowing that a latency shorter than one decision
        window is invisible to the agent -- the same action is being written on every sub-tick,
        so delaying it by a tick or two inside that window changes nothing it can observe."""
        state, cfg = self.state, self.cfg
        L = state.act_buf.shape[1]

        write_idx = state.act_head.view(-1, 1, 1).expand(-1, 1, 2)
        state.act_buf.scatter_(1, write_idx, action.unsqueeze(1))

        read_idx = ((state.act_head - cfg.action_latency_ticks) % L).view(-1, 1, 1).expand(-1, 1, 2)
        effective_action = state.act_buf.gather(1, read_idx).squeeze(1)

        state.act_head.copy_((state.act_head + 1) % L)
        return effective_action

    # -- Section 4 phase 2 (regen folded in here; see class docstring) --
    def _tick_timers(self) -> torch.Tensor:
        """Regen isn't its own numbered phase in Section 4, but it reads ent_out_of_combat_t (a
        timer) and only ever heals -- and running it here, BEFORE this tick's combat, means a
        fatal hit dealt later this same tick can never be "un-fatal-ed" by a regen tick that
        hasn't seen it yet. Running it later (e.g. at bookkeeping) would risk exactly that.

        Returns the (N,E) HP regen actually restored, for `info["hp_healed_tick"]`."""
        hero.tick_timers(self.state, self.params, self.cfg)
        return combat.apply_regen(self.state, self.params, self.cfg)

    # -- Section 4 phase 3 --
    def _decode(self, effective_action: torch.Tensor):
        return hero.decode_action(effective_action, self.state, self.params, self.cfg)

    # -- Section 4 phase 4 --
    def _bot_phase(self, hero_move_dir: torch.Tensor, hero_fire: torch.Tensor,
                   hero_super: torch.Tensor):
        state, cfg = self.state, self.cfg
        vis = perception.visibility(state, self.bank, self.params, cfg)
        intent = policy.all_bot_intents(state, vis, self.bank, self.params, cfg, self.gen)

        move_dir = intent.move_dir.clone()
        move_dir[:, 0] = hero_move_dir
        fire = intent.fire.clone()
        fire[:, 0] = hero_fire
        # `BotIntent.super_fire` is all-False today -- no bot kind configures a super. It exists so
        # that giving one to a bot is a decision inside its combat rule plus a brawlers.yaml block,
        # with no plumbing change here (bot_overhaul.md D2).
        super_fire = intent.super_fire.clone()
        super_fire[:, 0] = hero_super
        # The hero has no separate ranged aim (N02/N03/R02: dash inherits move_dir) -- these
        # placeholders are only ever read for entity 0 by spawn_volley, whose own proj_count
        # for hero_mortis is 0 (unset in brawlers.yaml), so they're provably inert.
        aim_dir = intent.aim_dir.clone()
        aim_dir[:, 0] = geo.from_angle(state.ent_facing[:, 0])
        aim_point = intent.aim_point.clone()
        aim_point[:, 0] = state.ent_pos[:, 0]
        return move_dir, fire, super_fire, aim_dir, aim_point

    # -- Section 4 phase 5 --
    def _override_phase(self, move_dir: torch.Tensor, fire: torch.Tensor, override: torch.Tensor | None):
        """override[..., 0] == -1 means "no override for this entity slot" (valid move bins are
        0..n_move_bins, so -1 is an unambiguous sentinel); otherwise override[n,e] is decoded
        exactly like the hero's own action and REPLACES that entity's move_dir/fire for this
        tick. Does not touch aim_dir/aim_point -- override drives movement/attack-timing, not
        targeting; an overridden bot still aims via its own archetype logic, and the hero's
        "aim" is just its (now overridden) move_dir via the dash."""
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

    # -- Section 4 phase 6 --
    def _attack_phase(self, move_dir: torch.Tensor, fire: torch.Tensor,
                      super_fire: torch.Tensor, aim_dir: torch.Tensor, aim_point: torch.Tensor):
        """consume ammo; START DASH (clip path now); spawn volleys; melee hitscan.

        Returns (dmg_by (N,E,E), healed (N,E)) -- the second is melee lifesteal as ACTUALLY
        applied, on the same contract as `_projectile_phase`'s: `apply_heal` drops it for dead
        entities and clips it at max HP, so it is what `info["hp_healed_tick"]` must be paid on.

        Ammo/cooldown/shots_fired bookkeeping for ranged and melee attacks is done HERE, by
        hand -- neither projectiles.spawn_volley nor combat.melee_hitscan mutates ent_ammo/
        ent_attack_cd/ent_shots_fired (only hero.start_dash does, for dashes). This also closes
        a gap core/events.py (Step 27) flagged explicitly: ent_shots_fired was previously only
        ever incremented by dashes, so ranged/melee bots never counted as having "fired" in the
        cumulative counter or the observation. They do now."""
        state, bank, params, cfg = self.state, self.bank, self.params, self.cfg

        # Safety net: bot-sourced and hero-sourced fire are already gated on this (fire_gate /
        # action_mask), but override-sourced fire has no such gate -- this makes it impossible
        # for ANY source to force an illegal attack through.
        can_attack = state.ent_alive & (state.ent_ammo >= 1.0) & (state.ent_attack_cd <= 0) & (state.ent_dash_t <= 0)
        fire = fire & can_attack

        # A SUPER costs charge, not ammo, so its gate deliberately omits the ammo term -- an empty
        # clip must not block it. It keeps the cooldown and no-dashing terms, so it cannot be used
        # to sidestep either. Same safety-net role as `can_attack` above: hero-sourced supers are
        # already gated by `action_mask`, but this makes an illegal one impossible from ANY source,
        # including a future bot or an override.
        can_super = (
            state.ent_alive & (state.ent_attack_cd <= 0) & (state.ent_dash_t <= 0)
            & hero.super_ready(state, params)
        )
        super_fire = super_fire & can_super

        dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)
        is_dash_attack = fire & (dash_distance > 0)
        is_other_attack = fire & ~is_dash_attack

        hero.start_dash(state, fire, move_dir, bank, params, cfg)

        # Attacking breaks concealment (Step 41). `perception.reveal_after_attack` has been a
        # SimParams field since Step 3 and is read by bots/perception.visibility, but nothing ever
        # WROTE ent_reveal_t -- hero.tick_timers only ever decremented it -- so the parameter was
        # inert and firing from a bush left you hidden. Applied to `fire` (every attack source,
        # dashes included) rather than only to bots, so the hero cannot attack out of a bush with
        # impunity either; that asymmetry was part of what made bush-sitting a winning policy.
        # torch.maximum, not assignment: a fresh shot must never SHORTEN a longer reveal already
        # running. tick_timers decrements at phase 2 and this is phase 6, so a reveal set here
        # lasts its full duration.
        # `attacked` -- not `fire` -- for everything that means "I just took an offensive
        # action": a super breaks concealment, breaks out-of-combat for regen, and spends the
        # long-dash charge exactly like an ordinary attack does. Missing any of these would
        # make the super a free way to shoot from a bush, out-heal a fight, or keep a charged
        # long dash banked.
        attacked = fire | super_fire
        reveal = params.reveal_after_attack.unsqueeze(-1).expand_as(state.ent_reveal_t)
        state.ent_reveal_t.copy_(
            torch.where(attacked, torch.maximum(state.ent_reveal_t, reveal), state.ent_reveal_t)
        )

        # Attacking also breaks you OUT OF COMBAT for regen purposes, exactly like being hit does
        # (combat.apply_damage owns that side). Without this, an entity could hold a target down
        # while out-healing the chip damage it was taking, because only the VICTIM's stopwatch ever
        # reset -- "regen after 4s of not taking damage or attacking" needs both halves, and this is
        # the only phase that knows an attack actually happened.
        state.ent_out_of_combat_t.copy_(
            torch.where(attacked, torch.zeros_like(state.ent_out_of_combat_t), state.ent_out_of_combat_t)
        )

        # Long-dash charge (Step D1). Reset by attacking and by NOTHING ELSE -- notably not by
        # taking damage, unlike ent_out_of_combat_t above.
        #
        # Placed after `hero.start_dash`, which is load-bearing: start_dash reads the charge to
        # decide whether THIS dash is a long one, so resetting before it would mean the long dash
        # could never actually fire. The ordering is the same reason out_of_combat_t is reset here
        # rather than earlier in the phase.
        state.ent_attack_idle_t.copy_(
            torch.where(attacked, torch.zeros_like(state.ent_attack_idle_t), state.ent_attack_idle_t)
        )

        attack_cooldown = stats.gather_kind(params.attack_cooldown, state.ent_kind)
        state.ent_ammo.copy_(torch.where(is_other_attack, state.ent_ammo - 1.0, state.ent_ammo))
        state.ent_attack_cd.copy_(torch.where(is_other_attack, attack_cooldown, state.ent_attack_cd))
        state.ent_shots_fired.copy_(torch.where(is_other_attack, state.ent_shots_fired + 1, state.ent_shots_fired))

        # --- super: spend the whole charge, take the cooldown, launch the bolt ---
        # Charge is zeroed rather than decremented by `super_charge_hits`: it is capped at that
        # value by `add_super_charge`, so the two are equivalent today, but "firing spends the
        # meter" stays true if a future brawler ever banks more than one.
        state.ent_super_charge.copy_(
            torch.where(super_fire, torch.zeros_like(state.ent_super_charge), state.ent_super_charge)
        )
        state.ent_attack_cd.copy_(torch.where(super_fire, attack_cooldown, state.ent_attack_cd))
        state.ent_shots_fired.copy_(torch.where(super_fire, state.ent_shots_fired + 1, state.ent_shots_fired))
        projectiles.spawn_supers(state, super_fire, state.ent_pos, move_dir, params, cfg)

        damage = stats.effective_damage(state.ent_kind, state.ent_cubes, params)
        projectiles.spawn_volley(state, is_other_attack, state.ent_pos, aim_dir, aim_point, state.ent_kind, damage, params, cfg)

        # --- melee, single-cone and SWEPT, resolved in ONE hitscan call (Step C2) ---
        # Buzz's attack is five cones fanned across his attack_cooldown rather than one at trigger
        # time. `sweep_cone_dir` reads the schedule off `ent_attack_cd`, which the block above just
        # set for anyone firing this tick -- so on the trigger tick elapsed is 0 and sub-swing 0
        # lands immediately, and the remaining four land on later ticks of the same cooldown.
        #
        # A swept kind is masked OUT of the ordinary trigger-time cone (`& ~swept`), because
        # sub-swing 0 already covers that tick; letting both through would double the first swing.
        #
        # Deliberately ONE melee_hitscan call with a per-entity cone direction rather than two
        # calls: melee_hitscan is a dense (N,E,E) cone + LOS march and the single most expensive
        # thing in this phase (~1.95 ms/tick at n_envs=1024 even after Step A1), so a second call
        # would have doubled it to buy nothing -- the two populations are disjoint per entity.
        sweep_fire, sweep_dir = melee_sweep.sweep_cone_dir(state, params, cfg)
        swept = melee_sweep.is_swept(state, params)
        melee_fire = (is_other_attack & ~swept) | sweep_fire
        cone_dir = torch.where(sweep_fire, sweep_dir, state.ent_facing)

        dmg_ent, dmg_by, dmg_box = combat.melee_hitscan(state, melee_fire, bank, params, cfg, cone_dir=cone_dir)
        combat.apply_damage(state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), params, cfg)
        # MELEE LIFESTEAL (Edgar). After apply_damage, for the same reason _projectile_phase heals
        # after its own damage: a trade that kills the victim resolves the kill first. It also has
        # to be after it to read the post-damage `ent_invuln_t` that combat.melee_lifesteal mirrors
        # -- though nothing in apply_damage writes that field, so the two orderings agree today and
        # this ordering is the one that keeps agreeing if that ever changes.
        #
        # Every kind but Edgar has `melee_lifesteal_fraction: 0`, so this is a masked multiply
        # producing an all-zero heal for the rest of the roster, not a branch.
        healed = combat.apply_heal(state, combat.melee_lifesteal(state, dmg_by, params))
        boxes.damage_boxes(state, dmg_box)
        return dmg_by, healed

    # -- Section 4 phase 7 --
    def _movement_phase(self, move_dir: torch.Tensor) -> None:
        """ONLY non-dashing entities -- apply_movement's own `active = alive & dash_t<=0` gate
        already excludes anyone who just started a dash in _attack_phase (their dash_t is
        already > 0 by the time this runs), which is exactly what makes the dash replace the
        walk this same tick (N02/N03/R02)."""
        movement.apply_movement(self.state, move_dir, self.bank, self.params, self.cfg)

    # -- Section 4 phase 8 --
    def _dash_phase(self) -> torch.Tensor:
        dmg_ent, dmg_by, dmg_box = hero.advance_dash(self.state, self.params, self.cfg)
        combat.apply_damage(self.state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), self.params, self.cfg)
        boxes.damage_boxes(self.state, dmg_box)
        return dmg_by

    # -- Section 4 phase 9 --
    def _projectile_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(dmg_by (N,E,E), healed (N,E)) -- the second is the lifesteal ACTUALLY applied, which
        is not `heal_ent`: apply_heal drops it for dead owners and clips it at max HP."""
        dmg_ent, dmg_by, dmg_box, heal_ent = projectiles.step_projectiles(self.state, self.bank, self.params, self.cfg)
        combat.apply_damage(self.state, dmg_ent, int(DeathCause.COMBAT), combat.dominant_attacker(dmg_by), self.params, self.cfg)
        # Lifesteal AFTER damage, so a super that trades with something lethal still resolves
        # the death first -- healing a corpse is rejected by apply_heal's own alive gate.
        healed = combat.apply_heal(self.state, heal_ent)
        boxes.damage_boxes(self.state, dmg_box)
        return dmg_by, healed

    # -- Section 4 phase 10 --
    def _zone_phase(self) -> None:
        state, params, cfg = self.state, self.params, self.cfg
        dmg = zone.zone_damage(state, params, cfg)
        no_attacker = torch.full_like(state.ent_last_hit_by, -1)
        combat.apply_damage(state, dmg, int(DeathCause.ZONE), no_attacker, params, cfg)

    # -- Section 4 phase 11 (box HP already decremented per-source in phases 6/8/9) --
    def _box_phase(self) -> torch.Tensor:
        return boxes.resolve_broken_boxes(self.state, self.params, self.cfg)

    # -- Section 4 phase 12 --
    def _pickup_phase(self) -> torch.Tensor:
        return combat.collect_pickups(self.state, self.params, self.cfg)

    # -- Section 4 phase 13 --
    def _death_phase(self) -> torch.Tensor:
        newly_dead = combat.resolve_deaths(self.state, self.cfg)
        combat.drop_cubes_on_death(self.state, newly_dead, self.params, self.cfg)
        return newly_dead

    # -- Section 4 phase 14 --
    def _zone_schedule(self) -> None:
        zone.step_zone(self.state, self.params, self.cfg)

    # -- Section 4 phase 15 --
    def _bookkeeping(self, dmg_by_total: torch.Tensor) -> None:
        """time/step_count/n_alive/most cumulative counters are already current by this point
        (resolve_deaths updates n_alive/kills, resolve_broken_boxes updates boxes_broken,
        _attack_phase updates shots_fired, apply_damage updates damage_taken -- each as its own
        event happens). ent_damage_dealt is the one exception: nothing upstream ever writes the
        ATTACKER's side of a hit (apply_damage only updates the victim's damage_taken), so it's
        aggregated here from this tick's combined attacker x victim matrix -- combat damage
        only, matching core/events.py's damage_dealt_tick definition (zone damage has no
        attacker to credit)."""
        state, cfg = self.state, self.cfg
        state.ent_damage_dealt.copy_(state.ent_damage_dealt + dmg_by_total.sum(dim=2))

        # SUPER CHARGE (Step D2): one per (attacker, victim) pair that connected this tick.
        # Derived from the attacker x victim matrix rather than a simpler "did I deal damage" flag
        # for two reasons: it is the only place that distinguishes hits on PLAYERS from hits on
        # boxes (box damage never enters `dmg_by`), so a hero cannot farm his super off crates; and
        # it counts a multi-target hit -- Buzz's sweep, Shelly's spread, Grom's cross -- as one
        # charge per victim, which is what "5 hits against other players" means.
        hits = (dmg_by_total > 0).sum(dim=2).to(torch.int32)
        hero.add_super_charge(state, hits, self.params)

        state.time += cfg.dt
        state.step_count += 1

    # -- Section 4 phase 16 --
    def _observe(self, dmg_by_total, newly_dead, newly_broken, cubes_gained, hp_healed, decision):
        """Runs ONCE per decision, not once per sub-tick -- `reward_fn` therefore sees the
        summed deltas and the latched outcome fields for the whole decision, and is called
        exactly once per `step()` no matter what `action_repeat` is. That keeps total episode
        return invariant to `action_repeat` (per-tick reward terms integrate over
        `info["alive_ticks"]`/`info["in_zone_ticks"]` rather than firing once per call), which
        is what makes the decision rate tunable without re-tuning every reward weight."""
        obs = self._build_observation()
        info = events.compute_info(
            self.state, dmg_by_total, newly_dead, newly_broken, cubes_gained, self.cfg,
            decision=decision, hp_healed=hp_healed,
        )
        reward = self.reward_fn(obs, info, self.cfg)
        return obs, info, reward

    # -- Section 4 phase 17 --
    def _autoreset(self, info: dict):
        """Resets done envs in place via a boolean mask -- reset_envs is torch.where-masked
        internally, so calling it every tick (even when nobody's done) is sync-free and
        correct, never needing a `.any()` host check first.

        Does NOT take obs_before_reset: `step()` already cloned it into `final_observation`/
        `final_info` and dropped its own reference before calling this, specifically so those
        large buffers (obs["world"] alone is ~177 MB at n_envs=4096) can be freed before the
        `_build_observation()` call below allocates a same-sized SECOND copy for the post-reset
        `obs` -- see step()'s own comment for why the clone had to happen there, not here."""
        terminated, truncated = info["terminated"], info["truncated"]
        done = terminated | truncated
        spawn.reset_envs(self.state, done, self.bank, self.params, self.cfg, self.gen,
                         self.spec, self.params_hook)
        obs = self._build_observation()
        return obs, terminated, truncated
