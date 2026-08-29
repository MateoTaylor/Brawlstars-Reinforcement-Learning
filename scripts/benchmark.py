"""Throughput/memory/observation-footprint benchmarking harness for BrawlVecEnv and
BrawlSB3VecEnv. See BRAWL_SIM_BUILD_PLAN.md Step 40.

**Three separate things get measured, each with its own function, not one do-everything loop:**
`throughput_native`/`throughput_sb3` (env-steps/sec, ms/step, peak VRAM, for one `n_envs` and
one path), `phase_breakdown` (per-Section-4-phase timing at one representative `n_envs`), and
`observation_footprint` (a purely analytical byte count from `obs_schema.obs_spec`, no env
construction needed at all). `main()` just calls each with the plan's requested sweep and
prints a report -- there's no shared "benchmark runner" abstraction because these three measure
genuinely different things (wall-clock throughput vs. per-phase device-side timing vs. static
schema arithmetic) and forcing them through one interface would obscure that, not simplify it.

**Actions are pre-generated ONCE, outside every timed region, and reused every step.** Sampling
fresh random actions inside the loop would pollute the throughput measurement with action-RNG
cost that has nothing to do with the sim; reusing one fixed action tensor is safe here because
every simulation branch here is expressed as `torch.where` over the full batch (CONVENTIONS.md),
so which specific action was chosen doesn't change which code PATHS run, only which VALUES flow
through them -- unlike a real training loop, this benchmark doesn't care about the values, only
the cost of computing them.

**Peak memory is reset AFTER warmup, not before.** `torch.cuda.reset_peak_memory_stats` right
before the timed region means the reported `peak_mem_bytes` reflects steady-state operation,
not first-call allocator ramp-up (lazy buffer allocation, CUDA context/kernel caching) that a
real training run only ever pays once.

**`phase_breakdown`'s per-phase timing works by monkeypatching `BrawlVecEnv`'s own bound phase
methods on the INSTANCE (see `_instrumented_phases`), not by hand-copying `_run_tick`'s call
sequence into this script.** `env.step()` -> `self._tick_fn` -> `self._run_tick` calls
`self._pop_action_buffer(...)`, `self._tick_timers()`, etc. by attribute lookup on `self` at
CALL time, not at `_tick_fn`-construction time -- so wrapping those same names on the env
INSTANCE with a timing wrapper is picked up transparently by the real, unmodified production
code path. A hand-copied mirror of the phase sequence was considered and rejected: it would
silently drift the moment `_run_tick`'s own phase order changed, and this repo's own established
pattern (Steps 37-39) is to reuse production code for debug tooling wherever that's possible
rather than risk a second copy. This only works because `cfg.compile=False` (the plan's own
fully-supported default, see Step 31) -- with `torch.compile` enabled, `_run_tick` gets traced
into a graph on first call and per-phase attribution via instance-attribute monkeypatching would
no longer mean anything; `main()` never exposes a `--compile` flag for exactly this reason.

**Section 4 has 17 phases; `_run_tick` (Step 29's own docstring) covers phases 1-15, with 16
(`_observe`) and 17 (`_autoreset`) run separately by `step()` itself.** `phase_breakdown` times
all 17 for a complete per-STEP picture. Note "per step", not "per tick": one `step()` is one
agent decision covering `cfg.action_repeat` sim ticks, so phases 1-15 are each invoked
`action_repeat` times per step and phases 16/17 once. Every number reported is the total time
that phase cost across ONE step, sub-ticks included -- which is what makes the phases comparable
to each other and to `throughput_native`'s ms/step. Raising `action_repeat` should therefore show
phases 1-15 growing roughly linearly while 16/17 stay flat; that flat part is exactly the cost
action repeat amortizes away. `step()`'s own small amount of work OUTSIDE any of those
17 methods -- the action tensor's device/dtype cast, and `final_observation`/`final_info`
cloning via `observation.clone_obs` (Step 29's own documented VRAM-saving early-clone, non-trivial
at scale since `obs["world"]` alone is ~162 MB at n_envs=4096) -- is NOT separately broken out,
since it isn't a bound method on `env` to monkeypatch. It shows up as the gap between "sum of
phases" and this same n_envs's own `throughput_native` ms/step number; printed as an explicit
residual line rather than silently absorbed into the last phase's number.
"""
import argparse
import contextlib
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

from brawl_sim.config import load_config
from brawl_sim.core import obs_schema, obs_select
from brawl_sim.core.reward import ExampleReward
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_AGENT_OBS_PATH = REPO_ROOT / "configs" / "agent_obs.yaml"

_DEFAULT_N_ENVS = (1, 64, 256, 1024, 4096, 16384)
_SB3_INFO_MODES = ("minimal", "full")
_NATIVE_TARGET_STEPS_PER_SEC = 100_000  # 5070 Ti, n_envs >= 1024 -- see plan's Step 40 text

# Section 4 phases 1-15 (_run_tick) + 16/17 (_observe/_autoreset), in the exact order env.py
# calls them -- see module docstring for why this list is real method NAMES to monkeypatch,
# not a copy of their logic.
_PHASE_METHOD_NAMES = (
    "_pop_action_buffer", "_tick_timers", "_decode", "_bot_phase", "_override_phase",
    "_attack_phase", "_movement_phase", "_dash_phase", "_projectile_phase", "_zone_phase",
    "_box_phase", "_pickup_phase", "_death_phase", "_zone_schedule", "_bookkeeping",
    "_observe", "_autoreset",
)

# hero.hp_frac(f32) + hero.pos(f32,2) + hero.alive(bool) + meta.step_count(i32) + meta.n_alive(i32)
# -- the extra per-env .to("cpu") transfers _maybe_attach_full_diagnostics does, analytically
# derived rather than measured since they're a fixed function of dtype, same reasoning as
# observation_footprint. Assumes the default info_every=1 (this script never overrides it).
_FULL_DIAG_BYTES_PER_ENV = 4 + 4 * 2 + 1 + 4 + 4

_DTYPE_BYTES = {"float32": 4, "bool": 1, "int64": 8, "int32": 4, "uint8": 1}


def _random_native_action(env: BrawlVecEnv) -> torch.Tensor:
    move = torch.randint(0, env.cfg.n_move_bins + 1, (env.n_envs,), device=env.device)
    fire = torch.randint(0, 2, (env.n_envs,), device=env.device)
    return torch.stack([move, fire], dim=1)


def _random_sb3_action(venv: BrawlSB3VecEnv) -> np.ndarray:
    move = np.random.randint(0, venv.cfg.n_move_bins + 1, size=venv.num_envs)
    fire = np.random.randint(0, 2, size=venv.num_envs)
    return np.stack([move, fire], axis=1).astype(venv.action_space.dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---- throughput ---------------------------------------------------------------------------------

def throughput_native(cfg, n_envs: int, device: str, steps: int, warmup: int, seed: int = 0) -> dict:
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed, verbose=False)
    env.reset()
    action = _random_native_action(env)

    for _ in range(warmup):
        env.step(action)
    _sync(env.device)
    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(action)
    _sync(env.device)
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated(env.device) if env.device.type == "cuda" else None
    return {
        "n_envs": n_envs,
        "steps_per_sec": (n_envs * steps) / elapsed,
        "ms_per_step": (elapsed / steps) * 1000.0,
        "peak_mem_bytes": peak_mem,
    }


def host_bytes_per_step(venv: BrawlSB3VecEnv) -> int:
    """Static byte count of the ALWAYS-paid per-step device->host transfer (the agent
    observation's host buffers + reward/terminated/truncated), computed directly from `venv`'s
    own persistent buffer tensors -- a fixed function of agent_spec/cfg/n_envs/info_mode, not
    something that needs to be measured. Does NOT include the CONDITIONAL
    `terminal_observation` transfer (same size as the observation itself) that only fires on
    ticks with >= 1 done env -- `main()` reports that worst case as a separate number."""
    obs_bytes = sum(t.numel() * t.element_size() for t in venv._host_buffers.values())
    scalar_bytes = sum(
        t.numel() * t.element_size() for t in (venv._reward_host, venv._terminated_host, venv._truncated_host)
    )
    total = obs_bytes + scalar_bytes
    if venv.info_mode == "full":
        total += _FULL_DIAG_BYTES_PER_ENV * venv.num_envs
    return total


def throughput_sb3(
    cfg, agent_spec, n_envs: int, device: str, steps: int, warmup: int, info_mode: str, seed: int = 0,
) -> dict:
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed, verbose=False)
    venv = BrawlSB3VecEnv(env, agent_spec, ExampleReward(), info_mode=info_mode)
    venv.reset()
    action = _random_sb3_action(venv)

    for _ in range(warmup):
        venv.step(action)
    _sync(venv.device)
    if venv.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(venv.device)

    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(action)
    _sync(venv.device)
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated(venv.device) if venv.device.type == "cuda" else None
    obs_bytes = sum(t.numel() * t.element_size() for t in venv._host_buffers.values())
    return {
        "n_envs": n_envs,
        "info_mode": info_mode,
        "steps_per_sec": (n_envs * steps) / elapsed,
        "ms_per_step": (elapsed / steps) * 1000.0,
        "peak_mem_bytes": peak_mem,
        "host_bytes_per_step": host_bytes_per_step(venv),
        "host_bytes_worst_case": host_bytes_per_step(venv) + obs_bytes,
    }


# ---- per-phase breakdown ------------------------------------------------------------------------

@contextlib.contextmanager
def _instrumented_phases(env: BrawlVecEnv, phase_names=_PHASE_METHOD_NAMES):
    """See module docstring. Restores the original (class-resolved) methods on exit regardless
    of how the `with` block exits."""
    use_cuda = env.device.type == "cuda"
    events: dict = {name: [] for name in phase_names}

    def _make_wrapper(name, fn):
        def wrapper(*args, **kwargs):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = fn(*args, **kwargs)
                end.record()
                events[name].append((start, end))
            else:
                t0 = time.perf_counter()
                result = fn(*args, **kwargs)
                events[name].append((t0, time.perf_counter()))
            return result
        return wrapper

    for name in phase_names:
        env.__dict__[name] = _make_wrapper(name, getattr(env, name))
    try:
        yield events
    finally:
        for name in phase_names:
            del env.__dict__[name]


def phase_breakdown(cfg, n_envs: int, device: str, ticks: int, warmup: int, seed: int = 0) -> dict:
    env = BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed, verbose=False)
    env.reset()
    action = _random_native_action(env)

    for _ in range(warmup):
        env.step(action)

    with _instrumented_phases(env) as events:
        for _ in range(ticks):
            env.step(action)
        _sync(env.device)

    result = {}
    for name, pairs in events.items():
        if env.device.type == "cuda":
            total_ms = sum(s.elapsed_time(e) for s, e in pairs)
        else:
            total_ms = sum((e - s) * 1000.0 for s, e in pairs)
        # Divided by `ticks` (the number of env.step() CALLS), NOT by len(pairs) (the number of
        # times that phase was invoked). With cfg.action_repeat > 1 those differ: phases 1-15 run
        # once per SUB-TICK (action_repeat calls per step) while 16/17 run once per step. Per-call
        # averages would silently make the two groups incomparable and would no longer sum to
        # anything -- the whole point of this breakdown is that its total lines up with
        # throughput_native's ms/step, so every phase is normalized to the same per-step unit.
        result[name] = total_ms / ticks
    return result


# ---- observation footprint (purely analytical, no env needed) ----------------------------------

def _bytes_per_env(cfg) -> dict:
    spec = obs_schema.obs_spec(cfg)
    out = {}
    for name, field in spec.items():
        count = 1
        for d in field.shape[1:]:  # drop the "N" leading dim
            count *= d
        out[name] = count * _DTYPE_BYTES[field.dtype]
    return out


def observation_footprint(cfg, n_envs: int) -> dict:
    with_grid = _bytes_per_env(dataclasses.replace(cfg, obs_include_world_grid=True))
    without_grid = _bytes_per_env(dataclasses.replace(cfg, obs_include_world_grid=False))
    per_env_with = sum(with_grid.values())
    per_env_without = sum(without_grid.values())
    return {
        "per_env_bytes_with_world_grid": per_env_with,
        "per_env_bytes_without_world_grid": per_env_without,
        "total_bytes_with_world_grid": per_env_with * n_envs,
        "total_bytes_without_world_grid": per_env_without * n_envs,
        "world_grid_bytes_per_env": with_grid.get("world", 0),
    }


# ---- CLI / report --------------------------------------------------------------------------------

def _fmt_bytes(n) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _check_gpu(device: str) -> str:
    """Returns the DEVICE STRING TO ACTUALLY USE. Warns loudly (plan's own requirement) on CPU
    or a non-sm_120 GPU, but only ever falls back CPU<-cuda when cuda was requested but truly
    unavailable -- an sm_120 mismatch is a warning, not a fallback, since the numbers are still
    real, just not comparable to the plan's 5070 Ti-specific target."""
    if device != "cuda":
        print("!" * 78)
        print("WARNING: benchmarking on CPU. Every steps/sec number below is meaningless as a")
        print("GPU throughput reference -- this run only exercises the CODE PATH. Re-run with")
        print("--device cuda for real numbers.")
        print("!" * 78)
        return device
    if not torch.cuda.is_available():
        print("!" * 78)
        print("WARNING: --device cuda requested but torch.cuda.is_available() is False.")
        print("Falling back to CPU -- see the CPU warning above for what that means.")
        print("!" * 78)
        return "cpu"
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        print("!" * 78)
        print(f"WARNING: GPU is {torch.cuda.get_device_name()} (sm_{major}{minor}), not sm_120")
        print("(Blackwell). The plan's '100k+ env-steps/sec at n_envs>=1024' target is specific")
        print("to a 5070 Ti -- treat it as a rough reference on this GPU, not a pass/fail bar.")
        print("!" * 78)
    return device


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--agent-obs", default=str(DEFAULT_AGENT_OBS_PATH))
    p.add_argument("--preset", default=None, help="optional configs/presets/*.yaml overrides file")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--n-envs-list", default=",".join(str(n) for n in _DEFAULT_N_ENVS),
                    help="comma-separated n_envs sweep")
    p.add_argument("--sb3-info-modes", default=",".join(_SB3_INFO_MODES))
    p.add_argument("--steps", type=int, default=30, help="measured steps per throughput config")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--phase-n-envs", type=int, default=1024)
    p.add_argument("--phase-steps", type=int, default=20)
    p.add_argument("--mem-n-envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    overrides = None
    if args.preset:
        import yaml
        overrides = yaml.safe_load(Path(args.preset).read_text())
    cfg = load_config(args.config, overrides=overrides)
    agent_spec = obs_select.load_agent_spec(args.agent_obs, cfg)

    device = _check_gpu(args.device)
    n_envs_list = [int(x) for x in args.n_envs_list.split(",")]
    info_modes = [x.strip() for x in args.sb3_info_modes.split(",")]

    print(f"\n=== native BrawlVecEnv.step (steps={args.steps}, warmup={args.warmup}) ===")
    for n_envs in n_envs_list:
        r = throughput_native(cfg, n_envs, device, args.steps, args.warmup, seed=args.seed)
        note = ""
        if device == "cuda" and n_envs >= 1024:
            passed = r["steps_per_sec"] > _NATIVE_TARGET_STEPS_PER_SEC
            note = f"  [target >{_NATIVE_TARGET_STEPS_PER_SEC:,}: {'PASS' if passed else 'BELOW TARGET'}]"
        print(f"  n_envs={n_envs:>6}  {r['steps_per_sec']:>13,.0f} steps/sec  "
              f"{r['ms_per_step']:>9.3f} ms/step  peak_mem={_fmt_bytes(r['peak_mem_bytes']):>9}{note}")

    for info_mode in info_modes:
        print(f"\n=== BrawlSB3VecEnv.step (info_mode={info_mode!r}, steps={args.steps}, warmup={args.warmup}) ===")
        for n_envs in n_envs_list:
            r = throughput_sb3(cfg, agent_spec, n_envs, device, args.steps, args.warmup, info_mode, seed=args.seed)
            print(f"  n_envs={n_envs:>6}  {r['steps_per_sec']:>13,.0f} steps/sec  "
                  f"{r['ms_per_step']:>9.3f} ms/step  peak_mem={_fmt_bytes(r['peak_mem_bytes']):>9}  "
                  f"host_xfer={_fmt_bytes(r['host_bytes_per_step'])}/step "
                  f"(worst case w/ a done env: {_fmt_bytes(r['host_bytes_worst_case'])})")

    print(f"\n=== per-tick phase breakdown (n_envs={args.phase_n_envs}, ticks={args.phase_steps}) ===")
    phases = phase_breakdown(cfg, args.phase_n_envs, device, args.phase_steps, args.warmup, seed=args.seed)
    total_phases_ms = sum(phases.values())
    for name, ms in phases.items():
        pct = (100.0 * ms / total_phases_ms) if total_phases_ms > 0 else 0.0
        print(f"  {name:<22} {ms:>9.4f} ms  ({pct:>5.1f}%)")
    print(f"  {'sum of 17 phases':<22} {total_phases_ms:>9.4f} ms")
    print("  (residual vs. this n_envs's own native ms/step above is step()'s own action cast +")
    print("   final_observation/final_info cloning -- not individually broken out, see module docstring)")

    print(f"\n=== observation footprint (n_envs={args.mem_n_envs}) ===")
    fp = observation_footprint(cfg, args.mem_n_envs)
    print(f"  per-env   WITH world grid: {_fmt_bytes(fp['per_env_bytes_with_world_grid'])}   "
          f"WITHOUT: {_fmt_bytes(fp['per_env_bytes_without_world_grid'])}   "
          f"(world grid alone: {_fmt_bytes(fp['world_grid_bytes_per_env'])}/env)")
    print(f"  total     WITH world grid: {_fmt_bytes(fp['total_bytes_with_world_grid'])}   "
          f"WITHOUT: {_fmt_bytes(fp['total_bytes_without_world_grid'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
