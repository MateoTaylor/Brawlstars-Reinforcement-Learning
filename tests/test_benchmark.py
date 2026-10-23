import numpy as np
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_sim.env import BrawlVecEnv
from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv
from scripts import benchmark

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _cfg(overrides=None):
    merged = dict(CONFIGS_TINY)
    if overrides:
        merged = {**merged, **overrides}
    return load_config(CONFIGS_DEFAULT, overrides=merged)


# ---- throughput_native --------------------------------------------------------------------------

def test_throughput_native_returns_sane_fields_on_cpu():
    cfg = _cfg()
    r = benchmark.throughput_native(cfg, n_envs=4, device="cpu", steps=3, warmup=1)
    assert r["n_envs"] == 4
    assert r["steps_per_sec"] > 0
    assert r["ms_per_step"] > 0
    assert r["peak_mem_bytes"] is None  # no CUDA on CPU


def test_throughput_native_samples_one_action_and_the_sim_never_mutates_it(monkeypatch):
    """The action tensor is built once and handed to every env.step() call, warmup included, and
    the sim reads it without ever writing into the caller's tensor."""
    sampled, stepped = [], []
    real_sample, real_step = benchmark._random_native_action, BrawlVecEnv.step

    def sample(env):
        action = real_sample(env)
        sampled.append((action, action.clone()))
        return action

    def step(self, action, *args, **kwargs):
        stepped.append(action)
        return real_step(self, action, *args, **kwargs)

    monkeypatch.setattr(benchmark, "_random_native_action", sample)
    monkeypatch.setattr(BrawlVecEnv, "step", step)
    benchmark.throughput_native(_cfg(), n_envs=4, device="cpu", steps=3, warmup=1)
    (action, original), = sampled
    assert action.shape == (4, 2)
    assert len(stepped) == 4 and all(a is action for a in stepped)
    assert torch.equal(action, original)


# ---- throughput_sb3 + host_bytes_per_step --------------------------------------------------------

def test_throughput_sb3_minimal_returns_sane_fields_on_cpu():
    cfg = _cfg()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    r = benchmark.throughput_sb3(cfg, spec, n_envs=4, device="cpu", steps=3, warmup=1, info_mode="minimal")
    assert r["n_envs"] == 4 and r["info_mode"] == "minimal"
    assert r["steps_per_sec"] > 0
    assert r["host_bytes_per_step"] > 0
    assert r["host_bytes_worst_case"] > r["host_bytes_per_step"]  # terminal-obs case adds strictly more


def test_full_info_mode_adds_exactly_the_bytes_the_diagnostics_read():
    """The gap between the modes is _FULL_DIAG_BYTES_PER_ENV, a hand-kept constant; it is checked
    against the tensors _maybe_attach_full_diagnostics actually reads, so a diagnostic added
    there without updating the constant fails here."""
    cfg = _cfg()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    venv = BrawlSB3VecEnv(BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0), spec,
                          benchmark.ExampleReward(), info_mode="full")
    venv.reset()
    read = {}

    class _Recording(dict):
        def __getitem__(self, key):
            value = dict.__getitem__(self, key)
            read[id(self), key] = value
            return value

    full_obs = venv._last_full_obs
    venv._maybe_attach_full_diagnostics([{} for _ in range(4)],
                                        {"hero": _Recording(full_obs["hero"]), "meta": _Recording(full_obs["meta"])})
    per_env = sum(t[0].numel() * t.element_size() for t in read.values())
    full = benchmark.host_bytes_per_step(venv)
    venv.info_mode = "minimal"
    assert per_env > 0
    assert full - benchmark.host_bytes_per_step(venv) == per_env * 4


def test_host_bytes_per_step_matches_actual_buffer_sizes():
    cfg = _cfg()
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    env = BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0)
    venv = BrawlSB3VecEnv(env, spec, benchmark.ExampleReward(), info_mode="minimal")
    venv.reset()
    computed = benchmark.host_bytes_per_step(venv)
    # Counted from the declared observation space, not from the buffers host_bytes_per_step sums:
    # one row per env of every group, plus a float32 reward and the two bool done flags.
    expected = venv.num_envs * (
        sum(int(np.prod(s.shape)) * s.dtype.itemsize for s in venv.observation_space.spaces.values())
        + 4 + 1 + 1
    )
    assert computed == expected


# ---- phase_breakdown + _instrumented_phases -------------------------------------------------------

def test_phase_breakdown_covers_all_17_named_phases_with_nonnegative_timings():
    cfg = _cfg()
    result = benchmark.phase_breakdown(cfg, n_envs=4, device="cpu", ticks=2, warmup=1)
    assert set(result.keys()) == set(benchmark._PHASE_METHOD_NAMES)
    assert len(result) == 17
    for name, ms in result.items():
        assert ms >= 0.0, f"{name} had a negative average timing"


def test_instrumented_phases_restores_original_methods_on_exit():
    cfg = _cfg()
    env = BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0)
    original = env._pop_action_buffer
    with benchmark._instrumented_phases(env) as events:
        assert "_pop_action_buffer" in env.__dict__  # monkeypatched onto the instance
        assert env._pop_action_buffer is not original
        assert set(events.keys()) == set(benchmark._PHASE_METHOD_NAMES)
    assert "_pop_action_buffer" not in env.__dict__  # restored to class-level resolution
    assert env._pop_action_buffer.__func__ is original.__func__


def test_instrumented_phases_restores_methods_even_if_the_body_raises():
    cfg = _cfg()
    env = BrawlVecEnv(cfg, n_envs=4, device="cpu", seed=0)
    try:
        with benchmark._instrumented_phases(env):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "_pop_action_buffer" not in env.__dict__


# ---- observation_footprint (purely analytical) -----------------------------------------------------

def test_observation_footprint_with_world_grid_is_larger_than_without():
    cfg = _cfg()
    fp = benchmark.observation_footprint(cfg, n_envs=4)
    assert fp["per_env_bytes_with_world_grid"] > fp["per_env_bytes_without_world_grid"]
    assert fp["world_grid_bytes_per_env"] > 0
    assert fp["total_bytes_with_world_grid"] == fp["per_env_bytes_with_world_grid"] * 4
    assert fp["total_bytes_without_world_grid"] == fp["per_env_bytes_without_world_grid"] * 4


def test_observation_footprint_does_not_construct_an_env():
    """Purely analytical -- must work even for an n_envs no machine could actually allocate,
    proving it never touches BrawlVecEnv/SimState."""
    cfg = _cfg()
    fp = benchmark.observation_footprint(cfg, n_envs=10_000_000)
    assert fp["total_bytes_with_world_grid"] == fp["per_env_bytes_with_world_grid"] * 10_000_000


# ---- _check_gpu / CLI ---------------------------------------------------------------------------

def test_check_gpu_falls_back_to_cpu_when_cuda_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = benchmark._check_gpu("cuda")
    assert device == "cpu"
    assert "WARNING" in capsys.readouterr().out


def test_check_gpu_returns_cpu_unchanged_and_warns(capsys):
    device = benchmark._check_gpu("cpu")
    assert device == "cpu"
    assert "WARNING: benchmarking on CPU" in capsys.readouterr().out


def test_fmt_bytes_handles_none_and_scales_units():
    assert benchmark._fmt_bytes(None) == "n/a"
    assert benchmark._fmt_bytes(500) == "500.0B"
    assert benchmark._fmt_bytes(2048) == "2.0KB"


def test_main_cli_runs_end_to_end_on_cpu_with_a_tiny_sweep(capsys):
    exit_code = benchmark.main([
        "--preset", "configs/presets/debug_tiny.yaml",
        "--device", "cpu",
        "--n-envs-list", "2",
        "--steps", "1", "--warmup", "0",
        "--phase-n-envs", "2", "--phase-steps", "1",
        "--mem-n-envs", "8",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "native BrawlVecEnv.step" in out
    assert "info_mode='minimal'" in out and "info_mode='full'" in out
    assert "per-tick phase breakdown" in out
    assert "observation footprint" in out
