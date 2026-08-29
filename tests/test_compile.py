import yaml
import torch

from brawl_sim.config import load_config
from brawl_sim.env import BrawlVecEnv

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _tiny_env(n_envs=4, seed=0, compile=False, device="cpu"):
    merged = dict(CONFIGS_TINY)
    merged["engine"] = {"compile": compile}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    return BrawlVecEnv(cfg, n_envs=n_envs, device=device, seed=seed)


def _random_action(n_envs, device, gen=None):
    move = torch.randint(0, 17, (n_envs,), generator=gen, device=device)
    fire = torch.randint(0, 2, (n_envs,), generator=gen, device=device)
    return torch.stack([move, fire], dim=1)


def _assert_obs_close(a, b):
    assert set(a.keys()) == set(b.keys())
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_obs_close(va, vb)
        elif va.is_floating_point():
            assert torch.allclose(va, vb, atol=1e-4), f"float mismatch at {key!r}"
        else:
            assert torch.equal(va, vb), f"exact mismatch at {key!r} (dtype {va.dtype})"


# ---- compile: false is completely unaffected --------------------------------------------

def test_compile_false_uses_the_raw_uncompiled_tick():
    env = _tiny_env(compile=False)
    # bound methods aren't `is`-identical across separate attribute accesses even when they
    # wrap the same function+instance -- `==` is the correct identity check here.
    assert env._tick_fn == env._run_tick  # no wrapping at all -- zero overhead, zero behavior change


def test_compile_false_rollout_behaves_normally():
    from brawl_sim.core import obs_schema
    env = _tiny_env(n_envs=8, compile=False)
    env.reset()
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(_random_action(8, "cpu", gen))
    obs_schema.validate_obs(obs, env.cfg)


# ---- compile: true either matches eager, or fails with a clear, documented error ---------

def test_compile_true_matches_eager_or_fails_with_a_clear_documented_error():
    n_envs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eager_env = _tiny_env(n_envs=n_envs, seed=0, compile=False, device=device)
    eager_env.reset()
    gen = torch.Generator(device=device).manual_seed(0) if device == "cuda" else torch.Generator().manual_seed(0)
    actions = [_random_action(n_envs, device, gen) for _ in range(5)]

    eager_traj = []
    for a in actions:
        eager_traj.append(eager_env.step(a.clone()))

    compiled_env = _tiny_env(n_envs=n_envs, seed=0, compile=True, device=device)
    compiled_env.reset()

    try:
        compiled_traj = []
        for a in actions:
            compiled_traj.append(compiled_env.step(a.clone()))
    except RuntimeError as e:
        msg = str(e)
        # Acceptance: the failure must be CLEAR and DOCUMENTED, not an opaque crash -- assert
        # it names torch.compile, points at the supported fallback, and chains the real cause.
        assert "torch.compile" in msg
        assert "compile: false" in msg
        assert e.__cause__ is not None
        cause_mod = type(e.__cause__).__module__ or ""
        assert cause_mod.startswith("torch._dynamo") or cause_mod.startswith("torch._inductor")
        return

    # torch.compile actually worked on this machine -- rollouts must match eager: exact on
    # int/bool, 1e-4 tolerance on float (per the plan's own acceptance criterion).
    for (obs_e, r_e, t_e, tr_e, _), (obs_c, r_c, t_c, tr_c, _) in zip(eager_traj, compiled_traj):
        assert torch.equal(t_e, t_c)
        assert torch.equal(tr_e, tr_c)
        assert torch.allclose(r_e, r_c, atol=1e-4)
        _assert_obs_close(obs_e, obs_c)


def test_compile_true_construction_itself_never_raises():
    """torch.compile(fn) is lazy -- wrapping must not eagerly compile or fail at __init__/reset
    time, only on the first actual tick call. Constructing + resetting a compile:true env must
    succeed unconditionally (independent of whether compilation itself will later succeed)."""
    env = _tiny_env(n_envs=4, compile=True)
    env.reset()
    assert env._tick_fn is not env._run_tick  # it IS wrapped, just not yet invoked
