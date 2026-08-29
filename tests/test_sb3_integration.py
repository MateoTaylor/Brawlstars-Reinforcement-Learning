import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from scripts import sb3_smoke

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())
AGENT_OBS_YAML = "configs/agent_obs.yaml"


def _cfg_and_spec():
    cfg = load_config(CONFIGS_DEFAULT, overrides=CONFIGS_TINY)
    spec = obs_select.load_agent_spec(AGENT_OBS_YAML, cfg)
    return cfg, spec


# ---- the script runs start to finish, learn completes without shape/dtype/space errors ------

def test_run_smoke_completes_all_8_steps_on_cpu(tmp_path):
    cfg, spec = _cfg_and_spec()
    results = sb3_smoke.run_smoke(
        cfg, spec, device="cpu", n_envs=4, n_steps=32, batch_size=32,
        total_timesteps=128, verbose=0, save_dir=tmp_path, log=lambda *a: None,
    )
    for key in (
        "check_env", "maskable_ppo_steps_per_sec", "predict_masked",
        "save_load_roundtrip", "plain_ppo_steps_per_sec", "predict_unmasked",
    ):
        assert key in results
    assert results["check_env"] == "OK"
    assert results["predict_masked"] == "OK"
    assert results["save_load_roundtrip"] == "OK"
    assert results["predict_unmasked"] == "OK"
    assert results["maskable_ppo_steps_per_sec"] > 0
    assert results["plain_ppo_steps_per_sec"] > 0


# ---- CUDA variant (skipped if unavailable) ---------------------------------------------------

def test_run_smoke_completes_all_8_steps_on_cuda(tmp_path):
    import torch
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")
    cfg, spec = _cfg_and_spec()
    results = sb3_smoke.run_smoke(
        cfg, spec, device="cuda", n_envs=4, n_steps=32, batch_size=32,
        total_timesteps=128, verbose=0, save_dir=tmp_path, log=lambda *a: None,
    )
    assert results["save_load_roundtrip"] == "OK"


# ---- CLI entry point (main()) works too, not just run_smoke() directly -----------------------

def test_main_cli_runs_end_to_end(capsys):
    argv = [
        "--preset", "configs/presets/debug_tiny.yaml",
        "--n-envs", "4", "--n-steps", "32", "--batch-size", "32",
        "--total-timesteps", "128", "--verbose", "0",
    ]
    exit_code = sb3_smoke.main(argv)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ALL 8 STEPS PASSED" in out
