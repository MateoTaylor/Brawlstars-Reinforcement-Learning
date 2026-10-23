import pytest
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

def test_main_cli_runs_all_8_steps_end_to_end(tmp_path, capsys, monkeypatch):
    """The CLI (`main()`) and `run_smoke()` in one run: `main` hands its arguments to the real
    `run_smoke`, whose results are kept and checked here, and then prints the banner. Until
    2026-10-20 two tests ran this smoke once each, at 128 timesteps; it is now one 16-step
    rollout (2 envs x 8 steps) and one update, which reaches every step. The pass-through also
    points the save/load round trip at `tmp_path`: the CLI passes no `save_dir`, so
    `run_smoke` would leave a model zip in a fresh system temp dir on every run."""
    real_run_smoke = sb3_smoke.run_smoke
    kept = {}

    def keep(*args, **kwargs):
        kwargs.setdefault("save_dir", tmp_path)
        kept["results"] = real_run_smoke(*args, **kwargs)
        return kept["results"]

    monkeypatch.setattr(sb3_smoke, "run_smoke", keep)
    argv = [
        "--preset", "configs/presets/debug_tiny.yaml",
        "--n-envs", "2", "--n-steps", "8", "--batch-size", "16",
        "--total-timesteps", "16", "--verbose", "0",
    ]
    assert sb3_smoke.main(argv) == 0
    assert "ALL 8 STEPS PASSED" in capsys.readouterr().out
    assert (tmp_path / "sb3_smoke_maskable_ppo.zip").exists()
    results = kept["results"]
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

@pytest.mark.gpu
def test_run_smoke_completes_all_8_steps_on_cuda(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg, spec = _cfg_and_spec()
    results = sb3_smoke.run_smoke(
        cfg, spec, device="cuda", n_envs=2, n_steps=8, batch_size=16,
        total_timesteps=16, verbose=0, save_dir=tmp_path, log=lambda *a: None,
    )
    assert results["save_load_roundtrip"] == "OK"
