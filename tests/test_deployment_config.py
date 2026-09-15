"""`brawl_deployment.config` and the file it loads.

Two things are being pinned. One is the ordinary loader contract every config module in this repo
shares (absent key keeps the default, dotted paths resolve, coercions run). The other is the part
specific to deployment: the rates are **derived** from the training config rather than configured,
and `validate` cross-checks them against the vision config. Those cross-checks catch mismatches
that otherwise produce a running, plausible-looking agent instead of an error.
"""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import yaml

from brawl_deployment import config as C
from brawl_sim.config import load_config
from brawl_vision.config import load_vision_config

SIM = load_config("configs/default.yaml")
VISION = load_vision_config()


def _sim(dt=0.05, action_repeat=5, max_episode_steps=3000):
    return SimpleNamespace(dt=dt, action_repeat=action_repeat,
                           max_episode_steps=max_episode_steps)


# ---------------------------------------------------------------------------------------------
# The shipped file.
# ---------------------------------------------------------------------------------------------

def test_the_shipped_file_loads_and_validates_against_the_real_configs():
    cfg = C.load_deployment_config()
    C.validate(cfg, SIM, VISION)


def test_every_configurable_field_appears_in_the_shipped_yaml():
    """A setting that exists in the dataclass but not in the file is one nobody knows they have.
    This file's whole purpose is to be read, so a new knob has to be documented in it -- the same
    standard configs/vision.yaml holds itself to."""
    raw = yaml.safe_load(open("configs/deployment.yaml").read())
    missing = [dotted for dotted, _, _ in C._DEPLOYMENT_CONFIG_FIELDS
               if C._dget(raw, dotted, default=C._ABSENT) is C._ABSENT]
    assert not missing, f"configs/deployment.yaml never mentions {missing}"


def test_every_dataclass_field_is_reachable_from_yaml():
    """The other direction: a field with no dotted path is silently unconfigurable, which reads as
    "the setting does not work" rather than as "the setting does not exist"."""
    mapped = {name for _, name, _ in C._DEPLOYMENT_CONFIG_FIELDS}
    unmapped = [f.name for f in fields(C.DeploymentConfig) if f.name not in mapped]
    assert not unmapped, f"no YAML path reaches {unmapped}"


def test_an_absent_key_keeps_the_dataclass_default(tmp_path):
    """The failure this prevents is documented in `brawl_sim.config.load_config`: an unreachable
    fallback branch once made every config file silently required to spell out all ~40 fields, so
    adding one broke every previously-valid YAML."""
    path = tmp_path / "partial.yaml"
    path.write_text("policy:\n  device: cuda\n")
    cfg = C.load_deployment_config(path)
    assert cfg.policy_device == "cuda"
    assert cfg.run_checkpoint == C.DeploymentConfig.run_checkpoint


def test_an_empty_file_is_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert C.load_deployment_config(path) == C.DeploymentConfig()


def test_the_attack_tap_arrives_as_floats_and_the_monitor_may_be_null(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("capture:\n  monitor: null\ncontrol:\n  attack_tap: [0.9, 0.5]\n")
    cfg = C.load_deployment_config(path)
    assert cfg.capture_monitor is None
    assert cfg.control_attack_tap == (0.9, 0.5)
    assert all(isinstance(v, float) for v in cfg.control_attack_tap)


# ---------------------------------------------------------------------------------------------
# The rates are derived, not configured.
# ---------------------------------------------------------------------------------------------

def test_the_decision_period_is_derived_and_time_frac_with_it():
    """The two numbers that are NOT settings: 250 ms per decision and 150 s for time_frac, read
    off the training config rather than restated here, which is the point."""
    assert C.agent_seconds(SIM) == pytest.approx(0.25)
    assert C.match_seconds(SIM) == pytest.approx(150.0)


def test_the_shipped_rate_is_12_hz_with_the_decision_period_intact():
    """The perception rate moved off the sim's dt; the decision period did not, and must not."""
    rates = C.resolve_rates(SIM, C.DeploymentConfig())
    assert rates.tick_hz == pytest.approx(12.0)
    assert rates.decision_every == 3
    assert rates.tick_seconds * rates.decision_every == pytest.approx(0.25)
    assert rates.decision_hz == pytest.approx(4.0)


def test_every_admitted_rate_divides_the_decision_period_exactly():
    """The whole reason `resolve_rates` exists. 20/16/12/8 all land on 250 ms; the failure it
    prevents is a rate that *nearly* does and gets rounded."""
    for hz, n in ((20.0, 5), (16.0, 4), (12.0, 3), (8.0, 2), (4.0, 1)):
        r = C.resolve_rates(SIM, C.DeploymentConfig(loop_tick_hz=hz))
        assert r.decision_every == n
        assert r.tick_seconds * n == pytest.approx(C.agent_seconds(SIM), abs=1e-12)


def test_a_rate_that_does_not_divide_the_decision_period_is_refused():
    """13 Hz would be 3.25 ticks per decision. Rounding to 3 steps the policy every 231 ms, on a
    rhythm it never trained on, with nothing on screen to show for it -- so it raises, and the
    message names the rates that do work."""
    with pytest.raises(ValueError, match="not a whole number"):
        C.resolve_rates(SIM, C.DeploymentConfig(loop_tick_hz=13.0))
    with pytest.raises(ValueError, match="16"):        # the message lists the valid neighbours
        C.resolve_rates(SIM, C.DeploymentConfig(loop_tick_hz=17.0))


def test_no_yaml_path_can_override_the_DECISION_rate():
    """The perception rate became a setting deliberately (loop.tick_hz) and is guarded by
    `resolve_rates`. The decision cadence and time_frac's denominator did NOT, and this is what
    stops someone adding a second source of truth for the one thing that must match training."""
    paths = {dotted for dotted, _, _ in C._DEPLOYMENT_CONFIG_FIELDS}
    for forbidden in ("loop.decision_hz", "loop.decision_every", "loop.action_repeat",
                      "loop.agent_seconds", "loop.match_seconds", "policy.action_repeat"):
        assert forbidden not in paths
    assert "loop.tick_hz" in paths


def test_a_slower_capture_rate_is_rejected_against_the_odometry_shift_bound():
    """`odometry.max_shift_tiles` bounds motion between consecutive FRAMES and was sized at 20 Hz
    against Mortis's charged dash. At 12 Hz the dash covers 1.48 tiles/frame; at 8 Hz it is 2.22
    and past the 2.0 bound. Crossing it is not a dropped frame -- `Odometry._cut` bumps the
    segment, which resets the occupancy map and drops every track."""
    C.validate(C.DeploymentConfig(loop_tick_hz=20.0), SIM, VISION)   # 0.89 tiles/frame
    C.validate(C.DeploymentConfig(loop_tick_hz=12.0), SIM, VISION)   # 1.48, inside 2.0
    with pytest.raises(ValueError, match="read as a camera cut"):
        C.validate(C.DeploymentConfig(loop_tick_hz=8.0), SIM, VISION)   # 2.22, past the bound


def test_the_shipped_rate_keeps_real_headroom_on_the_shift_bound():
    """Pin the actual margin rather than only the pass/fail. 12 Hz is 74% of the bound, which is
    the number the choice was made on -- if a future edit takes it past ~90% that is a different
    decision and should look like one."""
    rates = C.resolve_rates(SIM, C.DeploymentConfig())
    dash = C.DASH_TILES_PER_SECOND * rates.tick_seconds
    assert dash == pytest.approx(1.48, abs=0.01)
    assert dash / VISION.odometry_max_shift_tiles < 0.80


def test_the_cross_check_is_skipped_when_the_other_configs_are_not_supplied():
    """`validate(cfg)` alone still checks everything self-contained. Making the sim config
    mandatory would push it into places that legitimately only want to check the file."""
    C.validate(C.DeploymentConfig())


# ---------------------------------------------------------------------------------------------
# validate.
# ---------------------------------------------------------------------------------------------

def test_an_attack_tap_on_the_left_half_is_rejected_by_name():
    """The left half is the movement joystick's half. Attack only fires from the right side, so
    this is not a preference -- a tap there moves the hero instead of firing, silently."""
    cfg = C.DeploymentConfig(control_attack_tap=(0.3, 0.55))
    with pytest.raises(ValueError, match="LEFT half"):
        C.validate(cfg)


@pytest.mark.parametrize("tap", [(1.2, 0.5), (0.9, 1.5), (0.0, 0.5), (0.9, 0.0)])
def test_an_attack_tap_outside_the_unit_square_is_rejected(tap):
    """Catches the commonest transcription error: writing device pixels where a fraction goes."""
    with pytest.raises(ValueError, match="fraction of the screen"):
        C.validate(C.DeploymentConfig(control_attack_tap=tap))


@pytest.mark.parametrize("patch,match", [
    ({"policy_device": "mps"}, "policy.device"),
    ({"control_backend": "sendinput"}, "control.backend"),
    ({"window_check_every_n_ticks": 0}, "at least 1"),
    ({"shadow_ammo_strikes": 0}, "at least 1"),
    ({"safety_capture_stall_seconds": 0.0}, "must be positive"),
    ({"safety_max_match_seconds": -1.0}, "must be positive"),
    ({"telemetry_frame_dump": -1}, "must be >= 0"),
])
def test_validate_rejects(patch, match):
    with pytest.raises(ValueError, match=match):
        C.validate(C.DeploymentConfig(**patch))


def test_an_occlusion_probe_more_than_ten_decisions_apart_is_rejected():
    """The probe costs microseconds, so a large value is never a cost decision -- it is how long
    the agent may keep playing into a covered screen, and ten decisions is a whole engagement."""
    # Ten decisions is `10 * decision_every` ticks, which moves with the tick rate -- 30 at the
    # shipped 12 Hz, 50 at 20 Hz. The bound is in decisions because that is the unit the agent
    # acts in; expressing it in ticks would silently loosen when the tick rate rose.
    C.validate(C.DeploymentConfig(window_check_every_n_ticks=30), _sim())
    with pytest.raises(ValueError, match="ten"):
        C.validate(C.DeploymentConfig(window_check_every_n_ticks=31), _sim())
    C.validate(C.DeploymentConfig(window_check_every_n_ticks=50, loop_tick_hz=20.0), _sim())


def test_the_null_backend_is_a_supported_setting_not_a_test_hook():
    """Running the whole loop and emitting nothing is the mode for the first run of any perception
    or policy change. It has to be reachable from the config file, not only from a test."""
    C.validate(C.DeploymentConfig(control_backend="null"), SIM, VISION)


# ---------------------------------------------------------------------------------------------
# The specific settings the design argued about.
# ---------------------------------------------------------------------------------------------

def test_the_shipped_attack_tap_clears_the_super_and_gadget_buttons():
    """Design 6.8's correction: attack is an AREA and the tap point is chosen for CLEARANCE. Pin
    the actual clearance against the measured button centres, so moving the tap point has to keep
    clearing them -- the failure it prevents (firing a Super uncommanded) is silent."""
    import json
    data = json.loads(open("brawl_deployment/data/control_calibration.json").read())
    w, h = data["screen"]
    cfg = C.load_deployment_config()
    tx, ty = cfg.control_attack_tap[0] * w, cfg.control_attack_tap[1] * h

    for name in ("super", "gadget"):
        b = data["buttons"][name]
        gap = ((tx - b["x"]) ** 2 + (ty - b["y"]) ** 2) ** 0.5
        assert gap > 4 * b["r"], f"the attack tap is only {gap:.0f} px from {name} (r={b['r']})"
    assert tx > w / 2, "the attack tap must be on the right half of the screen"
    assert 0 < tx < w and 0 < ty < h, "the attack tap must be on screen"


# ---------------------------------------------------------------------------------------------
# `--run`: choosing a run by name.
# ---------------------------------------------------------------------------------------------

def _runs(tmp_path, *dirs):
    """A runs/ directory of `(dir name, run.name)` pairs, each with the files a real run has."""
    for dirname, trained_as in dirs:
        run = tmp_path / dirname
        run.mkdir()
        (run / "train.yaml").write_text(yaml.safe_dump({"run": {"name": trained_as}}))
        (run / "best_model.zip").write_bytes(b"")
        (run / "final_model.zip").write_bytes(b"")
    return tmp_path


# The two real runs this was written for: one name is a prefix of the other.
BASE = ("mortis_deploy3-20260911-203322", "mortis_deploy3")
ELITE = ("mortis_deploy3_elite-20260913-015933", "mortis_deploy3_elite")


def test_a_run_resolves_by_the_name_it_was_trained_as(tmp_path):
    runs = _runs(tmp_path, BASE, ELITE)
    assert C.resolve_run("mortis_deploy3", runs).endswith(BASE[0])
    assert C.resolve_run("mortis_deploy3_elite", runs).endswith(ELITE[0])


def test_a_run_resolves_by_directory_name_and_by_path(tmp_path):
    runs = _runs(tmp_path, BASE, ELITE)
    assert C.resolve_run(BASE[0], runs).endswith(BASE[0])
    assert C.resolve_run(str(runs / ELITE[0]), runs).endswith(ELITE[0])


def test_a_name_trained_twice_is_refused_with_both_listed(tmp_path):
    """Resolving to the newest would be a guess about which checkpoint played, in exactly the
    setting -- comparing runs -- where that is the fact being recorded."""
    runs = _runs(tmp_path, BASE, ("mortis_deploy3-20260915-090000", "mortis_deploy3"))
    with pytest.raises(ValueError, match="2 runs were trained as 'mortis_deploy3'") as err:
        C.resolve_run("mortis_deploy3", runs)
    assert BASE[0] in str(err.value) and "mortis_deploy3-20260915-090000" in str(err.value)


def test_a_prefix_is_not_a_name(tmp_path):
    """`mortis_deploy3` must not also mean `mortis_deploy3_elite`, and `mortis` must mean nothing."""
    runs = _runs(tmp_path, ELITE)
    with pytest.raises(ValueError, match="no run called 'mortis_deploy3'") as err:
        C.resolve_run("mortis_deploy3", runs)
    assert "--run mortis_deploy3_elite" in str(err.value), "the error lists what does exist"


def test_a_directory_without_train_yaml_is_not_a_run(tmp_path):
    """`runs/deploy` holds telemetry, not a checkpoint; offering it would fail later and worse."""
    runs = _runs(tmp_path, BASE)
    (runs / "deploy").mkdir()
    with pytest.raises(ValueError, match="no run called 'deploy'"):
        C.resolve_run("deploy", runs)


def _args(**kw):
    base = {"config": "configs/deployment.yaml", "run": None, "checkpoint": None}
    return SimpleNamespace(**{**base, **kw})


def test_deploy_run_takes_the_run_and_checkpoint_from_the_command_line(tmp_path, monkeypatch):
    from scripts import deploy_run

    runs = _runs(tmp_path, BASE, ELITE)
    monkeypatch.setattr(deploy_run, "resolve_run", lambda name: C.resolve_run(name, runs))
    cfg = deploy_run._config_from_args(_args(run="mortis_deploy3", checkpoint="final"))
    assert cfg.run_dir.endswith(BASE[0])
    assert cfg.run_checkpoint == "final_model.zip"
    # Everything else still comes from the file.
    assert cfg.loop_tick_hz == C.load_deployment_config().loop_tick_hz


def test_deploy_run_without_flags_is_the_config_file(monkeypatch):
    from scripts import deploy_run

    shipped = C.load_deployment_config()
    if not (C.REPO_ROOT / shipped.run_dir / shipped.run_checkpoint).is_file():
        pytest.skip("runs/ is gitignored; the configured checkpoint is not on every machine")
    monkeypatch.chdir(C.REPO_ROOT)
    assert deploy_run._config_from_args(_args()) == shipped


def test_deploy_run_names_a_missing_checkpoint_before_building_anything(tmp_path, monkeypatch):
    from scripts import deploy_run

    runs = _runs(tmp_path, BASE)
    monkeypatch.setattr(deploy_run, "resolve_run", lambda name: C.resolve_run(name, runs))
    with pytest.raises(SystemExit, match="no checkpoint .*best_model_v2.zip"):
        deploy_run._config_from_args(_args(run="mortis_deploy3", checkpoint="best_model_v2.zip"))


def test_deploy_run_turns_an_unknown_run_into_a_message_not_a_traceback(tmp_path, monkeypatch):
    from scripts import deploy_run

    runs = _runs(tmp_path, BASE)
    monkeypatch.setattr(deploy_run, "resolve_run", lambda name: C.resolve_run(name, runs))
    with pytest.raises(SystemExit, match="no run called 'elite'"):
        deploy_run._config_from_args(_args(run="elite"))
