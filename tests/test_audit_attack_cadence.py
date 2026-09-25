"""Step A1.4 (SIM_OVERHAUL_STEPS.md): the cadence audit's recorder and statistics, exercised by a
hand-scripted policy that attacks at every legal decision with one enemy pinned in dash reach.

This is also the regression pin for the STRUCTURAL cadence cap itself: Mortis's 0.35 s
attack_cooldown against a 0.25 s decision period means the earliest legal dash after a dash is the
second decision boundary, 10 ticks later. The `10` below is a literal on purpose -- Step A3 (the
fire latch) is what changes it, and it must change this test when it does.
"""
from pathlib import Path

import pytest
import torch
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import hero
from brawl_sim.env import BrawlVecEnv
from scripts.audit_attack_cadence import CadenceRecorder, FIELDS, render_report, summarize

CONFIGS_DEFAULT = "configs/default.yaml"
CONFIGS_TINY = yaml.safe_load(open("configs/presets/debug_tiny.yaml").read())


def _env(n_envs=1, seed=0):
    cfg = load_config(CONFIGS_DEFAULT, overrides=CONFIGS_TINY)
    return BrawlVecEnv(cfg, n_envs=n_envs, device="cpu", seed=seed, verbose=False)


def _pin_world(env) -> None:
    """One enemy 1.2 tiles east of the hero, the other dead, both survivors immortal and the
    hero's clip full -- so the ONLY thing deciding whether an attack is legal is the cooldown /
    dash timer pair this audit exists to measure. Re-applied every decision: a dash moves the
    hero and costs ammo, and neither may leak into the cadence."""
    st = env.state
    st.ent_pos[:, 0] = torch.tensor([10.0, 10.0])
    st.ent_pos[:, 1] = torch.tensor([11.2, 10.0])
    st.ent_vel.zero_()
    st.ent_alive[:, 2] = False
    st.ent_hp[:, 2] = 0.0
    st.ent_max_hp[:, :2] = 1.0e6
    st.ent_hp[:, :2] = 1.0e6
    st.ent_ammo[:, 0] = 3.0


def _scripted_rollout(n_decisions=40):
    env = _env()
    env.reset()
    bots_idle = torch.zeros(1, env.cfg.n_entities, 2, dtype=torch.int64)
    bots_idle[:, 0, 0] = -1  # no override for the hero slot
    recorder = CadenceRecorder()
    for _ in range(n_decisions):
        _pin_world(env)
        legal = bool(hero.action_mask(env.state, env.params, env.cfg)["attack"][0, 1])
        action = torch.tensor([[1, 1 if legal else 0]], dtype=torch.int64)  # move bin 1, attack if legal
        recorder.record(env, action)
        env.step(action, bots_idle)
    return env, recorder


def test_recorder_holds_one_row_per_decision_and_env_with_every_field():
    env = _env(n_envs=3)
    env.reset()
    recorder = CadenceRecorder()
    idle = torch.zeros(3, 2, dtype=torch.int64)
    for _ in range(4):
        recorder.record(env, idle)
        env.step(idle)
    rows = recorder.rows()
    assert recorder.n_decisions == 4 and len(rows) == 12
    assert all(set(row) == set(FIELDS) for row in rows)
    assert [row["env"] for row in rows[:3]] == [0, 1, 2]
    assert [row["step_count"] for row in rows[::3]] == [0, 5, 10, 15]  # ticks, action_repeat 5


def test_scripted_policy_attacks_at_exactly_the_structural_cap():
    env, recorder = _scripted_rollout(n_decisions=40)
    rows = recorder.rows()
    assert all(row["enemy_in_reach"] for row in rows), "the pinned enemy must be in reach throughout"

    summary = summarize(rows)
    assert summary["utilization"]["value"] == 1.0
    assert summary["utilization"]["n_opportunities"] == 20  # every other decision is legal
    # The cap: one dash per 10 ticks (0.50 s), never 5, even though the cooldown is 7 ticks.
    assert set(summary["interval_hist"]) == {10}
    assert summary["interval_hist"][10] == 19
    # ...and its footprint: the decision after every dash has 0.15 s of cooldown left.
    assert summary["phasing_loss"]["value"] == pytest.approx(0.5)
    assert summary["long_dash_waiting"]["value"] == 0.0
    assert summary["ammo_at_first_attack"]["hist"] == {3: 1}
    assert summary["n_fights"] == 1 and summary["n_episodes"] == 1


def test_summarize_segments_fights_by_reach_and_episode():
    """Hand-built rows: two envs; env 0 has one fight interrupted by a reset (step_count back to
    0), env 1 never has an enemy in reach. Intervals never span the reset."""
    def row(env, step, in_reach, attacked, legal=True, cd=0.0, ammo=3.0, idle=0.0):
        return {"env": env, "step_count": step, "can_attack": legal, "attack_cd": cd,
                "dash_t": 0.0, "ammo": ammo, "long_dash_ready": idle >= 4.5,
                "attack_idle_t": idle, "attack_col": 1 if attacked else 0,
                "enemy_in_reach": in_reach}
    rows = [
        row(0, 0, True, True), row(1, 0, False, False),
        row(0, 5, True, False, legal=False, cd=0.15), row(1, 5, False, False),
        row(0, 10, True, True), row(1, 10, False, False),
        row(0, 0, True, True, ammo=2.0), row(1, 15, False, False),     # env 0 reset here
        row(0, 5, True, False, legal=True, idle=4.2), row(1, 20, False, False),  # held fire, charging
    ]
    s = summarize(rows)
    assert s["n_envs"] == 2 and s["n_episodes"] == 3 and s["n_fights"] == 2
    assert s["interval_hist"] == {10: 1}                 # 0 -> 10 in the first fight only
    assert s["utilization"] == {"value": pytest.approx(3 / 4), "n_opportunities": 4, "n_taken": 3}
    assert s["phasing_loss"]["n_phased"] == 1
    assert s["long_dash_waiting"]["n_waiting"] == 1
    assert s["ammo_at_first_attack"]["hist"] == {2: 1, 3: 1}
    text = render_report(s, {"run": "hand-built"})
    assert "| 1 | utilization" in text and "10 ticks" in text


def test_a_gadget_decision_is_not_counted_as_an_attack():
    """Attack-column value 3 is the gadget (SIM_OVERHAUL Step G3): its own timer, no cooldown, no
    ammo. `>= 1` would have scored it as an attack taken; a super (2) still counts."""
    def row(step, col):
        return {"env": 0, "step_count": step, "can_attack": True, "attack_cd": 0.0, "dash_t": 0.0,
                "ammo": 3.0, "long_dash_ready": False, "attack_idle_t": 0.0, "attack_col": col,
                "enemy_in_reach": True}
    s = summarize([row(0, 1), row(5, 3), row(10, 2), row(15, 0)])
    assert s["utilization"] == {"value": pytest.approx(2 / 4), "n_opportunities": 4, "n_taken": 2}
    assert s["interval_hist"] == {10: 1}          # attack at 0, super at 10; the gadget at 5 is not one


# ---- the pre-gadget checkpoint refusal (SIM_OVERHAUL Step G3 review) ----------------------------
# The sim's attack column went 3 -> 4 wide, so every checkpoint trained before it carries a
# MultiDiscrete([17, 3]) head. Both sim-side checkpoint consumers (scripts/watch.py and this
# script's --run mode) must say so, not die inside `predict` on a 20-vs-21 mask shape. The widths
# are literals: built from `cfg.action_nvec` they would follow the next widening silently.

def _spaces_stub(attack_width, obs_dim=7):
    import numpy as np
    from gymnasium import spaces
    from types import SimpleNamespace

    return SimpleNamespace(
        observation_space=spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32),
        action_space=spaces.MultiDiscrete([17, attack_width]),
    )


def test_watch_check_spaces_refuses_a_pre_gadget_checkpoint_and_passes_a_current_one():
    from scripts import watch

    with pytest.raises(SystemExit, match="action space mismatch"):
        watch._check_spaces(_spaces_stub(3), _spaces_stub(4), Path("train.yaml"))
    # Equal spaces: no refusal. And the OBSERVATION refusal still wins when both differ, because
    # a wrong train.yaml is the likelier cause and the one the operator can fix.
    assert watch._check_spaces(_spaces_stub(4), _spaces_stub(4), Path("train.yaml")) is None
    with pytest.raises(SystemExit, match="observation space mismatch"):
        watch._check_spaces(_spaces_stub(3, obs_dim=9), _spaces_stub(4), Path("train.yaml"))


def test_the_sim_audit_refuses_a_pre_gadget_checkpoint_before_the_first_predict(tmp_path, monkeypatch):
    """`main --run` with everything heavy stubbed out: the model is a (17, 3) checkpoint, the env
    a (17, 4) build. The refusal must come from `main` itself, before `run_audit` is reached."""
    from types import SimpleNamespace

    import scripts.audit_attack_cadence as audit
    from scripts import watch

    (tmp_path / "best_model.zip").write_bytes(b"")
    tcfg = SimpleNamespace(run=SimpleNamespace(name="stub", algo="maskable_ppo", env_overrides=None))
    fake_watch = SimpleNamespace(
        find_train_config=lambda model_path: tmp_path / "train.yaml",
        load_model=lambda model_path, algo, device: (_spaces_stub(3), True),
        maybe_wrap_vecnormalize=lambda venv, model_path, tcfg, verbose=False: venv,
        _check_spaces=watch._check_spaces,
    )
    monkeypatch.setattr(audit, "_import_watch", lambda: fake_watch)
    monkeypatch.setattr("brawl_sim.training.config.load_train_config", lambda path, **kw: tcfg)
    sim = SimpleNamespace(cfg=SimpleNamespace(action_repeat=5, dt=0.05))
    monkeypatch.setattr(audit, "build_audit_env", lambda *a, **k: (sim, _spaces_stub(4), None))

    def _must_not_run(*a, **k):
        raise AssertionError("run_audit was reached with a mismatched checkpoint")

    monkeypatch.setattr(audit, "run_audit", _must_not_run)

    with pytest.raises(SystemExit, match="action space mismatch"):
        audit.main(["--run", str(tmp_path), "--out", str(tmp_path / "report.md")])
    assert not (tmp_path / "report.md").exists()


def test_watch_and_the_audit_load_an_archived_run_without_the_holdout_check(tmp_path, monkeypatch):
    """Neither script evaluates on the holdout maps, so neither may refuse an archived run because
    one of its `eval.holdout_maps` has since left the map registry: the opt-out
    `DeployedPolicy.from_run` already takes (Step M4 review). `scripts/train.py --resume` stays
    strict, because it re-runs the holdout eval. A run whose TRAINING map left the registry is
    still refused, by `BrawlVecEnv`'s own validation."""
    from types import SimpleNamespace

    import scripts.audit_attack_cadence as audit
    from scripts import watch

    class _Loaded(Exception):
        pass

    seen = []

    def _load(path, **kw):
        seen.append(kw)
        raise _Loaded

    (tmp_path / "best_model.zip").write_bytes(b"")
    monkeypatch.setattr(watch, "load_train_config", _load)
    with pytest.raises(_Loaded):
        watch.main([str(tmp_path / "best_model.zip"), "--train-config", str(tmp_path / "train.yaml"),
                    "--no-view"])

    monkeypatch.setattr(audit, "_import_watch",
                        lambda: SimpleNamespace(find_train_config=lambda model_path: tmp_path / "train.yaml"))
    monkeypatch.setattr("brawl_sim.training.config.load_train_config", _load)
    with pytest.raises(_Loaded):
        audit.main(["--run", str(tmp_path), "--out", str(tmp_path / "report.md")])

    assert seen == [{"check_holdout": False}, {"check_holdout": False}]


def test_summarize_on_no_rows_reports_nothing_rather_than_dividing_by_zero():
    s = summarize([])
    assert s["utilization"]["value"] is None and s["interval_hist"] == {}
    assert "n/a" in render_report(s, {})


# -- Step A2.3: the same statistics from deployment telemetry ---------------------------------
#
# Synthetic `TickRow`s at the shipped rates: 12 Hz perception, a decision every 3 ticks, and the
# sim's 5 ticks per decision. The chain below attacks on every legal decision and the shadow makes
# every other one illegal (0.15 s of cooldown left, plan section 1.1), which is the 0.50 s chain.
# Every expected number is a literal.

DECISION_EVERY = 3        # perception ticks per decision at 12 Hz
TICKS_PER_DECISION = 5    # the sim's action_repeat
DECISION_SECONDS = 0.25


def _tick(index, phase="playing", **kw):
    from brawl_deployment.loop import TickRow

    return TickRow(index=index, t=index / 12.0, grab_ms=1.0, phase=phase, **kw)


def _match(start, n_decisions, *, resync_at=(), skip_at=(), in_reach=True):
    """One match's rows: `n_decisions` decision slots, three ticks each. A slot in `skip_at`
    is a decision tick that made no decision (no hero box); one in `resync_at` tripped the
    canary with a +1.0 pip error before its mask was read."""
    ticks = []
    index = start
    for i in range(n_decisions):
        legal = i % 2 == 0
        if i in skip_at:
            ticks.append(_tick(index, note="no hero box"))
        else:
            ticks.append(_tick(
                index, decision=True, move_bin=1, attack=1 if legal else 0,
                # Bit 3 is a charged gadget, set on real rows since SIM_OVERHAUL Step G5.
                attack_legal=0b1011 if legal else 0b1001,
                attack_cd_shadow=0.0 if legal else 0.15, attack_idle_t_shadow=0.0,
                ammo_shadow=3.0, ammo_cv=3.0, enemy_in_reach=in_reach,
                resync=i in resync_at, resync_error=1.0 if i in resync_at else 0.0))
        index += 1
        for _ in range(DECISION_EVERY - 1):
            ticks.append(_tick(index))
            index += 1
    return ticks


def _rows(ticks):
    from scripts.audit_attack_cadence import telemetry_rows

    return telemetry_rows(ticks, decision_every=DECISION_EVERY,
                          ticks_per_decision=TICKS_PER_DECISION)


def _write_csv(ticks, path):
    """What `DeployLoop.write_csv` writes: `fields(TickRow)` as the header, `asdict` per row."""
    import csv
    from dataclasses import asdict, fields

    from brawl_deployment.loop import TickRow

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, [f.name for f in fields(TickRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in ticks)


def test_a_half_second_chain_in_telemetry_lands_in_the_ten_tick_bin():
    from scripts.audit_attack_cadence import summarize_telemetry

    ticks = [_tick(0, phase="waiting")] + _match(1, 40, resync_at=(7,))
    rows = _rows(ticks)
    assert len(rows) == 40
    assert [row["step_count"] for row in rows[:4]] == [0, 5, 10, 15]
    assert [row["can_attack"] for row in rows[:4]] == [True, False, True, False]
    assert set(rows[0]) >= set(FIELDS)

    s = summarize_telemetry(rows, decision_seconds=DECISION_SECONDS)
    assert s["utilization"] == {"value": 1.0, "n_opportunities": 20, "n_taken": 20}
    assert s["interval_hist"] == {10: 19}
    assert s["phasing_loss"]["value"] == pytest.approx(0.5)
    assert s["long_dash_waiting"]["value"] == 0.0
    assert s["ammo_at_first_attack"]["hist"] == {3: 1}
    assert s["n_fights"] == 1 and s["n_episodes"] == 1
    # 40 fight decisions = 10 s = 1/6 min; one resync inside them.
    assert s["resyncs"]["per_minute_in_fights"] == pytest.approx(6.0)
    assert s["resyncs"]["errors"] == [1.0]
    assert s["resyncs"]["n_in_fights"] == 1 and s["resyncs"]["n_total"] == 1


def test_a_skipped_decision_slot_still_advances_the_index_and_a_new_match_restarts_it():
    """A no-decision tick is 0.25 s the loop spent, so the attacks either side of it stay 10
    ticks apart; a row out of `playing` ends the match and the next one starts at step 0, which
    is what `_segment` reads as a new episode."""
    ticks = (_match(0, 6, skip_at=(3,))
             + [_tick(18, phase="waiting")]
             + _match(19, 4))
    rows = _rows(ticks)
    assert [row["step_count"] for row in rows] == [0, 5, 10, 20, 25, 0, 5, 10, 15]
    s = summarize(rows)
    assert s["n_episodes"] == 2 and s["n_fights"] == 2
    assert s["interval_hist"] == {10: 3}      # 0-10, 10-20 across the skip; 0-10 in match two
    assert s["utilization"]["n_opportunities"] == 5


def test_the_telemetry_cli_writes_the_same_report_with_a_sixth_row(tmp_path):
    """Through the CSV and the real rates (`configs/default.yaml`, `configs/deployment.yaml`,
    read relative to the repo root pytest runs from)."""
    from scripts.audit_attack_cadence import main

    path = tmp_path / "match1.csv"
    _write_csv([_tick(0, phase="waiting")] + _match(1, 40, resync_at=(7,)), path)
    out = tmp_path / "report.md"
    assert main(["--telemetry", str(path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Attack cadence audit (deployment side, Step A2)")
    assert "| 1 | utilization P(attack \\| legal & enemy in reach) | 1.000 | 20 / 20 |" in text
    assert "| 2 | inter-attack interval, ticks (mode) | 10 | 19 intervals |" in text
    assert "| 6 | shadow resyncs per minute inside fights | 6.00 | 1 in 0.17 fight-min (1 in the file) |" in text
    assert "+1.00" in text


def test_the_telemetry_cli_refuses_a_file_from_before_step_a2(tmp_path):
    from scripts.audit_attack_cadence import load_telemetry

    path = tmp_path / "old.csv"
    path.write_text("index,t,grab_ms,phase,decision,attack\n0,0.0,1.0,playing,True,1\n",
                    encoding="utf-8")
    with pytest.raises(SystemExit, match="before Step A2"):
        load_telemetry(path)


def test_a_missing_telemetry_file_is_a_clear_refusal(tmp_path, capsys):
    from scripts.audit_attack_cadence import main

    assert main(["--telemetry", str(tmp_path / "nope.csv")]) == 1
    assert "no such telemetry file" in capsys.readouterr().err


# -- Step A2 review ----------------------------------------------------------------------------

def test_the_resync_count_covers_the_file_when_the_raw_ticks_are_handed_in():
    """The canary runs before the brawlers-left read can skip a decision, so a resync can sit on
    a decision TICK that made no decision; it is in the CSV but not in the audit's rows. Handed
    the raw count, the sixth row says "in the file"; without it, "on decisions"."""
    from scripts.audit_attack_cadence import render_report, summarize_telemetry

    ticks = _match(0, 8, resync_at=(2,))
    skipped = 5 * DECISION_EVERY                          # slot 5's decision tick
    ticks[skipped] = _tick(skipped, note="no brawlers-left read", resync=True, resync_error=-1.0)
    rows = _rows(ticks)
    assert len(rows) == 7

    n_in_file = sum(1 for t in ticks if t.resync)
    assert n_in_file == 2
    s = summarize_telemetry(rows, decision_seconds=DECISION_SECONDS, n_resyncs_in_file=n_in_file)
    assert s["resyncs"]["n_in_fights"] == 1 and s["resyncs"]["errors"] == [1.0]
    assert s["resyncs"]["n_total"] == 2 and s["resyncs"]["n_total_scope"] == "in the file"
    assert "| 1 in 0.03 fight-min (2 in the file) |" in render_report(s, {}, "t")

    s = summarize_telemetry(rows, decision_seconds=DECISION_SECONDS)
    assert s["resyncs"]["n_total"] == 1 and s["resyncs"]["n_total_scope"] == "on decisions"
    assert "(1 on decisions) |" in render_report(s, {}, "t")


def test_a_decision_row_with_the_ammo_sentinel_is_refused_not_summarized_as_a_clip():
    """`ammo_shadow` is the audit's `ammo`; `-1.0` beside a valid mask is a row the current loop
    never writes, and passing it through would put a -1 bin in `ammo_at_first_attack`."""
    ticks = _match(0, 4)
    ticks[DECISION_EVERY].ammo_shadow = -1.0              # slot 1's decision row
    with pytest.raises(ValueError, match="row 3: attack_legal recorded but ammo_shadow is -1.0"):
        _rows(ticks)


def test_deployment_rates_come_from_the_repo_configs_from_any_cwd(tmp_path, monkeypatch):
    """`--telemetry` is run from wherever the CSV is; the rates are the repo's defaults
    (12 Hz / 0.25 s decisions, action_repeat 5 at dt 0.05), found off the script's own path."""
    from scripts.audit_attack_cadence import deployment_rates

    monkeypatch.chdir(tmp_path)
    assert deployment_rates() == (3, 5, 0.25)
