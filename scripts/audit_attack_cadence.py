"""Sim-side attack-cadence audit (SIM_OVERHAUL_PLAN.md Phase A, Step A1).

    python scripts/audit_attack_cadence.py --run runs/mortis_deploy3_elite-20260913-015933 \\
        --checkpoint best_model.zip --episodes 200 --tier elite --device cpu

Five numbers that say whether the trained sim policy attacks whenever it can, and where the
structural cadence cap shows up. The cap: Mortis's `attack_cooldown` is 0.35 s and a decision is
`action_repeat * dt` = 0.25 s, so after a dash the next decision boundary lands at 0.25 s with
0.15 s of cooldown left, the attack is illegal, and the earliest legal one is at 0.50 s -- one
dash per 10 ticks, when the weapon itself would allow one per 7. If the agent learned to attack at
every legal opportunity the histogram below sits at 10 ticks and utilization is ~1.0; if it learned
to over-conserve, utilization is low and the intervals spread out. Which of those is true decides
whether Phase A's fire latch (Step A3) is worth building.

The five statistics (`summarize`), all measured at DECISION boundaries, i.e. on the state the
policy's observation was built from:

  1. utilization         P(attacked | attack legal & an enemy in dash reach)
  2. interval_hist       inter-attack interval in TICKS during fights (consecutive decisions with an
                         enemy in reach throughout), a histogram
  3. phasing_loss        share of fight decisions with ammo >= 1 and 0 < attack_cd <= 0.20 s -- the
                         decisions the cooldown alone made illegal, i.e. the cap's footprint
  4. long_dash_waiting   share of fight decisions where the attack was legal, the long dash was
                         charging (attack_idle_t >= 4.0 s) and the agent did NOT attack -- holding
                         fire for the doubled range
  5. ammo_at_first_attack  ammo at the first attack of each fight, a histogram

"Enemy in reach" is any enemy alive, revealed to the hero (bots/perception.visibility row 0) and
within `dash_distance + dash_radius + unit_radius` tiles -- the uncharged dash; a charged one
reaches further, so this is the conservative "certainly could have hit" criterion.

The recorder is called by the driver at each decision boundary with the action the policy chose
(`CadenceRecorder.record(sim, action)`, BEFORE `step()`), not installed as `BrawlVecEnv.tick_hook`:
the hook fires after every sub-tick, mid-decision, and never sees the action. Recording before
`step()` reads exactly the state the observation came from, with no host sync in the sim itself.
One host transfer per decision -- an audit, not the hot path.

Deployment side (Step A2.3):

    python scripts/audit_attack_cadence.py --telemetry runs/deploy/<match>.csv

reads a `deploy_run.py --telemetry` CSV (`brawl_deployment.loop.TickRow` rows) and computes the
SAME five statistics with the SAME `summarize`, so the two sides are compared on identical
definitions rather than on two re-implementations. `telemetry_rows` converts each decision row
into the dict `CadenceRecorder.rows()` would have produced:

    env               0 -- one telemetry file is one loop
    step_count        `ticks_per_decision * i`, i = this decision's index within its match; the
                      deployment decides every `decision_every` perception ticks (3 at 12 Hz) and
                      the sim every `action_repeat` sim ticks (5 at dt 0.05), both 0.25 s, so the
                      mapping puts a 0.50 s attack chain in the SAME 10-tick histogram bin the sim's
                      lands in. Decision INDEX, not wall time: a decision tick that skipped (no hero
                      box, odometry lost) still spent its slot and still advances i, but tick
                      jitter does not smear the bins. A match boundary (a row whose phase is not
                      "playing") restarts i at 0, which is exactly `_segment`'s new-episode rule
                      (step_count <= the previous), so a file spanning several matches is several
                      episodes.
    can_attack        bit 1 of `attack_legal` -- the dash column of the mask the policy got
    attack_cd         `attack_cd_shadow`, the shadow's timer at the decision
    attack_idle_t     `attack_idle_t_shadow`
    ammo              `ammo_shadow`, the shadow's clip as read BEFORE the ammo canary check --
                      pre-resync on a resync row, while the mask on that row is post-resync;
                      harmless, since the resync reseeds a full cooldown and the row is then
                      neither an opportunity nor phased, so no statistic reads its ammo. Written
                      on every decision that reached the bars (hero box or not); a decision row
                      still carrying the `-1.0` sentinel is refused, not passed through as a clip
    attack_col        `attack` -- what was SENT (the shadow refuses an illegal pick to 0)
    enemy_in_reach    `enemy_in_reach`, the same radius off the tracks the policy was handed
    dash_t, long_dash_ready   not recorded on the deployment side and not read by `summarize`;
                      carried as None so the row has the recorder's shape

A sixth, deployment-only statistic: resyncs per minute INSIDE FIGHTS (the shadow's ammo canary
tripping while an enemy is in reach, each costing up to one cooldown of refused attacks) and the
ammo error that tripped each (CV minus shadow, in pips). Report: `runs/audit/cadence_<file stem>.md`,
same format, plus a resync block. Plan section 2 Step A2's table reads the two reports together.

The rates behind the mapping (`deployment_rates`) come from the REPO's `configs/default.yaml` and
`configs/deployment.yaml`, resolved off this file's location so the command works from any CWD.
The loop itself resolves them from the deployed run's env config (`policy.cfg`, i.e. train.yaml's
`env_overrides` applied), which this does not read: a run that overrode `dt` or `action_repeat`
would be mapped here at the defaults. No run has, and the report header prints the values used so
a mismatch is visible rather than silent.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# `python scripts/audit_attack_cadence.py` puts scripts/, not the repo root, on sys.path;
# `brawl_sim` is editable-installed but `brawl_deployment` (the `--telemetry` side) may not be
# in a stale finder, and `--telemetry` is run from wherever the CSV is. Same as deploy_run.py.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brawl_sim.bots import perception  # noqa: E402 -- after the sys.path insert above
from brawl_sim.core import geometry as geo
from brawl_sim.core import hero, obs_select, stats

FIELDS = ("env", "step_count", "can_attack", "attack_cd", "dash_t", "ammo",
          "long_dash_ready", "attack_idle_t", "attack_col", "enemy_in_reach")

PHASING_CD_SECONDS = 0.20   # attack_cd in (0, this] at a decision = "the cooldown alone blocked it"
LONG_DASH_WAIT_SECONDS = 4.0  # attack_idle_t at or past this = "the long dash is (nearly) charged"


class CadenceRecorder:
    """One row per (decision, env). `record(sim, action)` at each decision boundary, before the
    env consumes `action`; `rows()` returns the list of dict rows `summarize` reads."""

    def __init__(self) -> None:
        self._cols: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}

    def record(self, sim, action) -> None:
        state, params, cfg = sim.state, sim.params, sim.cfg
        if not torch.is_tensor(action):
            action = torch.as_tensor(np.asarray(action))
        attack_col = action[:, 1].to(torch.int64)

        mask = hero.action_mask(state, params, cfg)["attack"]
        vis = perception.visibility(state, sim.bank, params, cfg)

        hero_pos = state.ent_pos[:, 0:1, :]
        dist = geo.safe_norm(state.ent_pos - hero_pos, dim=-1)              # (N,E)
        dash_distance = stats.gather_kind(params.dash_distance, state.ent_kind)[:, 0]
        dash_radius = stats.gather_kind(params.dash_radius, state.ent_kind)[:, 0]
        reach = (dash_distance + dash_radius + params.unit_radius).unsqueeze(-1)  # (N,1)
        in_reach = state.ent_alive & vis[:, 0, :] & (dist <= reach)
        in_reach[:, 0] = False
        enemy_in_reach = in_reach.any(dim=-1)

        n = state.ent_alive.shape[0]
        self._push("env", np.arange(n, dtype=np.int64))
        self._push("step_count", state.step_count)
        self._push("can_attack", mask[:, 1])
        self._push("attack_cd", state.ent_attack_cd[:, 0])
        self._push("dash_t", state.ent_dash_t[:, 0])
        self._push("ammo", state.ent_ammo[:, 0])
        self._push("long_dash_ready", hero.long_dash_ready(state, params)[:, 0])
        self._push("attack_idle_t", state.ent_attack_idle_t[:, 0])
        self._push("attack_col", attack_col)
        self._push("enemy_in_reach", enemy_in_reach)

    def _push(self, name: str, value) -> None:
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        self._cols[name].append(np.array(arr, copy=True))

    @property
    def n_decisions(self) -> int:
        return len(self._cols["env"])

    def rows(self) -> list[dict]:
        """Every (decision, env) as a dict, in decision order (env-major within a decision)."""
        if not self._cols["env"]:
            return []
        flat = {name: np.concatenate(chunks) for name, chunks in self._cols.items()}
        n = flat["env"].shape[0]
        return [{name: flat[name][i].item() for name in FIELDS} for i in range(n)]


# ---- statistics ----------------------------------------------------------------------------

def _segment(rows: list[dict]) -> tuple[list[list[dict]], int]:
    """(fights, n_episodes). A fight is a maximal run of consecutive decisions, within one env
    and one episode, that all have an enemy in reach. Episode boundaries are read off
    `step_count`, which resets to 0 with the env: a non-increasing step_count starts a new
    episode (and an episode still in flight when the recording stops counts as started)."""
    by_env: dict[int, list[dict]] = {}
    for row in rows:
        by_env.setdefault(row["env"], []).append(row)

    fights: list[list[dict]] = []
    n_episodes = 0
    for env_rows in by_env.values():
        current: list[dict] = []
        prev_step = None
        for row in env_rows:
            new_episode = prev_step is None or row["step_count"] <= prev_step
            n_episodes += int(new_episode)
            prev_step = row["step_count"]
            if row["enemy_in_reach"] and not new_episode:
                current.append(row)
                continue
            if current:
                fights.append(current)
            current = [row] if row["enemy_in_reach"] else []
        if current:
            fights.append(current)
    return fights, n_episodes


def _attacked(row: dict) -> bool:
    # 1 = dash, 2 = super: both are "used the attack column" for cadence purposes. A super shares
    # the cooldown, so counting it keeps a super-happy policy from reading as over-conserving.
    # 3 = gadget (SIM_OVERHAUL Step G3) is NOT an attack: it has its own 18 s timer, shares no
    # cooldown and spends no ammo, so a gadget decision is an attack opportunity that was passed up.
    return row["attack_col"] in (1, 2)


def _share(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def summarize(rows: list[dict]) -> dict:
    """The five statistics documented at the top of this module, plus the counts behind them."""
    fights, n_episodes = _segment(rows)
    fight_rows = [row for fight in fights for row in fight]

    opportunities = [row for row in rows if row["can_attack"] and row["enemy_in_reach"]]
    taken = [row for row in opportunities if _attacked(row)]

    intervals: Counter = Counter()
    ammo_first: Counter = Counter()
    for fight in fights:
        attack_steps = [row["step_count"] for row in fight if _attacked(row) and row["can_attack"]]
        for earlier, later in zip(attack_steps, attack_steps[1:]):
            intervals[int(later - earlier)] += 1
        first = next((row for row in fight if _attacked(row) and row["can_attack"]), None)
        if first is not None:
            ammo_first[int(np.floor(first["ammo"] + 1e-6))] += 1

    phased = [row for row in fight_rows
              if row["ammo"] >= 1.0 and 0.0 < row["attack_cd"] <= PHASING_CD_SECONDS]
    waiting = [row for row in fight_rows
               if row["can_attack"] and row["attack_idle_t"] >= LONG_DASH_WAIT_SECONDS
               and not _attacked(row)]

    n_ammo = sum(ammo_first.values())
    return {
        "n_rows": len(rows),
        "n_envs": len({row["env"] for row in rows}),
        "n_episodes": n_episodes,
        "n_fights": len(fights),
        "n_fight_decisions": len(fight_rows),
        "utilization": {
            "value": _share(len(taken), len(opportunities)),
            "n_opportunities": len(opportunities), "n_taken": len(taken),
        },
        "interval_hist": dict(sorted(intervals.items())),
        "phasing_loss": {
            "value": _share(len(phased), len(fight_rows)),
            "n_fight_decisions": len(fight_rows), "n_phased": len(phased),
        },
        "long_dash_waiting": {
            "value": _share(len(waiting), len(fight_rows)),
            "n_fight_decisions": len(fight_rows), "n_waiting": len(waiting),
        },
        "ammo_at_first_attack": {
            "hist": dict(sorted(ammo_first.items())),
            "mean": (sum(k * v for k, v in ammo_first.items()) / n_ammo) if n_ammo else None,
            "n_fights_with_attack": n_ammo,
        },
    }


# ---- deployment telemetry (Step A2.3) ---------------------------------------------------------

TELEMETRY_ENV = 0
# `TickRow.attack_legal` on a row that made no decision.
_NO_DECISION = -1


def telemetry_rows(ticks, *, decision_every: int, ticks_per_decision: int) -> list[dict]:
    """`brawl_deployment.loop.TickRow`s (file order) -> the dict rows `summarize` reads, one per
    DECISION, under the mapping in the module docstring. `decision_every` is the loop's
    perception ticks per decision (`Rates.decision_every`), `ticks_per_decision` the sim's
    `action_repeat`; both are handed in rather than read here so a test pins them.

    The decision index is rebuilt from the row stream the way `DeployLoop._ticks_in_match` counts:
    every row with phase "playing" is one tick of its match, any other phase ends it. A file that
    starts mid-match (the telemetry ring is bounded) may be offset by a tick or two from the
    loop's own count; the floor division below keeps consecutive decisions one index apart
    regardless, which is all the intervals need.
    """
    rows: list[dict] = []
    ticks_in_match = 0
    for tick in ticks:
        if tick.phase != "playing":
            ticks_in_match = 0
            continue
        if tick.decision and tick.attack_legal != _NO_DECISION:
            if tick.ammo_shadow < 0.0:
                # The loop writes `ammo_shadow` on every decision that reaches the bars, before the
                # cadence columns; a row with the mask but not the clip is a loop this audit does
                # not know, and `-1.0` would otherwise be summarized as a clip size.
                raise ValueError(f"telemetry row {tick.index}: attack_legal recorded but "
                                 f"ammo_shadow is {tick.ammo_shadow} -- the loop that wrote this "
                                 f"file predates the Step A2 review; play the match again")
            rows.append({
                "env": TELEMETRY_ENV,
                "step_count": int(ticks_per_decision * (ticks_in_match // decision_every)),
                "can_attack": bool(tick.attack_legal & 0b10),
                "attack_cd": float(tick.attack_cd_shadow),
                "dash_t": None,
                "ammo": float(tick.ammo_shadow),
                "long_dash_ready": None,
                "attack_idle_t": float(tick.attack_idle_t_shadow),
                "attack_col": int(tick.attack),
                "enemy_in_reach": bool(tick.enemy_in_reach),
                # Deployment-only, for `summarize_telemetry`; `summarize` ignores them.
                "resync": bool(tick.resync),
                "resync_error": float(tick.resync_error),
                "t": float(tick.t),
            })
        ticks_in_match += 1
    return rows


def summarize_telemetry(rows: list[dict], *, decision_seconds: float,
                        n_resyncs_in_file: int | None = None) -> dict:
    """`summarize` plus the sixth statistic: resyncs per minute inside fights, with the ammo
    error that tripped each. Fight time is counted in decisions (`decision_seconds` each, the
    sim's `action_repeat * dt`), the same clock the intervals are in, not wall time.

    `n_resyncs_in_file` is the count over the RAW ticks (`sum(t.resync for t in ticks)`): the
    canary runs before the brawlers-left read can skip a decision, so a resync can sit on a tick
    that made no decision and is not in `rows`. Without it the count is over decisions only, and
    the report labels it so."""
    summary = summarize(rows)
    fights, _ = _segment(rows)
    in_fights = [row for fight in fights for row in fight if row.get("resync")]
    fight_minutes = summary["n_fight_decisions"] * decision_seconds / 60.0
    on_decisions = sum(1 for row in rows if row.get("resync"))
    summary["resyncs"] = {
        "per_minute_in_fights": (len(in_fights) / fight_minutes) if fight_minutes else None,
        "n_in_fights": len(in_fights),
        "n_total": on_decisions if n_resyncs_in_file is None else int(n_resyncs_in_file),
        "n_total_scope": "on decisions" if n_resyncs_in_file is None else "in the file",
        "fight_minutes": fight_minutes,
        "errors": [round(row["resync_error"], 2) for row in in_fights],
    }
    return summary


def load_telemetry(path):
    """The CSV as `TickRow`s, refusing one written before Step A2 by name: an older file loads
    (every missing column defaults) but carries no mask and no reach, and a report of "no fights"
    from it would read as a finding."""
    import csv

    from brawl_deployment.loop import read_telemetry_csv

    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    if "attack_legal" not in header:
        raise SystemExit(f"{path}: no `attack_legal` column -- telemetry from before Step A2; "
                         f"play the match again with the current loop")
    return read_telemetry_csv(path)


# The repo's configs, off this file's location, the way `brawl_deployment.loop._CONFIGS_DIR`
# finds them: `--telemetry` must not need the repo as CWD.
_CONFIGS_DIR = REPO / "configs"


def deployment_rates() -> tuple[int, int, float]:
    """`(decision_every, ticks_per_decision, decision_seconds)` through the loop's own
    `resolve_rates`, from the REPO DEFAULTS: `configs/default.yaml`'s `dt` / `action_repeat` and
    `configs/deployment.yaml`'s `loop.tick_hz`. The loop resolves the same pair from the deployed
    run's env config (`env_overrides` applied), which this does not read -- see the module
    docstring; the report header prints what was used."""
    from brawl_deployment.config import load_deployment_config, resolve_rates
    from brawl_sim.config import load_config

    sim_cfg = load_config(_CONFIGS_DIR / "default.yaml")
    rates = resolve_rates(sim_cfg, load_deployment_config())
    return int(rates.decision_every), int(sim_cfg.action_repeat), float(rates.agent_seconds)


# ---- report ---------------------------------------------------------------------------------

def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _hist_block(hist: dict, unit: str) -> str:
    if not hist:
        return "(empty)"
    total = sum(hist.values())
    width = max(len(str(k)) for k in hist)
    lines = []
    for key, count in hist.items():
        bar = "#" * int(round(40 * count / total))
        lines.append(f"{str(key).rjust(width)} {unit}  {str(count).rjust(6)}  {bar}")
    return "\n".join(lines)


def render_report(summary: dict, header: dict, title: str = "sim side, Step A1") -> str:
    u, p, w, a = (summary["utilization"], summary["phasing_loss"],
                  summary["long_dash_waiting"], summary["ammo_at_first_attack"])
    lines = [f"# Attack cadence audit ({title})", ""]
    lines += [f"- **{key}**: {value}" for key, value in header.items()]
    lines += [
        "",
        f"Rows: {summary['n_rows']} decisions x envs over {summary['n_envs']} envs, "
        f"{summary['n_episodes']} episodes started, {summary['n_fights']} fights "
        f"({summary['n_fight_decisions']} fight decisions).",
        "",
        "| # | statistic | value | counts |",
        "|---|---|---|---|",
        f"| 1 | utilization P(attack \\| legal & enemy in reach) | {_fmt(u['value'])} | "
        f"{u['n_taken']} / {u['n_opportunities']} |",
        f"| 2 | inter-attack interval, ticks (mode) | "
        f"{max(summary['interval_hist'], key=summary['interval_hist'].get) if summary['interval_hist'] else 'n/a'} | "
        f"{sum(summary['interval_hist'].values())} intervals |",
        f"| 3 | phasing loss (fight decisions with ammo and 0 < cd <= {PHASING_CD_SECONDS:.2f} s) | "
        f"{_fmt(p['value'])} | {p['n_phased']} / {p['n_fight_decisions']} |",
        f"| 4 | long-dash waiting (legal, idle >= {LONG_DASH_WAIT_SECONDS:.1f} s, no attack) | "
        f"{_fmt(w['value'])} | {w['n_waiting']} / {w['n_fight_decisions']} |",
        f"| 5 | ammo at the first attack of a fight (mean) | {_fmt(a['mean'], 2)} | "
        f"{a['n_fights_with_attack']} fights |",
    ]
    r = summary.get("resyncs")
    if r is not None:
        lines += [
            f"| 6 | shadow resyncs per minute inside fights | {_fmt(r['per_minute_in_fights'], 2)} | "
            f"{r['n_in_fights']} in {r['fight_minutes']:.2f} fight-min "
            f"({r['n_total']} {r.get('n_total_scope', 'on decisions')}) |",
        ]
    lines += [
        "",
        "## Inter-attack interval during fights (ticks; 5 ticks = one decision)",
        "",
        "```",
        _hist_block(summary["interval_hist"], "ticks"),
        "```",
        "",
        "## Ammo at the first attack of each fight",
        "",
        "```",
        _hist_block(a["hist"], "ammo"),
        "```",
        "",
    ]
    if r is not None:
        lines += [
            "## Shadow resyncs inside fights (ammo error that tripped each, CV minus shadow, pips)",
            "",
            "```",
            " ".join(f"{err:+.2f}" for err in r["errors"]) if r["errors"] else "(none)",
            "```",
            "",
            "A positive error is an attack the shadow modelled and the game never took (a press "
            "that did not land, or landed after the canary read); each resync reseeds the "
            "cooldown and refuses up to one decision of attacks, which lengthens deploy intervals "
            "without the policy having declined anything.",
            "",
        ]
    lines += [
        "Reading it: a histogram sitting at 10 ticks with utilization near 1.0 means the policy "
        "attacks at every legal decision and the 0.35 s cooldown against the 0.25 s decision "
        "period is the whole cap (plan section 2; Step A3 fixes that). Low utilization or intervals "
        "spread past 10 means the policy itself learned to hold fire.",
        "",
    ]
    return "\n".join(lines)


def write_report(summary: dict, header: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, header), encoding="utf-8")
    return path


# ---- CLI -------------------------------------------------------------------------------------

def _import_watch():
    """scripts/watch.py already knows how to find a run's train.yaml, load its model and apply
    its observation statistics; reuse that rather than restate it. Both entry points, like
    watch.py itself: `python scripts/...` puts scripts/ on sys.path, pytest puts the repo root."""
    try:
        from scripts import watch
    except ModuleNotFoundError:
        import watch  # type: ignore[no-redef]
    return watch


def build_audit_env(tcfg, tier: str | None, n_envs: int, seed: int, device: str):
    """`n_envs` envs pinned to `tier` with autoreset ON (the audit wants many episodes, not one
    frozen match), otherwise built the way scripts/watch.py builds its env."""
    from brawl_sim.config import load_config
    from brawl_sim.env import BrawlVecEnv
    from brawl_sim.training.builder import _resolve, build_spec
    from brawl_sim.training.curriculum import FixedTierHook
    from brawl_sim.training.reward import ShapedReward
    from brawl_sim.wrappers.sb3_vecenv import BrawlSB3VecEnv

    overrides = dict(tcfg.run.env_overrides or {})
    env_cfg = load_config(_resolve(tcfg.run.env_config), overrides=overrides or None)
    agent_spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), env_cfg)
    sim = BrawlVecEnv(
        env_cfg, n_envs=n_envs, device=device, seed=seed,
        reward_fn=ShapedReward(tcfg.reward, track_terms=False),
        spec=build_spec(tcfg), verbose=False, autoreset=True,
    )
    if tier is not None:
        if not tcfg.curriculum.tiers:
            raise ValueError("--tier needs `curriculum.tiers` defined in the training config")
        sim.params_hook = FixedTierHook.uniform(tcfg.curriculum.tiers, sim.device, n_envs, tier)
    venv = BrawlSB3VecEnv(sim, agent_spec, sim.reward_fn, info_mode="episode")
    return sim, venv, env_cfg


def run_audit(model, venv, sim, uses_masks: bool, episodes: int, deterministic: bool = True,
              log_every: int = 0) -> CadenceRecorder:
    recorder = CadenceRecorder()
    obs = venv.reset()
    episodes_done = 0
    decisions = 0
    while episodes_done < episodes:
        kwargs = {"action_masks": venv.action_masks()} if uses_masks else {}
        action, _ = model.predict(obs, deterministic=deterministic, **kwargs)
        recorder.record(sim, action)
        obs, _, dones, _ = venv.step(action)
        episodes_done += int(np.asarray(dones).sum())
        decisions += 1
        if log_every and decisions % log_every == 0:
            print(f"[audit] {decisions} decisions, {episodes_done}/{episodes} episodes", flush=True)
    return recorder


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory, e.g. runs/mortis_deploy3_elite-20260913-015933")
    src.add_argument("--telemetry", help="deployment telemetry CSV (deploy_run.py --telemetry) "
                                         "instead of a sim rollout; see the module docstring")
    p.add_argument("--checkpoint", default="best_model.zip", help="model file inside --run (default: best_model.zip)")
    p.add_argument("--episodes", type=int, default=200, help="episodes to complete before summarizing")
    p.add_argument("--tier", default="elite", help="difficulty tier to pin the bots to; 'none' = brawlers.yaml verbatim")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--out", default=None, help="report path (default: runs/audit/cadence_<run name>.md)")
    p.add_argument("--log-every", type=int, default=200, help="progress line every N decisions (0 = quiet)")
    return p.parse_args(argv)


def main_telemetry(args) -> int:
    path = Path(args.telemetry)
    if not path.is_file():
        print(f"[audit] no such telemetry file: {path}", file=sys.stderr)
        return 1
    ticks = load_telemetry(path)
    decision_every, ticks_per_decision, decision_seconds = deployment_rates()
    rows = telemetry_rows(ticks, decision_every=decision_every,
                          ticks_per_decision=ticks_per_decision)
    summary = summarize_telemetry(rows, decision_seconds=decision_seconds,
                                  n_resyncs_in_file=sum(1 for t in ticks if t.resync))
    header = {
        "telemetry": str(path), "ticks": len(ticks), "decisions": len(rows),
        "decision_every (perception ticks)": decision_every,
        "ticks_per_decision (sim ticks)": ticks_per_decision,
        "decision_seconds": f"{decision_seconds:.2f} s",
        "date": _dt.date.today().isoformat(),
    }
    out = Path(args.out) if args.out else Path("runs") / "audit" / f"cadence_{path.stem}.md"
    title = "deployment side, Step A2"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(summary, header, title), encoding="utf-8")
    print(render_report(summary, header, title))
    print(f"[audit] report -> {out}")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.telemetry:
        return main_telemetry(args)
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        print(f"[audit] no such run directory: {run_dir}", file=sys.stderr)
        return 1
    model_path = run_dir / args.checkpoint
    if not model_path.exists():
        print(f"[audit] no such checkpoint: {model_path}", file=sys.stderr)
        return 1

    watch = _import_watch()
    from brawl_sim.training.config import load_train_config

    train_config = watch.find_train_config(model_path)
    tcfg = load_train_config(train_config, check_holdout=False)   # as scripts/watch.py: no holdout eval here
    tier = None if args.tier == "none" else args.tier
    run_name = getattr(getattr(tcfg, "run", None), "name", None) or run_dir.name
    out = Path(args.out) if args.out else Path("runs") / "audit" / f"cadence_{run_name}.md"

    print(f"[audit] {model_path}  vs  {tier or 'brawlers.yaml (no tier)'} bots  "
          f"[{args.episodes} episodes, {args.n_envs} envs, {args.device}]")
    model, uses_masks = watch.load_model(model_path, tcfg.run.algo, args.device)
    sim, venv, _ = build_audit_env(tcfg, tier, args.n_envs, args.seed, args.device)
    venv = watch.maybe_wrap_vecnormalize(venv, model_path, tcfg, verbose=True)
    # Same refusal scripts/watch.py gives: a checkpoint from before an action-space change (the
    # attack column went 3 -> 4 wide in SIM_OVERHAUL Step G3) otherwise dies inside `predict` on a
    # bare "shape '[-1, 20]' is invalid for input of size 42" from the mask.
    watch._check_spaces(model, venv, train_config)

    recorder = run_audit(model, venv, sim, uses_masks, args.episodes,
                         deterministic=not args.stochastic, log_every=args.log_every)
    summary = summarize(recorder.rows())
    header = {
        "run": str(run_dir), "checkpoint": args.checkpoint, "tier": tier or "none",
        "episodes": args.episodes, "n_envs": args.n_envs, "seed": args.seed,
        "device": args.device, "deterministic": not args.stochastic,
        "action_repeat x dt": f"{sim.cfg.action_repeat} x {sim.cfg.dt} s",
        "hero attack_cooldown": f"{float(sim.params.attack_cooldown[0, 0]):.2f} s",
        "date": _dt.date.today().isoformat(),
    }
    path = write_report(summary, header, out)
    print(render_report(summary, header))
    print(f"[audit] report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
