"""`DeploymentConfig`: the tunables of the live loop, loaded from `configs/deployment.yaml`.

Mirrors `brawl_vision/config.py`, which mirrors `brawl_sim/config.py`: a frozen dataclass of
static Python values, a `(dotted YAML path, field name, coercion)` table, a loader where an absent
key keeps the dataclass default, and a separate `validate`. Go read `brawl_sim/config.py` when in
doubt about a convention here.

**Settings here; measurements in `brawl_deployment/data/*.json`.** Same split the vision package
documents. The joystick anchor, its saturation radius and the button centres are outputs of
`scripts/deploy_calibrate.py` and are meaningless to hand-tune; the numbers below are ones where a
different value is a different judgment call.

**What is deliberately NOT configurable, and why that matters more than what is.**

- **The DECISION period.** `cfg.dt * cfg.action_repeat` = 250 ms, from the training config the
  checkpoint was produced under. This is the one rate that has to match training exactly: the
  policy chose its actions on that rhythm and the failure mode of a different one is an agent
  stepping at a cadence it never saw while every number on screen still looks plausible.
  **The PERCEPTION period is a setting** (`loop.tick_hz`) and always was, in the only sense that
  matters -- nothing in the checkpoint knows or cares how often we refresh the world between
  decisions. It was pinned to `cfg.dt` here originally, and that conflated two different
  questions; `resolve_rates()` now answers them separately and refuses any pairing where the
  decision period would drift off the trained one, or where a Mortis dash would cross
  `odometry.max_shift_tiles` (a bound whose own comment says it is per FRAME and sized at 20 Hz,
  and says to raise it if the capture rate drops).
- **`time_frac`'s denominator.** `cfg.max_episode_steps * cfg.dt`, from the same place, for the
  same reason (§6.5).
- **The match-gate refinement.** `MatchState.refine()` always runs at startup; radii do not
  transfer across frame sources and skipping it is how a 0.98 becomes a 0.08.
- **The capture viewport.** Read from the shipped camera model by `capture.calibrated_viewport()`.
- **The monitor index.** Resolved from the emulator window by `window.monitor_index_for()`.
  `capture.monitor` exists only as an override for the case where that resolution is wrong, and
  its default of `null` means "ask the window".
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "deployment.yaml"

_MISSING = object()
_ABSENT = object()


@dataclass(frozen=True)
class DeploymentConfig:
    # --- which policy plays ---------------------------------------------------------------
    # The run directory carries its own `train.yaml`, which names the env config and the agent_obs
    # spec. `policy.DeployedPolicy.from_run` reads those rather than taking them here -- a spec
    # path in this file could disagree with the one the checkpoint was trained against, and that
    # disagreement is exactly the silent-wrong-column failure the guards in `policy.py` exist to
    # prevent.
    run_dir: str = "runs/mortis_deploy-20260907-041522"
    run_checkpoint: str = "best_model.zip"
    # MEASURED (§6.7): 0.96 ms on CPU against 7.56 ms on CUDA for one batch-1 decision. CPU wins
    # 8x at this size and leaves all 16 GB of VRAM to the detectors, which is where it is scarce.
    policy_device: str = "cpu"
    # Deterministic at inference. The stochastic policy is a training-time exploration device; a
    # deployed agent sampling from its own action distribution is adding noise to a decision it
    # already made, and makes every failure unreproducible.
    policy_deterministic: bool = True

    # --- the emulator window (§6.9) --------------------------------------------------------
    window_exe: str = "HD-Player.exe"
    window_require_fullscreen: bool = True
    window_check_occlusion: bool = True
    # Ticks between occlusion probes: about once a second, which at `loop.tick_hz` = 12 is 12.
    # Five `WindowFromPoint` calls are microseconds, so this is not a cost tuning -- it is how long
    # the agent may keep playing into a covered screen, and a second of that is recoverable. It is
    # expressed in ticks rather than seconds because the check runs on the tick loop; if the tick
    # rate moves, this should move with it.
    window_check_every_n_ticks: int = 12

    # --- capture ---------------------------------------------------------------------------
    # `null` means "the monitor the emulator window is on". An explicit index is an override for
    # when that resolution is wrong, not the normal path.
    capture_monitor: int | None = None

    # --- control ---------------------------------------------------------------------------
    # "adb" drives the emulator; "null" runs the whole loop and emits nothing, which is the mode
    # to use for the first run of any change to perception or the policy.
    control_backend: str = "adb"
    # `null` asks `find_adb_serial` to read the single running instance out of bluestacks.conf.
    control_adb_instance: str | None = None
    # Where to tap to fire, as a fraction of the DEVICE screen. A SETTING, not a measurement, and
    # this is the correction from §6.8: attack is an AREA, not a button. Any right-side tap that
    # misses the Super and gadget buttons fires, and the attack stick floats to the touch point --
    # so this is chosen for CLEARANCE, the same way §4.3 chose the joystick anchor, and pointing
    # it at the button sprite would be a downgrade. At 1920x1080 this is (1690, 594): ~330 px from
    # gadget, ~450 px from super, above the button cluster and inside the right edge.
    # UNVERIFIED IN A REAL MATCH -- `scripts/deploy_calibrate.py` confirms it by tapping once and
    # asserting `read_ammo` decrements, which is the venue-independent check.
    control_attack_tap: tuple[float, float] = (0.88, 0.55)

    # --- loop rates (§6.13) -----------------------------------------------------------------
    # Perception ticks per second. A SETTING, unlike the decision period, which is derived and
    # fixed at cfg.dt * cfg.action_repeat = 250 ms. `resolve_rates` allows only values that divide
    # that period into a whole number of ticks, so 20 / 16 / 12 / 8 Hz are the candidates and
    # anything else raises rather than quietly stretching the agent's step.
    #
    # 12, not the 20 the sim ticks at, and this is a MEASURED choice (§6.13). At 20 Hz the loop
    # does not fit its own period: 27.1 ms of stage work per plain tick and 34.7 per decision
    # tick, plus a ~24 ms live grab, against 50 ms. The pacer does not sleep on an overrun, so the
    # result is not a slow loop but a JITTERING one -- ~51 ms plain, ~59 ms decision, and a
    # decision window ~263 ms instead of the trained 250. At 12 Hz (83.3 ms) both fit with ~20 ms
    # to spare even on the worst measured frame, the decision window lands on 250 ms exactly, and
    # the spare covers this machine being used for other things at the same time.
    #
    # What it costs, both halves measured rather than asserted (§6.13):
    #
    #  1. A Mortis charged dash moves 1.48 tiles per frame instead of 0.89, against
    #     odometry.max_shift_tiles = 2.0 -- 1.35x headroom, down from 2.2x. Crossing that bound is
    #     not a dropped frame; Odometry._cut resets the occupancy map and drops every track.
    #  2. Projectiles get the two looks a velocity needs on 21% of shots, against 44% at 20 Hz and
    #     33% at 16. That sounds worse than it is: stratified by time on screen, shots lasting
    #     >=200 ms are 100% covered at ALL of 12/16/20 Hz, shots under 100 ms are near-hopeless at
    #     all three (17% even at 20 Hz), and the entire difference is the 100-200 ms band, which
    #     goes 99% -> 42%. Those are the ~20 shots per 25 s of combat this rate actually gives up.
    #
    # Raise to 16 if a live run shows the budget is kinder than the estimate, or if dodging looks
    # bad in that middle band; do not raise to 20 without re-measuring, and do not lower to 8 --
    # that is 2.22 tiles per dash frame and every dash reads as a scene cut.
    loop_tick_hz: float = 12.0

    # --- shadow-state desync (§6.3) --------------------------------------------------------
    # Consecutive ammo mismatches before a resync. Three, not one: a single mismatch is as likely
    # to be a misread pip bar mid-reload as a real desync, and resyncing on it would reseed the
    # unobservable timers conservatively for no reason.
    shadow_ammo_strikes: int = 3
    # Resyncs inside one match before the run is called lost. Sustained desync means our actions
    # are not landing, and an agent whose model of itself is wrong is worse than a stopped one.
    shadow_max_resyncs: int = 5

    # --- safety backstops ------------------------------------------------------------------
    # The match gate is the primary interlock; this is the backstop for the gate itself being
    # wrong. Generous on purpose -- a Showdown match runs ~150 s and this should never be what
    # ends one.
    safety_max_match_seconds: float = 300.0
    # A gap this long between frames means capture has stalled, and the odometry shift bound
    # (per FRAME, not per second) is already violated. Release and stop.
    safety_capture_stall_seconds: float = 1.0

    # --- telemetry -------------------------------------------------------------------------
    # Scalars only, append-only, one row per decision. Never frames: at 20 Hz a 1440p BGR history
    # is ~124 MB/s, so a three-minute match is ~22 GB (§7).
    telemetry_dir: str = "runs/deploy"
    # Frames to dump for debugging, hard-capped. 0 = off, which is the default and the only value
    # that should survive into an unattended run.
    telemetry_frame_dump: int = 0


# (dotted YAML path, DeploymentConfig field name, type coercion)
_DEPLOYMENT_CONFIG_FIELDS = (
    ("run.dir", "run_dir", str),
    ("run.checkpoint", "run_checkpoint", str),
    ("policy.device", "policy_device", str),
    ("policy.deterministic", "policy_deterministic", bool),
    ("window.exe", "window_exe", str),
    ("window.require_fullscreen", "window_require_fullscreen", bool),
    ("window.check_occlusion", "window_check_occlusion", bool),
    ("window.check_every_n_ticks", "window_check_every_n_ticks", int),
    ("capture.monitor", "capture_monitor", lambda v: None if v is None else int(v)),
    ("control.backend", "control_backend", str),
    ("control.adb_instance", "control_adb_instance", lambda v: None if v is None else str(v)),
    ("control.attack_tap", "control_attack_tap", lambda v: tuple(float(x) for x in v)),
    ("loop.tick_hz", "loop_tick_hz", float),
    ("shadow.ammo_strikes", "shadow_ammo_strikes", int),
    ("shadow.max_resyncs", "shadow_max_resyncs", int),
    ("safety.max_match_seconds", "safety_max_match_seconds", float),
    ("safety.capture_stall_seconds", "safety_capture_stall_seconds", float),
    ("telemetry.dir", "telemetry_dir", str),
    ("telemetry.frame_dump", "telemetry_frame_dump", int),
)


# Tiles per second of the fastest thing in the game: Mortis's charged dash, `dash_distance 2.67
# * long_dash_multiplier 2.0 = 5.34` tiles in `dash_duration 0.30` s (`configs/brawlers.yaml`).
# `configs/vision.yaml` sized `odometry.max_shift_tiles` against this exact number.
DASH_TILES_PER_SECOND = 17.8

# How far the realised decision period may sit from the trained one before `resolve_rates` refuses
# the pairing. Zero would be right in principle and useless in practice -- 250/83.333.. is not
# exact in binary -- so this is a rounding tolerance, not a drift allowance: every rate that
# divides 250 ms evenly lands inside 1e-9 of it, and every rate that does not misses by
# milliseconds.
AGENT_PERIOD_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class Rates:
    """The loop's two clocks, resolved together because only the pairing is checkable.

    `agent_seconds` is the invariant; `tick_seconds` and `decision_every` are two ways of spending
    it. Handing these around as one object is what stops a caller reading the tick rate off the
    config and the decision cadence off the sim, which is how they would drift apart.
    """
    tick_seconds: float
    decision_every: int
    agent_seconds: float

    @property
    def tick_hz(self) -> float:
        return 1.0 / self.tick_seconds

    @property
    def decision_hz(self) -> float:
        return 1.0 / self.agent_seconds


def agent_seconds(sim_cfg) -> float:
    """The trained decision period: `dt * action_repeat`. The one rate that is not negotiable.

    `env._held` repeats one action across `action_repeat` sub-steps of `dt` and zeroes the fire
    column on all but the first, so this is the wall-clock interval the policy actually chose
    actions on. 250 ms on the shipped config.
    """
    return float(sim_cfg.dt) * float(sim_cfg.action_repeat)


def resolve_rates(sim_cfg, cfg=None) -> Rates:
    """Perception and decision cadence together, or a refusal.

    **The decision period is derived and the perception period is chosen**, which is the split the
    original version of this file got wrong by deriving both from `dt`. The policy was trained to
    act every 250 ms; it has no opinion whatever about how many times the world model is refreshed
    in between, because it never sees those refreshes. What does have an opinion is odometry,
    whose `max_shift_tiles` is a per-FRAME bound.

    Three things are enforced, and each is a silent failure if left to the caller:

    1. **The chosen rate must divide the decision period into a whole number of ticks.** 13 Hz
       would put a decision every 3.25 ticks, and rounding that to 3 stretches the agent's step to
       231 ms -- a policy acting on a rhythm it never trained on, with nothing on screen to show
       for it.
    2. **The realised decision period must equal the trained one.** Implied by (1), asserted
       anyway, because it is the actual invariant and (1) is only the arithmetic that delivers it.
    3. **A Mortis dash must stay inside `odometry.max_shift_tiles`.** Crossing it is not a dropped
       frame: `Odometry._cut` increments the segment, which resets the occupancy map and drops
       every track. A rate that reads dashes as scene cuts wipes the world model several times a
       match.

    `cfg` may be omitted, in which case the dataclass default rate is used -- so a caller that has
    a sim config but no deployment config still gets a checked pairing rather than a guess.
    """
    from .config import DeploymentConfig       # noqa: F401  (self-import kept for clarity)

    cfg = cfg if cfg is not None else DeploymentConfig()
    agent = agent_seconds(sim_cfg)
    if agent <= 0:
        raise ValueError(f"the training config gives dt={sim_cfg.dt} and "
                         f"action_repeat={sim_cfg.action_repeat}; cannot derive a decision period")
    hz = float(cfg.loop_tick_hz)
    if hz <= 0:
        raise ValueError(f"loop.tick_hz must be positive, got {hz}")

    ticks = agent * hz
    n = int(round(ticks))
    if n < 1 or abs(ticks - n) > 1e-6:
        options = ", ".join(f"{k / agent:g}" for k in range(1, 7))
        raise ValueError(
            f"loop.tick_hz={hz:g} puts a decision every {ticks:.3f} perception ticks, which is "
            f"not a whole number. The decision period is fixed at {agent * 1e3:.0f} ms by the "
            f"training config (dt={sim_cfg.dt} x action_repeat={sim_cfg.action_repeat}) and "
            f"rounding this would step the policy at a rhythm it never trained on. Whole-number "
            f"rates for this checkpoint: {options} Hz."
        )
    tick = agent / n
    if abs(tick * n - agent) > AGENT_PERIOD_TOLERANCE_S:
        raise ValueError(f"{n} ticks of {tick:.6f}s is {tick * n:.6f}s, not the trained {agent}s")
    return Rates(tick_seconds=tick, decision_every=n, agent_seconds=agent)


def match_seconds(sim_cfg) -> float:
    """`meta.time_frac`'s denominator: `max_episode_steps * dt` (§6.5).

    150 s on the shipped config. Derived, because a different value here would silently rescale a
    column the policy was trained to read -- and the clamp at 1.0 means the error only shows up as
    slightly-off behaviour late in a match, never as an exception.
    """
    return float(sim_cfg.max_episode_steps) * float(sim_cfg.dt)


def _dget(d: dict, dotted: str, default=_MISSING):
    """Duplicated from `brawl_sim/config.py` for the reason `brawl_vision/config.py` gives at
    length: it is ten lines, and importing it would make this package depend on the simulator's
    config machinery to read a YAML file."""
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(dotted)
            return default
        node = node[part]
    return node


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_deployment_config(path=DEFAULT_CONFIG_PATH,
                           overrides: dict | None = None) -> DeploymentConfig:
    """A field absent from the YAML keeps its dataclass default rather than raising."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)

    kwargs = {}
    for dotted, field_name, coerce in _DEPLOYMENT_CONFIG_FIELDS:
        value = _dget(raw, dotted, default=_ABSENT)
        if value is _ABSENT:
            continue
        kwargs[field_name] = coerce(value)
    return DeploymentConfig(**kwargs)


def validate(cfg: DeploymentConfig, sim_cfg=None, vision_cfg=None) -> None:
    """Sanity-check the tunables, and cross-check the derived rates against the other two configs.

    The cross-checks are the valuable half. Each one catches a mismatch that produces a running,
    plausible-looking agent rather than an error: a capture rate that violates the odometry shift
    bound, or a decision cadence the policy never trained at.
    """
    if cfg.policy_device not in ("cpu", "cuda"):
        raise ValueError(f"policy.device must be 'cpu' or 'cuda', got {cfg.policy_device!r}")
    if cfg.control_backend not in ("adb", "null"):
        raise ValueError(f"control.backend must be 'adb' or 'null', got {cfg.control_backend!r}")
    fx, fy = cfg.control_attack_tap
    if not (0.0 < fx < 1.0 and 0.0 < fy < 1.0):
        raise ValueError(
            f"control.attack_tap is a fraction of the screen in each axis, got {(fx, fy)}"
        )
    if fx <= 0.5:
        raise ValueError(
            f"control.attack_tap x={fx} is on the LEFT half of the screen, which is the movement "
            f"joystick's half. Attack only fires from the right side (§6.8)."
        )
    if cfg.window_check_every_n_ticks < 1:
        raise ValueError("window.check_every_n_ticks must be at least 1")
    if cfg.shadow_ammo_strikes < 1:
        raise ValueError("shadow.ammo_strikes must be at least 1")
    if cfg.safety_capture_stall_seconds <= 0 or cfg.safety_max_match_seconds <= 0:
        raise ValueError("safety timeouts must be positive")
    if cfg.telemetry_frame_dump < 0:
        raise ValueError("telemetry.frame_dump must be >= 0")

    if cfg.loop_tick_hz <= 0:
        raise ValueError(f"loop.tick_hz must be positive, got {cfg.loop_tick_hz}")

    if sim_cfg is not None:
        rates = resolve_rates(sim_cfg, cfg)     # raises on a rate that does not divide evenly
        if cfg.window_check_every_n_ticks > 10 * rates.decision_every:
            raise ValueError(
                f"window.check_every_n_ticks={cfg.window_check_every_n_ticks} is more than ten "
                f"decisions apart, which is long enough to play a whole engagement into a "
                f"covered screen."
            )

    if sim_cfg is not None and vision_cfg is not None:
        # `odometry.max_shift_tiles` bounds motion between consecutive FRAMES and was sized at
        # 20 Hz against Mortis's charged dash (17.8 tiles/s -> 0.89 tiles/frame). Crossing it is
        # not a dropped frame: `Odometry._cut` bumps the segment, which resets the occupancy map
        # and drops every track. So this is a correctness failure, not a slow one.
        rates = resolve_rates(sim_cfg, cfg)
        dash_tiles_per_frame = DASH_TILES_PER_SECOND * rates.tick_seconds
        if dash_tiles_per_frame > vision_cfg.odometry_max_shift_tiles:
            raise ValueError(
                f"loop.tick_hz={rates.tick_hz:.1f} puts a Mortis dash at "
                f"{dash_tiles_per_frame:.2f} tiles per frame, past "
                f"odometry.max_shift_tiles={vision_cfg.odometry_max_shift_tiles}. Every dash "
                f"would read as a camera cut and reset the world model. Raise the tick rate, or "
                f"raise max_shift_tiles -- its own comment says to, if the capture rate drops."
            )
