"""The driver. Screen -> perception -> policy -> touch, at two rates, with the safety interlock.

Everything else in this package is a stage that can be tested on its own. This is the one file
that decides *when* each of them runs, and it is therefore the one file that owns the rule the
whole project fails closed on: **the control layer may only emit while the match gate is true.**

**Two rates. One is derived and one is chosen, and knowing which is which is the point** (§6.13).
`config.resolve_rates` settles both together:

- **The DECISION period is derived and fixed**: `cfg.dt * cfg.action_repeat` = 250 ms, read out of
  the run's own env config. The policy chose actions on that rhythm; a different one is an agent
  stepping at a cadence it never trained on.
- **The PERCEPTION period is `loop.tick_hz`**, a setting, allowed only at values that divide the
  decision period into a whole number of ticks. The checkpoint has no opinion about it -- it never
  sees the refreshes between its own decisions. Odometry does: `odometry.max_shift_tiles` is a
  PER-FRAME bound, and crossing it is not a dropped frame but a `_cut`, which resets the occupancy
  map and drops every track.

Shipped: **12 Hz perception, 3 ticks per decision, 4 Hz decisions.** Not 20, because at 20 the loop
does not fit inside 50 ms and the pacer below does not sleep on an overrun -- the result is a
jittering period, not a slow one. §6.13 has the measurement.

**What runs at which rate, and why each is where it is:**

| every tick (12 Hz) | why not slower |
|---|---|
| grab, rectify, odometry | the shift bound above |
| match gate | exit hysteresis is counted in frames |
| projectile detect + track | a projectile crosses the screen inside one decision (§9.5) |
| crate/cube map, off the same boxes | the model call is already paid; more sightings confirm sooner |
| terrain classify + occupancy deposit | accumulating; a skipped frame is evidence thrown away |
| `grid.observe_zone` | same, and it is what `ZoneEstimator` reads |
| `shadow.advance` | it spends REAL elapsed seconds as whole sub-ticks, which is what keeps the two rates honest when a decision runs long |
| `buttons.settle` | walks an attack press one step: the tick after the down drags it along the aim, the tick after that lifts, which fires |

| every decision (4 Hz) | why not faster |
|---|---|
| entity detect, HP read, HUD read | ~9 ms of GPU, and nothing consumes them between decisions |
| tracker, grid build, assemble, policy | a decision's worth of work, by definition |

**The process shares eight cores with the emulator's VM, and idle thread pools are not idle.**
torch, ONNX Runtime and OpenCV each bring a pool whose workers spin-wait between jobs; with their
defaults this loop held 3.3-4.2 cores while sleeping a third of every tick, and the game -- not
the tick -- is what stuttered. `pin_thread_pools` caps them from `compute.*` before anything is
built (§7.2); 1.0 core after, with a faster tick.

**Held actions are the sim's `_held` semantics, reached from the device side.** The movement
contact stays down and only moves when the bin changes; the fire bit does not repeat, because
`Buttons.press` is called once per decision and `ShadowHero.act` queues exactly one attack for the
next sub-tick. The press is dragged along `ShadowHero.attack_bearing` -- the move bin, or `facing`
when idle -- because that is where the sim's dash goes, and a bare tap would auto-aim instead.
None of it is a simplification of the trained behaviour -- all of it is it.

**Failure routes to one place** (§8). `_stop` releases every contact, drops the gate, and refuses
to emit again. Occlusion is the single exception: it `_pause`s instead, because a window drawn
over the emulator is transient by nature and the world model survives it. A stopped agent is
recoverable by hand; an agent mashing inputs into a menu is not.

**What this file does NOT do.** It does not queue matches -- navigating menus is out of the MVP,
and the loop simply goes quiet between them. It does not detect a dropped movement contact by ring
score (§4.2's suggestion): that needs a rest-anchor radius nothing has measured yet, so the desync
signals that DO exist -- the ammo canary and hero-position divergence -- are what is wired instead,
and the gap is named here rather than silently skipped.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from brawl_sim.constants import Tile
from brawl_vision.config import EMULATOR_HUD_MASK_PATH

from .config import DeploymentConfig, resolve_rates
from .control import Buttons, Joystick
from .match_state import Calibration, MatchState
from .perception import (EntityTracker, GridBuilder, GridSpec, LatticePhase, LootMap,
                         ProjectileTracker, ShadowHero, ShadowParams, ZoneEstimator,
                         box_occlusion, crate_occlusion, require_loot_classes)

# The emulator's HUD, NOT `brawl_vision`'s default. That one is the iOS recordings' layout; on
# BlueStacks frames it masks floor and leaves every button in view, which the terrain map deposits
# as walls and odometry correlates as a static patch. See the file. It sits with the other masks in
# brawl_vision/data because labels drawn on emulator footage are checked against it too.
HUD_MASK_PATH = EMULATOR_HUD_MASK_PATH

# How long odometry may report anything but "ok" before the loop gives up.
#
# Sized against what a LEGITIMATE loss looks like: a death, a respawn fly-in or a match-start
# camera sweep is a *cut*, which opens a new segment and resets every world-frame consumer in one
# tick -- not a run of bad ticks. A sustained "lost" means the correlation itself is failing, and
# while it lasts nothing deposits, no track moves and every world position is frozen, so the
# observation quietly goes stale rather than wrong. Two seconds is long enough that no cut trips
# it and short enough that the agent is not playing off a two-second-old world.
#
# In SECONDS, converted to ticks per loop against the resolved rate. It used to be a flat 40 ticks,
# which was two seconds only while `loop.tick_hz` happened to be 20 -- a threshold whose stated
# rationale is a duration should not silently become 3.3 s when the perception rate moves (§6.13).
ODOMETRY_LOST_SECONDS = 2.0

# Telemetry ring depth: 4000 rows is >5 minutes at any admitted tick rate (333 s at 12 Hz), one
# match plus margin, and ~1 MB of scalars. Fixed-size on purpose (§7) -- the resource risk in this
# package is unbounded growth, not size.
TELEMETRY_ROWS = 4000

# The repo's configs, for `dash_reach_tiles`: the same directory `perception/shadow.py` and
# `perception/grid.py` resolve for the kit constants and the view.
_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

# The brawler the loop deploys: the `configs/brawlers.yaml` block the shadow dead-reckons from
# and `dash_reach_tiles` takes `dash_radius` from. One name, so the two cannot default apart
# (`ShadowParams` carries no `kind`, and shadow.py is not this package's to change for that).
# Only Mortis deploys today; a second kind would make this a `DeploymentConfig` field.
HERO_KIND = "hero_mortis"

# Cells `occupancy.best()` has never observed. Mirrors `terrain.occupancy.UNKNOWN` rather than
# importing it, so a terrain-side rename fails a test here instead of silently reading class 255.
_UNKNOWN = -1


class Phase(str, Enum):
    """Where the loop is. `PLAYING` is the only phase that emits input."""

    WAITING = "waiting"      # capturing, gate closed, emitting nothing
    PLAYING = "playing"      # in a match
    PAUSED = "paused"        # something is drawn over the emulator; capturing, not emitting
    STOPPED = "stopped"      # terminal, and deliberately not recoverable in-process


@dataclass
class TickRow:
    """One tick of bounded telemetry. **Scalars only** -- never a frame, never a tensor (§7).

    Mutable, and filled in as the tick proceeds rather than assembled at the end, so that a tick
    which bails out early still records how far it got. A frozen row would need a mutable twin to
    accumulate into, and the two would drift.

    `move_bin` and `attack` are `-1` on a tick with no decision, which is a different thing from
    `0` -- idle is a bin the policy chose and a contact the loop moved. Telemetry that conflated
    them would make a stalled loop and a stationary agent look identical.
    """

    index: int
    t: float
    grab_ms: float
    phase: str = ""
    tick_ms: float = 0.0
    odometry: str = ""
    # The one tick that pays `VisionStack.warm`'s ~1 s of PTX JIT. Flagged rather than left to
    # look like a 758 ms overrun in the summary: it happens before the gate opens, exactly so no
    # gameplay tick eats it, and reporting it inside the budget tail hides the real p95.
    warmup: bool = False
    decision: bool = False
    move_bin: int = -1
    attack: int = -1
    n_detections: int = 0
    # Both sides of the desync canary, `-1` where there is no value. `ammo_cv`: no decision this
    # tick, no hero box, or `read_ammo` returned None (it misses ~12% of frames). `ammo_shadow`:
    # no decision reached the bars this tick -- the shadow always has a clip, so it is written on
    # every decision that gets that far, hero box or not (the audit reads it as the row's ammo
    # next to `attack_legal`, and a sentinel there would count as a clip size). Kept as two columns
    # rather than one error, because the operator has said the kit constants in
    # `configs/brawlers.yaml` are APPROXIMATIONS and the live game is the better source -- so the
    # pair is what lets `reload_seconds` and `attack_cooldown` be REFIT from a run that was going
    # to happen anyway. An error column alone cannot: it hides which of the two moved.
    ammo_cv: float = -1.0
    ammo_shadow: float = -1.0
    # `LatticeResult.status` ("unlocked" / "locked" / "rebased", or odometry's own) and the phase
    # the world frame is shifted by. A "rebased" row is a tick on which every tracker reset.
    lattice: str = ""
    phase_x: float = 0.0
    phase_y: float = 0.0
    note: str = ""
    # --- attack cadence (SIM_OVERHAUL_PLAN.md Phase A, Step A2). What the policy was HANDED at
    # this decision, so `scripts/audit_attack_cadence.py --telemetry` can compute the sim-side
    # audit's five statistics from a live match on identical definitions. All read BEFORE
    # `policy.act`, from the same shadow state the mask came from; `-1` / `-1.0` on a tick with
    # no decision, like `move_bin` above. Appended after `note` so an older CSV's columns keep
    # their positions and `TickRow.from_record` can default the ones a file lacks.
    #
    # `attack_legal`: `ShadowHero.attack_mask()` as a bitmask, bit i = attack column i, so bit 0
    # (no-fire, always legal) is set on every decision row and `attack_legal & 0b10` is "the
    # dash was legal" -- the audit's `can_attack`. `attack_cd_shadow` / `attack_idle_t_shadow`
    # are the shadow's two timers the audit reads (phasing loss and long-dash waiting).
    # `enemy_in_reach` is the audit's own criterion applied to the tracks the assembler was given:
    # any enemy within `dash_distance + dash_radius + unit_radius` tiles of the hero
    # (`DeployLoop.reach_tiles`, from the configs -- not a literal).
    attack_legal: int = -1
    attack_cd_shadow: float = -1.0
    attack_idle_t_shadow: float = -1.0
    enemy_in_reach: bool = False
    # The ammo canary tripped on this decision and `ShadowHero.resync` ran BEFORE the mask above
    # was read (the row records what the policy saw, which is the post-resync "assume not ready"
    # state). `resync_error` is the `Desync.error` that tripped it, CV minus shadow in pips: a
    # positive value is a modelled attack the game never took. `0.0` when no resync happened --
    # unambiguous, because a zero error is inside every tolerance and can never trip the canary.
    resync: bool = False
    resync_error: float = 0.0

    @classmethod
    def from_record(cls, record: dict) -> "TickRow":
        """A row back from `write_csv`'s CSV (a `csv.DictReader` dict of strings), or from any
        dict of scalars. **A column the record lacks keeps the field's default**, so telemetry
        written before a column existed still loads; a record's extra keys are ignored for the
        same reason in the other direction. Values are coerced by the field's own annotation
        (`csv` hands everything back as text, and `bool("False")` is True)."""
        from dataclasses import MISSING, fields

        kwargs = {}
        for f in fields(cls):
            if f.name not in record:
                if f.default is MISSING:
                    raise KeyError(f"telemetry record has no {f.name!r} column")
                continue
            kwargs[f.name] = _coerce(f.type, record[f.name])
        return cls(**kwargs)


def _coerce(annotation, value):
    """`TickRow.from_record`'s per-field parse. Annotations are strings under
    `from __future__ import annotations`, so this matches on the name."""
    if isinstance(value, str):
        name = annotation if isinstance(annotation, str) else annotation.__name__
        if name == "bool":
            return value.strip().lower() in ("true", "1", "yes")
        if name == "int":
            return int(float(value))
        if name == "float":
            return float(value)
        return value
    return value


def read_telemetry_csv(path) -> list[TickRow]:
    """`write_csv`'s inverse: every row of a `--telemetry` CSV as a `TickRow`, in file order,
    with columns the file predates left at their defaults (see `TickRow.from_record`)."""
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        return [TickRow.from_record(record) for record in csv.DictReader(fh)]


def dash_reach_tiles(params: ShadowParams, *, kind: str = HERO_KIND,
                     brawlers_path=None, env_config_path=None) -> float:
    """`dash_distance + dash_radius + unit_radius` in tiles: the uncharged dash's "certainly could
    have hit" radius, which is `scripts/audit_attack_cadence.py`'s `enemy_in_reach` criterion
    (Step A1.1) restated from the deployment's own sources. `dash_distance` is the shadow's
    (`ShadowParams`, from `configs/brawlers.yaml`); `dash_radius` is read from the same brawler
    block, and `unit_radius` from `entities.unit_radius` in `configs/default.yaml`, the way
    `GridSpec.load` reads that file for the view -- neither is a field the shadow needs, so
    neither is on `ShadowParams`, and a literal here would drift the moment the sim's changed.
    A randomized range for either raises for the reason `ShadowParams.load` gives."""
    import yaml

    brawlers = yaml.safe_load(Path(brawlers_path or _CONFIGS_DIR / "brawlers.yaml").read_text())
    env = yaml.safe_load(Path(env_config_path or _CONFIGS_DIR / "default.yaml").read_text())
    try:
        dash_radius = brawlers[kind]["dash_radius"]
        unit_radius = env["entities"]["unit_radius"]
    except KeyError as exc:
        raise KeyError(f"dash reach needs {kind}.dash_radius and entities.unit_radius: {exc}") from exc
    for name, raw in (("dash_radius", dash_radius), ("unit_radius", unit_radius)):
        if isinstance(raw, (list, tuple, dict)):
            raise ValueError(f"{name} is a randomized range {raw!r}; deployment needs one value")
    return float(params.dash_distance) + float(dash_radius) + float(unit_radius)


def pin_thread_pools(cfg: DeploymentConfig) -> None:
    """Cap the CPU thread pools this process shares with the emulator. FIRST, before anything
    that builds a network -- torch's pool is process-global, and the detectors take their
    settings at construction (`VisionStack.build`, which the caller passes `cfg.compute_*`).

    Not a tick optimisation; a courtesy to the VM on the other cores, which is what lags when
    this process burns them. MEASURED 2026-09-16 by replaying the real `tick` at 12 Hz from a
    BlueStacks clip with process-scoped CPU sampling (§7.2): the library defaults held 3.3-4.2
    cores of a machine with eight, while the loop slept a third of every tick -- an idle pool
    thread spin-waits for its next job rather than sleeping, and every library brings its own
    pool. torch alone (8 OpenMP threads for a 1 ms policy call every 250 ms) was 1.5-2.4 of
    those cores. Pinned to the shipped `compute.*`: 1.0 core, and the tick got faster.
    """
    import cv2
    import torch

    torch.set_num_threads(cfg.compute_torch_threads)
    cv2.setNumThreads(cfg.compute_cv2_threads)


@dataclass
class VisionStack:
    """The `brawl_vision` stages, grouped because they share one `RectifyPlan` and one config.

    Held as one object so the loop's constructor takes a stack rather than nine parameters, and so
    a test can hand it fakes without also faking the deployment-side trackers -- which are the part
    worth exercising for real.
    """

    plan: object
    odometry: object
    occupancy: object
    classifier: object
    entities: object            # ObjectDetector
    projectiles: object         # ProjectileDetector
    health: object              # smooth.HealthTracker
    hud: object                 # HudReader
    cfg: object                 # VisionConfig

    @classmethod
    def build(cls, vision_cfg=None, *, detector_threads: int | None = None,
              detector_spin: bool = True) -> "VisionStack":
        """The real stack, loaded from `configs/vision.yaml` and the shipped weights.

        `detector_threads` / `detector_spin` are the ORT CPU-pool settings for both sessions, from
        `DeploymentConfig.compute_*` when the live builders call this (`pin_thread_pools`). The
        defaults are the library's, which is right for a script that runs alone.
        """
        from brawl_vision.camera import build_rectify_plan, load_camera_model, load_hud_mask
        from brawl_vision.config import load_vision_config
        from brawl_vision.hud import HudReader
        from brawl_vision.object_detection.detector import ObjectDetector
        from brawl_vision.object_detection.hp_detection.smooth import HealthTracker
        from brawl_vision.object_detection.projectile_detection.detect import ProjectileDetector
        from brawl_vision.terrain.classifier import TerrainClassifier
        from brawl_vision.terrain.occupancy import OccupancyMap
        from brawl_vision.terrain.odometry import Odometry

        from .perception.projectiles import require_projectile_class

        cfg = vision_cfg or load_vision_config()
        plan = build_rectify_plan(load_camera_model(), load_hud_mask(HUD_MASK_PATH))
        # Checked here, on the real model, because the tracker filters by label: a promoted model
        # with no `Projectile` class would otherwise load fine and feed the policy zero shots.
        pool = dict(cpu_threads=detector_threads, spin=detector_spin)
        projectiles = ProjectileDetector.from_config(cfg, **pool)
        require_projectile_class(projectiles.names)
        return cls(
            plan=plan,
            odometry=Odometry(plan, cfg),
            occupancy=OccupancyMap.from_config(cfg),
            classifier=TerrainClassifier.from_config(cfg),
            entities=ObjectDetector.from_config(cfg, **pool),
            projectiles=projectiles,
            health=HealthTracker.from_config(cfg),
            hud=HudReader.from_config(cfg),
            cfg=cfg,
        )

    def warm(self, image: np.ndarray) -> None:
        """Run every GPU stage once on a real frame before the gate is allowed to open.

        Not politeness. The 5070 Ti is Blackwell and the wheel's kernels JIT from PTX, so the first
        inference after construction takes ~1 s against a steady-state 7-11 ms (§7.1). Paying that
        on the first decision of a match would blow one 50 ms tick by a factor of twenty, which is
        an `odometry.max_shift_tiles` violation -- a correctness failure, not a slow start.

        **The terrain classifier is on this list, and was not always.** It moved to CUDA once the
        cost was measured (`configs/vision.yaml`), and its first call there took 49 ms against a
        6 ms steady state -- landing inside the first perception tick of the match, where every
        tick is already near budget. A warm-up covering two of the three GPU stages is exactly the
        kind of gap that shows up as one bad frame at the worst possible moment.
        """
        self.entities.predict(image)
        self.projectiles.predict(image)
        self.classifier.predict(self.plan.rectify(image), self.plan)


@dataclass
class Controls:
    """The input side, and the only object in the loop that can touch the device."""

    backend: object
    joystick: Joystick
    buttons: Buttons

    @classmethod
    def build(cls, cal: Calibration, dcfg: DeploymentConfig, n_move_bins: int) -> "Controls":
        """The real ADB backend plus the geometry, in DEVICE pixels.

        `n_move_bins` comes from the run's env config, not from a default: a mismatch does not
        raise anywhere, it silently rotates every commanded direction by a fraction of a turn.
        """
        from .control.adb import AdbTouchBackend, find_adb_serial

        if dcfg.control_backend != "adb":
            raise ValueError(f"control_backend={dcfg.control_backend!r} has no builder here; "
                             f"construct the backend yourself and pass a Controls in.")
        backend = AdbTouchBackend(serial=find_adb_serial(instance=dcfg.control_adb_instance),
                                  screen=cal.screen)
        w, h = cal.screen
        buttons = Buttons(backend,
                          attack=(dcfg.control_attack_tap[0] * w, dcfg.control_attack_tap[1] * h),
                          super_=cal.button("super"),
                          aim_radius_px=dcfg.control_aim_radius_px)
        try:
            buttons.require_on_screen(cal.screen)
        except ValueError:
            backend.close()
            raise
        return cls(
            backend=backend,
            joystick=Joystick(backend, cal.joystick_anchor, cal.joystick_radius_px,
                              n_bins=n_move_bins),
            buttons=buttons,
        )

    def release_all(self) -> None:
        """Every path out of `PLAYING` goes through here. Idempotent by construction: both
        `Joystick.release` and `Buttons.release` no-op when nothing is held. `release`, not
        `settle`: settle would only walk a press one step further, and this path lifts now."""
        self.buttons.release()
        self.joystick.release()


class DeployLoop:
    """One match's worth of driving, and the interlock that keeps it inside one.

    Components are injected rather than constructed, with `from_config` building the real ones.
    That split is what lets the deployment tests join the siloed fast set (§Testing): the trackers,
    the shadow, the assembler and the phase machine all run for real against a fake capture and a
    fake backend, with no emulator and no GPU.

    Everything derived is derived here and nowhere else: the tick period, the decision cadence, the
    entity slot count and the move-bin count all come from `policy.cfg`, which is the config the
    checkpoint was trained under. None of them is a setting.
    """

    def __init__(self, *, capture, guard, match: MatchState, vision: VisionStack, policy,
                 controls: Controls, cfg: DeploymentConfig, log=None):
        self.capture = capture
        self.guard = guard
        self.match = match
        self.vision = vision
        self.policy = policy
        self.controls = controls
        self.cfg = cfg
        self.log = log or (lambda msg: None)

        self.sim = policy.cfg
        # Resolved as a pair, and it raises rather than rounding: a tick rate that does not divide
        # the trained decision period evenly would stretch the agent's step silently.
        self.rates = resolve_rates(self.sim, cfg)
        self.tick_seconds = self.rates.tick_seconds
        self.decision_every = self.rates.decision_every
        # A duration, not a tick count: see ODOMETRY_LOST_SECONDS.
        self.odometry_lost_ticks = max(1, round(ODOMETRY_LOST_SECONDS / self.tick_seconds))
        self._grab_total = 0.0      # last read of the capture's CUMULATIVE grab_seconds

        self.assembler = policy.make_assembler()
        self.shadow = ShadowHero(ShadowParams.load(HERO_KIND), dt=self.sim.dt,
                                 n_move_bins=int(self.sim.n_move_bins),
                                 desync_strikes=cfg.shadow_ammo_strikes)
        # Telemetry only (`TickRow.enemy_in_reach`, Step A2): the cadence audit's "certainly
        # could have hit" radius, from the same configs the shadow and the sim read. The SAME
        # kind as the shadow's block: `ShadowParams` does not record which brawler it was loaded
        # for, so this is the one place both defaults are set.
        self.reach_tiles = dash_reach_tiles(self.shadow.p, kind=HERO_KIND)
        self.tracker = EntityTracker(n_slots=int(self.sim.n_entities) - 1)
        self.projectiles = ProjectileTracker()
        self.loot = LootMap()
        # Not `self.phase`, which is the loop's Phase. This is where the game's tiles are.
        self.lattice = LatticePhase()
        # The grid's channels come from the spec the policy was built with. The no-argument
        # `GridSpec.load()` reads agent_obs_deploy.yaml, which is two channels short of deploy3,
        # and the mismatch would only surface as a shape error in `assemble` on the first decision.
        self.grid = GridBuilder(vision.occupancy, GridSpec.from_agent_spec(policy.spec, self.sim))
        # Here rather than in `VisionStack.build`, because it takes both halves: a deploy3 policy
        # on a model with no crate class would read an empty `box` plane forever, silently.
        require_loot_classes(vision.projectiles.names, self.grid.spec.channels)
        self.zone = ZoneEstimator(self.sim)

        self.phase = Phase.WAITING
        self.stop_reason: str | None = None
        self.telemetry: deque[TickRow] = deque(maxlen=TELEMETRY_ROWS)

        self._tick_index = 0
        self._ticks_in_match = 0
        self._match_t0 = 0.0
        self._last_grab_t = 0.0
        self._odometry_lost = 0
        self._resyncs = 0
        self._dry_desync_noted = False
        self._n_enemies: int | None = None
        self._enemy_hp: dict[int, int] = {}
        self._slot_ids: dict[int, int] = {}
        self._last_decision = None
        self._warmed = False

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: DeploymentConfig, *, log=None) -> "DeployLoop":
        """The whole live stack, from one config. Order matters: the window is located first,
        because both the capture crop box and the monitor index come out of it (§6.11)."""
        import mss

        from .capture import DeployCapture
        from .policy import DeployedPolicy
        from .window import WindowGuard, find_emulator_window, set_dpi_aware

        pin_thread_pools(cfg)
        set_dpi_aware()
        window = find_emulator_window(cfg.window_exe)
        with mss.mss() as sct:
            monitors = [dict(m) for m in sct.monitors]

        guard = WindowGuard(window, monitors,
                            require_fullscreen=cfg.window_require_fullscreen,
                            check_occlusion=cfg.window_check_occlusion)
        capture = DeployCapture.from_window(window, monitors)
        policy = DeployedPolicy.from_run(cfg.run_dir, checkpoint=cfg.run_checkpoint,
                                         device=cfg.policy_device,
                                         deterministic=cfg.policy_deterministic)
        cal = Calibration.load()
        vision = VisionStack.build(detector_threads=cfg.compute_detector_threads,
                                   detector_spin=cfg.compute_detector_spin)
        return cls(capture=capture, guard=guard, match=MatchState(cal),
                   vision=vision, policy=policy,
                   controls=Controls.build(cal, cfg, int(policy.cfg.n_move_bins)),
                   cfg=cfg, log=log)

    # -- the run --------------------------------------------------------------

    def run(self, *, max_seconds: float | None = None, max_matches: int | None = 1) -> str:
        """Drive until something stops it. Returns the stop reason.

        `max_matches` defaults to 1 because match queueing is not in the MVP -- the agent plays the
        battle the operator started and then goes quiet. Passing `None` keeps it waiting for the
        next gate, which is the useful mode for a session of back-to-back friendly battles: it
        avoids reloading two detectors and a checkpoint between them, and it emits nothing in the
        menus either way, because that is the interlock and not a policy about matches.
        """
        matches = 0
        t0 = time.perf_counter()
        deadline = t0
        try:
            with self.capture:
                while self.phase is not Phase.STOPPED:
                    was_playing = self.phase is Phase.PLAYING
                    self.tick()
                    if was_playing and self.phase is not Phase.PLAYING:
                        matches += 1
                        if max_matches is not None and matches >= max_matches:
                            self._stop("match over")
                            break
                    if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
                        self._stop("max_seconds reached")
                        break
                    deadline = self._pace(deadline)
        finally:
            # Not `_stop`: this must run even when the loop is unwinding on an exception, and it
            # must not overwrite a stop reason that already explains what happened.
            self.controls.release_all()
            self.phase = Phase.STOPPED
        return self.stop_reason or "stopped"

    def _pace(self, deadline: float) -> float:
        """Hold the tick rate without accumulating drift, and without bursting to catch up.

        A tick that ran long has already cost the wall-clock time it cost -- firing the next four
        back to back would not recover the frames, it would just violate the same shift bound in
        the other direction. So a missed deadline resets to now rather than replaying.
        """
        deadline += self.tick_seconds
        now = time.perf_counter()
        if deadline <= now:
            return now
        time.sleep(deadline - now)
        return deadline

    # -- one perception tick --------------------------------------------------

    def tick(self) -> None:
        """One perception tick. Never raises for an expected failure -- it stops instead.

        The tick index advances on every path that got a frame, including the ones that bailed
        early: the window check runs on a tick cadence, and a phase that never advances the counter
        is a phase that never re-checks the window it is waiting on.
        """
        t_start = time.perf_counter()
        elapsed = self._elapsed_since_last_tick(t_start)
        frame = self.capture.grab()
        # `capture.grab_seconds` is CUMULATIVE -- its own docstring says so, and reading it as a
        # per-frame duration is what made the first live run report a 3516 ms median grab against
        # a 55.8 ms tick. Store the delta; the total stays available on the capture.
        total = self.capture.grab_seconds
        row = TickRow(index=frame.index, t=frame.t,
                      grab_ms=(total - self._grab_total) * 1e3)
        self._grab_total = total

        if not self._check_stall(t_start):
            return
        self._last_grab_t = t_start

        try:
            if not self._check_window():
                row.note = "window"
                return
            # An attack press moves one step per tick -- down, drag, lift -- so each step has had a
            # full frame to register before the next. First thing, so on a decision tick the last
            # press has lifted before a new one can go down.
            self.controls.buttons.settle()

            rect = self.vision.plan.rectify(frame.image)
            odo = self.vision.odometry.update(rect)
            row.odometry = odo.status
            gate = self.match.update(frame.image)

            if self.phase is Phase.PLAYING:
                # Real seconds, spent as whole sub-ticks. Before the gate check, so the sub-ticks
                # of a match's last tick are not silently dropped.
                self.shadow.advance(elapsed)

            if not self._check_odometry(odo):
                row.note = f"odometry {odo.status}"
                return
            if not self._gate_transition(gate, frame, row):
                return
            detections, world = self._perceive(frame, rect, odo, row)
            if self._is_decision_tick():
                self._decide(frame, world, row, detections)
        finally:
            row.phase = self.phase.value
            row.tick_ms = (time.perf_counter() - t_start) * 1e3
            self.telemetry.append(row)
            self._tick_index += 1
            if self.phase is Phase.PLAYING:
                self._ticks_in_match += 1

    def _perceive(self, frame, rect, odo, row: TickRow | None = None) -> tuple[list, object]:
        """The accumulating every-tick stages. Everything here is gated on odometry inside its own
        `update`; the loop does not re-implement that gate, it relies on it (§6.1).

        Returns this tick's entity detections, which a decision on the same tick reuses rather
        than running the detector twice on one frame, and the WORLD odometry every consumer was
        given: `odo` moved onto the game's tile lattice (`perception/lattice.py`).
        """
        from brawl_vision.terrain.zone import detect_zone

        # First: the crates are both the lattice's evidence and the occupancy map's fifth gate
        # (a cell a crate stands on is not evidence about the floor under it).
        dets = self.vision.projectiles.predict(frame.image)
        # Before anything deposits, because a re-lock on this tick moves this tick's frame. From
        # here on nothing sees `odo`: every consumer gets `world`, whose `segment` is the lattice
        # epoch, so a re-lock resets them all through their own new-segment path.
        lattice = self.lattice.update(dets, self.vision.plan, odo)
        world = self.lattice.world(odo)
        if row is not None:
            row.lattice = lattice.status
            row.phase_x, row.phase_y = lattice.phase

        # `rect` came through the fixed plan, which is what odometry measures against. Everything
        # that deposits cells takes this frame re-rectified onto the map's own lattice instead, so
        # its cells are the map's cells rather than up to half a tile off (`RectifyPlan.registered`).
        # With the world frame on the game's lattice, the map's cells are the game's tiles.
        # Continuous consumers (projectiles, loot) read the same tiles through either plan.
        plan = self.vision.plan.registered(world.position_tiles)
        if plan is not self.vision.plan:
            rect = plan.rectify(frame.image)
        # Every tick, not only on decisions, for the same gate: a brawler sprite is not floor
        # either, and two thirds of the deposits would otherwise carry it (`loot.box_occlusion`).
        # A tick odometry did not place deposits nothing, and `_decide` returns before reading.
        entities = self.vision.entities.predict(frame.image) if odo.status == "ok" else []

        zone = detect_zone(rect, plan, self.vision.cfg)
        cells, _ = self.vision.classifier.predict(rect, plan)
        occluded = crate_occlusion(dets, plan) | box_occlusion(entities, plan)
        self.vision.occupancy.update(cells, world, plan, zone=zone.at_least(0.05),
                                     occluded=occluded, cfg=self.vision.cfg)
        self.grid.observe_zone(zone, plan, world)

        # One model, three classes, and each consumer keeps its own label.
        self.projectiles.update(dets, plan, world, frame.t)
        self.loot.update(dets, plan, world, frame.t)
        return entities, world

    # -- one decision ---------------------------------------------------------

    def _is_decision_tick(self) -> bool:
        return self._ticks_in_match % self.decision_every == 0

    def _decide(self, frame, odo, row: TickRow, detections: list) -> None:
        """One 4 Hz decision.

        **A supplier that has nothing to say skips the decision rather than filling a column.**
        The held movement bin persists and the fire bit does not repeat, so a skipped decision is
        exactly one action-repeat window of the previous action -- which is a state the sim
        produces constantly and the policy has seen. Substituting a zero is not. Every skip is
        named in the telemetry row, because "the agent stood still" and "no observation could be
        built" look identical from outside and mean opposite things.

        `odo` is the WORLD odometry `_perceive` returned, the frame every track and map is in.
        """
        if odo.status != "ok":
            # §8's "hold last action". The tracker would hand back its previous tracks unchanged
            # on this tick -- correct of it, and exactly why the loop must not then decide on them:
            # every world position is one tick stale, and the detector pass and policy call spent
            # getting there buy nothing. Holding is not a degraded decision, it is the sim's own
            # action-repeat.
            row.note = f"odometry {odo.status}"
            return

        image = frame.image
        row.n_detections = len(detections)
        readings = self.vision.health.update(image, detections)
        tracked = self.tracker.update(detections, self.vision.plan, odo, frame.t)

        if tracked.hero is None:
            row.note = "no hero box"
            return
        hero_pos = tracked.hero.pos
        hero_vel = tracked.hero.vel

        hero_hp = next((r.hp for r in readings if r.label == "player" and r.hp is not None), None)
        if hero_hp is None:
            row.note = "no hero hp"
            return

        self._read_own_bars(image, detections, row)
        self._update_enemy_hp(tracked, readings, odo)
        n_enemies = self._read_enemies_left(image)
        if n_enemies is None:
            row.note = "no brawlers-left read"
            return

        obs = self.assembler.assemble(
            hero_pos=hero_pos,
            hero_vel=hero_vel,
            shadow=self.shadow.observe(),
            hero_hp=hero_hp,
            n_enemies_alive=n_enemies,
            elapsed_s=time.perf_counter() - self._match_t0,
            hero_in_bush=self._in_bush(hero_pos),
            hero_in_zone=self._in_gas(hero_pos),
            enemies=tracked.enemies,
            enemy_hp=dict(self._enemy_hp),
            enemy_in_bush={slot: self._in_bush(tr.pos)
                           for slot, tr in enumerate(tracked.enemies) if tr is not None},
            projectiles=self.projectiles.snapshot(hero_pos),
            zone=self.zone.estimate(self.grid.gas, hero_pos),
            grid=self.grid.build(hero_pos, alive=self.shadow.alive,
                                 enemies=tracked.enemies,
                                 projectiles=self.projectiles.live(),
                                 crates=self.loot.crates(), cubes=self.loot.cubes()),
        )

        # Step A2 cadence columns, written BEFORE the policy call so a policy that raises still
        # leaves what it was handed on the row. `legal` is the mask the network gets, read once
        # and used for both, so the bitmask can never disagree with the decision it explains.
        legal = self.shadow.attack_mask()
        row.attack_legal = sum(int(bool(v)) << i for i, v in enumerate(legal))
        row.attack_cd_shadow = float(self.shadow.attack_cd)
        row.attack_idle_t_shadow = float(self.shadow.attack_idle_t)
        row.enemy_in_reach = self._enemy_in_reach(hero_pos, tracked.enemies)

        try:
            decision = self.policy.act(obs, legal)
        except Exception as exc:                      # noqa: BLE001 -- re-raised after failing closed
            self._stop(f"policy raised: {exc!r}")
            raise

        # `act` returns what the shadow will actually MODEL -- `ATTACK_NONE` if its own mask
        # refuses. Pressing the policy's choice instead would model a shot the game never took,
        # which is the one error the shadow must never make. The aim is read AFTER `act`, which is
        # what sets the move bin it is read from.
        modelled = self.shadow.act(decision.move_bin, decision.attack)
        self.controls.joystick.apply(decision.move_bin)
        self.controls.buttons.press(modelled, self.shadow.attack_bearing)
        self._last_decision = decision
        row.decision = True
        row.move_bin = decision.move_bin
        # What was SENT, not what was chosen. They differ when the shadow's mask refused the
        # policy's pick, and a telemetry row that recorded the choice would show a shot the game
        # never saw.
        row.attack = modelled

    def _enemy_in_reach(self, hero_pos, enemies) -> bool:
        """`scripts/audit_attack_cadence.py`'s `enemy_in_reach` (Step A1.1), on the deployment's
        inputs: any enemy track the assembler was handed within `reach_tiles` of the hero, both
        in WORLD tiles. The sim's "alive and revealed to the hero" is, on this side, "has a track"
        -- the tracker only holds enemies the detector saw (or is coasting for a few ticks), and
        those are exactly the enemies the observation carries. Telemetry only; nothing acts on it.
        """
        hx, hy = hero_pos
        for tr in enemies:
            if tr is None:
                continue
            if np.hypot(tr.pos[0] - hx, tr.pos[1] - hy) <= self.reach_tiles:
                return True
        return False

    def _read_own_bars(self, image, detections, row: "TickRow | None" = None) -> None:
        """Ammo and super off the hero's own box. Ammo is the desync canary (§6.3)."""
        from brawl_vision.object_detection.hp_detection.hero_bars import read_ammo, read_super

        # The shadow's clip is recorded FIRST, whether or not this frame has a hero box: it exists
        # regardless of the box, and `scripts/audit_attack_cadence.py --telemetry` reads it as the
        # `ammo` of every decision row (Step A2 review) -- a `-1.0` sentinel next to a valid
        # `attack_legal` would reach the audit as a clip size. Recorded BEFORE `check_ammo` and
        # before the resync below, so `ammo_cv` / `ammo_shadow` hold what the two sides actually
        # disagreed about rather than the corrected value (pre-resync on a resync row; the mask
        # `_decide` reads after this returns is the post-resync one, which refuses the dash, so no
        # audit statistic ever reads that row's ammo).
        if row is not None:
            row.ammo_shadow = float(self.shadow.ammo)
        hero_det = next((d for d in detections if d.label == "player"), None)
        if hero_det is None:
            return
        self.shadow.set_super(read_super(image, hero_det, self.vision.cfg))
        reading = read_ammo(image, hero_det, self.vision.cfg)
        if row is not None and reading is not None:
            row.ammo_cv = float(reading.ammo)
        desync = self.shadow.check_ammo(reading)
        if not desync.tripped:
            return
        # PASS THE READING. `resync()` with no argument reseeds the timers and resets the strike
        # count but leaves `ammo` at the value that just failed the comparison -- so the canary
        # detects the disagreement, declines to fix the one observable it is holding, and trips
        # again three reads later. That burns `shadow_max_resyncs` and stops the match without
        # ever converging. Only a DESYNC verdict reaches here and a DESYNC requires a real read
        # (a miss returns NO_READ, which is not `tripped`), so this is never None in practice.
        self.shadow.resync(reading)
        # Step A2: the row records that the shadow resynced on this decision and what tripped it,
        # whether or not the backend injects -- it is a fact about the shadow, and the audit's
        # "resyncs per minute in fights" is only meaningful on a live backend anyway.
        if row is not None:
            row.resync = True
            row.resync_error = float(desync.error)
        if not self.controls.backend.injects:
            # A dry run breaks the link the canary watches, on purpose: the shadow spends ammo for
            # attacks that were never injected, so the game keeps more than we predicted and the
            # error is positive every time. That is the canary WORKING, so it is still logged --
            # but counting it would stop every dry run longer than about half a minute for a
            # reason that is not real. The first live run reached 4/5 in 22 s.
            if not self._dry_desync_noted:
                self._dry_desync_noted = True
                self.log(f"shadow desync ({desync}) -- expected under a null backend, since the "
                         f"attacks it models were never injected; not counted")
            return
        self._resyncs += 1
        self.log(f"shadow desync ({desync}); resync {self._resyncs}/{self.cfg.shadow_max_resyncs}")
        if self._resyncs > self.cfg.shadow_max_resyncs:
            self._stop(f"shadow desynced {self._resyncs} times in one match")

    def _read_enemies_left(self, image) -> int | None:
        """`meta.n_enemies_alive`, holding the last successful read through a failed one.

        Holding is not fabrication: it is the same thing `HealthTracker` does for an occluded HP,
        and the count changes only on a kill. Before the FIRST successful read there is nothing to
        hold, and the decision is skipped rather than seeded.
        """
        left = self.vision.hud.brawlers_left(image)
        value = left.enemies_alive(self.shadow.alive) if left.ok else None
        if value is not None:
            self._n_enemies = int(value)
        return self._n_enemies

    def _update_enemy_hp(self, tracked, readings, odo) -> None:
        """`{slot: hp}` for `assemble`, under §6.5's promotion rule.

        A slot holds its last committed HP for the life of its track and is dropped the moment the
        track is retired or replaced -- carrying a dead enemy's number onto the next occupant of
        the slot would be a fabricated value wearing a real one's clothes. Slots are matched to
        readings by world position because `Track` deliberately keeps no reference to the
        `Detection` it came from; the two are the same frame's numbers, so the gate can be tight.
        """
        from brawl_vision.object_detection.project import to_tiles

        live = {slot: tr.id for slot, tr in enumerate(tracked.enemies) if tr is not None}
        for slot, track_id in list(self._slot_ids.items()):
            if live.get(slot) != track_id:
                self._slot_ids.pop(slot, None)
                self._enemy_hp.pop(slot, None)
        self._slot_ids.update(live)

        enemy = [r for r in readings if r.label == "enemy" and r.hp is not None]
        if not enemy:
            return
        placed = to_tiles([r.detection for r in enemy], self.vision.plan)
        for reading, (_, tile) in zip(enemy, placed):
            slot = self._nearest_slot(tracked.enemies, tile, odo)
            if slot is not None:
                self._enemy_hp[slot] = int(reading.hp)

    def _nearest_slot(self, enemies, tile, odo) -> int | None:
        """The slot whose track sits on this detection, or `None` if none is close enough.

        Camera-relative tiles compared against world tiles minus the camera position -- the same
        subtraction the tracker did on the way in, so a track seen this tick lands within float
        noise. The half-tile gate is there for the tracks that are COASTING, which should not
        claim a reading they were not detected for.
        """
        px, py = odo.position_tiles
        best, best_d = None, 0.5
        for slot, tr in enumerate(enemies):
            if tr is None:
                continue
            d = np.hypot(tr.pos[0] - px - tile[0], tr.pos[1] - py - tile[1])
            if d < best_d:
                best, best_d = slot, d
        return best

    # -- terrain lookups ------------------------------------------------------

    def _cell(self, world_xy) -> tuple[int, int] | None:
        ox, oy = self.vision.occupancy.origin
        col = int(np.floor(world_xy[0])) - ox
        row = int(np.floor(world_xy[1])) - oy
        if 0 <= col < self.vision.occupancy.width and 0 <= row < self.vision.occupancy.height:
            return (col, row)
        return None

    def _in_bush(self, world_xy) -> bool:
        """Bush from the accumulated map, not from this frame.

        An UNKNOWN cell reads `False`, which is the same optimistic direction `ZoneEstimator`
        documents and the same one `GridBuilder`'s `unknown_tile=FLOOR` already commits the grid
        group to. One convention, applied twice, rather than two.
        """
        from brawl_vision.terrain.labeling import CLASSES

        cell = self._cell(world_xy)
        if cell is None:
            return False
        idx = int(self.vision.occupancy.best()[cell[1], cell[0]])
        return idx != _UNKNOWN and CLASSES[idx] is Tile.BUSH

    def _in_gas(self, world_xy) -> bool:
        cell = self._cell(world_xy)
        if cell is None:
            return False
        return bool(self.grid.gas.gassed[cell[1], cell[0]])

    # -- phase and the interlock ----------------------------------------------

    def _gate_transition(self, gate: bool, frame, row: TickRow) -> bool:
        """Apply the match gate. Returns whether the tick should continue into perception."""
        if self.phase is Phase.PLAYING:
            if not gate:
                self._end_match("match gate closed")
                return False
            if time.perf_counter() - self._match_t0 > self.cfg.safety_max_match_seconds:
                self._end_match("match watchdog expired")
                self._stop("match ran past safety_max_match_seconds")
                return False
            return True

        if self.phase is not Phase.WAITING or not gate:
            return False
        return self._begin_match(frame, row)

    def _begin_match(self, frame, row: TickRow) -> bool:
        """The gate has opened. Refine it, reset every accumulator, take the contact.

        **Refinement is not optional and it is not configurable** (§5). The stored radius was
        measured on an OBS recording and `ring_score_at` is sharply radius-sensitive -- 6.7 px of
        error turns 0.984 into 0.083. A refine that cannot reach threshold means this frame is not
        gameplay, so the gate is dropped and the loop keeps waiting rather than starting a match
        against a menu.
        """
        if not self._warmed:
            self.vision.warm(frame.image)
            self._warmed = True
            row.warmup = True        # this row, not telemetry[-1] -- it is appended at tick end
        try:
            score = self.match.refine(frame.image)
        except ValueError as exc:
            self.match.force_exit()
            self.log(f"gate opened but refine failed, staying out: {exc}")
            return False

        self.log(f"match started (gate refined to {score:.3f})")
        self.shadow.reset()
        # A new match is a new world frame even if odometry never cut: a new epoch at phase 0, and
        # every consumer adopts it. The occupancy map, which has no reset here, sees the new epoch
        # on its next deposit and drops the last match's votes.
        self.lattice.reset(self.vision.odometry.segment)
        self.tracker.reset(self.lattice.epoch)
        self.projectiles.reset(self.lattice.epoch)
        self.loot.reset(self.lattice.epoch)
        self.grid.reset(self.lattice.epoch)
        self.vision.health.reset()
        self._enemy_hp.clear()
        self._slot_ids.clear()
        self._n_enemies = None
        self._resyncs = 0
        self._dry_desync_noted = False
        self._odometry_lost = 0
        self._ticks_in_match = 0
        self._match_t0 = time.perf_counter()
        self.phase = Phase.PLAYING
        self.controls.joystick.acquire()
        return True

    def _end_match(self, reason: str) -> None:
        """Out of `PLAYING` without stopping the loop. Contacts go up first, always."""
        self.controls.release_all()
        self.match.force_exit()
        self.phase = Phase.WAITING
        self.log(f"match ended: {reason}")

    def _check_window(self) -> bool:
        """`WindowGuard`, on a cadence. Occlusion pauses; everything else stops.

        Occlusion is transient by nature -- a notification, a dragged window -- and the world model
        survives it because odometry and the occupancy map are both frozen rather than corrupted
        while nothing deposits. A moved or closed window is not transient, and every geometry
        constant in the package is wrong the moment it happens.
        """
        due = self._tick_index % self.cfg.window_check_every_n_ticks == 0
        if not due:
            return self.phase is not Phase.PAUSED
        reason = self.guard.check()
        if reason is None:
            if self.phase is Phase.PAUSED:
                self.phase = Phase.WAITING
                self.log("occlusion cleared")
            return True
        # `reason.kind`, never a substring of `reason`. This branch was `"occlud" in reason`
        # against a message that says "drawn over", so it never fired and every occlusion took
        # the terminal path below -- found by a live run on 2026-09-09, not by a test.
        if reason.recoverable:
            if self.phase is not Phase.PAUSED:
                self._pause(reason)
            return False
        self._stop(reason)
        return False

    def _check_stall(self, now: float) -> bool:
        if self._last_grab_t and now - self._last_grab_t > self.cfg.safety_capture_stall_seconds:
            self._stop(f"capture stalled for {now - self._last_grab_t:.2f}s")
            return False
        return True

    def _check_odometry(self, odo) -> bool:
        """Sustained loss stops the loop; a cut does not. See `ODOMETRY_LOST_SECONDS`."""
        if odo.status == "ok":
            self._odometry_lost = 0
            return True
        self._odometry_lost += 1
        if self._odometry_lost >= self.odometry_lost_ticks:
            self._stop(f"odometry {odo.status} for {self._odometry_lost} ticks")
            return False
        return True

    def _elapsed_since_last_tick(self, now: float) -> float:
        """Real seconds, for `ShadowHero.advance` to spend as whole sub-ticks.

        Deliberately wall-clock and not `self.tick_seconds`: a tick that ran long really did let
        the game's cooldowns run, and feeding the nominal period instead is how a shadow drifts
        slower than the thing it is shadowing.
        """
        return self.tick_seconds if not self._last_grab_t else now - self._last_grab_t

    def _pause(self, reason: str) -> None:
        self.controls.release_all()
        self.phase = Phase.PAUSED
        self.log(f"paused: {reason}")

    def _stop(self, reason: str) -> None:
        """Terminal. Release everything, drop the gate, refuse to emit again."""
        if self.phase is Phase.STOPPED:
            return
        self.controls.release_all()
        self.match.force_exit()
        self.phase = Phase.STOPPED
        self.stop_reason = reason
        self.log(f"STOPPED: {reason}")

    # -- telemetry ------------------------------------------------------------

    def write_csv(self, path) -> None:
        """Dump the ring to CSV. Scalars only, so this stays small however long the run was."""
        import csv
        from dataclasses import asdict, fields

        rows = list(self.telemetry)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, [f.name for f in fields(TickRow)])
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)
