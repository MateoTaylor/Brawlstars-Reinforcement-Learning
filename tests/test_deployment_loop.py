"""`brawl_deployment.loop` -- the phase machine and the interlock.

**The thing under test is when input is emitted, not what the pixels said.** So the vision stages
are fakes and the deployment-side state is real: the shadow, the trackers, the gas map, the
joystick and the buttons all run for real against a recording backend, because those are where the
emit rules actually live. No emulator, no GPU, no model -- this joins the siloed fast set.

Every assertion about "no input" is against `backend.events`, which is the only place a touch can
come from. Checking `phase` instead would pass for a loop that set the right phase and pressed the
button anyway, which is the exact bug the interlock exists to make impossible.
"""
from pathlib import Path
from types import SimpleNamespace

import math

import cv2
import numpy as np
import pytest

from brawl_deployment.control.backend import SLOT_MOVE, SLOT_TAP
from brawl_deployment.control.buttons import (ATTACK_AUTO, ATTACK_FIRE, ATTACK_GADGET, ATTACK_NONE,
                                              ATTACK_SUPER)
from brawl_deployment.loop import (ODOMETRY_LOST_SECONDS, TELEMETRY_ROWS, Controls, DeployLoop,
                                   Phase, TickRow, VisionStack)
from brawl_deployment.config import DeploymentConfig
from brawl_deployment.control.buttons import Buttons
from brawl_deployment.control.joystick import Joystick
from brawl_deployment.perception.assemble import ObservationAssembler
from brawl_deployment.perception.known_map import KnownMap
from brawl_deployment.perception.lattice import LatticePhase
from brawl_deployment.perception.loot import crate_occlusion
from brawl_deployment.policy import Decision
from brawl_deployment.window import WindowFault
from brawl_sim.config import load_config
from brawl_sim.constants import (TILE_BLOCKS_PROJ, TILE_BLOCKS_UNIT, TILE_IS_BUSH, TILE_IS_WATER,
                                 Tile)
from brawl_sim.core.obs_select import agent_obs_index_map, agent_space, load_agent_spec
from brawl_vision.camera import HERO_ANCHOR_TILES
from brawl_vision.capture import Frame
from brawl_vision.hud import BrawlersLeft
from brawl_vision.object_detection.detector import Detection
from brawl_vision.object_detection.projectile_detection.classes import (CUBE_BOX, CUBE_DROPPED,
                                                                        PROJECTILE)
from brawl_vision.terrain.labeling import CLASS_INDEX
from brawl_vision.terrain.odometry import OdometryResult

CFG = load_config("configs/default.yaml")
VIEWPORT = (2002, 1126)

# The shipped projectile model's classes. The ids are the export's (`classes.py` says why they are
# not pinned anywhere); only the set matters to the loop.
PROJECTILE_NAMES = {0: CUBE_BOX, 1: CUBE_DROPPED, 2: PROJECTILE}


# -- fakes -------------------------------------------------------------------------------------

class _Backend:
    """Records instead of injecting. The whole interlock is an assertion about this list."""

    def __init__(self):
        self.events = []
        self.log = []          # the same touches with their coordinates, `NullBackend.log`'s shape

    def down(self, slot, x, y):
        self.events.append(("down", slot))
        self.log.append(("down", slot, x, y))

    def move(self, slot, x, y):
        self.events.append(("move", slot))
        self.log.append(("move", slot, x, y))

    def up(self, slot):
        self.events.append(("up", slot))
        self.log.append(("up", slot))

    def release_all(self):
        self.events.append(("release_all",))

    def close(self):
        pass

    def is_alive(self):
        return True

    @property
    def injects(self) -> bool:
        """True: this stub stands in for the REAL backend, so the loop must treat its actions as
        having landed. `NullBackend` is the one that answers False, and the difference is what
        decides whether an ammo desync spends the fail-closed budget."""
        return True

    @property
    def touches(self):
        """Everything that would reach the device. `up` counts: it is still an event."""
        return [e for e in self.events if e[0] in ("down", "move", "up")]


class _Capture:
    """`grab_seconds` ACCUMULATES here, because it accumulates in the real capture.

    It used to be a fixed 0.02, and that is why the suite could not catch the loop reading a
    cumulative counter as a per-frame duration -- the first live run reported a 3516 ms median
    grab against a 55.8 ms tick. A stub that is constant where the real thing grows tests nothing
    about the difference.
    """

    PER_GRAB = 0.02

    def __init__(self):
        self.image = np.zeros((VIEWPORT[1], VIEWPORT[0], 3), np.uint8)
        self.grab_seconds = 0.0
        self.index = 0
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return False

    def grab(self):
        frame = Frame(image=self.image, t=self.index * 0.05, index=self.index)
        self.index += 1
        self.grab_seconds += self.PER_GRAB
        return frame


class _Guard:
    """Returns REAL `WindowFault`s, not lookalike strings.

    This stub used to hand back plain prose the test author invented ("occluded by 2 window(s)"),
    which happened to contain the substring `loop.py` was sniffing for. The real
    `WindowGuard.check` says "drawn over the emulator" and contains no such substring, so the
    pause path was dead in production while these tests reported it working. A stub that produces
    a different type than the thing it stands in for tests the stub.
    """

    def __init__(self, reason=None):
        self.reason = reason

    def check(self):
        return self.reason


class _Match:
    def __init__(self):
        self.gate = False
        self.refine_ok = True
        self.refines = 0
        self.exits = 0

    def update(self, frame):
        return self.gate

    def refine(self, frame, min_score=None):
        self.refines += 1
        if not self.refine_ok:
            raise ValueError("this frame is not gameplay")
        return 0.9

    def force_exit(self):
        # The real `MatchState.force_exit` resets the hysteresis counters; it does not decide the
        # ring score. Modelling it as "the gate is now closed" would make every fail-closed path
        # look self-latching, which is the opposite of what it does.
        self.exits += 1

    @property
    def in_match(self):
        return self.gate


class _Odometry:
    def __init__(self):
        self.segment = 0
        self.status = "ok"
        self.position = (0.0, 0.0)

    def update(self, rect):
        return OdometryResult(delta_tiles=(0.0, 0.0), position_tiles=self.position,
                              response=1.0, agreement_tiles=0.0, inlier_ratio=1.0,
                              n_windows=11, status=self.status, segment=self.segment)


class _Detection:
    """Duck-typed to what `to_tiles` and the label filters read off a real `Detection`."""

    def __init__(self, label, point):
        self.label = label
        self.point = point
        # A sprite-sized box standing on the point, for `loot.box_occlusion`: one tile wide, two
        # tall, feet at the anchor.
        x, y = point
        self.xyxy = (x - 24.0, y - 88.0, x + 24.0, y + 8.0)

    def anchor(self, frac=0.30):
        return self.point


class _Plan:
    """A real-arithmetic plan with an identity homography.

    `to_tiles` runs for real against it -- viewport pixels straight through
    `perspectiveTransform`, then the genuine `rect_to_tile` formula -- so the projection and the
    tracker are exercised rather than stubbed. Only the warp is trivial.
    """

    size_tiles = (21, 13)
    origin_tile = (-10, -6)
    pixels_per_tile = 48
    size_px = (21 * 48, 13 * 48)
    viewport = VIEWPORT
    M = np.eye(3, dtype=np.float64)
    M_inv = np.eye(3, dtype=np.float64)
    valid = np.ones((13 * 48, 21 * 48), bool)

    def rectify(self, image):
        return image

    def rect_to_tile(self, rect):
        r = np.asarray(rect, np.float64).reshape(-1, 2)
        return r / self.pixels_per_tile + np.asarray(self.origin_tile, np.float64)

    def tile_to_rect(self, tiles):
        t = np.asarray(tiles, np.float64).reshape(-1, 2)
        return (t - np.asarray(self.origin_tile, np.float64)) * self.pixels_per_tile

    def registered(self, position_tiles):
        """`RectifyPlan.registered`'s arithmetic: the content shifted by the remainder of
        `origin_tile + position_tiles`, and the origin moved to match. Only the warp stays trivial."""
        at = np.asarray(self.origin_tile, np.float64) + np.asarray(position_tiles, np.float64)
        f = at - np.round(at)
        if np.abs(f).max() < 1e-9:
            return self
        out = type(self)()
        T = np.array([[1.0, 0.0, f[0] * self.pixels_per_tile],
                      [0.0, 1.0, f[1] * self.pixels_per_tile], [0.0, 0.0, 1.0]])
        out.M = T @ self.M
        out.M_inv = np.linalg.inv(out.M)
        out.origin_tile = tuple(np.asarray(self.origin_tile, np.float64) - f)
        out.valid = cv2.warpAffine(self.valid.astype(np.uint8), T[:2], self.size_px,
                                   flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
        return out


HERO_PX = (10 * 48, 6 * 48)          # -> camera-relative tile (0, 0)


class _Reading:
    """What `HealthTracker.update` hands back, reduced to what the loop reads."""

    def __init__(self, label, hp, detection=None):
        self.label = label
        self.hp = hp
        self.detection = detection


class _Zone:
    """What `detect_zone` returns, reduced to what `GasMap.update` reads off it."""

    def __init__(self, tiles=(21, 13), gassed=False):
        cols, rows = tiles
        self.cells = np.full((rows, cols), gassed, bool)
        self.observed = np.ones((rows, cols), bool)

    def at_least(self, fraction):
        return self.observed & self.cells


class _Occupancy:
    """A real-shaped map with an inert `update`. `GridBuilder` and the terrain lookups read it;
    the deposit itself is `brawl_vision`'s and is tested there."""

    def __init__(self, size=128):
        self.height = self.width = size
        self._best = np.full((size, size), -1, np.int8)
        self.deposits = 0
        self.occluded = None
        self.plan = None

    @property
    def origin(self):
        return (-(self.width // 2), -(self.height // 2))

    def best(self):
        return self._best

    def update(self, cells, odometry, plan, zone=None, occluded=None, cfg=None):
        self.deposits += 1
        self.occluded = occluded
        self.plan = plan


class _Detector:
    def __init__(self, detections=(), names=None):
        self.detections = list(detections)
        self.names = dict(names or {})
        self.calls = 0

    def predict(self, image, conf=None):
        self.calls += 1
        return list(self.detections)


class _Health:
    def __init__(self, readings=()):
        self.readings = list(readings)
        self.resets = 0

    def update(self, image, detections):
        return list(self.readings)

    def reset(self):
        self.resets += 1


class _Hud:
    def __init__(self, count=4):
        self.count = count

    def brawlers_left(self, image):
        return BrawlersLeft(count=self.count, status="ok" if self.count is not None else "no-read")


class _Classifier:
    def predict(self, rect, plan):
        cols, rows = _Plan.size_tiles
        return np.zeros((rows, cols), np.int8), np.ones((rows, cols), np.float32)


class _Assembler:
    def __init__(self):
        self.calls = []

    def assemble(self, **kw):
        self.calls.append(kw)
        return {"obs": np.zeros(1, np.float32)}


class _Policy:
    """Real config and a real agent-obs spec -- the loop derives its rates, slot counts and grid
    channels from them -- and a scripted act.

    `real_assembler=True` swaps the recording `_Assembler` for the real `ObservationAssembler`,
    which is what `DeployedPolicy.make_assembler` returns. The recorder accepts any keyword at all,
    so it cannot see a field the spec needs and the loop never supplies; the real one raises.
    """

    def __init__(self, decision=None, spec_path="configs/agent_obs_deploy.yaml",
                 real_assembler=False, cfg=None):
        self.cfg = CFG if cfg is None else cfg
        self.spec = load_agent_spec(spec_path, self.cfg)
        self.assembler = (ObservationAssembler(self.spec, self.cfg) if real_assembler
                          else _Assembler())
        self.obs = []
        self.decision = decision or Decision(move_bin=3, attack=ATTACK_NONE,
                                             legal=(True, True, False, True))
        self.calls = 0
        self.raises = None
        self.move_legal = []

    def make_assembler(self, **kw):
        return self.assembler

    def act(self, obs, attack_legal, move_legal=None):
        self.calls += 1
        self.obs.append(obs)
        self.move_legal.append(move_legal)
        if self.raises is not None:
            raise self.raises
        # `DeployedPolicy.act`'s own contract: one legal per attack column (four, or five under
        # `action.auto_aim`), and the move half either absent or one flag per bin plus idle.
        assert len(tuple(attack_legal)) == self.cfg.action_nvec[1], attack_legal
        assert move_legal is None or len(move_legal) == self.cfg.n_move_bins + 1, move_legal
        return self.decision


def _patch_vision(monkeypatch):
    """The three vision reads the fixture stubs out: the zone, and the ammo and super bars."""
    monkeypatch.setattr("brawl_vision.terrain.zone.detect_zone",
                        lambda rect, plan, cfg=None: _Zone())
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_ammo",
                        lambda image, det, cfg=None: None)
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_super",
                        lambda image, det, cfg=None: None)


@pytest.fixture
def loop(monkeypatch):
    """A `DeployLoop` wired to fakes on the vision side and the real thing everywhere else."""
    _patch_vision(monkeypatch)
    return _build_loop(_Policy())


def _build_loop(policy):
    """The fixture's loop around `policy`; `_patch_vision` first."""
    backend = _Backend()
    controls = Controls(backend=backend,
                        joystick=Joystick(backend, (345.6, 669.6), 110.0, n_bins=16),
                        buttons=Buttons(backend, attack=(1690.0, 594.0), super_=(1462.5, 1000.5),
                                        gadget=(1559.9, 901.1)))
    vision = VisionStack(plan=_Plan(), odometry=_Odometry(), occupancy=_Occupancy(),
                         classifier=_Classifier(),
                         entities=_Detector([_Detection("player", HERO_PX)]),
                         projectiles=_Detector(names=PROJECTILE_NAMES),
                         health=_Health([_Reading("player", 8000)]),
                         hud=_Hud(), cfg=None)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=vision,
                    policy=policy, controls=controls, cfg=DeploymentConfig())
    lp.backend = backend                       # test handle; the loop never reads it
    lp.vision.warm = lambda image: None         # no detectors to JIT
    return lp


def _play(loop, ticks=1):
    """Open the gate and run, which is how every emitting test starts."""
    loop.match.gate = True
    for _ in range(ticks):
        loop.tick()


def _decisions(loop):
    """The telemetry rows for ticks that actually reached the policy.

    Not `telemetry[-1]`: a decision lands on the FIRST tick of its window, so the last row of any
    run is almost always a held tick with nothing to say.
    """
    return [r for r in loop.telemetry if r.decision]


def _last_attempt(loop):
    """The last row that was a decision TICK, whether or not a decision came out of it."""
    return [r for r in loop.telemetry if r.decision or r.note][-1]


# -- the interlock -----------------------------------------------------------------------------

def test_nothing_is_emitted_before_the_gate_opens():
    """The single most important assertion in the package. A loop started on a menu must be
    completely silent, and silence is a property of the BACKEND, not of a phase flag."""
    lp = _loop_fixture_value()
    for _ in range(20):
        lp.tick()
    assert lp.phase is Phase.WAITING
    assert lp.backend.touches == []


def test_nothing_is_emitted_after_a_stop(loop):
    _play(loop, 3)
    before = len(loop.backend.touches)
    loop._stop("test")
    for _ in range(10):
        loop.tick()
    assert loop.phase is Phase.STOPPED
    # The stop itself releases; nothing after it presses.
    assert all(e[0] == "up" for e in loop.backend.touches[before:])


def test_the_gate_closing_releases_every_contact(loop):
    _play(loop, 3)
    assert ("down", SLOT_MOVE) in loop.backend.events
    loop.match.gate = False
    loop.tick()
    assert loop.phase is Phase.WAITING
    assert loop.backend.events[-1] == ("up", SLOT_MOVE)
    assert not loop.controls.joystick.is_down


def test_a_gate_that_cannot_be_refined_does_not_start_a_match(loop):
    """`refine` raising means the frame is not gameplay -- a false gate open. Starting anyway
    would have the agent playing a menu, which is precisely what the gate is for."""
    loop.match.refine_ok = False
    _play(loop, 5)
    assert loop.phase is Phase.WAITING
    assert loop.backend.touches == []
    assert loop.match.refines == 5           # it keeps trying, it just never believes it


# -- phases ------------------------------------------------------------------------------------

def test_the_gate_opening_refines_once_and_takes_the_contact(loop):
    _play(loop, 4)
    assert loop.phase is Phase.PLAYING
    assert loop.match.refines == 1           # once at the transition, not per tick
    assert ("down", SLOT_MOVE) in loop.backend.events


def _occluded() -> WindowFault:
    """Exactly what `WindowGuard.check` emits for a covering window, message and all."""
    return WindowFault("occluded", "1 window(s) are drawn over the emulator (hwnd [67844]). "
                                   "The capture is showing them, not the game.")


# -- the ammo canary's fail-closed budget --------------------------------------------------

class _Ammo:
    """What `read_ammo` returns. Only `.ammo` is read."""

    def __init__(self, ammo):
        self.ammo = ammo


def _force_desync(loop, monkeypatch, ammo=3.0):
    """Make every read say the clip is full while the shadow spends it, which is exactly the
    signature of attacks that never landed: error is CV minus shadow, so it goes positive."""
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_ammo",
                        lambda image, det, cfg=None: _Ammo(ammo))
    loop.shadow.ammo = 0.0
    loop.shadow._last_attack_at = -1e9          # past the grace window
    loop.shadow.strikes = loop.cfg.shadow_ammo_strikes


def test_a_sustained_ammo_desync_eventually_stops_the_loop(loop, monkeypatch):
    """§8: the ammo canary is the earliest signal that our actions are not landing, and a run
    where they never land is a run that must not keep playing."""
    _play(loop, 2)
    for _ in range(loop.cfg.shadow_max_resyncs + 2):
        _force_desync(loop, monkeypatch)
        loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)])
    assert loop.phase is Phase.STOPPED
    assert "desynced" in loop.stop_reason


def test_a_dry_run_does_not_spend_the_resync_budget_on_its_own_null_backend(loop, monkeypatch):
    """The bug the first live run exposed: a dry run injects nothing, so the shadow spends ammo
    for attacks that never happened and the canary trips every time -- correctly. Counting those
    stopped the run at 5, for a disconnection we created deliberately. It reached 4/5 in 22 s."""
    from brawl_deployment.control.backend import NullBackend

    messages: list[str] = []
    loop.log = messages.append
    _play(loop, 2)
    object.__setattr__(loop.controls, "backend", NullBackend())
    for _ in range(loop.cfg.shadow_max_resyncs + 5):
        _force_desync(loop, monkeypatch)
        loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)])
    assert loop.phase is not Phase.STOPPED
    assert loop._resyncs == 0
    assert sum("expected under a null backend" in m for m in messages) == 1


def test_a_resync_actually_adopts_the_ammo_it_just_read(loop, monkeypatch):
    """`resync()` takes an optional reading, and the loop was calling it WITHOUT one -- so the
    canary noticed the disagreement, reseeded the timers, reset the strikes, and left `ammo` at
    the value that had just failed. It then re-tripped three reads later, every time, spending
    `shadow_max_resyncs` and stopping the match on a gap it was holding the fix for.

    That matters more now than it looks: the operator has said the kit constants are
    approximations, so a `reload_seconds` that is merely a little wrong is exactly what produces
    a steady one-sided ammo gap. Converging on CV is what keeps that a logged note instead of a
    stopped match."""
    _play(loop, 2)
    _force_desync(loop, monkeypatch, ammo=3.0)
    loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)])
    assert float(loop.shadow.ammo) == 3.0
    assert loop.shadow.strikes == 0


def test_both_sides_of_the_canary_reach_telemetry_at_their_pre_resync_values(loop, monkeypatch):
    """The operator has said `configs/brawlers.yaml`'s move/attack/reload numbers are
    APPROXIMATIONS and that the live game is the better source. `ammo_cv` and `ammo_shadow` are
    what let `reload_seconds` be refit from an ordinary run, so they must record the values that
    DISAGREED -- writing them after `resync()` would log the correction and lose the evidence."""
    from brawl_deployment.loop import TickRow

    _play(loop, 2)
    _force_desync(loop, monkeypatch, ammo=3.0)      # shadow at 0.0, CV says a full clip
    row = TickRow(index=0, t=0.0, grab_ms=0.0)
    loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)], row)

    assert (row.ammo_cv, row.ammo_shadow) == (3.0, 0.0)
    assert float(loop.shadow.ammo) == 3.0, "the resync still ran; only the logging order changed"


def test_a_tick_with_no_ammo_read_records_no_ammo_rather_than_a_plausible_zero(loop, monkeypatch):
    """`read_ammo` returns None on ~12% of frames. A missing read is not an empty clip, and a
    refit that averaged those zeros in would put Mortis's reload somewhere it has never been."""
    from brawl_deployment.loop import TickRow

    _play(loop, 2)
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_ammo",
                        lambda image, det, cfg=None: None)
    row = TickRow(index=0, t=0.0, grab_ms=0.0)
    loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)], row)

    assert row.ammo_cv == -1.0
    assert row.ammo_shadow >= 0.0, "the shadow always has a value, even when the read misses"


def test_the_warmup_is_paid_while_waiting_and_flagged_on_the_row_that_paid_for_it(loop):
    """`VisionStack.warm` costs ~1 s of PTX JIT, once. It is paid on the run's first tick, while
    the gate is still closed, so no match tick carries it: on the gate's tick it held up the first
    decision, and with the gate's own work that tick reached 1.22 s. It has to be attributed to
    THAT row -- telemetry is appended at tick end, so reaching for `[-1]` while inside the tick
    flags the previous one and the summary then excludes a healthy tick and keeps the slow one."""
    warmed = []
    loop.vision.warm = lambda image: warmed.append(image)
    for _ in range(3):
        loop.tick()                            # the gate is closed
    _play(loop, 4)
    rows = [r for r in loop.telemetry if r.warmup]
    assert len(warmed) == 1 and len(rows) == 1
    assert rows[0].index == 0 and rows[0].phase == "waiting"
    assert loop.phase is Phase.PLAYING


def test_the_telemetry_grab_time_is_per_tick_not_the_running_total(loop):
    """`capture.grab_seconds` is cumulative and the row wants a duration. Read straight through,
    every row reports the total so far -- which climbs forever and looks like a catastrophic grab
    while the tick it is nested inside stays fast. That is what the first live run printed."""
    _play(loop, 8)
    rows = list(loop.telemetry)
    assert len(rows) >= 8
    for row in rows:
        assert row.grab_ms == pytest.approx(_Capture.PER_GRAB * 1e3)
    assert loop.capture.grab_seconds > _Capture.PER_GRAB      # the source really was accumulating


def test_the_real_occlusion_message_is_what_triggers_the_pause(loop):
    """The regression test for the bug the live run of 2026-09-09 found. The loop branched on
    `"occlud" in reason` against a message reading "drawn over", so occlusion took the terminal
    stop path -- the run ended for good the moment the operator alt-tabbed away. Asserting on the
    PRODUCER's own message is the part that matters; a hand-written stand-in is what hid it."""
    fault = _occluded()
    assert "occlud" not in fault.lower()      # the substring the old code looked for
    assert fault.recoverable
    _play(loop, 3)
    loop.guard.reason = fault
    loop._tick_index = 0
    loop.tick()
    assert loop.phase is Phase.PAUSED
    assert loop.stop_reason is None           # paused, not stopped


def test_occlusion_pauses_and_releases_but_keeps_capturing(loop):
    """§8: a window over the emulator is transient and the world model survives it, so this is the
    one failure that does not end the run. It must still let go of the stick."""
    _play(loop, 3)
    loop.guard.reason = _occluded()
    loop._tick_index = 0                     # land on the guard's cadence
    loop.tick()
    assert loop.phase is Phase.PAUSED
    assert not loop.controls.joystick.is_down

    grabs = loop.capture.index
    loop.tick()
    assert loop.capture.index > grabs        # still capturing
    assert loop.phase is Phase.PAUSED

    loop.guard.reason = None
    loop._tick_index = 0
    loop.tick()
    assert loop.phase is not Phase.PAUSED


def test_a_moved_window_stops_rather_than_pausing(loop):
    """Occlusion is recoverable; a moved window invalidates every geometry constant in the
    package, and the crop box was taken from the window at startup (6.11)."""
    _play(loop, 3)
    loop.guard.reason = WindowFault("moved", "the emulator window moved or resized: ...")
    loop._tick_index = 0
    loop.tick()
    assert loop.phase is Phase.STOPPED
    assert "moved" in loop.stop_reason


def test_a_capture_stall_stops_the_loop(loop, monkeypatch):
    _play(loop, 2)
    loop._last_grab_t -= loop.cfg.safety_capture_stall_seconds + 1.0
    loop.tick()
    assert loop.phase is Phase.STOPPED
    assert "stall" in loop.stop_reason


def test_the_warmup_is_not_a_capture_stall_but_a_later_gap_is(loop, monkeypatch):
    """The first K5 dry run (2026-09-28) stopped with "capture stalled for 1.22s" on the tick
    after the one that paid the warm-up. The guard is for a capture that stopped delivering, and
    the JIT is the loop's own cost, once: it comes out of the next tick's gap, and only that one.
    A clock the test moves, so the 5 s warm-up costs no real time."""
    clock = [100.0]
    monkeypatch.setattr("brawl_deployment.loop.time",
                        SimpleNamespace(perf_counter=lambda: clock[0], sleep=lambda s: None))

    def slow_warm(image):
        clock[0] += 5.0

    loop.vision.warm = slow_warm
    loop.tick()
    clock[0] += loop.tick_seconds
    loop.tick()
    assert loop.phase is Phase.WAITING and loop.stop_reason is None
    clock[0] += loop.cfg.safety_capture_stall_seconds + 1.0
    loop.tick()
    assert loop.phase is Phase.STOPPED and "stall" in loop.stop_reason


def test_the_match_watchdog_stops_a_match_that_never_ends(loop):
    """A gate stuck true is indistinguishable from a very long match, and the difference matters:
    `meta.time_frac` is clamped at 1, so past 150 s the agent is already out of distribution."""
    _play(loop, 2)
    loop._match_t0 -= loop.cfg.safety_max_match_seconds + 1.0
    loop.tick()
    assert loop.phase is Phase.STOPPED
    assert not loop.controls.joystick.is_down


# -- odometry ----------------------------------------------------------------------------------

def test_a_brief_odometry_loss_does_not_stop_the_loop(loop):
    """A cut is one tick and opens a new segment; it is not a fault. Stopping on it would end the
    run on every death and respawn."""
    _play(loop, 2)
    loop.vision.odometry.status = "lost"
    for _ in range(loop.odometry_lost_ticks - 1):
        loop.tick()
    assert loop.phase is Phase.PLAYING
    loop.vision.odometry.status = "ok"
    loop.tick()
    assert loop.phase is Phase.PLAYING


def test_the_odometry_patience_is_a_duration_not_a_tick_count(loop):
    """It reads two seconds in the docstring, so it has to be two seconds at whatever rate the
    loop actually runs. As a flat 40 it was two seconds only at 20 Hz and 3.3 s at the shipped 12,
    which is the kind of drift that survives a rate change unnoticed."""
    assert loop.odometry_lost_ticks * loop.tick_seconds == pytest.approx(ODOMETRY_LOST_SECONDS,
                                                                        abs=loop.tick_seconds)
    assert loop.odometry_lost_ticks == 24        # 2.0 s at the shipped 12 Hz


def test_a_sustained_odometry_loss_stops_the_loop(loop):
    """Past the threshold every world position is frozen, so the observation is stale rather than
    wrong -- which is worse, because nothing downstream can tell."""
    _play(loop, 2)
    loop.vision.odometry.status = "lost"
    for _ in range(loop.odometry_lost_ticks + 1):
        loop.tick()
    assert loop.phase is Phase.STOPPED
    assert "odometry" in loop.stop_reason


def test_odometry_ticks_do_not_reach_the_decision(loop):
    """The tracker refuses a bad-odometry tick itself; the loop must not have already spent a
    detector pass and a policy call on it."""
    _play(loop, 1)
    loop.vision.odometry.status = "uncertain"
    calls = loop.policy.calls
    for _ in range(loop.decision_every * 2):
        loop.tick()
    assert loop.policy.calls == calls
    assert _last_attempt(loop).note == "odometry uncertain"


# -- the two rates -----------------------------------------------------------------------------

def test_the_decision_PERIOD_is_the_configs_and_the_tick_rate_is_not(loop):
    """The split §6.13 makes. `decision_every` is no longer `action_repeat` -- it is however many
    ticks of the configured rate fill one trained decision period, and THAT is what has to match
    `dt * action_repeat`. Asserting the tick count instead would pin the wrong invariant."""
    assert loop.decision_every * loop.tick_seconds == pytest.approx(CFG.dt * CFG.action_repeat)
    assert loop.rates.agent_seconds == pytest.approx(CFG.dt * CFG.action_repeat)
    assert loop.decision_every == 3 and loop.rates.tick_hz == pytest.approx(12.0)
    # And explicitly NOT the sim's dt any more. This line used to assert the opposite; it is kept
    # inverted rather than deleted so the change is visible to whoever reads the test next.
    assert loop.tick_seconds != pytest.approx(CFG.dt)

    _play(loop, loop.decision_every * 3)
    assert loop.policy.calls == 3


def test_the_detectors_run_every_tick_and_a_decision_reuses_the_boxes(loop):
    """§1.2's split, asserted rather than assumed. Both detectors and the terrain deposit run at
    the perception rate. The entity detector used to run only on decisions; its boxes now mask the
    deposit on every tick, and a decision on the same tick reads those boxes instead of paying the
    ~8 ms twice for one frame."""
    n = loop.decision_every * 2
    _play(loop, n)
    assert loop.vision.entities.calls == n
    assert loop.vision.projectiles.calls == n
    assert loop.vision.occupancy.deposits == n
    assert loop.policy.calls == 2
    assert [r.n_detections for r in _decisions(loop)] == [1, 1]


def test_the_shadow_advances_on_wall_clock_not_on_the_nominal_period(loop):
    """A tick that ran long really did let the cooldowns run. Feeding `tick_seconds` instead is how
    a shadow drifts slower than the thing it is shadowing."""
    _play(loop, 1)
    ticks = loop.shadow.ticks
    # Half a second, which is under `safety_capture_stall_seconds` -- a longer gap is a stall, and
    # the loop is right to stop on it rather than to spend ten sub-ticks catching up.
    loop._last_grab_t -= 0.5
    loop.tick()
    assert loop.shadow.ticks - ticks >= int(0.5 / CFG.dt)


# -- the decision ------------------------------------------------------------------------------

def test_a_decision_moves_the_stick_and_the_stick_stays_down(loop):
    """Idle is a position, not a release -- the whole floating-joystick design turns on it."""
    _play(loop, loop.decision_every + 1)
    assert ("move", SLOT_MOVE) in loop.backend.events
    assert ("up", SLOT_MOVE) not in loop.backend.events
    assert loop.controls.joystick.is_down


def test_an_idle_decision_does_not_release_the_contact(loop):
    loop.policy.decision = Decision(move_bin=0, attack=ATTACK_NONE, legal=(True, True, False, True))
    _play(loop, loop.decision_every * 2)
    assert ("up", SLOT_MOVE) not in loop.backend.events
    assert loop.controls.joystick.is_down


def test_the_fire_bit_does_not_repeat_across_the_decision_window(loop):
    """`env._held` zeroes the fire column on sub-ticks 2..K. One press per decision, and it is
    finished inside the window: down, drag, lift, and the slot is free for the next decision."""
    loop.policy.decision = Decision(move_bin=1, attack=ATTACK_FIRE, legal=(True, True, False, True))
    _play(loop, loop.decision_every)
    taps = [e for e in loop.backend.events if e[1:] == (SLOT_TAP,)]
    assert taps == [("down", SLOT_TAP), ("move", SLOT_TAP), ("up", SLOT_TAP)]
    assert not loop.controls.buttons.is_held


def _tap_steps_per_tick(loop, ticks):
    """SLOT_TAP touches, grouped by the tick that sent them."""
    per_tick = []
    for _ in range(ticks):
        before = len(loop.backend.log)
        loop.tick()
        per_tick.append([e for e in loop.backend.log[before:] if e[1] == SLOT_TAP])
    return per_tick


def test_an_attack_is_dragged_along_the_move_bin_one_step_per_tick(loop):
    """The drag IS the aim, because a bare tap auto-aims and the sim's dash goes along the move
    bin. Bin 5 is a quarter turn from +x, which is DOWN the screen (no y flip): the press goes down
    on the attack point on the decision tick, drags `aim_radius_px` straight down on the next, and
    lifts -- which fires -- on the last tick of the window."""
    loop.policy.decision = Decision(move_bin=5, attack=ATTACK_FIRE, legal=(True, True, False, True))
    loop.match.gate = True
    ax, ay = loop.controls.buttons.attack
    r = loop.controls.buttons.aim_radius_px
    down, drag, lift = _tap_steps_per_tick(loop, loop.decision_every)
    assert down == [("down", SLOT_TAP, ax, ay)]
    assert len(drag) == 1 and drag[0][:2] == ("move", SLOT_TAP)
    assert drag[0][2:] == pytest.approx((ax, ay + r), abs=0.01)
    assert lift == [("up", SLOT_TAP)]


def test_an_idle_attack_is_dragged_the_way_the_hero_last_faced(loop):
    """`action.dash_on_idle: facing`. Walk down for one decision, then fire standing still: the
    dash goes the way the hero faces, so the drag must still go down -- NOT screen-right, which is
    where `facing` starts, and not a bare tap."""
    loop.match.gate = True
    loop.policy.decision = Decision(move_bin=5, attack=ATTACK_NONE, legal=(True, True, False, True))

    def tick_in_real_time():
        # The shadow spends WALL-CLOCK seconds as sub-ticks, and a test tick takes microseconds.
        # Backdating the last grab by one period is what lets its `facing` turn at all. (Not
        # before the first grab: 0 means "none yet", and backdating it would read as a stall.)
        if loop._last_grab_t:
            loop._last_grab_t -= loop.tick_seconds
        loop.tick()

    for _ in range(loop.decision_every):
        tick_in_real_time()
    loop.policy.decision = Decision(move_bin=0, attack=ATTACK_FIRE, legal=(True, True, False, True))
    before = len(loop.backend.log)
    for _ in range(loop.decision_every):
        tick_in_real_time()
    drags = [e for e in loop.backend.log[before:] if e[:2] == ("move", SLOT_TAP)]
    ax, ay = loop.controls.buttons.attack
    assert len(drags) == 1
    assert drags[0][2:] == pytest.approx((ax, ay + loop.controls.buttons.aim_radius_px), abs=0.01)


def test_a_press_in_flight_is_lifted_when_the_match_ends(loop):
    """Every path out of `PLAYING` lifts SLOT_TAP immediately -- `release`, not `settle`, which
    would only have dragged it one step further and left it down."""
    loop.policy.decision = Decision(move_bin=1, attack=ATTACK_FIRE, legal=(True, True, False, True))
    _play(loop, 1)
    assert loop.controls.buttons.is_held
    loop._end_match("test")
    assert not loop.controls.buttons.is_held
    assert loop.backend.log[-2:] == [("up", SLOT_TAP), ("up", SLOT_MOVE)]


def test_only_what_the_shadow_modelled_is_tapped(loop):
    """The shadow's mask is the last word: an uncharged super never reaches the device, because
    modelling a shot the game did not take is the one error the shadow must never make."""
    loop.policy.decision = Decision(move_bin=1, attack=ATTACK_SUPER, legal=(True, True, True, True))
    _play(loop, loop.decision_every)
    assert not loop.shadow.super_ready
    assert ("down", SLOT_TAP) not in loop.backend.events
    assert _decisions(loop)[-1].attack == ATTACK_NONE


def test_a_gadget_decision_is_a_bare_tap_on_the_gadget_button(loop):
    """Attack value 3 reaches the device as a tap on the gadget button,
    down on the decision tick and up on the next, never a drag, and the row records 3. The next
    decision's bitmask has lost bit 3 to the cooldown and kept bit 1, because a throw spends no
    ammo and takes no attack cooldown; the second throw it asks for is refused, and nothing is
    tapped."""
    loop.policy.decision = Decision(move_bin=5, attack=ATTACK_GADGET,
                                    legal=(True, True, False, True))
    loop.match.gate = True
    gx, gy = loop.controls.buttons.gadget
    down, lift, quiet = _tap_steps_per_tick(loop, loop.decision_every)
    assert down == [("down", SLOT_TAP, gx, gy)]
    assert lift == [("up", SLOT_TAP)]
    assert quiet == []
    first = _decisions(loop)[0]
    assert first.attack == ATTACK_GADGET and first.attack_legal == 0b1011

    loop.shadow.advance(0.25)    # the shadow spends REAL time; the fixture ticks back to back
    before = len(loop.backend.log)
    loop.tick()
    second = _decisions(loop)[1]
    assert second.attack_legal == 0b0011
    assert second.attack == ATTACK_NONE
    assert [e for e in loop.backend.log[before:] if e[1] == SLOT_TAP] == []


def test_a_policy_exception_fails_closed_before_it_propagates(loop):
    loop.policy.raises = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        _play(loop, loop.decision_every)
    assert loop.phase is Phase.STOPPED
    assert not loop.controls.joystick.is_down


# -- skipped decisions -------------------------------------------------------------------------

def test_a_missing_hero_box_skips_the_decision_instead_of_filling_it(loop):
    """9.8's rule at the loop level. A skipped decision is one more action-repeat window of the
    held action, which the sim produces constantly; a zeroed column is not."""
    _play(loop, loop.decision_every)
    assert loop.policy.calls == 1
    loop.vision.entities.detections = []     # the detector stops seeing the hero
    loop.tracker.max_misses = -1             # and the track is dropped rather than coasting
    for _ in range(loop.decision_every):
        loop.tick()
    assert loop.policy.calls == 1
    assert _last_attempt(loop).note == "no hero box"


def test_a_skipped_decision_holds_the_previous_action(loop):
    """Nothing is re-sent and nothing is released: the contact simply stays where the last
    decision put it."""
    _play(loop, loop.decision_every)
    loop.vision.entities.detections = []
    loop.tracker.max_misses = -1
    before = list(loop.backend.events)
    for _ in range(loop.decision_every):
        loop.tick()
    assert [e for e in loop.backend.events[len(before):] if e[0] != "up"] == []


def test_the_first_brawlers_left_read_is_awaited_rather_than_seeded(loop):
    """Holding a previous count through a failed read is legitimate -- it changes only on a kill.
    Seeding one before any read ever succeeded would be inventing the number."""
    loop.vision.hud.count = None
    _play(loop, loop.decision_every)
    assert loop.policy.calls == 0
    assert _last_attempt(loop).note == "no brawlers-left read"


# -- the slot rule -----------------------------------------------------------------------------

# Both off the tracker's own defaults (promote 2, coast 3), which are also today's config: a loop
# that dropped the run's numbers would fall back to those and pass on the shipped yaml.
SLOTS_CFG = load_config("configs/default.yaml",
                        overrides={"slots": {"promote_hits": 4, "max_misses": 1}})


def test_the_tracker_takes_the_runs_slot_rule_not_its_own_defaults(monkeypatch):
    """`slots.*` is the rule the sim's `core/slots.py` trained the enemy rows on, so the live
    tracker reads the run's numbers. An enemy standing in view reaches the policy's slots on its
    4th consecutive sighting, not its 2nd, and leaves on its 2nd unseen decision, not its 4th."""
    _patch_vision(monkeypatch)
    lp = _build_loop(_Policy(cfg=SLOTS_CFG))
    assert (lp.tracker.promote_hits, lp.tracker.max_misses) == (4, 1)

    hero = _Detection("player", HERO_PX)
    lp.vision.entities.detections = [hero, _Detection("enemy", (HERO_PX[0] + 2 * 48, HERO_PX[1]))]
    _play(lp, 4 * lp.decision_every)
    lp.vision.entities.detections = [hero]
    _play(lp, 2 * lp.decision_every)
    slotted = [any(t is not None for t in call["enemies"]) for call in lp.policy.assembler.calls]
    assert slotted == [False, False, False, True, True, False]


# -- enemy HP bookkeeping ----------------------------------------------------------------------

def test_a_slots_hp_is_dropped_when_its_track_is_replaced(loop):
    """§6.5 promotes a track only once it has a committed HP and holds that value for the life of
    the track. Carrying it onto the slot's next occupant would be a fabricated number wearing a
    real one's clothes."""
    from brawl_deployment.perception.tracker import Track

    _play(loop, 1)
    tracked = type("R", (), {"enemies": [Track(id=7, slot=0, label="enemy", pos=(1.0, 1.0))]})()
    loop._update_enemy_hp(tracked, [], loop.vision.odometry.update(None))
    loop._enemy_hp[0] = 4000
    assert loop._enemy_hp == {0: 4000}

    tracked.enemies = [Track(id=8, slot=0, label="enemy", pos=(1.0, 1.0))]
    loop._update_enemy_hp(tracked, [], loop.vision.odometry.update(None))
    assert loop._enemy_hp == {}


# -- terrain lookups ---------------------------------------------------------------------------

def test_an_unobserved_cell_is_not_a_bush(loop):
    """The same optimistic direction `GridBuilder`'s `unknown_tile=FLOOR` already commits to. One
    convention applied twice, not two conventions."""
    assert loop._in_bush((0.0, 0.0)) is False
    assert loop._in_bush((10_000.0, 0.0)) is False     # off the canvas entirely


def test_gas_is_read_from_the_accumulated_map_not_the_frame(loop):
    assert loop._in_gas((0.0, 0.0)) is False
    loop.grid.gas.gassed[64, 64] = True                # world (0, 0) on a 128x128 canvas
    assert loop._in_gas((0.0, 0.0)) is True


def test_zone_active_outlives_a_new_segment_and_a_new_match_clears_it(loop, monkeypatch):
    """The sim's `zone_seen` is cleared by an episode reset and nothing else. Live, a new segment
    empties the gas map, so a flag read off the map alone would fall back to 0 mid-match with the
    gas out of view, which is the case here, and it holds. The next match starts at 0 again."""
    on_screen = {"gas": True}
    monkeypatch.setattr("brawl_vision.terrain.zone.detect_zone",
                        lambda rect, plan, cfg=None: _Zone(gassed=on_screen["gas"]))
    _play(loop, 1)
    assert loop.policy.assembler.calls[-1]["zone"]["active"] is True

    on_screen["gas"] = False
    loop.vision.odometry.segment += 1
    _play(loop, 2 * loop.decision_every)              # the tracker's reset tick is skipped
    assert loop.policy.calls == 2, _last_attempt(loop).note
    assert not loop.grid.gas.gassed.any()
    assert loop.policy.assembler.calls[-1]["zone"]["active"] is True

    loop.match.gate = False
    loop.tick()                                        # the match ends...
    _play(loop, 1)                                     # ...and the next one begins
    assert loop.policy.calls == 3, _last_attempt(loop).note
    assert loop.policy.assembler.calls[-1]["zone"]["active"] is False


# -- the real assembler ------------------------------------------------------------------------

# Globbed, so a new deploy spec is exercised here the day it is added.
DEPLOY_SPECS = sorted(str(p).replace("\\", "/") for p in Path("configs").glob("agent_obs_deploy*.yaml"))


@pytest.mark.parametrize("spec_path", DEPLOY_SPECS)
def test_a_decision_reaches_the_policy_through_the_real_assembler(loop, spec_path):
    """Every supplier the loop wires up, fed through the assembler the checkpoint would get.

    Every other test in this file hands the loop `_Assembler`, which accepts any keyword and any
    value, so none of them can see a field the spec names and the loop never supplies. That is how
    the first live decision on an agent_obs_deploy3.yaml run raised "no value supplied for
    'zone.hero_margin_local'": the zone estimator only answered to the first spec's field names.
    """
    policy = _Policy(spec_path=spec_path, real_assembler=True)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
                    policy=policy, controls=loop.controls, cfg=DeploymentConfig())
    _play(lp, 1)
    assert policy.calls == 1, _last_attempt(lp).note

    space = agent_space(policy.spec, CFG)
    obs = policy.obs[0]
    assert set(obs) == set(space.spaces)
    for name, sub in space.spaces.items():
        assert obs[name].shape == sub.shape, name
        assert obs[name].dtype == sub.dtype, name


# -- hero_offset and hero.near_edge -----------------------------------------------------------

def _near_edge_spec() -> str:
    """deploy4 with `hero.near_edge` after `hero.in_zone`, the shape deploy5 takes."""
    import tempfile
    import yaml
    doc = yaml.safe_load(open("configs/agent_obs_deploy4.yaml").read())
    for g in doc["groups"]:
        if "hero.in_zone" in g.get("fields", ()):
            g["fields"].insert(g["fields"].index("hero.in_zone") + 1, "hero.near_edge")
    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w") as f:
        yaml.dump(doc, f)
    return path


_OLD_RECORD = {"index": "7", "t": "0.35", "grab_ms": "1.5", "phase": "playing", "tick_ms": "40.0",
               "odometry": "ok", "warmup": "False", "decision": "True", "move_bin": "3",
               "attack": "1", "n_detections": "2", "ammo_cv": "2.0", "ammo_shadow": "2.0",
               "lattice": "locked", "phase_x": "0.1", "phase_y": "-0.2", "note": ""}


def test_the_trackers_hero_offset_reaches_the_assembler_and_the_telemetry(loop):
    """`hero_offset` is the player box's tiles from its nominal screen anchor: the fixture's box
    at HERO_PX against the viewport centre, minus the ring's measured (0.09, 0.80) and the box's
    measured (-0.08, -0.78) from the ring, both through the identity plan at 48 px per tile, so
    the plan's `origin_tile` cancels."""
    _play(loop, 1)
    want = ((HERO_PX[0] - VIEWPORT[0] / 2) / 48 - 0.01, (HERO_PX[1] - VIEWPORT[1] / 2) / 48 - 0.02)
    assert loop.policy.assembler.calls[-1]["hero_offset"] == pytest.approx(want, abs=1e-6)
    row = _decisions(loop)[-1]
    assert (row.hero_offset_x, row.hero_offset_y) == pytest.approx(want, abs=1e-6)
    assert row.near_edge == int(max(abs(want[0]), abs(want[1])) > CFG.camera_edge_flag_tiles)
    old = TickRow.from_record(_OLD_RECORD)
    assert (old.hero_offset_x, old.hero_offset_y, old.near_edge) == (0.0, 0.0, -1), "older CSVs"


def test_a_spec_that_names_near_edge_decides_through_the_real_assembler(loop):
    policy = _Policy(spec_path=_near_edge_spec(), real_assembler=True)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
                    policy=policy, controls=loop.controls, cfg=DeploymentConfig())
    _play(lp, 1)
    assert policy.calls == 1, _last_attempt(lp).note


# -- the history -------------------------------------------------------------------------------

def _deploy4_loop(loop, decision=None, real_assembler=True):
    """The fixture's loop with a deploy4 policy, the first spec that reads the history. The real
    assembler by default, so the `hist` group is the one the checkpoint would read; the recorder
    when a test wants the snapshots and the grid exactly as the loop handed them over."""
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
                    policy=_Policy(decision=decision, spec_path="configs/agent_obs_deploy4.yaml",
                                   real_assembler=real_assembler),
                    controls=loop.controls, cfg=DeploymentConfig())
    lp.backend = loop.backend
    return lp


def _valid(lp) -> list:
    """`hist.valid` of every observation the policy was handed, in order."""
    return [obs["history"][:3].tolist() for obs in lp.policy.obs]


def test_the_first_decisions_after_the_gate_fill_the_history_one_slot_at_a_time(loop):
    """Through the real assembler: [0,0,0], [1,0,0],
    [1,1,0], and then the ring is full. A different HP read each decision shows the order:
    newest first, as the sim's ring keeps it."""
    lp = _deploy4_loop(loop)
    for i, hp in enumerate((8000, 7000, 6000, 5000)):
        lp.vision.health.readings = [_Reading("player", hp)]
        _play(lp, lp.decision_every if i else 1)
    assert _valid(lp) == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    np.testing.assert_allclose(lp.policy.obs[3]["history"][66:69], [0.3, 0.35, 0.4])


def test_a_snapshot_holds_the_modelled_attack_and_what_its_observation_was_built_from(loop):
    """The policy asks for a super the shadow's mask refuses at the gate, so nothing is pressed,
    and the next observation's history says nothing was: `hist.attack_onehot` is what the device
    did. The rest is what the first observation was handed: move bin 3, the 8000 read, the full
    clip a match starts with, and the hero's world position."""
    lp = _deploy4_loop(loop, Decision(move_bin=3, attack=ATTACK_SUPER,
                                      legal=(True, True, False, True)))
    _play(lp, lp.decision_every + 1)
    assert lp.policy.calls == 2, _last_attempt(lp).note
    first = lp._history[1]
    assert (first.move_bin, first.attack, first.hp, first.ammo_frac, first.pos) == (
        3, ATTACK_NONE, 8000.0, 1.0, (0.0, 0.0))
    hist = lp.policy.obs[1]["history"]
    assert hist[3 + 3] == 1.0                    # slot 0's move one-hot, bin 3
    assert hist[54] == 1.0 and hist[56] == 0.0   # slot 0's attack one-hot: no-fire, not super
    assert hist[66] == pytest.approx(0.4)        # 8000 / 20000


def test_the_history_holds_the_ammo_an_observation_saw_not_what_its_shot_left(loop):
    """The sim pushes `hist_ammo` at the top of `env.step`, before the action lands, so slot 0
    is the clip the previous observation was built from. Here that observation's answer is a
    shot, the shadow spends it before the next decision, and the history still says full."""
    lp = _deploy4_loop(loop, Decision(move_bin=3, attack=ATTACK_FIRE,
                                      legal=(True, True, False, True)))
    _play(lp, 1)
    lp.shadow.advance(0.05)    # one sub-tick; the fixture ticks back to back, in no real time
    assert lp.shadow.observe()["ammo_frac"] < 1.0, "the shot was not modelled"
    _play(lp, lp.decision_every)
    assert lp.policy.calls == 2, _last_attempt(lp).note
    assert lp._history[1].attack == ATTACK_FIRE
    assert lp.policy.obs[1]["history"][69] == 1.0      # slot 0's ammo_frac
    assert lp._history[0].ammo_frac == pytest.approx(2 / 3)


def test_a_snapshot_holds_the_heros_world_position_x_then_y(loop):
    """World (3, 1), where the two axes differ, so a transposed snapshot shows. The hero never
    moves, so the next decision's displacement for slot 0 must read zero."""
    lp = _deploy4_loop(loop)
    lp.vision.odometry.position = (3.0, 1.0)
    _play(lp, lp.decision_every + 1)
    assert lp.policy.calls == 2, _last_attempt(lp).note
    assert lp._history[1].pos == (3.0, 1.0)
    assert lp.policy.obs[1]["history"][72:74].tolist() == [0.0, 0.0]   # slot 0's displacement


def test_a_new_match_starts_with_no_history(loop):
    lp = _deploy4_loop(loop)
    _play(lp, 2 * lp.decision_every + 1)
    assert _valid(lp)[-1] == [1.0, 1.0, 0.0]
    lp.match.gate = False
    lp.tick()
    _play(lp, 1)
    assert lp.policy.calls == 4, _last_attempt(lp).note
    assert _valid(lp)[-1] == [0.0, 0.0, 0.0]


def test_a_new_segment_drops_the_history(loop):
    """A snapshot's positions are world-frame, and a new segment has no defined offset to the old
    one, so the ring empties the way every tracker does. The first decision in the new segment is
    the tracker's reset tick and is skipped; the one after reads an empty history."""
    lp = _deploy4_loop(loop)
    _play(lp, 2 * lp.decision_every + 1)
    assert _valid(lp)[-1] == [1.0, 1.0, 0.0]
    lp.vision.odometry.segment += 1
    _play(lp, 2 * lp.decision_every)
    assert [r.note for r in lp.telemetry if r.note] == ["no hero box"]
    assert lp.policy.calls == 4
    assert _valid(lp)[-1] == [0.0, 0.0, 0.0]


def test_a_skipped_decision_takes_no_snapshot(loop):
    """No hero HP read, no decision, and no snapshot. The next decision's first slot is the last
    decision that reached the policy, two windows back, and `valid` stays a prefix: the sim never
    produces an invalid slot in front of a valid one."""
    lp = _deploy4_loop(loop)
    _play(lp, 1)
    lp.vision.health.readings = []
    _play(lp, lp.decision_every)
    assert _last_attempt(lp).note == "no hero hp"
    lp.vision.health.readings = [_Reading("player", 7000)]
    _play(lp, lp.decision_every)
    assert lp.policy.calls == 2, _last_attempt(lp).note
    assert _valid(lp) == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert lp.policy.obs[1]["history"][66] == pytest.approx(0.4)   # 8000, the decision before


def test_a_snapshot_keeps_only_the_enemies_seen_that_decision(loop):
    """The sim's `hist_enemy_seen` is alive AND revealed, the set its `enemy_revealed` plane draws;
    on this side that set is `seen_now`. A coasted track is the tracker's prediction, not a
    sighting, so the snapshot of a decision it coasted through leaves it out while the track
    itself lives on. The planes follow the snapshots, oldest last."""
    lp = _deploy4_loop(loop, real_assembler=False)
    enemy = _Detection("enemy", (HERO_PX[0] + 2 * 48, HERO_PX[1]))     # world (2, 0)
    lp.vision.entities.detections.append(enemy)
    _play(lp, lp.decision_every + 1)      # two decisions: pending, then promoted and seen
    lp.vision.entities.detections.remove(enemy)
    _play(lp, 2 * lp.decision_every)      # one decision coasting, then one that reads them all
    calls = lp.policy.assembler.calls
    assert len(calls) == 4
    # The track lives on, coasting. Read off the LAST call only: the recorder keeps the tracker's
    # own Track objects, and every later update rewrites their `seen_now` in place.
    assert [t.seen_now for t in calls[3]["enemies"] if t is not None] == [False]
    assert [s.enemies for s in calls[3]["history"]] == [(), ((2.0, 0.0),), ()]

    grid, channels = calls[3]["grid"], lp.grid.spec.channels
    assert grid[channels.index("enemy_hist1")].sum() == 0
    # The hero is world (0, 0) at crop (row 6, col 10); the enemy two columns right.
    assert np.argwhere(grid[channels.index("enemy_hist2")]).tolist() == [[6, 12]]
    assert grid[channels.index("enemy_hist3")].sum() == 0


# -- crates and cubes --------------------------------------------------------------------------

# A real `Detection`, since `LootMap` reads the box as well as the anchor. Anchor (600, 360) through
# the identity plan is camera-relative tile (2.5, 1.5): world cell (2, 1), crop (row 7, col 12)
# with the hero at world (0, 0).
CRATE = Detection(CUBE_BOX, 0.9, (580.0, 290.0, 620.0, 390.0))


def _deploy3_loop(loop, detections=(), names=PROJECTILE_NAMES):
    """The fixture's loop again with a deploy3 policy. Rebuilt rather than patched, because the
    grid's channels and the class check are both settled at construction."""
    loop.vision.projectiles = _Detector(detections, names=names)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
                    policy=_Policy(spec_path="configs/agent_obs_deploy3.yaml"),
                    controls=loop.controls, cfg=DeploymentConfig())
    lp.backend = loop.backend
    return lp


def test_the_grid_is_built_for_the_policys_spec_not_a_default_one(loop):
    """`GridSpec.load()` with no arguments is deploy's eight planes. A deploy3 policy built with
    that would fail its first `assemble` on the shape, mid-match."""
    assert loop.grid.spec.shape == (8, 13, 21)
    assert _deploy3_loop(loop).grid.spec.shape == (10, 13, 21)


def test_a_deploy3_policy_sees_a_crate_in_its_box_plane_once_it_is_confirmed(loop):
    lp = _deploy3_loop(loop, [CRATE])
    _play(lp, lp.decision_every + 1)
    channels = lp.grid.spec.channels
    grids = [call["grid"] for call in lp.policy.assembler.calls]
    assert len(grids) == 2
    # The first decision is the first tick: one sighting, not yet on the map.
    assert grids[0][channels.index("box")].sum() == 0
    box = grids[1][channels.index("box")]
    assert box[7, 12] == 1 and box.sum() == 1
    assert grids[1][channels.index("pickup")].sum() == 0


def test_each_class_goes_to_its_own_consumer(loop):
    """One model, one call, three classes. A crate in the projectile tracker would be a still
    projectile for its whole life; a shot in the loot map would be a crate."""
    shot = Detection(PROJECTILE, 0.9, (700.0, 300.0, 720.0, 320.0))
    lp = _deploy3_loop(loop, [CRATE, shot])
    _play(lp, 4)
    assert lp.vision.projectiles.calls == 4
    assert lp.loot.crates() == [(2.5, 1.5)] and lp.loot.cubes() == []
    assert len(lp.projectiles.tracks) == 1


def test_a_crate_box_keeps_the_terrain_map_off_the_floor_under_it(loop):
    """The occupancy map's fifth gate, and independent of the spec: a deploy policy never reads a
    crate, but its terrain map still must not lock the crate as a wall."""
    loop.vision.entities.detections = []         # the crate alone; brawlers are the next test
    _play(loop, 1)
    assert loop.vision.occupancy.occluded.shape == (13, 21)
    assert not loop.vision.occupancy.occluded.any()
    loop.vision.projectiles.detections = [CRATE]
    loop.tick()
    occluded = loop.vision.occupancy.occluded
    assert occluded[7, 12]
    assert occluded[:, 12].sum() == occluded.sum() <= 3     # the box is 40 px wide, inside col 12


def test_a_brawler_box_keeps_the_terrain_map_off_the_floor_under_it_on_every_tick(loop):
    """The hero's sprite is not floor. Checked on the ticks BETWEEN decisions too, which are two
    of every three deposits and the ones a decision-only detector left unmasked."""
    loop.vision.entities.detections = [_Detection("enemy", (5 * 48, 3 * 48))]
    _play(loop, 2)                        # the second tick is not a decision tick
    occluded = loop.vision.occupancy.occluded
    # The box is x [216, 264], y [56, 152]: half of each of cols 4-5, rows 1-2 almost whole. Row 3
    # gets its top 8 px, 8% of each cell and under the gate's 10%.
    assert set(zip(*np.nonzero(occluded))) == {(1, 4), (1, 5), (2, 4), (2, 5)}
    loop.vision.entities.detections = []
    loop.tick()
    assert not loop.vision.occupancy.occluded.any()


def test_terrain_is_deposited_through_a_plan_registered_at_this_ticks_position(loop):
    """The map is a whole-tile grid and the camera almost never sits on a whole tile. A deposit
    through the fixed plan lands up to half a tile off, differently every tick, while the hero is
    placed continuously. So what the occupancy map is handed must put the frame's pixel (0, 0)
    on a whole world tile at THIS tick's position, and the crate mask must come from the same plan."""
    loop.vision.entities.detections = []
    loop.vision.projectiles.detections = [CRATE]
    loop.vision.odometry.position = (3.3, -1.8)
    _play(loop, 1)
    plan = loop.vision.occupancy.plan
    at = np.asarray(plan.origin_tile, np.float64) + (3.3, -1.8)
    assert np.allclose(at, np.round(at))
    assert plan is not loop.vision.plan
    # The crate mask moves with the content. The remainder here is (+0.3, +0.2) tile, and 0.3 of
    # a tile pushes the 40 px box (x 580-620, inside col 12 on the fixed plan) across into col 13.
    occluded = loop.vision.occupancy.occluded
    assert occluded[:, 13].any() and not crate_occlusion([CRATE], loop.vision.plan)[:, 13].any()
    assert np.array_equal(occluded, crate_occlusion([CRATE], plan))


def _lattice_crate(tile):
    """A crate box at a real crate's on-screen height, anchored on camera-relative `tile`.

    `CRATE` above is 100 px tall, which `perception/lattice.py` reads as a crate half hidden
    behind something and ignores: every other test here runs at phase 0 because of that.
    """
    from brawl_deployment.perception.lattice import CRATE_HEIGHT_PX
    from brawl_vision.object_detection.project import CRATE_ANCHOR_FRAC

    a, b = CRATE_HEIGHT_PX
    ax = (tile[0] - _Plan.origin_tile[0]) * _Plan.pixels_per_tile
    ay = (tile[1] - _Plan.origin_tile[1]) * _Plan.pixels_per_tile
    y1 = (ay + CRATE_ANCHOR_FRAC * a) / (1.0 - CRATE_ANCHOR_FRAC * b)   # anchor = y1 - 0.3 h
    return Detection(CUBE_BOX, 0.9, (ax - 60.0, y1 - (a + b * y1), ax + 60.0, y1))


def test_every_consumer_moves_onto_the_game_lattice_the_crates_give(loop):
    """The lattice through the real loop. The crate reads (+0.3, -0.25) off a tile centre in the
    odometry frame. On the third sighting the world frame moves by exactly that. The deposit is
    registered at the moved position, every tracker starts over in the new frame, and the crate the
    loot map confirms sits on a tile centre: the cell the sim keeps a crate in."""
    lp = _deploy3_loop(loop, [_lattice_crate((2.8, 1.25))])
    _play(lp, 7)
    rows = list(lp.telemetry)
    assert [r.lattice for r in rows[:4]] == ["unlocked", "unlocked", "rebased", "locked"]
    assert (rows[2].phase_x, rows[2].phase_y) == pytest.approx((0.3, -0.25))

    plan = lp.vision.occupancy.plan
    at = np.asarray(plan.origin_tile, np.float64) + (-0.3, 0.25)      # the WORLD position
    assert np.allclose(at, np.round(at))
    assert lp.loot.crates() == [pytest.approx((2.5, 1.5))]

    # The decision on the tick after the re-lock is the tracker's reset tick, and is skipped like
    # any segment change; the next one goes through. Two decisions of three, not three.
    assert rows[3].note == "no hero box" and lp.policy.calls == 2
    # The hero is camera-relative (0, 0), so world (-0.3, 0.25): game tile (-1, 0). The crate is
    # game tile (2, 1), three columns right of the hero and one row down.
    channels = lp.grid.spec.channels
    box = lp.policy.assembler.calls[-1]["grid"][channels.index("box")]
    assert box.sum() == 1 and box[6 + 1, 10 + 3] == 1


def test_a_new_match_is_a_new_lattice_and_its_first_decision_still_goes_through(loop):
    """The match start hands every tracker the NEW epoch. Handed the odometry segment instead,
    each would see the lattice's epoch as a change on its first update, reset, and skip the
    match's first decision, every match."""
    lp = _deploy3_loop(loop, [_lattice_crate((2.8, 1.25))])
    _play(lp, 4)
    assert lp.lattice.phase != (0.0, 0.0)
    lp.match.gate = False
    lp.tick()
    calls = lp.policy.calls
    lp.vision.projectiles.detections = []
    _play(lp, 1)
    assert lp.lattice.phase == (0.0, 0.0)
    assert lp.policy.calls == calls + 1, _last_attempt(lp).note


def test_a_deploy3_policy_refuses_a_projectile_model_that_cannot_see_crates(loop):
    with pytest.raises(ValueError, match="Power Cube Box"):
        _deploy3_loop(loop, names={0: PROJECTILE})


def test_a_policy_that_reads_no_crates_does_not_need_the_crate_class(loop):
    loop.vision.projectiles = _Detector(names={0: PROJECTILE})
    DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
               policy=_Policy(), controls=loop.controls, cfg=DeploymentConfig())


def test_a_new_match_starts_with_no_crates(loop):
    lp = _deploy3_loop(loop, [CRATE])
    _play(lp, 4)
    assert lp.loot.crates()
    lp.match.gate = False
    lp.tick()                                            # the gate closes; out of PLAYING
    lp.vision.projectiles.detections = []
    _play(lp, 1)
    assert lp.phase is Phase.PLAYING and lp.loot.crates() == []


# -- a known map (KNOWN_MAP_LOCALIZATION_PLAN.md step K3) --------------------------------------

# Where the fakes put the camera on Dark Passage: map world = odometry + TRUE_OFFSET. With odometry
# at (0.5, 0.5) the hero, camera-relative (-4, 0), stands on the centre of the spawn at col 6,
# row 24, world (-23.5, -5.5), and every plan is a registered one. The camera has stopped 4 tiles
# east of it, at map col 10.5: 1.6 tiles past where the sim's stops, as the game's does there (K5
# measured up to 1.9 on the west edge). The 21 x 13 grid around the hero holds wall, bush, water
# and floor, and its four westmost columns are past the map's edge.
TRUE_OFFSET = (-20.0, -6.0)
SPAWN = (6, 24)
HERO_AT = (-4, 0)            # the hero's camera-relative tile
PAD = 30                     # tiles of WALL around the label in the arrays read below


class _MapPlan(_Plan):
    """`_Plan` with a viewport whose centre projects to `-HERO_ANCHOR_TILES`, so the camera's
    nominal point (`localize._nominal`) is camera-relative (0, 0), the hero's tile while the
    game's camera tracks it. The landing prior puts that point on a spawn, or where the camera
    stops short of one; on `_Plan` it is 11 tiles from (0, 0)."""

    viewport = tuple(2.0 * (-o - a) * _Plan.pixels_per_tile
                     for o, a in zip(_Plan.origin_tile, HERO_ANCHOR_TILES))


class _MapClassifier:
    """The terrain classifier on the labelled map: the label's classes under the view, read at the
    TRUE camera pose (odometry plus `TRUE_OFFSET`), never at the frame the loop hands out, so a
    wrong fix would see terrain that disagrees with it. WALL past the map's edge."""

    def __init__(self, odometry, known):
        self.odometry = odometry
        self.classes = np.pad(known.classes, PAD, constant_values=CLASS_INDEX[Tile.WALL])

    def predict(self, rect, plan):
        cols, rows = plan.size_tiles
        # The view's top-left map tile (world + 30), then into the padding.
        at = np.asarray(plan.origin_tile) + self.odometry.position + TRUE_OFFSET + 30.0 + PAD
        assert np.allclose(at, np.round(at)), f"the view is off the map's tiles: {at}"
        c, r = np.round(at).astype(int)
        return self.classes[r:r + rows, c:c + cols].copy(), np.ones((rows, cols), np.float32)


def _map_loop(loop, spec_path="configs/agent_obs_deploy.yaml", real_assembler=True):
    """The fixture's loop again with `map.name: dark_passage`, the camera as above, and its log
    kept on `lp.messages`. The commit margin is pinned to the provisional 60, which the clean view
    clears on its first tick, so tuning the default in step K4 cannot move the commit."""
    vision = loop.vision
    vision.plan = _MapPlan()
    vision.odometry.position = (0.5, 0.5)
    vision.entities.detections = [_Detection("player", (HERO_PX[0] + HERO_AT[0] * 48,
                                                        HERO_PX[1] + HERO_AT[1] * 48))]
    vision.classifier = _MapClassifier(vision.odometry, KnownMap.load("dark_passage"))
    messages = []
    cfg = DeploymentConfig(map_name="dark_passage", map_commit_margin=60.0)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=vision,
                    policy=_Policy(spec_path=spec_path, real_assembler=real_assembler),
                    controls=loop.controls, cfg=cfg, log=messages.append)
    lp.backend, lp.messages = loop.backend, messages
    return lp


def _until(lp, done, limit=40):
    """Tick with the gate open until `done()` holds, at most `limit` ticks."""
    lp.match.gate = True
    for _ in range(limit):
        if done():
            return
        lp.tick()
    assert done(), f"not done after {limit} ticks; log {lp.messages}"


def test_on_a_known_map_the_grid_reads_the_label_and_pos_norm_the_map_column(loop):
    """Once the localizer has fixed on the label, the static planes are the label's cells under
    the view read through the sim's tile tables, WALL past the edge as the sim pads its maps, and
    `hero.pos_norm` is the hero's map column over 60.

    The clean view commits on the first tick, which is also the first decision. That decision is
    still in the lattice frame the tick handed out, so its planes must be the occupancy map's: the
    label read there would be the map at the wrong place."""
    lp = _map_loop(loop)
    _until(lp, lambda: lp.localizer.state == "fixed")
    assert "map: fixed on dark_passage at offset (-20.00, -6.00)" in lp.messages
    assert lp.telemetry[0].decision and lp.telemetry[0].map_state == "fixed"    # the premise
    # The fix is a new epoch, whose first decision the tracker's reset skips. The next goes through.
    calls = lp.policy.calls
    _play(lp, 2 * lp.decision_every)
    assert lp.policy.calls > calls, _last_attempt(lp).note

    tiles = np.pad(KnownMap.load("dark_passage").tiles, PAD, constant_values=int(Tile.WALL))
    c0, r0 = SPAWN[0] - 10 + PAD, SPAWN[1] - 6 + PAD            # the hero's cell is the centre
    crop = tiles[r0:r0 + 13, c0:c0 + 21]
    assert not (crop == Tile.FENCE).any()        # the one tile the grid does not read as the sim
    first, last, channels = lp.policy.obs[0], lp.policy.obs[-1], lp.grid.spec.channels
    for name, table in (("blocks_unit", TILE_BLOCKS_UNIT), ("blocks_projectile", TILE_BLOCKS_PROJ),
                        ("is_bush", TILE_IS_BUSH), ("is_water", TILE_IS_WATER)):
        plane = channels.index(name)
        np.testing.assert_array_equal(last["grid"][plane], table[crop], err_msg=name)
        assert not first["grid"][plane].any(), "the commit tick: an unseen occupancy map, floor"
    assert last["grid"][channels.index("blocks_unit")][:, :4].all()      # off the map: wall

    start, end = agent_obs_index_map(lp.policy.spec, CFG)["self"]["hero.pos_norm"]
    assert last["self"][start:end] == pytest.approx([(SPAWN[0] + 0.5) / 60, (SPAWN[1] + 0.5) / 60])
    row = lp.telemetry[-1]
    assert row.map_state == "fixed" and (row.map_dx, row.map_dy) == TRUE_OFFSET
    assert math.isnan(row.map_lead_dx) and math.isnan(row.map_lead_dy)    # K5: fixed, no leader

    # The hero in raw odometry reads the same in the lattice frame (the commit tick) and the map
    # frame (every tick after), and plus the offset it is the spawn's centre: what K4's report
    # scores a fix against.
    placed = [r for r in lp.telemetry if not math.isnan(r.hero_odo_x)]
    assert placed[0] is lp.telemetry[0] and len(placed) >= 2
    spawn_world = (SPAWN[0] + 0.5 - 30.0, SPAWN[1] + 0.5 - 30.0)
    for r in placed:
        assert (r.hero_odo_x + TRUE_OFFSET[0], r.hero_odo_y + TRUE_OFFSET[1]) == \
            pytest.approx(spawn_world, abs=1e-6)


def test_with_no_map_named_the_loop_is_wired_as_it_was(loop):
    """`map.name: null`, the default: the lattice hands out the frame and the grid reads the
    occupancy map, the objects every other test in this file runs on, and the map columns stay
    empty."""
    assert DeploymentConfig().map_name is None
    assert loop.localizer is None and type(loop.lattice) is LatticePhase
    assert loop.terrain is loop.vision.occupancy and loop.grid.occupancy is loop.vision.occupancy
    _play(loop, 3)
    for row in loop.telemetry:
        assert (row.map_state, row.map_crate_resid) == ("", "")
        assert all(math.isnan(v) for v in (row.map_dx, row.map_dy, row.map_agree, row.map_margin,
                                           row.map_lead_dx, row.map_lead_dy))


def test_a_tick_without_a_fix_logs_where_the_search_leader_sits(loop):
    """K5: a search that never commits still logs where its leader puts the map, as the offset a
    commit on it would take, so a match that never fixes can be scored against its spawn. Margins
    no lead can reach hold the clean view off a commit it would take on the first tick."""
    lp = _map_loop(loop)
    lp.localizer.commit_margin = lp.localizer.whole_map_margin = math.inf
    _play(lp, 3)
    assert len(lp.telemetry) == 3
    for row in lp.telemetry:
        assert row.map_state == "search" and math.isnan(row.map_dx)
        assert (row.map_lead_dx, row.map_lead_dy) == TRUE_OFFSET


def test_a_crate_seen_before_a_confirmed_cut_is_still_in_the_box_plane_after_it(loop):
    """A cut the localizer confirms keeps its epoch, so the loot map, the tracks and the history
    keep their state across it. The crate leaves the screen before the cut, so only the loot map's
    memory can put it in the box plane afterwards."""
    lp = _map_loop(loop, "configs/agent_obs_deploy3.yaml", real_assembler=False)
    _until(lp, lambda: lp.localizer.state == "fixed")
    # Camera-relative (-2, 1), a true tile centre: map world (-21.5, -4.5), two columns right of
    # the hero and one row down. Crate check 3 finds the fix on the game's lattice and leaves it.
    lp.vision.projectiles.detections = [_lattice_crate((-2.0, 1.0))]
    _until(lp, lambda: lp.loot.crates())
    assert lp.loot.crates() == [pytest.approx((-21.5, -4.5))]
    resid = [float(v) for r in lp.telemetry for v in r.map_crate_resid.replace(";", " ").split()]
    assert resid and max(map(abs, resid)) < 1e-6

    epoch = lp.localizer.epoch
    lp.vision.projectiles.detections = []
    # The real odometry's cut: a new segment on a lost tick, the position carried over.
    lp.vision.odometry.segment += 1
    lp.vision.odometry.status = "lost"
    lp.tick()
    lp.vision.odometry.status = "ok"
    _until(lp, lambda: lp.localizer.state == "fixed")
    assert "map: cut confirmed, fix kept" in lp.messages and lp.localizer.epoch == epoch

    calls = lp.policy.calls
    _until(lp, lambda: lp.policy.calls > calls)
    box = lp.policy.assembler.calls[-1]["grid"][lp.grid.spec.channels.index("box")]
    assert box.sum() == 1 and box[6 + 1, 10 + 2] == 1


# -- telemetry ---------------------------------------------------------------------------------

def test_telemetry_is_bounded(loop):
    assert loop.telemetry.maxlen == TELEMETRY_ROWS
    _play(loop, 30)
    assert len(loop.telemetry) == 30
    assert all(isinstance(r, TickRow) for r in loop.telemetry)


def test_telemetry_carries_no_pixels(loop):
    """§7's rule as a test: 1920x1080 BGR at 20 Hz is ~124 MB/s, so a frame reference on a row is
    a three-minute match at 22 GB."""
    _play(loop, 5)
    for row in loop.telemetry:
        for value in vars(row).values():
            assert isinstance(value, (int, float, str, bool)), value


def test_a_non_decision_tick_is_not_recorded_as_an_idle_move(loop):
    """`-1`, not `0`. Idle is a bin the policy chose; a tick with no decision is not."""
    _play(loop, loop.decision_every)
    decided = [r for r in loop.telemetry if r.decision]
    idle = [r for r in loop.telemetry if not r.decision]
    assert len(decided) == 1
    assert len(idle) == loop.decision_every - 1
    assert all(r.move_bin == -1 and r.attack == -1 for r in idle)


def test_the_run_loop_enters_the_capture_and_always_releases(loop):
    reason = loop.run(max_seconds=0.0, max_matches=None)
    assert loop.capture.entered
    assert loop.phase is Phase.STOPPED
    assert reason
    assert not loop.controls.joystick.is_down


# ------------------------------------------------------------------------------------------------
# `test_nothing_is_emitted_before_the_gate_opens` runs without the fixture's monkeypatching to
# prove the silence does not depend on it: a loop that never opens the gate never reaches a vision
# stage at all, which is the property being claimed.
# ------------------------------------------------------------------------------------------------

def _loop_fixture_value():
    backend = _Backend()
    controls = Controls(backend=backend,
                        joystick=Joystick(backend, (345.6, 669.6), 110.0, n_bins=16),
                        buttons=Buttons(backend, attack=(1690.0, 594.0), super_=(1462.5, 1000.5),
                                        gadget=(1559.9, 901.1)))
    vision = VisionStack(plan=_Plan(), odometry=_Odometry(), occupancy=_Occupancy(),
                         classifier=_Classifier(), entities=_Detector(),
                         projectiles=_Detector(names=PROJECTILE_NAMES), health=_Health(),
                         hud=_Hud(), cfg=None)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=vision,
                    policy=_Policy(), controls=controls, cfg=DeploymentConfig())
    lp.backend = backend
    lp.vision.warm = lambda image: None
    return lp


# -- the process shares its cores with the emulator ---------------------------------------------

def test_pin_thread_pools_caps_torch_and_cv2_from_the_config():
    """`from_config` and the calibration rig call this before building anything, because torch's
    pool is process-global and the detectors take theirs at construction. The caps come from
    `compute.*`, not from constants here: the measurement lives in the config."""
    import cv2
    import torch

    from brawl_deployment.loop import pin_thread_pools

    torch_before, cv2_before = torch.get_num_threads(), cv2.getNumThreads()
    try:
        pin_thread_pools(DeploymentConfig(compute_torch_threads=1, compute_cv2_threads=3))
        assert torch.get_num_threads() == 1
        assert cv2.getNumThreads() == 3
    finally:
        torch.set_num_threads(torch_before)
        cv2.setNumThreads(cv2_before)


def test_vision_stack_build_hands_the_detector_pool_settings_to_both_sessions(monkeypatch):
    """Both detectors, not one: each ORT session has its own pool, and a spinning pool on the
    session that was forgotten costs the same cores as the one that was not."""
    import brawl_deployment.loop as loop_mod

    seen = {}

    class _Det:
        names = {0: "player", 1: "enemy"}

        @classmethod
        def from_config(cls, cfg, **kw):
            seen[cls.__name__] = kw
            return cls()

    class _Proj(_Det):
        names = {0: "Projectile", 1: "cube_box", 2: "power_cube"}

    class _Any:
        def __init__(self, *a, **kw):
            pass

        @classmethod
        def from_config(cls, cfg):
            return cls()

    monkeypatch.setattr("brawl_vision.object_detection.detector.ObjectDetector", _Det)
    monkeypatch.setattr("brawl_vision.object_detection.projectile_detection.detect.ProjectileDetector",
                        _Proj)
    for target in ("brawl_vision.terrain.classifier.TerrainClassifier",
                   "brawl_vision.terrain.occupancy.OccupancyMap",
                   "brawl_vision.object_detection.hp_detection.smooth.HealthTracker",
                   "brawl_vision.hud.HudReader"):
        monkeypatch.setattr(target, _Any)
    monkeypatch.setattr("brawl_vision.terrain.odometry.Odometry", _Any)
    monkeypatch.setattr("brawl_vision.camera.build_rectify_plan", lambda *a: _Plan())
    monkeypatch.setattr("brawl_vision.camera.load_camera_model", lambda: None)
    monkeypatch.setattr("brawl_vision.camera.load_hud_mask", lambda *a: None)
    monkeypatch.setattr("brawl_vision.config.load_vision_config", lambda: None)

    loop_mod.VisionStack.build(detector_threads=2, detector_spin=False)
    assert seen == {"_Det": {"cpu_threads": 2, "spin": False},
                    "_Proj": {"cpu_threads": 2, "spin": False}}

    seen.clear()
    loop_mod.VisionStack.build()
    assert seen["_Det"] == {"cpu_threads": None, "spin": True}, "a script running alone keeps the library defaults"



# -- attack-cadence telemetry ------------------------------------------------------------------
#
# The four columns `scripts/audit_attack_cadence.py --telemetry` reads, plus the two the audit's
# own definitions turned out to need (the shadow's idle timer for long-dash waiting, and the ammo
# error behind a resync). Numbers are literals: the reach is 2.67 + 0.70 + 0.40 from
# configs/brawlers.yaml (hero_mortis) and configs/default.yaml (entities.unit_radius), and the
# test would be worthless if it re-added them.

def test_dash_reach_is_the_audits_radius_from_the_configs(loop):
    from brawl_deployment.loop import dash_reach_tiles

    assert loop.reach_tiles == pytest.approx(3.77)
    assert dash_reach_tiles(loop.shadow.p) == pytest.approx(3.77)


def test_a_decision_with_a_legal_attack_and_an_enemy_two_tiles_away_records_both(loop):
    """`attack_legal & 0b10` and `enemy_in_reach` on the decision row.

    The SECOND decision: `EntityTracker` confirms a track on its second sighting, and the column
    follows the tracks the policy was handed, so an enemy is "in reach" from the decision after
    it first appears -- one decision later than the sim's `vis`, and the same lag the observation
    itself has."""
    loop.vision.entities.detections = [_Detection("player", HERO_PX),
                                       _Detection("enemy", (HERO_PX[0] + 2 * 48, HERO_PX[1]))]
    _play(loop, loop.decision_every + 1)
    first, row = _decisions(loop)
    assert first.enemy_in_reach is False, "one sighting is not yet a track"
    assert row.attack_legal & 0b10, "a fresh shadow has a full clip and no cooldown"
    assert row.attack_legal & 0b01, "no-fire is always legal"
    assert not row.attack_legal & 0b100, "no super charge at match start"
    assert row.enemy_in_reach is True
    assert row.attack_cd_shadow == 0.0
    assert row.attack_idle_t_shadow == 0.0
    assert row.resync is False and row.resync_error == 0.0


def test_an_enemy_past_the_dash_reach_is_not_in_reach(loop):
    loop.vision.entities.detections = [_Detection("player", HERO_PX),
                                       _Detection("enemy", (HERO_PX[0] + 4 * 48, HERO_PX[1]))]
    _play(loop, loop.decision_every + 1)
    row = _decisions(loop)[1]
    assert row.enemy_in_reach is False      # 4.0 tiles > 3.77, and the track is confirmed
    assert row.attack_legal & 0b10


def test_the_columns_are_the_shadows_state_the_policy_was_handed(loop):
    """After a dash the next decision (0.25 s later) has 0.15 s of cooldown left in the shadow
    (the cooldown is set on the sub-tick after the decision)
    and the attack column is illegal: the bitmask and the timer on that row must agree with each
    other and with what `policy.act` received. The shadow spends REAL elapsed time and the fixture
    ticks back to back, so the decision period is advanced by hand."""
    loop.policy.decision = Decision(move_bin=3, attack=ATTACK_FIRE, legal=(True, True, False, True))
    _play(loop, 1)
    first = _decisions(loop)[0]
    assert first.attack == ATTACK_FIRE and first.attack_legal & 0b10
    loop.shadow.advance(0.25)                # five sub-ticks: the dash runs, 0.15 s of cooldown left
    for _ in range(loop.decision_every):
        loop.tick()
    second = _decisions(loop)[1]
    assert second.attack_legal == 0b1001, ("0.35 s cooldown, 0.25 s later: only no-fire, and the "
                                           "gadget, which neither the cooldown nor the dash gates")
    assert second.attack_cd_shadow == pytest.approx(0.15, abs=1e-6)
    assert second.attack == ATTACK_NONE, "the shadow refused the pick, and the row says what was sent"
    # The dash zeroed the long-dash timer on sub-tick 1; the four sub-ticks after it count up.
    assert second.attack_idle_t_shadow == pytest.approx(0.20, abs=1e-6)


def test_a_resync_lands_on_the_row_with_the_ammo_error_that_tripped_it(loop, monkeypatch):
    """`resync` is a fact about the shadow (it reseeds `attack_cd` to a full cooldown, so the
    mask on the same row is the post-resync one), and `resync_error` is CV minus shadow: +3.0
    here, a full clip the game kept while the shadow had spent it."""
    from brawl_deployment.loop import TickRow

    _play(loop, 2)
    _force_desync(loop, monkeypatch, ammo=3.0)
    row = TickRow(index=0, t=0.0, grab_ms=0.0)
    loop._read_own_bars(loop.capture.image, [_Detection("player", HERO_PX)], row)
    assert row.resync is True
    assert row.resync_error == 3.0
    assert float(loop.shadow.attack_cd) == pytest.approx(0.35)


def test_a_telemetry_record_without_the_cadence_columns_still_loads():
    """Old `--telemetry` CSVs have no cadence columns. Missing columns take the field defaults; the
    required positional fields still have to be there."""
    from brawl_deployment.loop import TickRow

    old = {"index": "7", "t": "0.35", "grab_ms": "1.5", "phase": "playing", "tick_ms": "40.0",
           "odometry": "ok", "warmup": "False", "decision": "True", "move_bin": "3",
           "attack": "1", "n_detections": "2", "ammo_cv": "2.0", "ammo_shadow": "2.0",
           "lattice": "locked", "phase_x": "0.1", "phase_y": "-0.2", "note": ""}
    row = TickRow.from_record(old)
    assert (row.index, row.t, row.decision, row.move_bin, row.attack) == (7, 0.35, True, 3, 1)
    assert row.warmup is False
    assert (row.attack_legal, row.attack_cd_shadow, row.attack_idle_t_shadow) == (-1, -1.0, -1.0)
    assert row.enemy_in_reach is False and row.resync is False and row.resync_error == 0.0
    with pytest.raises(KeyError):
        TickRow.from_record({"t": "0.0", "grab_ms": "0.0"})
    # A stringly "False" is False, not truthy text.
    assert TickRow.from_record(dict(old, enemy_in_reach="False", resync="True")).resync is True
    assert TickRow.from_record(dict(old, enemy_in_reach="False")).enemy_in_reach is False


def test_write_csv_and_read_telemetry_csv_round_trip_the_cadence_columns(loop, tmp_path):
    from brawl_deployment.loop import read_telemetry_csv

    loop.vision.entities.detections = [_Detection("player", HERO_PX),
                                       _Detection("enemy", (HERO_PX[0] + 2 * 48, HERO_PX[1]))]
    _play(loop, 4)
    path = tmp_path / "run.csv"
    loop.write_csv(path)
    back = read_telemetry_csv(path)
    assert [r.index for r in back] == [r.index for r in loop.telemetry]
    decisions = [r for r in back if r.decision]
    assert len(decisions) == 2
    assert decisions[1].attack_legal == 0b1011 and decisions[1].enemy_in_reach is True
    assert decisions[1].attack_cd_shadow == 0.0
    held = [r for r in back if not r.decision]
    assert all(r.attack_legal == -1 and r.enemy_in_reach is False for r in held)


def test_write_csv_makes_its_folder(loop, tmp_path):
    """It runs after the match, so a folder that is not there yet lost two K5 dry runs' telemetry
    (2026-09-28) after the whole match had been played."""
    from brawl_deployment.loop import read_telemetry_csv

    _play(loop, 2)
    path = tmp_path / "deploy" / "k5" / "match.csv"
    loop.write_csv(path)
    assert len(read_telemetry_csv(path)) == len(loop.telemetry)


# -- the audit's `ammo` and the reach's brawler kind -------------------------------------------

def test_the_shadows_clip_is_recorded_even_when_the_frame_has_no_hero_box(loop):
    """`ammo_shadow` is what `scripts/audit_attack_cadence.py --telemetry` reads as the row's
    ammo, next to `attack_legal`. The shadow has a clip whether or not the detector found the
    hero this frame, so the column is written before the hero-box early return -- a `-1.0`
    sentinel beside a valid mask would reach the audit as a clip size."""
    from brawl_deployment.loop import TickRow

    _play(loop, 2)
    row = TickRow(index=0, t=0.0, grab_ms=0.0)
    loop._read_own_bars(loop.capture.image, [], row)     # no `player` box at all
    assert row.ammo_shadow == 3.0, "a fresh shadow's full clip, not the sentinel"
    assert row.ammo_cv == -1.0, "and no CV read to put beside it"
    assert row.resync is False


def test_a_decision_taken_while_the_hero_track_coasts_carries_the_shadows_ammo(loop):
    """`EntityTracker` keeps the hero track for up to three misses, so a
    decision can proceed on a frame with no `player` box. The cadence columns on that row are
    valid and its `ammo_shadow` must be the shadow's clip, not `-1.0`. (The real `HealthTracker`
    emits no hero reading on such a frame and the loop skips with "no hero hp" first; this
    fixture's health fake keeps reading, which is exactly what lets the row be exercised.)"""
    _play(loop, 1)                                        # first decision: hero seen, track made
    loop.vision.entities.detections = []                  # the box drops; the track coasts
    for _ in range(loop.decision_every):
        loop.tick()
    first, second = _decisions(loop)
    assert second.attack_legal == 0b1011 and second.note == ""
    assert second.ammo_shadow == 3.0
    assert second.ammo_cv == -1.0


def test_the_reach_and_the_shadow_share_one_brawler_kind(loop):
    """`ShadowParams` does not record which brawlers.yaml block it came from, so the loop names
    the kind once (`HERO_KIND`) and hands it to both `ShadowParams.load` and `dash_reach_tiles`;
    a second default in either would let `dash_distance` and `dash_radius` come from two
    brawlers. Numbers are Mortis's, as literals."""
    from brawl_deployment.loop import HERO_KIND, dash_reach_tiles

    assert HERO_KIND == "hero_mortis"
    assert float(loop.shadow.p.dash_distance) == pytest.approx(2.67)
    assert dash_reach_tiles(loop.shadow.p, kind=HERO_KIND) == pytest.approx(3.77)
    with pytest.raises(KeyError, match="hero_nobody"):
        dash_reach_tiles(loop.shadow.p, kind="hero_nobody")


# -- the dead-bin move mask (design 6.16) ------------------------------------------------------

def test_a_wall_beside_the_hero_masks_the_bin_into_it(loop):
    """`move_mask.py` through the loop, on the `blocks_unit` plane the policy is handed and in
    that plane's units. The hero stands at world (0.5, 0.5), mid-cell. The first decision sees an
    unobserved map -- UNKNOWN reads as floor, so every bin is legal and the row says 0x1FFFF --
    then a WALL is written into the cell west of him and the next decision has bin 9 (west) dead
    with idle, east and north still legal. The row's bitmask is the tuple the policy got."""
    from brawl_sim.constants import Tile
    from brawl_vision.terrain.labeling import CLASS_INDEX

    lp = _deploy4_loop(loop, real_assembler=False)
    lp.vision.entities.detections = [_Detection("player", (HERO_PX[0] + 24, HERO_PX[1] + 24))]
    _play(lp, 1)
    assert lp.policy.calls == 1, _last_attempt(lp).note
    assert lp.policy.move_legal[0] == (True,) * 17
    assert _decisions(lp)[-1].move_legal == 0x1FFFF

    occ = lp.vision.occupancy
    ox, oy = occ.origin
    occ._best[0 - oy, -1 - ox] = CLASS_INDEX[Tile.WALL]          # world tile (-1, 0)
    _play(lp, lp.decision_every)
    assert lp.policy.calls == 2, _last_attempt(lp).note
    legal = lp.policy.move_legal[1]
    assert not legal[9]
    assert legal[0] and legal[1] and legal[13]
    assert legal.count(False) == 1
    assert _decisions(lp)[-1].move_legal == sum(int(v) << i for i, v in enumerate(legal))
    assert _decisions(lp)[-1].move_legal == 0x1FFFF & ~(1 << 9)


def test_the_dead_bin_mask_is_off_by_config_and_the_row_says_so(loop):
    """`policy.dead_bin_mask: false` is the A/B switch: the policy gets None -- `act` then hands
    the network the training-time all-True move half -- and the row records -1, not 0x1FFFF,
    because "every bin was legal" and "nothing was checked" are different facts. A CSV written
    before the column existed loads to the same -1."""
    from brawl_deployment.loop import TickRow

    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=loop.vision,
                    policy=_Policy(spec_path="configs/agent_obs_deploy4.yaml"),
                    controls=loop.controls, cfg=DeploymentConfig(policy_dead_bin_mask=False))
    lp.backend = loop.backend
    assert lp.move_mask is None
    _play(lp, 1)
    assert lp.policy.calls == 1, _last_attempt(lp).note
    assert lp.policy.move_legal == [None]
    assert _decisions(lp)[-1].move_legal == -1
    assert TickRow.from_record({"index": 0, "t": 0.0, "grab_ms": 0.0}).move_legal == -1


# -- the auto-aimed attack (action.auto_aim, attack value 4; the lead, 2026-09-26) ---------------
#
# A run trained with the flag has a 5-wide attack column. The loop reads the flag off the run's own
# config (`policy.cfg`), gives its shadow the fifth mask column, sends value 4 to the device as a
# bare tap on the attack point (the game aims it) and hands the shadow its own estimate of the
# game's target, so the modelled dash goes where the real one does.

AUTO_CFG = load_config("configs/default.yaml", overrides={"action": {"auto_aim": True}})
AUTO_LEGAL = (True, True, False, True, True)


def test_an_auto_aimed_attack_is_a_bare_tap_aimed_at_the_nearest_track(monkeypatch):
    """The row records 4 and a 5-bit mask; the device gets the tap, down on the decision tick and
    up on the next, no drag; and the shadow dashes at the nearest enemy track in reach, 2 tiles
    +x here, not along bin 5 (+y). The track is confirmed on its second sighting, so the first
    window only looks."""
    _patch_vision(monkeypatch)
    lp = _build_loop(_Policy(cfg=AUTO_CFG))
    assert lp.shadow.auto_aim and lp.shadow.attack_mask() == AUTO_LEGAL
    lp.vision.entities.detections = [_Detection("player", HERO_PX),
                                     _Detection("enemy", (HERO_PX[0] + 2 * 48, HERO_PX[1]))]
    _play(lp, lp.decision_every)
    assert _decisions(lp)[0].attack_legal == 0b11011

    lp.policy.decision = Decision(move_bin=5, attack=ATTACK_AUTO, legal=AUTO_LEGAL)
    ax, ay = lp.controls.buttons.attack
    before = len(lp.backend.log)
    lp.tick()
    assert [e for e in lp.backend.log[before:] if e[1] == SLOT_TAP] == [("down", SLOT_TAP, ax, ay)]
    row = _decisions(lp)[1]
    assert row.attack == ATTACK_AUTO and row.attack_legal == 0b11011
    assert row.enemy_in_reach is True
    assert lp.shadow.attack_bearing == pytest.approx(0.0, abs=1e-6)
    lp.shadow.advance(0.05)                 # the sub-tick that starts the modelled dash
    s = lp.shadow.observe()
    assert s["dash_dir"] == pytest.approx((1.0, 0.0), abs=1e-6) and s["ammo"] == 2.0

    lift, *quiet = _tap_steps_per_tick(lp, lp.decision_every - 1)
    assert lift == [("up", SLOT_TAP)] and all(q == [] for q in quiet)


def test_an_auto_aimed_attack_with_nothing_in_reach_dashes_along_the_move_bin(monkeypatch):
    """No track and no crate within the dash's reach: the tap still goes out (the game then
    dashes along the joystick, which is the sim's fallback too) and the shadow's aim is the bin."""
    _patch_vision(monkeypatch)
    lp = _build_loop(_Policy(cfg=AUTO_CFG, decision=Decision(move_bin=5, attack=ATTACK_AUTO,
                                                             legal=AUTO_LEGAL)))
    lp.match.gate = True
    ax, ay = lp.controls.buttons.attack
    before = len(lp.backend.log)
    lp.tick()
    assert [e for e in lp.backend.log[before:] if e[1] == SLOT_TAP] == [("down", SLOT_TAP, ax, ay)]
    row = _decisions(lp)[0]
    assert row.attack == ATTACK_AUTO and row.enemy_in_reach is False
    assert lp.shadow.attack_bearing == pytest.approx(math.pi / 2)
    lp.shadow.advance(0.05)
    assert lp.shadow.observe()["dash_dir"] == pytest.approx((0.0, 1.0), abs=1e-6)


@pytest.mark.parametrize("move_bin", [0, 5])
def test_an_idle_super_is_a_bare_tap_and_a_moving_one_a_drag(monkeypatch, move_bin):
    """The sim aims an idle super like the game's tap-to-fire and a moving one along the bin
    (`hero.super_aim_target`), so the device taps the super button on the idle bin, down on the
    decision tick and up on the next, and drags it along the bin otherwise (bin 5 is +y)."""
    from brawl_vision.object_detection.hp_detection.hero_bars import SuperReading

    _patch_vision(monkeypatch)
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_super",
                        lambda image, det, cfg=None: SuperReading(
                            charge=1.0, ready=True, state="ready", track_px=118, row=120))
    lp = _build_loop(_Policy())
    lp.vision.entities.detections = [_Detection("player", HERO_PX)]
    _play(lp, lp.decision_every)
    assert lp.shadow.super_ready

    lp.policy.decision = Decision(move_bin=move_bin, attack=ATTACK_SUPER,
                                  legal=(True, True, True, True))
    btn = lp.controls.buttons
    sx, sy = btn.super_
    down, *rest = _tap_steps_per_tick(lp, lp.decision_every)
    assert down == [("down", SLOT_TAP, sx, sy)]
    assert _decisions(lp)[1].attack == ATTACK_SUPER
    if move_bin == 0:
        assert rest[0] == [("up", SLOT_TAP)]
        assert all(q == [] for q in rest[1:])
    else:
        (move,) = rest[0]                     # the bin's angle is float32 in the shadow
        assert move[:2] == ("move", SLOT_TAP)
        assert move[2:] == pytest.approx(btn.aim_point(ATTACK_SUPER, math.pi / 2), abs=1e-3)
        assert rest[1] == [("up", SLOT_TAP)]


def test_a_run_trained_without_the_flag_never_taps_an_auto_aimed_attack(loop):
    """The fixture's run has the 4-wide column. A 4 from its policy is an illegal value: the
    shadow refuses it, nothing is tapped, and the row says what was sent."""
    loop.policy.decision = Decision(move_bin=5, attack=ATTACK_AUTO, legal=(True, True, False, True))
    _play(loop, loop.decision_every)
    assert ("down", SLOT_TAP) not in loop.backend.events
    assert _decisions(loop)[0].attack == ATTACK_NONE
