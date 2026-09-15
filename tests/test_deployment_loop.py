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

import numpy as np
import pytest

from brawl_deployment.control.backend import SLOT_MOVE, SLOT_TAP
from brawl_deployment.control.buttons import ATTACK_FIRE, ATTACK_NONE, ATTACK_SUPER
from brawl_deployment.loop import (ODOMETRY_LOST_SECONDS, TELEMETRY_ROWS, Controls, DeployLoop,
                                   Phase, TickRow, VisionStack)
from brawl_deployment.config import DeploymentConfig
from brawl_deployment.control.buttons import Buttons
from brawl_deployment.control.joystick import Joystick
from brawl_deployment.perception.assemble import ObservationAssembler
from brawl_deployment.policy import Decision
from brawl_deployment.window import WindowFault
from brawl_sim.config import load_config
from brawl_sim.core.obs_select import agent_space, load_agent_spec
from brawl_vision.capture import Frame
from brawl_vision.hud import BrawlersLeft
from brawl_vision.object_detection.detector import Detection
from brawl_vision.object_detection.projectile_detection.classes import (CUBE_BOX, CUBE_DROPPED,
                                                                        PROJECTILE)
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

    def down(self, slot, x, y):
        self.events.append(("down", slot))

    def move(self, slot, x, y):
        self.events.append(("move", slot))

    def up(self, slot):
        self.events.append(("up", slot))

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

    @property
    def origin(self):
        return (-(self.width // 2), -(self.height // 2))

    def best(self):
        return self._best

    def update(self, cells, odometry, plan, zone=None, occluded=None, cfg=None):
        self.deposits += 1
        self.occluded = occluded


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
                 real_assembler=False):
        self.cfg = CFG
        self.spec = load_agent_spec(spec_path, CFG)
        self.assembler = (ObservationAssembler(self.spec, CFG) if real_assembler
                          else _Assembler())
        self.obs = []
        self.decision = decision or Decision(move_bin=3, attack=ATTACK_NONE,
                                             legal=(True, True, False))
        self.calls = 0
        self.raises = None

    def make_assembler(self, **kw):
        return self.assembler

    def act(self, obs, attack_legal):
        self.calls += 1
        self.obs.append(obs)
        if self.raises is not None:
            raise self.raises
        return self.decision


@pytest.fixture
def loop(monkeypatch):
    """A `DeployLoop` wired to fakes on the vision side and the real thing everywhere else."""
    monkeypatch.setattr("brawl_vision.terrain.zone.detect_zone",
                        lambda rect, plan, cfg=None: _Zone())
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_ammo",
                        lambda image, det, cfg=None: None)
    monkeypatch.setattr("brawl_vision.object_detection.hp_detection.hero_bars.read_super",
                        lambda image, det, cfg=None: None)

    backend = _Backend()
    controls = Controls(backend=backend,
                        joystick=Joystick(backend, (345.6, 669.6), 110.0, n_bins=16),
                        buttons=Buttons(backend, attack=(1690.0, 594.0), super_=(1462.5, 1000.5)))
    vision = VisionStack(plan=_Plan(), odometry=_Odometry(), occupancy=_Occupancy(),
                         classifier=_Classifier(),
                         entities=_Detector([_Detection("player", HERO_PX)]),
                         projectiles=_Detector(names=PROJECTILE_NAMES),
                         health=_Health([_Reading("player", 8000)]),
                         hud=_Hud(), cfg=None)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=vision,
                    policy=_Policy(), controls=controls, cfg=DeploymentConfig())
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


def test_the_warmup_tick_is_flagged_on_the_row_that_actually_paid_for_it(loop):
    """`VisionStack.warm` runs inside the gate-opening tick and costs ~1 s of PTX JIT. It has to
    be attributed to THAT row -- telemetry is appended at tick end, so reaching for `[-1]` while
    inside the tick flags the previous one and the summary then excludes a healthy tick and keeps
    the 758 ms one. Exactly one row carries the flag, and it is the one the gate opened on."""
    warmed = []
    loop.vision.warm = lambda image: warmed.append(image)
    _play(loop, 4)
    rows = [r for r in loop.telemetry if r.warmup]
    assert len(warmed) == 1 and len(rows) == 1
    assert rows[0].index == 0                  # the first tick is where the gate opened


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
    loop.policy.decision = Decision(move_bin=0, attack=ATTACK_NONE, legal=(True, True, False))
    _play(loop, loop.decision_every * 2)
    assert ("up", SLOT_MOVE) not in loop.backend.events
    assert loop.controls.joystick.is_down


def test_the_fire_bit_does_not_repeat_across_the_decision_window(loop):
    """`env._held` zeroes the fire column on sub-ticks 2..K. One tap per decision, and the release
    lands on the next perception tick rather than inside the same `sendevent` batch."""
    loop.policy.decision = Decision(move_bin=1, attack=ATTACK_FIRE, legal=(True, True, False))
    _play(loop, loop.decision_every)
    taps = [e for e in loop.backend.events if e == ("down", SLOT_TAP)]
    assert len(taps) == 1
    assert ("up", SLOT_TAP) in loop.backend.events


def test_only_what_the_shadow_modelled_is_tapped(loop):
    """The shadow's mask is the last word: an uncharged super never reaches the device, because
    modelling a shot the game did not take is the one error the shadow must never make."""
    loop.policy.decision = Decision(move_bin=1, attack=ATTACK_SUPER, legal=(True, True, True))
    _play(loop, loop.decision_every)
    assert not loop.shadow.super_ready
    assert ("down", SLOT_TAP) not in loop.backend.events
    assert _decisions(loop)[-1].attack == ATTACK_NONE


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
                        buttons=Buttons(backend, attack=(1690.0, 594.0), super_=(1462.5, 1000.5)))
    vision = VisionStack(plan=_Plan(), odometry=_Odometry(), occupancy=_Occupancy(),
                         classifier=_Classifier(), entities=_Detector(),
                         projectiles=_Detector(names=PROJECTILE_NAMES), health=_Health(),
                         hud=_Hud(), cfg=None)
    lp = DeployLoop(capture=_Capture(), guard=_Guard(), match=_Match(), vision=vision,
                    policy=_Policy(), controls=controls, cfg=DeploymentConfig())
    lp.backend = backend
    lp.vision.warm = lambda image: None
    return lp
