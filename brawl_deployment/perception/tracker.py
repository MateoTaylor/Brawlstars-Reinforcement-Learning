"""Anonymous per-frame boxes -> identified entities with velocity. BRAWL_DEPLOYMENT_DESIGN.md 6.1.

**The problem this solves is identity, and the observation spec is what creates it.**
`configs/agent_obs_deploy.yaml` asks for `entities.rel_vel` (a difference across time, so
something must know that box_t and box_t+1 are the same enemy) and declares enemy slots
"index-stable within an episode" (so slot 3 must mean the same enemy for as long as it lives). A
detector supplies neither: `Detection` carries a label, a confidence and a box, and nothing that
survives a frame boundary.

**Association happens in TILE space, not pixel space, and that is the whole reason this can work
at 4 Hz.** The camera pans constantly, so in pixels every enemy on screen moves whenever the hero
does -- an enemy standing perfectly still can slide hundreds of pixels between decisions, which is
indistinguishable from it sprinting. `Odometry` already removes exactly that: adding
`position_tiles` puts detections in the segment's world frame, where a stationary enemy has a
stationary coordinate and the only motion left is the motion we actually want to measure.

`hp_detection/smooth.py` associates too, and says of itself: *"This is not a tracker and must not
be mistaken for one... a Mortis dash (17.8 tiles/s, ~340 px between decisions at 4 Hz) breaks it,
and it is meant to. When a real tracker exists, `associate` is the one function to replace."* This
is that tracker. It does not replace that function -- HP smoothing wants box-space nearest-centre
for its own purposes and is welcome to it -- but it is where identity is supposed to come from.

#### The rate makes this harder than it looks

At 4 Hz a walking brawler covers `2.73 * 0.25 = 0.68` tiles between decisions (`configs/
brawlers.yaml`, the roster tops out at 2.73), and the projection anchor is worth roughly a tile of
noise on its own -- `detector.anchor_frac`'s own calibration table reports 5-24% of frames placing
a brawler in a provably impossible cell. So the gate has to be wide relative to the true step,
which is the opposite of the comfortable regime, and two enemies passing within ~1.5 tiles of each
other can plausibly swap. Running detection at 20 Hz would shrink the step fivefold and mostly
dissolve the problem; it was declined because it is 5x the YOLO cost for a policy that only acts
at 4 Hz. That trade is recorded here because it is the thing to revisit if identity proves flaky.

**The cost half of that trade is now known to be measuring the wrong machine.** `onnxruntime` is
installed without a CUDA provider, so the detector runs on the CPU -- 26.1 ms per 640x640 frame
while the GPU is idle (BRAWL_DEPLOYMENT_DESIGN.md 9.12). 5x of that does not fit in a 50 ms tick;
5x of a GPU inference very likely does. So the reason to stay at 4 Hz should be re-derived, not
inherited, once the provider is fixed. Note the *identity* argument would survive either way:
this tracker is correct at 4 Hz by design, and a faster rate is an improvement, not a repair.

#### What is deliberately NOT here

No Kalman filter and no motion model beyond constant velocity. With <=10 entities, a gate that has
to be wider than the step anyway, and 4 measurements a second, a filter's covariance would be
carrying less information than its parameters imply. Constant-velocity predict-then-gate is the
honest amount of machinery for this problem; if identity switches turn out to be the dominant
error, the fix is a faster detection rate, not a fancier filter.
"""
import math
from dataclasses import dataclass, field

from brawl_vision.object_detection.project import to_tiles

# Fastest sustained ground speed in the roster (`configs/brawlers.yaml`: Mortis 2.73, the rest
# 2.40-2.57), in tiles/s. Sizes the gate's motion allowance.
MAX_WALK_TILES_S = 2.73

# Gate slack for everything that is not motion: projection-anchor error, box breathing as auras
# come and go (`detector.anchor_frac` measured the player box swinging 264-380 px tall inside one
# clip), and odometry's sub-tile drift. Dominates the motion term at 4 Hz, which is worth seeing
# plainly rather than burying in one combined constant.
GATE_NOISE_TILES = 1.0


@dataclass
class Track:
    """One entity's identity across frames. Positions are WORLD tiles in the current odometry
    segment -- comparable across frames, meaningless across a segment change."""
    id: int
    slot: int
    label: str
    pos: tuple[float, float]
    vel: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    misses: int = 0
    t: float = 0.0
    seen_now: bool = True          # detected THIS tick, as opposed to coasting on a prediction
    history: list = field(default_factory=list, repr=False)

    def predict(self, t: float) -> tuple[float, float]:
        dt = max(0.0, t - self.t)
        return (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)


@dataclass(frozen=True)
class TrackerResult:
    """One tick's output. `hero` is the `player` box; `enemies` are slot-ordered."""
    hero: Track | None
    enemies: list
    status: str                    # "ok" | "reset" | odometry's own status when unusable
    segment: int
    n_detections: int


def _greedy_match(tracks, points, gate: float):
    """`({track_index: point_index}, unmatched_point_indices)`, nearest pair first.

    **Globally greedy, not per-track greedy.** Iterating tracks in order and giving each its own
    nearest point lets whichever track happens to be first claim a point that is a much better fit
    for a later one -- a real failure with several enemies in a brawl, and invisible unless you go
    looking. Sorting every candidate pair by distance and consuming them in that order costs
    nothing at this scale (<=10 x <=10) and cannot make that mistake.

    Not the Hungarian algorithm, which would minimise TOTAL distance. That is the right objective
    when every point must be assigned; here points appear and vanish constantly, and the optimal
    total can be bought by pairing two entities badly rather than leaving one unmatched.
    """
    pairs = []
    for ti, (tx, ty) in enumerate(tracks):
        for pi, (px, py) in enumerate(points):
            d = math.hypot(px - tx, py - ty)
            if d <= gate:
                pairs.append((d, ti, pi))
    pairs.sort()
    used_t, used_p, matched = set(), set(), {}
    for _, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matched[ti] = pi
    return matched, [i for i in range(len(points)) if i not in used_p]


class EntityTracker:
    """Feed it one tick's detections; read identified tracks. One instance per match.

    `promote_hits` exists because a single spurious box should not consume a slot: the observation
    has `n_enemies` of them and they are index-stable, so a slot handed to noise stays wrong until
    that noise track ages out. Two consecutive detections is cheap insurance at 4 Hz.
    """

    def __init__(self, n_slots: int = 9, max_misses: int = 3, promote_hits: int = 2,
                 vel_alpha: float = 0.5, anchor_frac: float | None = None):
        self.n_slots = n_slots
        self.max_misses = max_misses
        self.promote_hits = promote_hits
        # EMA weight on each new finite difference. At 4 Hz a raw difference divides ~1 tile of
        # anchor noise by 0.25 s, so it carries a couple of tiles/s of error against a 2.73
        # tiles/s signal -- unusable raw. 0.5 halves that at the cost of one tick of lag, which
        # for a velocity the policy uses to lead a target is the better end of the trade.
        self.vel_alpha = vel_alpha
        self.anchor_frac = anchor_frac
        self.hero: Track | None = None
        self.enemies: dict[int, Track] = {}      # slot -> Track
        self._pending: list[Track] = []          # seen once, not yet given a slot
        self._next_id = 0
        # None, not -1: the first tick ADOPTS whatever segment odometry is in rather than treating
        # it as a change. A sentinel that compares unequal to segment 0 would throw away the first
        # decision of every match, which is cheap but silent and surprising.
        self._segment: int | None = None

    # -- the tick ------------------------------------------------------------

    def update(self, detections, plan, odometry, t: float) -> TrackerResult:
        """`detections` are `Detection`s in viewport pixels; `plan` the `RectifyPlan` they were
        produced under; `odometry` that frame's `OdometryResult`; `t` its timestamp in seconds.

        **Odometry gates this exactly as it gates terrain deposits, and for the same reason.**
        `occupancy.update` refuses anything but `status == "ok"` because `uncertain` has a world
        frame that is not trustworthy evidence and `lost` has no world frame at all. Positions
        here are in that same frame, so a tick admitted on a bad estimate would move every track
        by the odometry's error and read as every entity accelerating at once.
        """
        if self._segment is None:
            self._segment = odometry.segment
        elif odometry.segment != self._segment:
            # A new segment is a new world frame with no defined offset to the old one, so every
            # stored position is now meaningless. Death, respawn and the match-start fly-in all
            # land here; carrying tracks across would teleport them.
            self.reset(odometry.segment)
            return TrackerResult(None, [], "reset", odometry.segment, len(detections))
        if odometry.status != "ok":
            return TrackerResult(self.hero, self.slots(), odometry.status, self._segment,
                                 len(detections))

        px, py = odometry.position_tiles
        placed = to_tiles(detections, plan, anchor_frac=self.anchor_frac) if self.anchor_frac \
            else to_tiles(detections, plan)
        world = [(d, (tx + px, ty + py)) for d, (tx, ty) in placed]

        self._update_hero([p for d, p in world if d.label == "player"], t)
        self._update_enemies([p for d, p in world if d.label == "enemy"], t)
        return TrackerResult(self.hero, self.slots(), "ok", self._segment, len(detections))

    def _update_hero(self, points, t: float) -> None:
        """The hero is a singleton, so there is no association problem -- take the one box.

        More than one `player` box means the detector is confused rather than that a second hero
        exists; the nearest to the current estimate is the right tie-break and the alternative
        (dropping the tick) throws away a frame over a transient.
        """
        if not points:
            if self.hero is not None:
                self.hero.misses += 1
                self.hero.seen_now = False
                if self.hero.misses > self.max_misses:
                    self.hero = None
            return
        if self.hero is None:
            self.hero = Track(id=self._take_id(), slot=-1, label="player", pos=points[0], t=t)
            return
        pred = self.hero.predict(t)
        best = min(points, key=lambda p: math.hypot(p[0] - pred[0], p[1] - pred[1]))
        self._advance(self.hero, best, t)

    def _update_enemies(self, points, t: float) -> None:
        live = list(self.enemies.values()) + self._pending
        gate = self.gate_for(t, live)
        matched, fresh = _greedy_match([tr.predict(t) for tr in live], points, gate)
        for ti, pi in matched.items():
            self._advance(live[ti], points[pi], t)
        for ti, tr in enumerate(live):
            if ti in matched:
                continue
            tr.misses += 1
            tr.seen_now = False
        for pi in fresh:
            self._pending.append(Track(id=self._take_id(), slot=-1, label="enemy",
                                       pos=points[pi], t=t))
        self._retire()
        self._promote()

    # -- track bookkeeping ---------------------------------------------------

    def _advance(self, tr: Track, pos: tuple[float, float], t: float) -> None:
        dt = t - tr.t
        if dt > 1e-6:
            vx = (pos[0] - tr.pos[0]) / dt
            vy = (pos[1] - tr.pos[1]) / dt
            a = self.vel_alpha if tr.hits > 1 else 1.0   # first measurement has nothing to blend
            tr.vel = (a * vx + (1 - a) * tr.vel[0], a * vy + (1 - a) * tr.vel[1])
        tr.pos = pos
        tr.t = t
        tr.hits += 1
        tr.misses = 0
        tr.seen_now = True

    def _retire(self) -> None:
        for slot, tr in list(self.enemies.items()):
            if tr.misses > self.max_misses:
                del self.enemies[slot]
        # A pending track has no slot to protect, so it gets no coast at all: one miss and it is
        # gone. That is what makes `promote_hits` mean "seen on CONSECUTIVE ticks".
        self._pending = [tr for tr in self._pending if tr.misses == 0]

    def _promote(self) -> None:
        for tr in list(self._pending):
            if tr.hits < self.promote_hits:
                continue
            slot = self._free_slot()
            if slot is None:
                continue      # every slot taken; leave it pending rather than evicting a live one
            tr.slot = slot
            self.enemies[slot] = tr
            self._pending.remove(tr)

    def _free_slot(self) -> int | None:
        for i in range(self.n_slots):
            if i not in self.enemies:
                return i
        return None

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    # -- interface -----------------------------------------------------------

    def gate_for(self, t: float, tracks) -> float:
        """Association radius in tiles, for the largest gap any live track is spanning.

        Scales with the ACTUAL elapsed time rather than a nominal 0.25 s: a dropped frame or a
        slow decision widens the true step, and a gate sized for the nominal rate would then
        reject every correct match at once -- losing identity precisely when the loop is already
        struggling.
        """
        dt = max((t - tr.t for tr in tracks), default=0.0)
        return GATE_NOISE_TILES + MAX_WALK_TILES_S * max(dt, 0.0)

    def slots(self) -> list:
        """Slot-ordered live enemies, `None` in empty slots. This is the shape the observation's
        `entities` group wants: index-stable, with holes."""
        return [self.enemies.get(i) for i in range(self.n_slots)]

    def reset(self, segment: int) -> None:
        self.hero = None
        self.enemies.clear()
        self._pending.clear()
        self._segment = segment
