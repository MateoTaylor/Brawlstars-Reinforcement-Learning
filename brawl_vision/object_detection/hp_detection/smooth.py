"""Holding an HP value steady across frames. The stateful half of this chunk.

`read.py` judges one frame in isolation and is deliberately memoryless. This remembers, and it is
where most of the consistency a policy sees actually comes from -- a single-frame reader has no
way to tell a real 8400 -> 7400 hit from a one-frame mis-segmentation, because both look like
"the number changed".

#### Why this is debouncing and not voting

The obvious temporal filter is a confidence-weighted vote over a window, and it is wrong here.
Voting assumes the quantity is CONSTANT and the disagreement is noise. HP is not constant -- in a
fight it changes almost every frame -- so a window vote collapses its own agreement score exactly
when the reading is most correct and most needed, and lags every real hit by half a window.

What the errors have in common instead is that they do not REPEAT. A heal popup covers a
different part of the number each frame; a mis-segmented digit is gone next frame. A real hit,
meanwhile, persists. So a changed value is held pending until a second frame agrees with it,
which costs one frame of latency on a genuine change and rejects essentially every transient.
A very confident read commits immediately and pays no latency at all.

#### The one game-specific rule, and why it earns its place

`_is_truncation` catches the failure this chunk was built around. When the readout is partly
covered, what survives is not a random number -- it is the true number with digits missing off
one end: `8400` reads as `400`, or `840`, or `8`. Those are the errors that are catastrophic
rather than annoying, because they are wrong by a factor of ten or a thousand while remaining a
perfectly plausible HP value that no range check would catch.

So a candidate that is a truncation of the value it would replace never commits on confidence
alone, however clean its glyphs looked -- the measured worst case scored 0.928 with a 0.181
margin. It must be confirmed by repetition like any other change. A real hit is essentially never
a digit-preserving truncation of the previous HP, so this costs almost nothing.

#### What survives a bad frame

A fatal read (`occluded`, `gap`, ...) does not clear the track. The last committed HP stays, with
its confidence decayed per frame, until `max_misses` frames have passed without a usable read --
because a brawler whose number is briefly covered by its own damage popup is a brawler whose HP
is still, to a good approximation, what it was 250 ms ago. Reporting nothing there would punch a
hole in the observation exactly when the agent is being shot at.

**This is not a tracker and must not be mistaken for one.** Association is nearest-centre within
a fraction of a box width, per class, greedy. It exists to carry an HP value across a few frames,
not to maintain identity: a Mortis dash (17.8 tiles/s, ~340 px between decisions at 4 Hz) breaks
it, and it is meant to. When a real tracker exists, `associate` is the one function to replace.

#### Where the remaining error lives, measured

Hand-scored samples during development turned up four trusted errors, and **all four came from
this module, none from the classifier** -- two were a value still awaiting `confirm_frames` while
the HP had genuinely changed, and two were a coast that ran stale through a sustained damage
flash. So the cost of everything above is paid in LATENCY, not in wrong magnitudes: the values
this emits are occasionally 250-750 ms out of date, and essentially never absurd. (The final
sample, after the gap and association bugs below were fixed, showed 57 of 57 unambiguous trusted
reads correct.)

That is the right way round for a policy -- an HP that lags by a decision tick is a much smaller
lie than an HP that reads 8 when it is 8400 -- but it is a real cost, and `confirm_frames` is the
dial. Set it to 1 to trade this latency back for transients.
"""
from collections import deque
from dataclasses import dataclass, field

from .read import HealthReading


def _is_truncation(a: int, b: int) -> bool:
    """True when one of these is the other with digits removed from an END.

    The signature of a partly-covered readout -- see the module docstring. Checked both ways
    because the covered frame can be either the incoming candidate or the value already held.
    """
    sa, sb = str(a), str(b)
    if len(sa) == len(sb):
        return False
    short, long = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    return long.startswith(short) or long.endswith(short)


@dataclass
class Track:
    """One brawler's HP as it is currently believed, plus the candidate trying to replace it."""
    track_id: int
    label: str
    xyxy: tuple[float, float, float, float]
    hp: int | None = None
    confidence: float = 0.0
    stable: int = 0
    misses: int = 0
    pending: int | None = None
    pending_count: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=16))


@dataclass(frozen=True)
class SmoothedReading:
    """A per-frame reading after the tracker has had its say.

    `hp` may be non-None while `raw.hp` is None: that is the tracker carrying the last committed
    value through an occluded frame, and `frames_since_seen` says how stale it is.
    """
    hp: int | None
    confidence: float
    detection: object
    raw: HealthReading
    track_id: int
    stable_frames: int = 0
    frames_since_seen: int = 0

    @property
    def label(self) -> str:
        return self.raw.detection.label

    def trusted(self, minimum: float) -> bool:
        return self.hp is not None and self.confidence >= minimum


def associate(readings, tracks, max_distance_frac: float):
    """`{track_id: reading}` plus the readings that matched nothing, greedy by nearest centre.

    Class-aware: a `player` box never inherits an `enemy` track, which in Solo Showdown is the
    one confusion that would be actively harmful. Distance is scaled by box width so the gate
    means the same thing for a big box and a small one.
    """
    pairs = []
    for ri, reading in enumerate(readings):
        rx0, ry0, rx1, ry1 = reading.detection.xyxy
        rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        limit = (rx1 - rx0) * max_distance_frac
        for track in tracks:
            if track.label != reading.detection.label:
                continue
            tx0, ty0, tx1, ty1 = track.xyxy
            d = (((tx0 + tx1) / 2 - rcx) ** 2 + ((ty0 + ty1) / 2 - rcy) ** 2) ** 0.5
            if d <= limit:
                pairs.append((d, ri, track.track_id))
    pairs.sort()
    taken_r, taken_t, matched = set(), set(), {}
    for _, ri, tid in pairs:
        if ri in taken_r or tid in taken_t:
            continue
        taken_r.add(ri)
        taken_t.add(tid)
        matched[tid] = readings[ri]
    return matched, [r for i, r in enumerate(readings) if i not in taken_r]


class HealthTracker:
    """Wraps a `HealthReader` with per-brawler memory. Call `update` once per processed frame.

    Frame-indexed, not time-indexed: every threshold here counts FRAMES at whatever rate the
    caller drives it. At the intended 4 Hz decision rate `confirm_frames=2` is 500 ms of latency
    on a change and `max_misses=3` is 750 ms of coasting.
    """

    def __init__(self, reader=None, *, confirm_frames: int = 2, max_misses: int = 3,
                 instant_confidence: float = 0.90, decay: float = 0.75,
                 match_distance_frac: float = 1.5):
        from .read import HealthReader
        self.reader = reader or HealthReader()
        self.confirm_frames = int(confirm_frames)
        self.max_misses = int(max_misses)
        self.instant_confidence = float(instant_confidence)
        self.decay = float(decay)
        self.match_distance_frac = float(match_distance_frac)
        self._tracks: dict[int, Track] = {}
        self._next_id = 0

    @classmethod
    def from_config(cls, cfg=None, reader=None, **overrides) -> "HealthTracker":
        from ...config import VisionConfig
        from .read import HealthReader
        cfg = cfg or VisionConfig()
        kwargs = dict(reader=reader or HealthReader.from_config(cfg),
                      confirm_frames=cfg.hp_confirm_frames, max_misses=cfg.hp_max_misses,
                      instant_confidence=cfg.hp_instant_confidence, decay=cfg.hp_decay,
                      match_distance_frac=cfg.hp_match_distance_frac)
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks.values())

    def reset(self) -> None:
        """Forget everything. Call between clips, or on a detected camera cut -- carrying a track
        across a cut is how a dead brawler's HP ends up attached to a live one."""
        self._tracks.clear()

    def update(self, image, detections) -> list[SmoothedReading]:
        """Read every box in `image` and fold each into its track. One call per frame."""
        return self.update_readings(self.reader.read_all(image, detections))

    def update_readings(self, readings) -> list[SmoothedReading]:
        """The same, given readings already produced by a `HealthReader`.

        Split out so a caller that already has readings (a render loop drawing both the raw and
        smoothed values, say) does not pay for a second pass over the pixels.
        """
        readings = list(readings)
        matched, fresh = associate(readings, self._tracks.values(), self.match_distance_frac)

        out: list[SmoothedReading] = []
        # `handled` must include tracks created THIS frame, not just pre-existing ones that
        # matched. Without them a brand-new track was immediately charged a miss on its own first
        # frame, so every reading came back one decay step below the confidence it had earned --
        # and a value corroborated over several frames climbed from a floor it should never have
        # been pushed to. Silent, and only visible as confidences that were mysteriously low.
        handled = set(matched)
        for tid, reading in matched.items():
            out.append(self._fold(self._tracks[tid], reading))
        for reading in fresh:
            track = Track(self._next_id, reading.detection.label, reading.detection.xyxy)
            self._tracks[self._next_id] = track
            self._next_id += 1
            handled.add(track.track_id)
            out.append(self._fold(track, reading))

        for track in self._tracks.values():
            if track.track_id in handled or track.hp is None:
                continue
            track.misses += 1
            track.confidence *= self.decay
        for tid in [t.track_id for t in self._tracks.values() if t.misses > self.max_misses]:
            del self._tracks[tid]
        return out

    def _fold(self, track: Track, reading: HealthReading) -> SmoothedReading:
        track.xyxy = reading.detection.xyxy

        if reading.hp is None:
            # Coast. The committed value survives a covered readout; only its confidence erodes.
            track.misses += 1
            track.confidence *= self.decay
            if track.misses > self.max_misses:
                track.hp, track.confidence, track.stable = None, 0.0, 0
            return SmoothedReading(track.hp, track.confidence, reading.detection, reading,
                                   track.track_id, track.stable, track.misses)

        track.misses = 0
        if track.hp is None:
            self._commit(track, reading.hp, reading.confidence)
        elif reading.hp == track.hp:
            track.stable += 1
            track.pending, track.pending_count = None, 0
            # Corroboration moves confidence toward 1 without ever asserting certainty.
            track.confidence += (1.0 - track.confidence) * reading.confidence * 0.5
        else:
            suspicious = _is_truncation(reading.hp, track.hp)
            if reading.confidence >= self.instant_confidence and not suspicious:
                self._commit(track, reading.hp, reading.confidence)
            else:
                if reading.hp == track.pending:
                    track.pending_count += 1
                else:
                    track.pending, track.pending_count = reading.hp, 1
                if track.pending_count >= self.confirm_frames:
                    self._commit(track, reading.hp, reading.confidence)
                else:
                    track.confidence *= self.decay
        track.history.append((reading.hp, reading.confidence))
        return SmoothedReading(track.hp, track.confidence, reading.detection, reading,
                               track.track_id, track.stable, 0)

    def _commit(self, track: Track, hp: int, confidence: float) -> None:
        track.hp = hp
        track.confidence = confidence
        track.stable = 0
        track.pending, track.pending_count = None, 0
