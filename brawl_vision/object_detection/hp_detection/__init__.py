"""brawl_vision.object_detection.hp_detection -- reading a brawler's HP out of its own box.

**A stage INSIDE `object_detection`, not a chunk beside it.** Unlike `terrain/` and
`object_detection/`, which share only `sources.Frame`, this one cannot run at all without the
detector: every number it reads is inside a box the detector found, and it never looks at a pixel
the detector did not point it at. That dependency is the reason it lives here rather than as a
sibling package -- the import direction says what is true.

**What it costs.** 1.7 ms/frame measured over `day10_gameplay.mp4` at 2.5 boxes/frame, i.e.
~0.7 ms per box, against the detector's own 12.9 ms and a 250 ms decision budget
(`sim.action_repeat: 5` at 20 Hz). It is not a stage anybody needs to optimize; see `read.py` for
why the effort went into being RIGHT instead.

**What it delivers.** Hand-checked on `day10_gameplay.mp4` -- crops read by eye against what the
pipeline said, so these are real accuracies and not self-consistency scores. 72 sampled reads on
the final code:

    trusted at confidence >= 0.5     61 reads, of which 57 unambiguous -- 57 correct
    below that threshold             11 reads, mostly genuinely wrong

("Unambiguous" excludes 4 crops where the padded window contained two brawlers' readouts and the
scorer could not tell which one the box meant -- a limit of the check, not a known error.) Over
the whole `--stop 6000` sweep, 82.8% of boxes yield a parse at all and 79.1% clear the confidence
threshold; the rest are honestly reported as `no-digits`, `occluded`, `gap` or `leading-zero`.

**The residual error is STALENESS, not misreading.** Earlier hand-scored samples caught four
trusted errors, and every one was the tracker reporting a value that had been correct one or two
frames before -- coasting through a damage flash, or waiting out `confirm_frames` on a genuine
change. None was a misread digit. That is the trade this chunk is tuned for and it is worth
knowing which way it leans: HP here runs up to ~750 ms late, and essentially never absurd.

#### The four modules

* `locate.py`  -- where on the pixels the readout is. Geometry and colour, no numbers.
* `glyphs.py`  -- the ten digit templates and a nearest-template NCC classifier.
* `read.py`    -- one box -> one HP with a confidence. Memoryless, and where the gates live.
* `smooth.py`  -- HP held steady across frames. Stateful, and where consistency comes from.

`draw.py` annotates and computes nothing, matching `object_detection/draw.py`.

#### The thing to know before using it

The dangerous failure of this pipeline is not a noisy number, it is a CONFIDENT WRONG one. The
game floats damage and healing popups -- white, same font, several times larger -- straight over
the health readout, and a covered `8400` reads as a clean, well-formed `8` that the classifier
scores 0.928. Every structural gate in `read.py` and the truncation rule in `smooth.py` exist for
that one failure. So: **use the confidence.** `HealthReading.trusted(minimum)` and
`SmoothedReading.trusted(minimum)` are the intended interface, and `hp` on its own is the raw
read, offered because suppressing it would throw away information the caller may be able to use.

    from brawl_vision.object_detection import ObjectDetector
    from brawl_vision.object_detection.hp_detection import HealthTracker

    detector = ObjectDetector.from_config(cfg)
    tracker = HealthTracker.from_config(cfg)
    for frame in source:
        dets = detector.predict(frame.image)
        for reading in tracker.update(frame.image, dets):
            if reading.trusted(cfg.hp_min_confidence):
                ...  # reading.hp is good
"""
from .glyphs import Glyph, GlyphBank
from .locate import Bar, DigitRow
from .hero_bars import AmmoReading, SuperReading, read_ammo, read_super
from .read import HealthReader, HealthReading
from .smooth import HealthTracker, SmoothedReading, Track
from .draw import draw_health

__all__ = ["AmmoReading", "Bar", "DigitRow", "Glyph", "GlyphBank", "HealthReader",
           "HealthReading", "HealthTracker", "SmoothedReading", "SuperReading", "Track",
           "draw_health", "read_ammo", "read_super"]
