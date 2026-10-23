import os

import pytest

# pytest-xdist: unpinned, every worker starts torch's OpenMP pool (and MKL's) on every core, eight
# workers x eight threads on this eight-core machine. Measured 2026-10-20 at `-n 8`: twice the
# wall time (3 min 26 s against 1 min 43 s) and a real-time budget failing
# (test_vision_classifier's four-parallel-games test). Set here, before any test module imports
# torch; a serial run keeps the library defaults, and an OMP_NUM_THREADS already set wins.
if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """tests/test_viewer.py and tests/test_play_manual.py each construct a real
    matplotlib Figure (ReplayViewer/ManualPlaySession) per test and never close it --
    harmless individually, but the accumulation trips matplotlib's "more than 20 figures
    open" warning partway through a full run. Closing everything after every test (not just
    the two files that need it) is simpler and safer than importing matplotlib here just to
    scope the fixture, and is a no-op for tests that never touched pyplot."""
    yield
    import sys
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")
