import pytest


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
