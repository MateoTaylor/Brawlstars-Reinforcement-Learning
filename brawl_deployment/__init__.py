"""brawl_deployment -- the live loop that lets a policy trained in `brawl_sim` play the real game
in BlueStacks. See BRAWL_DEPLOYMENT_DESIGN.md.

Structured the way `brawl_vision` is: this root holds what every part needs, each subpackage is one
concern. `control/` is the input side (touch injection, joystick geometry, buttons); `perception/`
is the observation side. `match_state.py` sits between them because it gates both.

**CONVENTIONS.md governs `brawl_sim`, not this package, and the boundary is deliberate** -- the
same boundary `brawl_vision/__init__.py` draws for the same reason. The leading-(N,) batch rule and
the no-host-sync rule scope to `BrawlVecEnv.step()`; there is exactly one screen, one match and one
agent here, so `.item()`, `.cpu()`, plain Python loops and numpy are all fine. Do not contort this
code to look batched.

What DOES carry over, because it is what lets the halves speak: **units** (tiles, seconds,
radians) and **coordinates** (`pos = (x, y)`, x = column, y = row, y increasing DOWNWARD). That
last one is load-bearing in `control/joystick.py` and is the single easiest thing to get silently
backwards -- the world's y and the screen's y both increase downward, so the bin-to-offset map
needs no sign flip on either axis.

**Measured on this setup, and the numbers that shaped the design** (see the design doc for the
full argument):

    mss grab, 2560x1440         18.1 ms mean, 19.3 p95     <- the capture path
    adb exec-out screencap      ~290 ms                     <- NOT the capture path
    persistent adb shell RTT     0.51 ms
    4 process spawns on device   5.77 ms                    <- one joystick MOVE costs this

`adb screencap` is 16x slower than grabbing the monitor and does not fit in a 50 ms perception
tick, let alone leave room for anything else. It is a fine diagnostic and a bad frame source.
"""
