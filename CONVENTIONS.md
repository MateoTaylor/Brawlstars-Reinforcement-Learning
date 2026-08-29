# 2. Global conventions

Step 1 writes this section verbatim to `CONVENTIONS.md`. **Every delegated task reads it first.**

## Stack
Python 3.11+, PyTorch 2.x, `numpy`, `PyYAML`, `pytest`.
Extras: `[sb3] stable-baselines3, sb3-contrib, gymnasium`, `[render] matplotlib`.
**No JAX, no numba, no custom CUDA.** Plain PyTorch tensor ops.

## The one rule that matters
**Everything is a batched tensor with a leading `(N,)` environment dimension, and there is
never a Python loop over environments.** If you write `for i in range(n_envs)`, the design is
wrong — **except** inside `wrappers/sb3_vecenv.py`, where SB3's list-of-dicts `infos` contract
makes it unavoidable. That file is the single sanctioned exception and must say so at the top.

Python loops are otherwise permitted only over compile-time constants: the 4 bot archetypes,
the fixed raycast step count, the fixed volley size, the fixed channel list.

## No host synchronization in the hot path
Forbidden inside `BrawlVecEnv.step()` and anything it calls:
`.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `bool(t)`, `int(t)`, `float(t)`,
`if some_tensor:`, `assert some_tensor`, `print(some_tensor)`, `torch.nonzero()`,
`t[bool_mask]` (data-dependent shape → forced sync).

Use `torch.where`, masked arithmetic, `masked_fill_`, `scatter_`,
`index_put_(accumulate=True)`, `clamp`, `topk`.

**Host transfers are legal in exactly one place: `wrappers/sb3_vecenv.py`.** Nowhere else.

Debug assertions go behind `if self.cfg.debug_checks:` where `debug_checks` is a Python bool,
off by default.

## Mutation model
Systems **mutate state in place** on preallocated buffers, and declare it:
```python
def apply_movement(state: SimState, move_dir: Tensor, ...) -> None:
    """MUTATES: state.ent_pos, state.ent_vel, state.ent_facing."""
```
Allocate nothing per step that can be preallocated.

## Dtypes and units
- Reals: `float32`. Indices: `int64`. Counters/enums: `int32`. Masks: `bool`.
  Observation occupancy grids: `uint8`.
- **Distance in tiles** (1 tile = 1.0). **Time in seconds.** **Angles in radians**,
  `atan2(dy, dx)`, range `(-pi, pi]`.
- **Coordinates:** `pos = (x, y)`, x = column, y = row. Tile `(col=i, row=j)` covers
  `[i, i+1) x [j, j+1)`. **y increases downward**, matching CSV row order.

## Shapes
`N` envs, `E` entities (`1 + n_enemies`), `P` projectile slots, `B` boxes, `U` pickups,
`M` maps, `H`/`W` map grid, `VH`/`VW` view grid, `K` = `N_KINDS` (5), `C` channels.

**Entity index 0 is always the hero. Entity slot order is stable for the whole episode.**
Never sort or compact entity slots.

## Devices and RNG
The env owns `self.device` and `self.gen = torch.Generator(device=device)`. Every random call
passes `generator=self.gen`. Never touch the global RNG.

## Testing
Every step ships `tests/test_<module>.py`, running on CPU by default with `@pytest.mark.gpu`
variants where relevant. Use `n_envs = 8` and `presets/debug_tiny.yaml`.

## Config
Nothing is hardcoded. Every number reaches code through `EnvConfig` (static Python values),
`SimParams` (per-env tensors), or `AgentObsSpec` (declarative agent observation).

---

# 4. Canonical tick order

`BrawlVecEnv.step()` executes exactly these phases, in order, once per `dt`.
**Attack resolution comes BEFORE movement** — that's what makes the dash replace the walk.

```
 1. action latency buffer      pop the action from L ticks ago, push the new one
 2. advance timers             ammo/reload, attack_cd, dash_t, invuln_t, reveal_t, react_t, no_damage_t
 3. decode hero action         move bin -> direction; attack bool ANDed with the action mask
 4. bot policies                all 4 archetypes computed batched, selected by kind
 5. external overrides         optional, drives any entity index
 6. ATTACK RESOLUTION          consume ammo; START DASH (clip path now); spawn volleys; melee hitscan
 7. movement                   ONLY non-dashing entities. intent -> terrain slide -> soft separation
 8. dash advance               dashers move the clipped path; capsule sweep; once-per-victim damage
 9. projectile update          integrate, wall collide, unit collide, artillery detonate, AoE
10. zone damage                outside the safe rect; i-frames do NOT block this
11. box damage                 destroyed boxes -> spawn pickups
12. pickup collection
13. death resolution           hp <= 0 -> dead, record rank + cause, drop cubes
14. zone shrink schedule
15. bookkeeping                time += dt, step_count += 1, n_alive, cumulative counters
16. perception + observation   visibility matrix (annotation only), full obs package, done, info
17. autoreset                  reset done envs in place; stash final obs/info
```

Deaths resolve after all damage sources, so simultaneous kills work.
