"""The trained checkpoint, loaded and asked for one decision at a time.

See BRAWL_DEPLOYMENT_DESIGN.md 6.7. This is a thin layer over `MaskablePPO.predict`, and thin is
the point -- everything interesting happened upstream in `perception/assemble.py`. What this file
actually contributes is a set of **load-time guards**, because every way of deploying a checkpoint
against the wrong observation is silent at inference.

#### The run is the only argument

`from_run(run_dir)` reads that run's own `train.yaml` and takes the observation spec, the env
config and the env overrides *from it*. Nothing here accepts a spec path, because the one thing
that must never happen is a checkpoint being fed an observation built to a different spec than it
was trained on. `configs/agent_obs_deploy.yaml` and `configs/agent_obs_deploy2.yaml` differ by two
columns in one group; a policy handed the wrong one gets a shape error if it is lucky and a
silently shifted `zone` group if it is not. The run knows which it was, so the run decides.

The same reasoning is why `make_assembler()` lives here: the assembler and the network are handed
the *same* loaded spec object, so they cannot disagree about anything.

#### `normalize.obs` is a hard refusal

A run trained with `normalize.obs: true` has its observation statistics in `best_vecnormalize.pkl`
and is **meaningless without them** -- SB3's own `VecNormalize` docs say so and
`training/callbacks.py`'s `VecNormalizeCheckpoint` exists to keep the two together. Deployment has
no `VecNormalize` in the path, so loading such a run here would feed the network raw values where
it expects whitened ones: no error, no crash, just a policy that is quietly wrong.

`runs/mortis_deploy-20260907-041522` has `normalize: {obs: false, reward: true}`, so the deployed
path needs no statistics at all -- reward normalization affects training only. That is a fact
about *this* run, not about runs in general, which is exactly why it is checked rather than
assumed.

#### The network runs on the CPU

MEASURED 2026-09-08 on this machine, batch of 1, 200 decisions after warmup: **0.96 ms on CPU
against 7.56 ms on CUDA.** The policy is 1.04M parameters, so a single-observation forward pass is
dominated entirely by kernel-launch and host-to-device transfer overhead, and the GPU loses by
8x. (Taken while the machine was otherwise in use, so treat the absolute numbers as loose -- the
*ratio* is far too large to be background noise.)

This is the opposite of the natural assumption and it is doubly convenient: the 250 ms decision
budget in 7.1 is untouched either way, and keeping the policy off the GPU leaves the whole 16 GB
of VRAM to the two YOLO models, which is where 7's actual pressure is.

#### What this file does NOT own

The interlock. `act()` will happily return a decision when the shadow is desynced or the match is
over, because deciding whether to *emit* it is `loop.py`'s job and putting half an interlock here
would make it possible to believe the wrong half is in force. The one thing `act()` does enforce
is the action mask, which is not an interlock but a promise the network was trained against.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brawl_sim.core import obs_select
from brawl_deployment.perception.assemble import MapFrame, ObservationAssembler

ATTACK_NONE, ATTACK_FIRE, ATTACK_SUPER = 0, 1, 2


def check_spaces(model, spec, cfg, *, label: str, spec_path="the spec", cfg_path="the config"):
    """The checkpoint and the spec must describe the same observation. Raises if they do not.

    Split out of `from_run` so it is testable without a 44 MB checkpoint (`runs/` is gitignored,
    so a test that needs one cannot run on a clean checkout). The two guards ahead of it -- the
    algo check and the `normalize.obs` refusal -- fire before `MaskablePPO.load` for the same
    reason, which makes the whole guard set reachable from a temp directory holding nothing but a
    `train.yaml`.

    A mismatch here has exactly one cause worth naming: a spec or env config edited after the run
    finished. That is why `configs/agent_obs_deploy2.yaml` is a new file rather than an edit of
    `agent_obs_deploy.yaml`, and why `tests/test_configs_files.py` pins both.
    """
    want = obs_select.agent_space(spec, cfg)
    if model.observation_space != want:
        raise ValueError(
            f"{label} expects {model.observation_space}, but {spec_path} under {cfg_path} builds "
            f"{want}. The run's own train.yaml names both, so one of the two files has been "
            f"edited since the run finished -- a changed spec must be a NEW file."
        )
    nvec = tuple(int(v) for v in getattr(model.action_space, "nvec", ()))
    if nvec != tuple(cfg.action_nvec):
        raise ValueError(f"{label} has action space {nvec}, config says {tuple(cfg.action_nvec)}")


@dataclass(frozen=True)
class Decision:
    """One decision, in the sim's own action encoding.

    `move_bin` is `action[:, 0]`: 0 is idle and 1..n_move_bins are `geo.dir_from_bin` directions.
    **Idle is a bin, not an absence** -- `control/joystick.py` moves the contact back to the anchor
    for it and does not release, which is the rule the whole floating-joystick design turns on.

    `attack` is `action[:, 1]`: 0 nothing, 1 attack, 2 super. `legal` is the mask the network was
    given, kept so telemetry can tell "the policy chose not to fire" from "the policy could not."
    Those look identical in the action alone and mean completely different things when a live run
    goes quiet.
    """

    move_bin: int
    attack: int
    legal: tuple[bool, bool, bool]

    @property
    def fired(self) -> bool:
        return self.attack != ATTACK_NONE


class DeployedPolicy:
    """A loaded checkpoint plus the spec and config it was trained under."""

    def __init__(self, model, spec, cfg, *, deterministic: bool = True):
        self.model = model
        self.spec = spec
        self.cfg = cfg
        self.deterministic = deterministic
        self._n_move = int(cfg.n_move_bins) + 1
        self._mask = np.ones((1, self._n_move + 3), dtype=bool)
        self._batched: dict = {}

    # -- loading --------------------------------------------------------------

    @classmethod
    def from_run(cls, run_dir, *, checkpoint: str = "best_model.zip", device: str = "cpu",
                 deterministic: bool = True) -> "DeployedPolicy":
        """Load a run's checkpoint together with the spec and config it was trained under.

        `device` defaults to the CPU on measured evidence, not on caution -- see the module
        docstring. `deterministic` matches `configs/train.yaml`'s `eval.deterministic: true`,
        which is how `best_model.zip` was *selected*; a stochastic deployment would not be the
        policy that scored 0.734.
        """
        from sb3_contrib import MaskablePPO

        from brawl_sim.config import load_config
        from brawl_sim.training.builder import _resolve
        from brawl_sim.training.config import load_train_config

        run_dir = Path(run_dir)
        tcfg = load_train_config(run_dir / "train.yaml")

        if tcfg.run.algo != "maskable_ppo":
            raise ValueError(
                f"{run_dir.name} was trained with algo={tcfg.run.algo!r}. Deployment passes an "
                f"action mask to `predict`, which only `maskable_ppo` accepts, and the mask is "
                f"not optional -- an uncharged super must be masked, not merely unchosen."
            )
        if tcfg.normalize.obs:
            raise ValueError(
                f"{run_dir.name} was trained with normalize.obs=true, so its observation "
                f"statistics live in best_vecnormalize.pkl and the checkpoint is meaningless "
                f"without them. The deployed path has no VecNormalize, so loading this run would "
                f"feed raw values to a network expecting whitened ones -- silently, with no "
                f"error. Retrain with normalize.obs=false or add the statistics to this path "
                f"deliberately."
            )

        cfg = load_config(_resolve(tcfg.run.env_config), overrides=tcfg.run.env_overrides or None)
        spec = obs_select.load_agent_spec(_resolve(tcfg.run.agent_obs), cfg)
        model = MaskablePPO.load(run_dir / checkpoint, device=device)

        check_spaces(model, spec, cfg, label=f"{run_dir.name}/{checkpoint}",
                     spec_path=tcfg.run.agent_obs, cfg_path=tcfg.run.env_config)
        return cls(model, spec, cfg, deterministic=deterministic)

    def make_assembler(self, *, map_frame: MapFrame | None = None) -> ObservationAssembler:
        """The assembler for THIS checkpoint, sharing its spec and config objects.

        Not a convenience: it is the only construction path the loop should use, because it makes
        an observation built to the wrong spec structurally impossible rather than merely checked.
        """
        return ObservationAssembler(self.spec, self.cfg, map_frame=map_frame)

    # -- the decision ---------------------------------------------------------

    def act(self, obs: dict, attack_legal) -> Decision:
        """One decision. `obs` is `ObservationAssembler.assemble`'s output; `attack_legal` is
        `ShadowHero.attack_mask()`.

        The mask is `[move (n_move_bins + 1), attack (3)]` flattened, which is what
        `wrappers/sb3_vecenv.py:action_masks` hands `MaskablePPO` during training. The move half
        is all-True because `core/hero.action_mask` builds it that way and nothing has ever
        narrowed it; the attack half comes from the shadow, which owns the hero's timers and ammo
        and already reproduces `action_mask`'s formula exactly. **Reimplementing that formula here
        would be a second copy of a rule that is only correct in one place.**
        """
        legal = tuple(bool(v) for v in attack_legal)
        if len(legal) != 3:
            raise ValueError(f"attack_legal must be (no_fire, attack, super), got {attack_legal!r}")
        if not legal[0]:
            # `no-fire` is unconditionally legal in `hero.action_mask` (`no_fire_ok = ones_like`).
            # An all-False row makes MaskablePPO's categorical distribution degenerate, which
            # surfaces as NaN logits rather than as an error, so refuse here instead.
            raise ValueError("attack_legal[0] (no-fire) must always be True; an all-illegal "
                             "dimension gives the policy a degenerate distribution, not an error")
        self._mask[0, self._n_move:] = legal

        for name, value in obs.items():
            slot = self._batched.get(name)
            if slot is None or slot.shape[1:] != value.shape:
                slot = np.empty((1, *value.shape), dtype=value.dtype)
                self._batched[name] = slot
            slot[0] = value

        action, _ = self.model.predict(self._batched, deterministic=self.deterministic,
                                       action_masks=self._mask)
        move_bin, attack = (int(v) for v in np.asarray(action).reshape(-1)[:2])
        return Decision(move_bin=move_bin, attack=attack, legal=legal)
