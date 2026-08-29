"""CurriculumManager: weighted bot-difficulty sampling, applied per env on every reset.

**What it does.** Every reset, each bot ARCHETYPE in each env independently draws a difficulty
tier from the current stage's weighted mixture, and that tier's multipliers are applied to that
archetype's `SimParams` columns. Archetype *selection* is untouched -- bots stay uniformly
randomly chosen by `core/spawn.sample_enemy_kinds` throughout the whole curriculum; what changes
is how well the bots you draw actually play.

**Where it plugs in.** As `BrawlVecEnv(params_hook=...)`, which `core/spawn.reset_envs` calls
immediately after `resample_params` and before anything reads `params` back -- see that
function's docstring for why that exact position is the only correct one (`max_hp` is derived
from `base_hp` right there, so a `hp` multiplier applied any later would leave entities spawning
with HP that disagrees with their own stats).

**Why a hook and not the {low, high} range syntax.** `resample_params` already resamples every
randomized field per env on every reset (N08) -- but a range is a single *continuous uniform*,
and a curriculum needs a *weighted discrete mixture* over named tiers ("70% easy, 25% medium,
5% hard"), which no `{low, high}` pair can express. The two compose cleanly and are independent:
`configs/randomization.yaml` jitters the base value, this then scales it by the drawn tier.

**Discipline.** Sync-free and fully batched, exactly like the simulator code it runs inside: one
`torch.rand` + `searchsorted` for the whole batch, `torch.where` over all N rows, no
`.item()`/`.cpu()`/boolean indexing/Python loop over envs. The only Python-level loop is over
the 4 archetypes' worth of *fields* (a compile-time constant list), which CONVENTIONS.md
explicitly permits.
"""
import torch

from ..constants import N_KINDS
from .config import CurriculumConfig, DifficultyTier

# Bot kinds occupy columns 1..N_KINDS-1 of every per-kind (N, K) SimParams tensor; column 0 is
# the hero and is never touched by the curriculum.
_N_BOT_KINDS = N_KINDS - 1

# (tier multiplier field, SimParams attribute(s) it scales). One tier knob may drive more than
# one field: `aim_noise` moves both the angular noise the sniper/melee/rifle archetypes use and
# the positional noise the artillery archetype uses, so a tier stays one number per concept
# rather than one number per archetype's implementation detail.
_FLOAT_TARGETS = (
    ("aim_noise", ("aim_noise_std_rad", "aim_noise_tiles")),
    ("reaction_delay", ("reaction_delay",)),
    ("move_speed", ("move_speed",)),
    ("hp", ("base_hp",)),
    ("damage", ("base_damage",)),
)


class TierApplier:
    """Turns a per-(env, archetype) tier index into `SimParams` multipliers.

    Split out from `CurriculumManager` so evaluation can reuse it: `FixedTierHook` (below) pins
    every env to one named tier, `CurriculumManager` samples the index from the current stage's
    mixture, and both must apply a tier the SAME way or an `eval/win_rate_hard` number would not
    describe the same bots the curriculum's `hard` stage trains against -- which is the entire
    point of having a stationary yardstick.
    """

    def __init__(self, tiers: dict, device) -> None:
        if not tiers:
            raise ValueError("TierApplier needs at least one tier")
        self.device = torch.device(device)
        self.tier_names = tuple(tiers)
        self._tiers = tuple(tiers[name] for name in self.tier_names)
        self._tables = _build_tables(self._tiers, self.device)
        self._ones_col: torch.Tensor | None = None
        self._bot_cols: torch.Tensor | None = None

    def tier_index(self, name: str) -> int:
        if name not in self.tier_names:
            raise ValueError(f"unknown tier {name!r}; defined tiers are {list(self.tier_names)}")
        return self.tier_names.index(name)

    def apply_tiers(self, params, reset_mask: torch.Tensor, tier_idx: torch.Tensor) -> None:
        """MUTATES (rebinds, matching `config.resample_params`' own convention): every per-kind
        SimParams tensor any tier multiplier targets, for masked rows only.

        `tier_idx` is `(N, _N_BOT_KINDS)` int64 -- one tier per (env, archetype).
        """
        n_envs = reset_mask.shape[0]
        device = reset_mask.device
        ones = self._ones_column(n_envs, device)  # (N,1), the hero's always-1.0 multiplier
        # Every write is gated on `mask_k & bot_cols`, so column 0 (the hero) is provably never
        # written -- for ANY field. The 1.0 multiplier in `ones` alone is NOT sufficient: the two
        # fields below additionally clamp, and `clamp(round(0 * 1.0), min=1)` would have quietly
        # moved the hero's own (unused, 0-valued) decision_period to 1. Caught by
        # tests/test_training.py::test_manager_never_touches_the_hero_column; the invariant is
        # cheap to hold exactly, so hold it exactly rather than argue the leak is harmless.
        write = reset_mask.unsqueeze(1) & self._bot_columns(device)   # (N,K)

        for tier_field, attrs in _FLOAT_TARGETS:
            mult = self._multiplier(tier_field, tier_idx, ones)
            for attr in attrs:
                current = getattr(params, attr)
                setattr(params, attr, torch.where(write, current * mult, current))

        # lead_target_fraction is a FRACTION: scaling it freely could push a tier past 1.0, which
        # would mean over-leading the target (aiming past where it will be) rather than "leading
        # perfectly", so it clamps rather than saturating into a different behavior.
        mult = self._multiplier("lead_target", tier_idx, ones)
        lead = params.lead_target_fraction
        params.lead_target_fraction = torch.where(
            write, torch.clamp(lead * mult, min=0.0, max=1.0), lead
        )

        # decision_period is in TICKS (int64) -- round after scaling, and floor at 1: a period of
        # 0 would mean "re-decide zero times per tick", which the archetype policies read as a
        # modulo divisor and would divide by zero on.
        mult = self._multiplier("decision_period", tier_idx, ones)
        period = params.decision_period
        scaled = torch.round(period.to(torch.float32) * mult).clamp(min=1.0).to(torch.int64)
        params.decision_period = torch.where(write, scaled, period)

    def _multiplier(self, tier_field: str, tier_idx: torch.Tensor, ones: torch.Tensor) -> torch.Tensor:
        """(N, K) multiplier column-aligned with a per-kind SimParams tensor: 1.0 for the hero
        in column 0, then each archetype's own drawn tier's value in columns 1..K-1."""
        bots = self._tables[tier_field][tier_idx]  # (T,)[(N,4)] -> (N,4)
        return torch.cat([ones, bots], dim=1)

    def _bot_columns(self, device) -> torch.Tensor:
        """(1, K) bool: False for the hero's column 0, True for every bot archetype column."""
        if self._bot_cols is None or self._bot_cols.device != device:
            cols = torch.ones((1, N_KINDS), dtype=torch.bool, device=device)
            cols[0, 0] = False
            self._bot_cols = cols
        return self._bot_cols

    def _ones_column(self, n_envs: int, device) -> torch.Tensor:
        if self._ones_col is None or self._ones_col.shape[0] != n_envs or self._ones_col.device != device:
            self._ones_col = torch.ones((n_envs, 1), dtype=torch.float32, device=device)
        return self._ones_col


class FixedTierHook(TierApplier):
    """A `params_hook` that pins every env to ONE named tier, for stationary evaluation.

    Optionally takes a per-env assignment so a single batched env can evaluate several tiers at
    once: `assignment[i]` is the tier index env `i` is pinned to. `training/evaluation.py` uses
    that to score every tier in one rollout -- envs `[0:k)` are easy, `[k:2k)` medium, and so on
    -- rather than building and stepping one env per tier.

    Unlike `CurriculumManager` this draws no randomness at all, which is what makes an eval
    number comparable across checkpoints: same seed, same maps, same spawns, same bot quality.
    """

    def __init__(self, tiers: dict, device, assignment: torch.Tensor) -> None:
        super().__init__(tiers, device)
        if assignment.ndim != 1:
            raise ValueError(f"assignment must be 1-D (N,), got shape {tuple(assignment.shape)}")
        self.assignment = assignment.to(device=self.device, dtype=torch.int64)
        # (N, 4): the same tier for all four archetypes in a given env.
        self._tier_idx = self.assignment.unsqueeze(1).expand(-1, _N_BOT_KINDS).contiguous()

    @classmethod
    def uniform(cls, tiers: dict, device, n_envs: int, tier_name: str) -> "FixedTierHook":
        index = tuple(tiers).index(tier_name)
        assignment = torch.full((n_envs,), index, dtype=torch.int64, device=torch.device(device))
        return cls(tiers, device, assignment)

    def __call__(self, params, reset_mask: torch.Tensor) -> None:
        self.apply_tiers(params, reset_mask, self._tier_idx.to(reset_mask.device))


class CurriculumManager(TierApplier):
    """Owns the stage pointer and applies the current stage's tier mixture. Advancement itself
    lives in `training/callbacks.py` (it needs episode outcomes, which only exist host-side);
    this class just does what it's told via `set_stage`/`advance`/`demote`."""

    def __init__(self, cfg: CurriculumConfig, device, gen: torch.Generator, stage_index: int = 0) -> None:
        if not cfg.stages:
            raise ValueError("CurriculumManager needs at least one stage")
        super().__init__(cfg.tiers, device)
        self.cfg = cfg
        self.gen = gen
        # Spawn histogram over tiers, accumulated on-device across resets and read (with one
        # sync) once per rollout by the callback -- the ground truth for "what mix is the agent
        # ACTUALLY facing", as opposed to what the configured weights claim.
        self._tier_spawns = torch.zeros(len(self._tiers), dtype=torch.float32, device=self.device)
        self.stage_index = -1
        self.set_stage(stage_index)

    # ---- stage control ------------------------------------------------------------------

    @property
    def stage(self):
        return self.cfg.stages[self.stage_index]

    @property
    def n_stages(self) -> int:
        return len(self.cfg.stages)

    @property
    def is_final_stage(self) -> bool:
        return self.stage_index >= self.n_stages - 1

    def set_stage(self, index: int) -> None:
        index = max(0, min(int(index), self.n_stages - 1))
        if index == self.stage_index:
            return
        self.stage_index = index
        weights = [float(self.stage.tier_weights.get(name, 0.0)) for name in self.tier_names]
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"stage {self.stage.name!r} has zero total weight over known tiers")
        probs = torch.tensor([w / total for w in weights], dtype=torch.float32, device=self.device)
        # Inverse-CDF sampling, the same shape core/spawn.sample_enemy_kinds already uses for
        # archetype selection -- one `searchsorted` for the whole batch, no multinomial, no sync.
        self._cdf = torch.cumsum(probs, dim=0)
        self._cdf[-1] = 1.0  # guard the last bucket against float32 cumsum landing at 0.9999994

    def advance(self) -> bool:
        """Moves up one stage. Returns False (and does nothing) at the final stage."""
        if self.is_final_stage:
            return False
        self.set_stage(self.stage_index + 1)
        return True

    def demote(self) -> bool:
        """Moves back one stage. Returns False (and does nothing) at stage 0."""
        if self.stage_index <= 0:
            return False
        self.set_stage(self.stage_index - 1)
        return True

    def tier_weights(self) -> dict:
        """Normalized {tier_name: probability} for the current stage -- what's configured."""
        return {name: float(p) for name, p in zip(self.tier_names, self._probs_list())}

    def tier_spawn_fractions(self, reset: bool = True) -> dict:
        """{tier_name: fraction} of bots actually spawned since the last call -- what's real.
        **One host sync**; call it once per rollout, never inside the step loop. Returns `{}`
        before any reset has happened."""
        counts = self._tier_spawns.tolist()   # the one sync
        total = sum(counts)
        if total == 0.0:
            return {}
        if reset:
            self._tier_spawns.zero_()
        return {name: c / total for name, c in zip(self.tier_names, counts)}

    # ---- the params_hook ------------------------------------------------------------------

    def __call__(self, params, reset_mask: torch.Tensor) -> None:
        """MUTATES (rebinds, matching `config.resample_params`' own convention): every per-kind
        SimParams tensor any tier multiplier targets, for masked rows only. Signature is
        `core/spawn.reset_envs`' `params_hook` contract."""
        n_envs = reset_mask.shape[0]
        device = reset_mask.device

        u = torch.rand((n_envs, _N_BOT_KINDS), generator=self.gen, device=device)
        tier_idx = torch.clamp(
            torch.searchsorted(self._cdf, u.contiguous(), right=True), max=len(self._tiers) - 1
        )  # (N, 4), one tier per (env, archetype)

        self.apply_tiers(params, reset_mask, tier_idx)

        # Histogram only the rows that actually reset; unmasked rows contribute weight 0.
        weights = reset_mask.to(torch.float32).unsqueeze(1).expand(n_envs, _N_BOT_KINDS)
        self._tier_spawns.scatter_add_(0, tier_idx.reshape(-1), weights.reshape(-1))

    # ---- checkpointing ---------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"stage_index": self.stage_index, "stage_name": self.stage.name}

    def load_state_dict(self, state: dict) -> None:
        """Restores the stage pointer. Matches by NAME first and falls back to the index, so
        reordering or renaming stages in configs/train.yaml between runs fails loudly here
        rather than silently resuming at the wrong difficulty."""
        name = state.get("stage_name")
        index = int(state.get("stage_index", 0))
        if name is not None:
            names = [s.name for s in self.cfg.stages]
            if name in names:
                self.set_stage(names.index(name))
                return
            raise ValueError(
                f"checkpoint was at curriculum stage {name!r}, which no longer exists in this "
                f"config (stages are {names}). Rename it back, or pass --curriculum-stage to "
                "pick one explicitly."
            )
        self.set_stage(index)

    # ---- internals ---------------------------------------------------------------------------

    def _probs_list(self):
        weights = [float(self.stage.tier_weights.get(name, 0.0)) for name in self.tier_names]
        total = sum(weights)
        return [w / total for w in weights]


def _build_tables(tiers: tuple[DifficultyTier, ...], device) -> dict:
    """{tier field: (T,) float32 tensor}, one row per tier, in `tier_names` order."""
    fields = [f for f, _ in _FLOAT_TARGETS] + ["lead_target", "decision_period"]
    return {
        name: torch.tensor([getattr(t, name) for t in tiers], dtype=torch.float32, device=device)
        for name in fields
    }
