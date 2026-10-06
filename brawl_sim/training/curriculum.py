"""CurriculumManager: weighted bot-difficulty sampling, applied per env on every reset.

Every reset, each bot KIND in each env independently draws a difficulty tier from the current
stage's weighted mixture, and that tier's multipliers scale the kind's `SimParams` columns. Which
kinds spawn is untouched (`core/spawn.sample_enemy_kinds`, weighted by
`entities.enemy_type_weights`); the curriculum changes how well the drawn bots play.

It plugs in as `BrawlVecEnv(params_hook=...)`, which `core/spawn.reset_envs` calls right after
`resample_params` and before anything reads `params` back: `max_hp` is derived from `base_hp`
there, so an `hp` multiplier applied later would spawn entities whose HP disagrees with their
stats.

A hook rather than the {low, high} range syntax because a range is one continuous uniform and a
curriculum needs a weighted discrete mixture over named tiers. The two compose:
`configs/randomization.yaml` jitters the base value, then the drawn tier scales it.

Sync-free and fully batched like the sim code it runs inside: one `torch.rand` + `searchsorted`
for the whole batch, `torch.where` over all N rows. The only Python loop is over the
compile-time list of target fields, which CONVENTIONS.md permits.
"""
import torch

from ..constants import N_KINDS
from .config import CurriculumConfig, DifficultyTier

# Bot kinds occupy columns 1..N_KINDS-1 of every per-kind (N, K) SimParams tensor; column 0 is
# the hero and is never touched by the curriculum.
_N_BOT_KINDS = N_KINDS - 1

# (tier multiplier field, SimParams attribute(s) it scales). `aim_noise` drives both the angular
# noise of LEAD-aimed kinds and the landing-point noise of LOB kinds (artillery, Spike), so a tier
# stays one number per concept.
_FLOAT_TARGETS = (
    ("aim_noise", ("aim_noise_std_rad", "aim_noise_tiles")),
    ("reaction_delay", ("reaction_delay",)),
    ("move_speed", ("move_speed",)),
    ("hp", ("base_hp",)),
    ("damage", ("base_damage",)),
    # Unclamped: bots/personality.py clamps each read. A 0 MULTIPLIER cannot happen (DifficultyTier
    # refuses it: core/stats.aggression_of reads 0 as the neutral 1.0). A 0 BASE (brawlers.yaml
    # omits the key) stays 0 and reads as that neutral 1.0, i.e. "unset" -- pinned by
    # tests/test_training.py::test_manager_clamps_hero_focus_to_one.
    ("aggression", ("aggression",)),
)

# (tier multiplier field, SimParams attribute) for per-kind values that are FRACTIONS in [0, 1]:
# the product is clamped back into that range after scaling, because past 1.0 each of them stops
# meaning "more of the same" and becomes a different behaviour.
#   lead_target_fraction > 1  over-leads: aims PAST where the target will be.
#   hero_focus > 1            `perception.select_target` scales the hero's distance by
#                             (1 - focus), so it goes NEGATIVE and argmin picks the hero from any
#                             range, ahead of a bot standing on top of the observer. 1.0 already
#                             means "the hero whenever visible"; there is nothing above it.
_UNIT_TARGETS = (
    ("lead_target", "lead_target_fraction"),
    ("hero_focus", "hero_focus"),
)


class TierApplier:
    """Turns a per-(env, bot kind) tier index into `SimParams` multipliers.

    Shared by `CurriculumManager` (samples the index from the stage's mixture) and `FixedTierHook`
    (pins one named tier, for evaluation), so an `eval/win_rate_hard` number describes the same
    bots the curriculum's `hard` tier trains against.
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

    def apply_tiers(self, params, reset_mask: torch.Tensor, tier_idx: torch.Tensor) -> None:
        """MUTATES (rebinds, matching `config.resample_params`' own convention): every per-kind
        SimParams tensor any tier multiplier targets, for masked rows only.

        `tier_idx` is `(N, _N_BOT_KINDS)` int64 -- one tier per (env, bot kind).
        """
        n_envs = reset_mask.shape[0]
        device = reset_mask.device
        ones = self._ones_column(n_envs, device)  # (N,1), the hero's always-1.0 multiplier
        # Every write is gated on the bot columns, so column 0 (the hero) is never written for ANY
        # field. The 1.0 multiplier in `ones` alone is not enough: the clamped fields below would
        # move the hero's unused 0 decision_period to `clamp(round(0 * 1.0), min=1)`.
        write = reset_mask.unsqueeze(1) & self._bot_columns(device)   # (N,K)

        for tier_field, attrs in _FLOAT_TARGETS:
            mult = self._multiplier(tier_field, tier_idx, ones)
            for attr in attrs:
                current = getattr(params, attr)
                setattr(params, attr, torch.where(write, current * mult, current))

        # Fractions clamp to [0, 1] (elite's 1.5 x the sniper's 0.9 lead would over-lead); see
        # _UNIT_TARGETS for what "past 1.0" would mean for each.
        for tier_field, attr in _UNIT_TARGETS:
            mult = self._multiplier(tier_field, tier_idx, ones)
            current = getattr(params, attr)
            setattr(params, attr, torch.where(
                write, torch.clamp(current * mult, min=0.0, max=1.0), current
            ))

        # decision_period is in TICKS (int64): round after scaling and floor at 1, because
        # bots/policy.py uses it as a modulo divisor.
        mult = self._multiplier("decision_period", tier_idx, ones)
        period = params.decision_period
        scaled = torch.round(period.to(torch.float32) * mult).clamp(min=1.0).to(torch.int64)
        params.decision_period = torch.where(write, scaled, period)

    def _multiplier(self, tier_field: str, tier_idx: torch.Tensor, ones: torch.Tensor) -> torch.Tensor:
        """(N, K) multiplier column-aligned with a per-kind SimParams tensor: 1.0 for the hero
        in column 0, then each bot kind's drawn tier's value in columns 1..K-1."""
        bots = self._tables[tier_field][tier_idx]  # (T,)[(N, K-1)] -> (N, K-1)
        return torch.cat([ones, bots], dim=1)

    def _bot_columns(self, device) -> torch.Tensor:
        """(1, K) bool: False for the hero's column 0, True for every bot kind column."""
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

    `assignment[i]` is the tier index env `i` is pinned to, so one batched env can score every
    tier in one rollout (`training/evaluation.py`: envs `[0:k)` get the first tier, `[k:2k)` the
    second, ...). It draws no randomness, which keeps an eval number comparable across
    checkpoints.
    """

    def __init__(self, tiers: dict, device, assignment: torch.Tensor) -> None:
        super().__init__(tiers, device)
        if assignment.ndim != 1:
            raise ValueError(f"assignment must be 1-D (N,), got shape {tuple(assignment.shape)}")
        self.assignment = assignment.to(device=self.device, dtype=torch.int64)
        # (N, _N_BOT_KINDS): the same tier for every bot kind in a given env.
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
        # Inverse-CDF sampling, as core/spawn.sample_enemy_kinds does: one `searchsorted` for the
        # whole batch, no multinomial, no sync.
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
        )  # (N, _N_BOT_KINDS), one tier per (env, bot kind)

        self.apply_tiers(params, reset_mask, tier_idx)

        # Histogram only the rows that actually reset; unmasked rows contribute weight 0.
        weights = reset_mask.to(torch.float32).unsqueeze(1).expand(n_envs, _N_BOT_KINDS)
        self._tier_spawns.scatter_add_(0, tier_idx.reshape(-1), weights.reshape(-1))

    # ---- checkpointing ---------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"stage_index": self.stage_index, "stage_name": self.stage.name}

    def load_state_dict(self, state: dict) -> None:
        """Restores the stage pointer. Matches by NAME (by index only when no name was recorded),
        so reordered stages still resume the right one and a renamed stage fails loudly rather
        than resuming at the wrong difficulty."""
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
    fields = [f for f, _ in _FLOAT_TARGETS] + [f for f, _ in _UNIT_TARGETS] + ["decision_period"]
    return {
        name: torch.tensor([getattr(t, name) for t in tiers], dtype=torch.float32, device=device)
        for name in fields
    }
