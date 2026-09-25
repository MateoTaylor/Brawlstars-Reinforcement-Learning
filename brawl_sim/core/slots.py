"""Tracker-style enemy slots for the hero's observation (OBS_PARITY_TASKS.md C8).

Live, `brawl_deployment/perception/tracker.EntityTracker` hands each enemy a SLOT: a track is
pending on its first sighting, promoted to the lowest free slot on its `promote_hits`-th
consecutive sighting (and shown from that decision, since `assemble` reads the slots after the
update), holds the slot while it coasts through up to `max_misses` unseen decisions, is retired
(slot freed) on the next one, and comes back later as a new track in whichever slot is lowest and
free then. The sim's enemy group used to be "slot k is entity k forever"; this module keeps the
four `SimState` slot fields (`state.py` `_SLOT_FIELDS`) on the tracker's rule, so a per-entity
group declared `slots: tracked` (obs_select, C9) reads the same permutation live and in training.

Time base: the tracker updates once per 4 Hz DECISION, so `cfg.slots_promote_hits` and
`cfg.slots_max_misses` are decision counts and `update` runs once per decision, from
`env._build_observation` after `hero_view` and before `build_obs`. Not from `step`: that would
lag the live promotion by a decision.

Two things are deliberately not modelled. The assembler's HP-commit rule (a slot is written only
once its HP numeral has been read, `assemble._put_entities`), which delays a live row by however
long the read takes and has no sim counterpart. And the order among several entities qualifying
for fewer free slots in the same decision: the sim promotes by entity index, the tracker by the
order its pending tracks were created, which is detection order -- the same whenever detections
come in entity order, and a tie no live match can tell apart. With E - 1 slots for E - 1 enemies
that case cannot even arise in the sim; the algorithm handles it for the sake of any smaller slot
count.
"""
import torch


def update(state, hero_view: torch.Tensor, cfg) -> None:
    """MUTATES slot_ent, ent_slot, ent_hits, ent_misses, one decision on. `hero_view`: (N,E)
    bool, the hero's reveal this decision (`core/camera.hero_view`); a dead entity is simply
    unseen and coasts out, as a brawler that stops appearing does live. Fully vectorised: no
    `.item()`, no loop over envs or entities, so no host sync on CUDA."""
    seen = hero_view & state.ent_alive
    seen[:, 0] = False                                    # the hero holds no slot
    N, E = seen.shape
    K = state.slot_ent.shape[1]
    device = seen.device
    zero_i64 = torch.zeros((), dtype=torch.int64, device=device)
    zero_i32 = torch.zeros((), dtype=torch.int32, device=device)

    # 1. Slotted rows coast. `misses` counts consecutive unseen decisions; past max_misses the
    #    slot is freed. scatter_add rather than scatter, because every unslotted row maps to
    #    index 0 and a plain scatter of zeros there would be a write too.
    slotted = state.ent_slot > 0
    misses = torch.where(seen, zero_i32, state.ent_misses + 1)
    misses = torch.where(slotted, misses, zero_i32)
    retire = slotted & (misses > cfg.slots_max_misses)
    slot_idx = (state.ent_slot - 1).clamp(min=0)
    retire_slot = torch.zeros((N, K), dtype=torch.int32, device=device)
    retire_slot.scatter_add_(1, slot_idx, retire.to(torch.int32))
    slot_ent = torch.where(retire_slot > 0, zero_i64, state.slot_ent)
    ent_slot = torch.where(retire, zero_i64, state.ent_slot)
    misses = torch.where(retire, zero_i32, misses)

    # 2. Unslotted rows count consecutive sightings; one miss and the count restarts, which is
    #    what makes "consecutive" mean consecutive (a pending track has no coast at all).
    unslotted = ent_slot == 0
    hits = torch.where(unslotted & seen, state.ent_hits + 1, zero_i32)

    # 3. Promotion: the needers, in entity order, take the free slots, in slot order, as far as
    #    they go. A slot freed in step 1 is free now, as the tracker retires before it promotes.
    need = unslotted & (hits >= cfg.slots_promote_hits)
    free = slot_ent == 0
    need_rank = torch.cumsum(need.to(torch.int64), 1) - 1            # (N,E)
    free_rank = torch.cumsum(free.to(torch.int64), 1) - 1            # (N,K)
    take = need & (need_rank < free.sum(1, keepdim=True))
    match = (take.unsqueeze(2) & free.unsqueeze(1)
             & (need_rank.unsqueeze(2) == free_rank.unsqueeze(1)))   # (N,E,K), one True per taker
    new_slot = match.to(torch.int64).argmax(2)                       # (N,E); 0 where nothing matched
    ent_slot = torch.where(take, new_slot + 1, ent_slot)
    e_index = torch.arange(E, device=device).view(1, E).expand(N, E)
    added = torch.zeros((N, K), dtype=torch.int64, device=device)
    added.scatter_add_(1, new_slot, torch.where(take, e_index + 1, zero_i64))
    slot_ent = slot_ent + added                                      # taken slots held 0
    hits = torch.where(take, zero_i32, hits)

    state.slot_ent.copy_(slot_ent)
    state.ent_slot.copy_(ent_slot)
    state.ent_hits.copy_(hits)
    state.ent_misses.copy_(misses)
