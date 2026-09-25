"""Observation history rings (SIM_OVERHAUL_PLAN.md Phase H, Step H1).

The policy has no recurrent memory, and that is fine -- what it gets instead is the last
`cfg.history_frames` DECISIONS of a few things, kept by the sim in per-env ring buffers
(core/state `hist_*`, newest at slot 0) and read back by the observation (Step H2: the hero's own
hp/ammo/action trail plus three local grid channels of where enemies were seen).

`push` is the only writer besides `state.zero_`, which is what a reset does to the rings and what
"no history yet" means: `hist_valid` all False. It is called from `env.step` BEFORE the world
advances, with the action the policy just chose and the visibility `_build_observation` computed
for the observation that action answered. Slot 0 therefore describes exactly the state the policy
looked at, paired with what it did about it -- which is the pairing a frame stack is for.

Per decision, not per sim tick: it runs outside `_run_tick`, once per `step()`, whatever
`action_repeat` is. Everything below is slice copies and boolean ops on the device; no host reads.
"""
import torch

RING_FIELDS = ("hist_valid", "hist_action", "hist_hp", "hist_ammo", "hist_pos",
               "hist_enemy_pos", "hist_enemy_seen")


def push(state, action: torch.Tensor, hero_view: torch.Tensor) -> None:
    """MUTATES every `hist_*` ring. Shifts slots 0..K-2 into 1..K-1 (K-1 non-overlapping slice
    copies per ring, oldest slot first, so nothing is read after it has been overwritten) and
    writes slot 0 from the PRE-step state:

        valid       True
        action      the policy's (N,2) action for this decision
        hp / ammo   the hero's, absolute (the observation normalizes)
        pos         the hero's position
        enemy_pos   EVERY entity's position (slot 0 of the E axis is the hero itself)
        enemy_seen  alive & hero_view, with the hero's own column forced False

    `hero_view` is `core/camera.hero_view`'s (N,E) for the observation this action answers:
    the hero's reveal, concealment AND the camera window, so a sighting the screen never showed
    is never remembered either (OBS_PARITY_TASKS.md C3).
    """
    depth = state.hist_valid.shape[1]
    for name in RING_FIELDS:
        ring = getattr(state, name)
        for slot in range(depth - 1, 0, -1):
            ring[:, slot].copy_(ring[:, slot - 1])

    state.hist_valid[:, 0].fill_(True)
    state.hist_action[:, 0].copy_(action)
    state.hist_hp[:, 0].copy_(state.ent_hp[:, 0])
    state.hist_ammo[:, 0].copy_(state.ent_ammo[:, 0])
    state.hist_pos[:, 0].copy_(state.ent_pos[:, 0])
    state.hist_enemy_pos[:, 0].copy_(state.ent_pos)
    seen = state.ent_alive & hero_view
    seen[:, 0] = False
    state.hist_enemy_seen[:, 0].copy_(seen)
