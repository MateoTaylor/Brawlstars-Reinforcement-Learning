import torch

from brawl_sim.config import load_config
from brawl_sim.env import BrawlVecEnv
from scripts import smoke_test


def test_the_smoke_battery_draws_the_whole_attack_column_so_it_throws_supers_and_gadgets():
    """SIM_OVERHAUL Step I4.1 (G3's review): `_random_action` drew the attack column from {0, 1},
    so the all-presets battery never threw a super or a gadget, and `check_invariants` never ran
    with either in flight. The column is four-valued since G3 (0 nothing, 1 attack, 2 super,
    3 gadget) and the move column is 16 bins plus idle. Literals, not `cfg.action_nvec`, so a
    narrowing in the script or in the config fails here. 400 draws per column: missing one of
    four values is a 1e-50 event, and the seed is fixed anyway."""
    cfg = load_config(smoke_test.DEFAULT_CONFIG_PATH)
    env = BrawlVecEnv(cfg, n_envs=8, device="cpu", seed=0, verbose=False)
    torch.manual_seed(0)
    draws = torch.cat([smoke_test._random_action(env) for _ in range(50)])
    assert draws.shape == (400, 2)
    assert set(draws[:, 0].tolist()) == set(range(17))
    assert set(draws[:, 1].tolist()) == {0, 1, 2, 3}
