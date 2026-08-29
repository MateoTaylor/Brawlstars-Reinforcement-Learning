"""Shared scaffolding for the bot tests (archetype combat, personality movement, dispatcher).

Not a conftest.py: these are plain helper FUNCTIONS that tests import by name, not pytest
fixtures, and `from bot_fixtures import ...` is unambiguous where a magically-injected fixture
would not be. tests/test_sniper.py etc. each used to carry their own private copy of `_FakeBank`/
`_grid`/`_cfg_and_params`/`_fresh_state`; Step 41 needed all of them plus a `Targeting` builder in
a sixth file, which is where five copies stopped being tolerable.
"""
import yaml
import torch

from brawl_sim.bots import perception, policy
from brawl_sim.config import build_params, load_config
from brawl_sim.constants import (
    TILE_BLOCKS_PROJ,
    TILE_BLOCKS_UNIT,
    TILE_IS_BUSH,
    Kind,
    Person,
    Tile,
)
from brawl_sim.core import stats
from brawl_sim.core.state import allocate
from brawl_sim.maps.loader import bush_waypoints

CONFIGS_DEFAULT = "configs/default.yaml"


class FakeBank:
    """The MapBank fields the bot layer actually touches, for a single hand-built map.

    Bush waypoints are derived from `tiles` with the REAL maps/loader.bush_waypoints rather than
    hand-listed, so a test map's waypoints are subsampled exactly the way a shipped map's are. A
    test that needs specific waypoints overrides `bush_wp`/`n_bush_wp` afterward."""

    def __init__(self, tiles, cell_tiles=10):
        self.blocks_unit = TILE_BLOCKS_UNIT[tiles].unsqueeze(0)
        self.blocks_proj = TILE_BLOCKS_PROJ[tiles].unsqueeze(0)
        self.is_bush = TILE_IS_BUSH[tiles].unsqueeze(0)
        pts = bush_waypoints(tiles.numpy(), cell_tiles)
        self.bush_wp = torch.as_tensor(pts, dtype=torch.float32).reshape(1, -1, 2)
        self.n_bush_wp = torch.tensor([len(pts)], dtype=torch.int64)


def grid(h, w, fill=Tile.FLOOR):
    t = torch.full((h, w), int(fill), dtype=torch.int64)
    t[0, :] = Tile.WALL
    t[-1, :] = Tile.WALL
    t[:, 0] = Tile.WALL
    t[:, -1] = Tile.WALL
    return t


def cfg_and_params(n_envs=1, n_enemies=1, map_h=20, map_w=20, overrides=None):
    merged = {
        "world": {"map_h": map_h, "map_w": map_w},
        "entities": {"n_enemies": n_enemies},
        "zone": {"enabled": False},
    }
    for section, values in (overrides or {}).items():
        merged[section] = {**merged.get(section, {}), **values}
    cfg = load_config(CONFIGS_DEFAULT, overrides=merged)
    spec = {
        **yaml.safe_load(open(CONFIGS_DEFAULT).read()),
        **yaml.safe_load(open("configs/brawlers.yaml").read()),
    }
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    params = build_params(cfg, n_envs=n_envs, device="cpu", gen=gen, spec=spec)
    return cfg, params, gen


def fresh_state(cfg, params, n_envs=1, enemy_kind=Kind.BOT_SNIPER, person=Person.RUSH):
    """A live state with slot 0 = hero and every other slot a bot of `enemy_kind`/`person`.

    `person` defaults to RUSH because RUSH is the personality with no preconditions -- it needs
    no bush, no memory and no zone -- so an archetype test that only cares about fire/aim gets
    predictable movement it can ignore. Tests that care set state.ent_person themselves.
    """
    state = allocate(cfg, n_envs=n_envs, device="cpu", verbose=False)
    state.ent_alive.fill_(True)
    state.ent_kind[:, 0] = int(Kind.HERO_MORTIS)
    state.ent_person.fill_(int(person))
    for e in range(1, cfg.n_entities):
        state.ent_kind[:, e] = int(enemy_kind)
    state.map_id.fill_(0)
    state.ent_target.fill_(-1)
    state.ent_hunt_t.fill_(cfg.bots_hunt_timeout_seconds)
    max_hp = stats.effective_max_hp(state.ent_kind, state.ent_cubes, params)
    state.ent_hp.copy_(max_hp)
    state.ent_max_hp.copy_(max_hp)
    max_ammo = stats.gather_kind(params.max_ammo, state.ent_kind)
    state.ent_ammo.copy_(max_ammo)
    return state


def build_targeting(state, bank, params, cfg):
    """visibility -> select_target -> target_los -> Targeting, i.e. exactly the preamble
    bots/policy.all_bot_intents runs before it calls any archetype's `combat()`. Returns
    (vis, tgt) since several tests assert on `vis` too.

    Mirrors production deliberately, including the (N,E) `target_los` rather than the (N,E,E)
    `raw_los` this used to build (Step A2) -- an archetype test that fed `targeting` a
    differently-shaped LOS than the real dispatcher does would be testing a code path that no
    longer exists."""
    vis = perception.bot_visibility(state, perception.visibility(state, bank, params, cfg), cfg)
    perception.select_target(state, vis, cfg)
    los = perception.target_los(state, bank, cfg)
    return vis, policy.targeting(state, vis, los, bank, params, cfg)
