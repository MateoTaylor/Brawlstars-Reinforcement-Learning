"""`policy.py` -- the checkpoint, its guards, and the action mask.

The module is a thin wrapper over `MaskablePPO.predict`, so there is little behaviour to test and
a lot of *refusal*. That is deliberate: every way of deploying a checkpoint against the wrong
observation is silent at inference, so the guards are the product and these tests are what say the
guards actually fire.

`runs/` is gitignored, so nothing here may *require* a 44 MB checkpoint. The guards are reachable
from a temp directory holding only a `train.yaml` (they run before `MaskablePPO.load`), `act()` is
exercised against a stub model that records what it was handed, and the one end-to-end test skips
itself when the run is absent.
"""
import numpy as np
import pytest
import yaml

from brawl_sim.config import load_config
from brawl_sim.core import obs_select
from brawl_deployment.policy import ATTACK_FIRE, ATTACK_SUPER, DeployedPolicy, Decision, check_spaces

RUN = "runs/mortis_deploy-20260907-041522"
SPEC = "configs/agent_obs_deploy.yaml"
CONFIGS = "configs/default.yaml"

# The run `scripts/deploy_run.py` loads by default, and every other local run it could be pointed
# at with `--run`. The end-to-end test used to cover RUN alone, so pointing configs/deployment.yaml
# at a newer spec was never exercised offline -- and the first live decision on it is where a
# missing supplier surfaced.
DEPLOYED_RUN = yaml.safe_load(open("configs/deployment.yaml").read())["run"]["dir"]


def _deployable_runs() -> list[str]:
    """RUN, DEPLOYED_RUN, and every run under runs/ whose train.yaml names a deploy spec."""
    from pathlib import Path

    found = {RUN, DEPLOYED_RUN}
    for train_yaml in Path("runs").glob("*/train.yaml"):
        spec = ((yaml.safe_load(train_yaml.read_text()) or {}).get("run") or {}).get("agent_obs")
        if spec and Path(spec).name.startswith("agent_obs_deploy"):
            found.add(train_yaml.parent.as_posix())
    return sorted(found)

# The smallest train.yaml `load_train_config` will validate. `eval.enabled: false` is only here
# because evaluation insists on curriculum tiers it has no use for in a load-time guard test.
BASE_RUN_YAML = {
    "run": {"algo": "maskable_ppo", "env_config": CONFIGS, "agent_obs": SPEC},
    "normalize": {"obs": False, "reward": True},
    "eval": {"enabled": False},
}


def _run_dir(tmp_path, **patch):
    raw = {k: dict(v) for k, v in BASE_RUN_YAML.items()}
    for section, fields in patch.items():
        raw[section].update(fields)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(raw))
    return tmp_path


class _StubModel:
    """Records the mask and observation it was given, and returns a fixed action.

    Deliberately not a real policy: what these tests check is what `act` HANDS the network and
    what it makes of the answer, not what the network decides. A real checkpoint would make the
    mask assertion probabilistic -- the policy usually declines to fire anyway, so "it did not
    fire" would prove nothing about masking.
    """

    def __init__(self, action=(3, 0), space=None, nvec=(17, 3)):
        self.action = np.array([action])
        self.observation_space = space
        self.action_space = type("A", (), {"nvec": np.array(nvec)})()
        self.seen = None

    def predict(self, obs, deterministic=True, action_masks=None):
        self.seen = {"obs": {k: v.copy() for k, v in obs.items()},
                     "mask": np.array(action_masks), "deterministic": deterministic}
        return self.action, None


def _policy(action=(3, 0), *, deterministic=True):
    cfg = load_config(CONFIGS)
    spec = obs_select.load_agent_spec(SPEC, cfg)
    return DeployedPolicy(_StubModel(action), spec, cfg, deterministic=deterministic)


# ---- the guards -------------------------------------------------------------------------------

def test_a_run_trained_without_masking_is_refused(tmp_path):
    """`predict` on a plain PPO has no `action_masks` parameter and would raise -- but the reason
    to refuse is not the TypeError. An unmasked policy can request a super it has not charged, and
    `MaskablePPO` was chosen precisely so that stays impossible."""
    with pytest.raises(ValueError, match="action mask"):
        DeployedPolicy.from_run(_run_dir(tmp_path, run={"algo": "ppo"}))


def test_a_run_that_normalized_its_observations_is_refused(tmp_path):
    """The silent one, and the reason this check exists at all.

    A `normalize.obs: true` run keeps its observation statistics in `best_vecnormalize.pkl`. The
    deployed path has no `VecNormalize`, so loading such a checkpoint feeds raw values to a network
    trained on whitened ones -- no shape error, no crash, just a policy that is quietly wrong in
    every column at once. The current run has `obs: false`, which is a fact about that run and not
    about runs in general.
    """
    with pytest.raises(ValueError, match="normalize.obs"):
        DeployedPolicy.from_run(_run_dir(tmp_path, normalize={"obs": True}))


def test_the_guards_fire_before_the_checkpoint_is_opened(tmp_path):
    """Not incidental -- it is what lets this file test them at all, since `runs/` is gitignored
    and no checkpoint exists on a clean checkout. Both directories below contain a `train.yaml` and
    nothing else; if either guard ran after `MaskablePPO.load` this would fail with a missing-file
    error rather than the ValueError."""
    for i, patch in enumerate(({"run": {"algo": "ppo"}}, {"normalize": {"obs": True}})):
        with pytest.raises(ValueError):
            DeployedPolicy.from_run(_run_dir(tmp_path / f"empty{i}", **patch))


def test_a_checkpoint_whose_observation_does_not_match_the_spec_is_refused():
    """The failure mode `agent_obs_deploy2.yaml` exists to avoid. The two deploy specs differ by
    two columns in one group, so a checkpoint paired with the wrong one gets a `zone` group of the
    wrong width -- caught here -- or, if a future edit ever preserved the width, a silently shifted
    one. This is the check that makes the first case loud."""
    cfg = load_config(CONFIGS)
    spec = obs_select.load_agent_spec(SPEC, cfg)
    other = obs_select.load_agent_spec("configs/agent_obs_deploy2.yaml", cfg)

    check_spaces(_StubModel(space=obs_select.agent_space(spec, cfg)), spec, cfg, label="ok")
    with pytest.raises(ValueError, match="NEW file"):
        check_spaces(_StubModel(space=obs_select.agent_space(other, cfg)), spec, cfg, label="bad")


def test_a_checkpoint_with_the_wrong_action_space_is_refused():
    """A two-valued attack column is the pre-super action space (`bot_overhaul.md` D1). A
    checkpoint from before that widening has an observation the current spec still matches, so
    this is the only check that catches it."""
    cfg = load_config(CONFIGS)
    spec = obs_select.load_agent_spec(SPEC, cfg)
    model = _StubModel(space=obs_select.agent_space(spec, cfg), nvec=(17, 2))
    with pytest.raises(ValueError, match="action space"):
        check_spaces(model, spec, cfg, label="stale")


# ---- the mask ---------------------------------------------------------------------------------

def test_the_mask_handed_to_the_network_is_the_sims_own_layout():
    """`[move (n_move_bins + 1), attack (3)]` flattened -- what `wrappers/sb3_vecenv.py`'s
    `action_masks()` produced on every training step. A different split here would mask the wrong
    dimension while remaining exactly the right width, which no shape check catches."""
    pol = _policy()
    obs = {k: np.zeros(v.shape, v.dtype) for k, v in
           obs_select.agent_space(pol.spec, pol.cfg).spaces.items()}
    pol.act(obs, (True, False, True))

    mask = pol.model.seen["mask"]
    n_move = pol.cfg.n_move_bins + 1
    assert mask.shape == (1, n_move + 3)
    assert mask[0, :n_move].all(), "hero.action_mask builds the move half all-True"
    assert list(mask[0, n_move:]) == [True, False, True]


def test_the_attack_mask_is_the_shadows_and_is_not_recomputed_here():
    """The formula (`alive & cd <= 0 & dash_t <= 0`, plus ammo for attack and charge for super)
    lives in `ShadowHero.attack_mask`, which owns the timers it reads. `act` takes that tuple
    verbatim. A second copy here would be a rule that is only correct in one place and silently
    diverges the day the shadow's dash handling changes."""
    pol = _policy()
    obs = {k: np.zeros(v.shape, v.dtype) for k, v in
           obs_select.agent_space(pol.spec, pol.cfg).spaces.items()}
    n_move = pol.cfg.n_move_bins + 1

    from brawl_deployment.perception.shadow import ShadowHero, ShadowParams

    shadow = ShadowHero(ShadowParams.load())
    shadow.reset()
    pol.act(obs, shadow.attack_mask())
    assert list(pol.model.seen["mask"][0, n_move:]) == list(shadow.attack_mask())


def test_an_all_illegal_attack_column_is_rejected_rather_than_producing_nan():
    """`hero.action_mask` makes no-fire unconditionally legal (`no_fire_ok = ones_like`), so an
    all-False attack row cannot arise from the sim and means the caller built the tuple wrong.
    MaskablePPO's response to one is a degenerate categorical -- NaN logits, not an exception --
    so it has to be caught before `predict`."""
    pol = _policy()
    obs = {k: np.zeros(v.shape, v.dtype) for k, v in
           obs_select.agent_space(pol.spec, pol.cfg).spaces.items()}
    with pytest.raises(ValueError, match="no-fire"):
        pol.act(obs, (False, True, True))
    with pytest.raises(ValueError, match="no_fire, attack, super"):
        pol.act(obs, (True, True))


def test_the_decision_keeps_the_mask_it_was_given():
    """"Did not fire" and "could not fire" are the same action and completely different problems.
    A live run that goes quiet is diagnosed by which one it was, and the action alone cannot say --
    so `Decision` carries the mask."""
    pol = _policy(action=(7, 0))
    obs = {k: np.zeros(v.shape, v.dtype) for k, v in
           obs_select.agent_space(pol.spec, pol.cfg).spaces.items()}

    chose_not_to = pol.act(obs, (True, True, True))
    could_not = pol.act(obs, (True, False, False))
    assert chose_not_to.attack == could_not.attack == 0
    assert chose_not_to.legal != could_not.legal
    assert not chose_not_to.fired


def test_the_batch_axis_is_added_without_disturbing_the_assemblers_output():
    """`assemble` returns un-batched arrays and `predict` wants a leading 1. The reusable staging
    buffer here is the same trade as `assemble`'s own: fine as long as it does not alias the
    caller's array, which is what a mutation-after-the-call would prove."""
    pol = _policy()
    space = obs_select.agent_space(pol.spec, pol.cfg)
    obs = {k: np.full(v.shape, 3, v.dtype) for k, v in space.spaces.items()}
    pol.act(obs, (True, True, True))

    seen = pol.model.seen["obs"]
    for name, value in obs.items():
        assert seen[name].shape == (1, *value.shape)
        assert seen[name].dtype == value.dtype
        np.testing.assert_array_equal(seen[name][0], value)

    obs["self"][:] = 99          # the caller reuses its array; the staged copy must not follow
    pol.act(obs, (True, True, True))
    assert pol.model.seen["obs"]["self"][0][0] == 99


def test_deterministic_is_passed_through_and_defaults_to_the_way_the_run_was_scored():
    """`configs/train.yaml` sets `eval.deterministic: true`, and `best_model.zip` was SELECTED by
    that evaluation -- a stochastic deployment would not be the policy that scored 0.734. The
    hazard is worth naming for `loop.py`: a deterministic policy whose input stops changing
    repeats one action forever, which in the sim self-corrects and in a real match does not."""
    obs_of = lambda p: {k: np.zeros(v.shape, v.dtype) for k, v in                    # noqa: E731
                        obs_select.agent_space(p.spec, p.cfg).spaces.items()}
    pol = _policy()
    pol.act(obs_of(pol), (True, True, True))
    assert pol.model.seen["deterministic"] is True

    stoch = _policy(deterministic=False)
    stoch.act(obs_of(stoch), (True, True, True))
    assert stoch.model.seen["deterministic"] is False


# ---- end to end, when the checkpoint is present -------------------------------------------------

@pytest.mark.parametrize("run", _deployable_runs())
def test_the_real_checkpoint_loads_and_decides_from_an_assembled_observation(run):
    """The one test that touches the actual artifact: load the run, build ITS assembler, and push
    a real assembled observation through. It asserts the seam rather than the answer -- that
    `assemble`'s output is exactly what `predict` accepts, with no reshaping in between, and that
    a masked-off super is never chosen over 30 varied frames.

    The zone group comes from the real `ZoneEstimator`, not a hand-written dict. A dict written here
    carried both margin names while the estimator produced one, which is how this test passed for a
    deploy3 run that could not make its first live decision."""
    from brawl_deployment.perception.grid import GasMap
    from brawl_deployment.perception.zone import ZoneEstimator

    if not __import__("pathlib").Path(f"{run}/best_model.zip").exists():
        pytest.skip("runs/ is gitignored; the deployed checkpoint is not on every machine")
    pol = DeployedPolicy.from_run(run)
    asm = pol.make_assembler()
    assert asm.spec is pol.spec and asm.cfg is pol.cfg, "one spec object, not two loads of it"
    gas = GasMap(128, 128)
    ox, oy = gas.origin
    gas.gassed[:, :5 - ox] = True                   # gas at world x <= 4: a real west margin
    zone = ZoneEstimator(pol.cfg)

    shadow_obs = {"facing_vec": (1.0, 0.0), "ammo_frac": 0.8, "ammo_whole": 2.0, "attack_cd": 0.0,
                  "can_attack": True, "dashing": False, "dash_t": 0.0, "dash_dir": (0.0, 0.0),
                  "invuln": False, "long_dash_ready": True, "long_dash_frac": 1.0,
                  "super_ready": False, "super_charge_frac": 0.4}
    rng = np.random.default_rng(0)
    supers = 0
    for i in range(30):
        obs = asm.assemble(
            hero_pos=(10.0 + i, 10.0), hero_vel=(1.0, 0.0), shadow=shadow_obs, hero_hp=8000.0,
            n_enemies_alive=4.0, elapsed_s=float(i) * 5.0, hero_in_bush=False, hero_in_zone=False,
            enemies=[None] * (pol.cfg.n_entities - 1), enemy_hp={}, enemy_in_bush={},
            projectiles=[],
            zone=zone.estimate(gas, (10.0 + i, 10.0)),
            grid=rng.integers(0, 2, (len(asm._grid_channels), pol.cfg.view_h, pol.cfg.view_w),
                              dtype=np.uint8),
        )
        d = pol.act(obs, (True, True, False))       # super masked off the whole way through
        assert isinstance(d, Decision)
        assert 0 <= d.move_bin <= pol.cfg.n_move_bins
        assert d.attack in (0, ATTACK_FIRE)
        supers += d.attack == ATTACK_SUPER
    assert supers == 0
