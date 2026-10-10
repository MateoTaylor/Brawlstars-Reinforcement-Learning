# Model size study: does a bigger network fix the jitter, or plan better?

**Status:** ready 2026-10-10, not launched. Nothing needed building: the network widths are already `policy.*` keys in configs/train.yaml. There is one arm for you to launch (§3), and a decision at 200M (§4).
**Control:** `runs/mortis_ppo-20261007-161117`, the deployed run. Its archived train.yaml equals today's configs/train.yaml, and no sim or training code has changed since it started (only configs/deployment.yaml). So the arm differs from it in network size alone.

---

## 1. The answer so far

- **The jitter halved without a bigger network, and what is left does not look like a capacity problem.**
  - Reversals fell from 28 % of decisions (the 09-30 policy against the new bots, same seeds) to 11 % in watch mode and 16 % in train mode.
  - The rest stopped falling early: 12 % / 16 % at 200M, 11 % / 16 % at 580M. Over the same stretch the six-tier win rate went from 0.25 to 0.33.
  - The policy reverses nearly as confidently as it makes any other move. At a reversal, its top two moves are 0.30–0.34 apart; across all moves the gap is 0.33–0.40. So reversing is something the reward allows, not a distinction too fine for the network.
  - Reversals no longer dodge anything (small samples): bot shots hit 57–63 % of the time after one, against 46–48 % otherwise. Some reversals are legitimate, such as turning on an enemy that ran past, and the probe cannot tell which are which.
- **Strategy might improve.** Nothing shows the network is saturated: the eval curve and the critic's explained variance were both still climbing at 600M. Only training a bigger network answers this.

## 2. Baseline: the deployed run

**Movement** (probe section [5], expert tier). Watch mode is 8 matches, seeds 0–7, argmax actions; train mode is 16 envs, seed 7, sampled actions.

| | 09-30 policy | this run, 200M | this run, 580M (best_model) |
|---|---|---|---|
| Reversals of 135° or more, watch / train | 28 % / 28 % | 12 % / 16 % | 11 % / 16 % |
| … with nobody in view, watch / train | — | 12 % / 19 % | 7 % / 19 % |
| Path efficiency over 2 s (net distance / walked), median, watch / train | — | 0.47 / 0.45 | 0.49 / 0.46 |
| Gap between the top two move probabilities, at a reversal / overall, watch | — | 0.30 / 0.33 | 0.33 / 0.40 |
| Bot shots that hit after a reversal / otherwise, watch | 55 / 52 % | 57 / 46 % | 63 / 48 % |

The 09-30 column comes from SIM_ISSUES_PLAN.md §B, with train.yaml's sim keys on. The bot-shot row rests on 8–25 shots per cell.

**Eval**, the average of the 160M, 180M and 200M evals (300 episodes a tier):

| six-tier mean | holdout | easy | medium | hard | veteran | expert | elite |
|---|---|---|---|---|---|---|---|
| 0.250 | 0.201 | 0.640 | 0.404 | 0.253 | 0.122 | 0.058 | 0.023 |

Successive evals of this run move by about 0.01 on the six-tier mean.

**PPO.**
- 100–200M: explained variance 0.48, approx KL 0.027, clip fraction 0.30.
- 500–600M: 0.50, 0.011 and 0.27.
- No update stopped before its 6th epoch. Training took 8.4 h to reach 200M and 25.1 h to reach 600M.

**Where the parameters are**, today's network and the arm's:

| | control (train.yaml) | arm: `wide2x` |
|---|---|---|
| Conv stack over the grid (32/64/64 channels, not a key) | 59,200 | 59,200 |
| Grid linear, `cnn_output_dim` | 1536 → 160: 245,920 | 1536 → 320: 491,840 |
| Vector MLP, `mlp_hidden_dims` | [256, 256]: 134,144 | [512, 512]: 399,360 |
| Policy head, `net_arch.pi` | [384, 384]: 307,968 | [768, 768]: 1,230,336 |
| Value head, `net_arch.vf` | [384, 384]: 307,968 | [768, 768]: 1,230,336 |
| Output layers | 8,855 | 17,687 |
| **Total** | **1,064,055** | **3,428,759** (3.2×) |

## 3. The arm

The arm doubles every width that is a key. One arm is enough to answer "does capacity help at all". If 3.2× shows nothing, bigger arms are unlikely to pay. If it does show a gain, it earns its full 600M.

```bash
.venv/Scripts/python.exe scripts/train.py --set run.name=mortis_wide2x --set policy.cnn_output_dim=320 --set "policy.mlp_hidden_dims=[512,512]" --set "policy.net_arch.pi=[768,768]" --set "policy.net_arch.vf=[768,768]"
```

- **Everything else is train.yaml's**, including the 600M `total_timesteps`. That way the learning rate, the clip range and the curriculum's 125M stage caps line up with the control's at every step, and stopping early at 200M compares like with like. The curriculum gates may open at different times in the two runs. eval/* uses fixed tiers, so the comparison holds anyway.
- **Verified 2026-10-10:**
  - The flags go through train.py's own parser and build the arm at 3,428,759 parameters.
  - `train.py --smoke` passes with them.
  - The smoke's checkpoint passes the real-checkpoint deploy test (`tests/test_deployment_policy.py`). So deploying a wide run needs only `configs/deployment.yaml` repointed. A 3.4M forward pass every 0.25 s is negligible on the deploy CPU.
- **Cost, estimated, not measured.** The arm does about 1.5× the network arithmetic per decision, because the conv stack, most of today's arithmetic, is unchanged. The sim is roughly half of each iteration, so expect about 10 h to 200M (the control took 8.4 h). The run's own fps will say.

## 4. The decision at 200M

Average the arm's 160M, 180M and 200M evals and compare them with the control's 0.250 (holdout 0.201):

| Arm's six-tier mean | Reading | Then |
|---|---|---|
| 0.280 or more (+0.03) | Capacity helps. | Let it run to 600M, then compare with the control's best 0.333 and final 0.328. |
| 0.270–0.280 | Unclear. | Run on to 300M and compare with the control's 260–300M window, 0.274. |
| 0.230–0.270 | No visible effect at 3.2×. | Stop it (Ctrl+C). Capacity is off the list for this setup. |
| below 0.230 | Worse. | Stop it. |

The +0.03 bar is about three times the eval-to-eval noise. Run-to-run variance from the training seed has never been measured here, so a smaller win from one seed per arm is not convincing.

At the same point I compare the following from the arm's 200M checkpoint (`checkpoints/model_199966720_steps.zip`):
- **Jitter:** the same probe and seeds as §2, against 12 % / 16 % reversals and 0.47 / 0.45 path efficiency.
- **Critic:** `train/explained_variance` against 0.48. A wider critic that climbs clearly above it would mean the critic was capacity-bound, which is the open question in the EV-plateau finding.
- **Step size:** approx KL and clip fraction against 0.027 and 0.30. Clearly higher would mean the shared learning rate is effectively too hot for the wider network. That is a confound to note, not a capacity result.
- **Time to 200M**, against 8.4 h.

To watch the arm against the control while it runs, fill in its timestamp:

```bash
.venv/Scripts/tensorboard.exe --logdir_spec control:runs/mortis_ppo-20261007-161117/logs,wide2x:runs/mortis_wide2x-STAMP/logs
```

## 5. Predictions, so the result can surprise us

- **Jitter: no change**, about 12 % / 16 % again. The remaining reversals are confident, and they survived 380M more decisions of training while everything else improved. A bigger network gets better at what the reward pays for; it does not change what the reward pays for.
- **Win rate: a modest gain at most.** The policy sees a 13 × 21-tile window, its margin to the gas, the clock, the count of enemies left and its last 3 decisions. A bigger network can only plan from what it is shown, and it is shown no crate or enemy that is off screen.

## 6. After the result

- **If it helps:** run the arm to 600M. If that gain holds up, the next steps would be:
  - a wider conv stack (its channels are hard-coded, so this is a small code change behind a key);
  - separate policy and value extractors (this also needs a new key);
  - a second seed.
- **If it does not:** capacity is settled for this setup. The remaining jitter would then answer to the reward or the sim, not the network.

How the §2 movement numbers were taken (CPU, writes nothing, about 2 minutes each):

```bash
.venv/Scripts/python.exe scripts/probes/sim_issues_probe.py runs/mortis_ppo-20261007-161117/checkpoints/model_199966720_steps.zip --tier expert --mode watch --episodes 8
```

For train mode, swap in `--mode train --envs 16 --seed 7`. For the 580M row, use `best_model.zip`.
