# Model size study: does a bigger network fix the jitter, or plan better?

**Status:**
- **wide2x** (`runs/mortis_wide2x-20261010-170815`) stopped at 216M. Its read, §7: slightly behind the control, with a learning-rate confound.
- **narrow** (§8, `runs/mortis_narrow-20261011-032708`) is running. Its 200M read, §9: clearly ahead of both, with the same confound in mirror image.
- Nothing needed building: the network widths are already `policy.*` keys in configs/train.yaml.
- Two folders are not arms: `mortis_wide2x-…-152617`, an earlier launch you stopped at about 30M, and `mortis_narrow_8k-…`, stopped right after launch.
**Control:** `runs/mortis_ppo-20261007-161117`, the deployed run. Its archived train.yaml equals today's configs/train.yaml, and no sim or training code has changed since it started (only configs/deployment.yaml). So the arm differs from it in network size alone.

---

## 1. The answer so far

- **The jitter halved without a bigger network, and what is left does not look like a capacity problem.**
  - Reversals fell from 28 % of decisions (the 09-30 policy against the new bots, same seeds) to 11 % in watch mode and 16 % in train mode.
  - The rest stopped falling early: 12 % / 16 % at 200M, 11 % / 16 % at 580M. Over the same stretch the six-tier win rate went from 0.25 to 0.33.
  - The policy reverses nearly as confidently as it makes any other move. At a reversal, its top two moves are 0.30–0.34 apart; across all moves the gap is 0.33–0.40. So reversing is something the reward allows, not a distinction too fine for the network.
  - Reversals no longer dodge anything (small samples): bot shots hit 57–63 % of the time after one, against 46–48 % otherwise. Some reversals are legitimate, such as turning on an enemy that ran past, and the probe cannot tell which are which.
- **Strategy might improve.** Nothing shows the network is saturated: the eval curve and the critic's explained variance were both still climbing at 600M. Only training a bigger network answers this.
- **Size so far:**
  - The 3.2× arm ran slightly behind until you stopped it at 216M (§7).
  - The 0.54× arm is clearly ahead at 200M: 0.304 against 0.250 (§9).
  - Bigger did not plan better. But each arm's effective learning rate moved with its width: hotter for the wider network, cooler for the narrower one. So the study cannot yet tell network size from step size.

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

To watch the arm against the control while it runs:

```bash
.venv/Scripts/tensorboard.exe --logdir_spec control:runs/mortis_ppo-20261007-161117/logs,wide2x:runs/mortis_wide2x-20261010-170815/logs
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

## 7. Read at 200M (2026-10-11)

**Slightly behind the control, in the "no visible effect" band, with a confound.** The arm's 160–200M evals average 0.236 against 0.250 (holdout 0.178 against 0.201), and every tier is a little lower. You kept it running.

| six-tier mean | 40M | 80M | 120M | 160M | 180M | 200M |
|---|---|---|---|---|---|---|
| wide2x | 0.065 | 0.154 | 0.199 | 0.233 | 0.234 | 0.240 |
| control | 0.119 | 0.207 | 0.228 | 0.247 | 0.253 | 0.251 |

From 40M to 100M it was about 0.05 behind; since 140M it has trailed by up to 0.02. Its 200M eval is what the control scored at about 150M.

**The step-size check fired.** §4 named this a confound, so the run so far measures "3.2× wider at the control's learning rate", not capacity alone. SB3 stops an update partway when a batch's approx KL passes 1.5 × target_kl = 0.0375.

| | wide2x | control |
|---|---|---|
| Updates stopped early by the KL brake, 150–200M | 84 of 95 | 0 of 95 |
| Epochs per update, 150–200M | 5.0 | 6.0 |
| Approx KL / clip fraction, 150–200M | 0.037 / 0.37 | 0.028 / 0.31 |
| Explained variance, 150–200M | 0.49 | 0.48 |
| Policy entropy, 200–212M | 1.84 | 1.52 |
| Curriculum stage 1 / stage 2 reached at | 102M / 189M | 62M / 116M |
| Hours to 200M | 9.1 | 8.4 |

- Adam moves each weight by about the learning rate per step, whatever the width. A unit with twice the inputs therefore changes about twice as much, so the same learning rate is hotter for the wider network. The brake catches that on most updates.
- The entropy is where the control's was at about 80M: the policy is sharpening slowly.
- The wider critic is no better. That points away from a critic capacity limit, but the conv trunk it shares kept its width, so it does not settle the question.
- Past 200M the brake still stops 20 of 24 updates. It should ease as the learning rate decays.

**Movement**, probe section [5] on both runs' 200M checkpoints, same seeds as §2:

| | wide2x | control |
|---|---|---|
| Reversals, watch / train | 13 % / 20 % | 12 % / 16 % |
| … with nobody in view, watch / train | 17 % / 22 % | 12 % / 19 % |
| Top move's probability, watch / train | 0.45 / 0.45 | 0.53 / 0.53 |
| Gap between the top two moves, at a reversal / overall, watch | 0.21 / 0.26 | 0.30 / 0.33 |
| Path efficiency, median, watch / train | 0.45 / 0.45 | 0.47 / 0.45 |
| 2 s windows under 0.3 efficiency, watch / train | 34 % / 27 % | 25 % / 26 % |
| p(attack) with an enemy in dash reach, watch / train | 0.57 / 0.65 | 0.66 / 0.62 |
| Bot shots that hit after a reversal / otherwise, watch | 60 / 59 % | 57 / 46 % |

The training logs from 150M agree: the reversal penalty lands on 16.4 % of decisions against 14.1 %, the gas penalty is about 35 % larger, and the attack-in-reach reward is 5–7 % smaller.

- **The jitter did not improve**, as §5 predicted. The arm reverses about as often in argmax play and more often when sampling.
- **Its reversals are closer to coin flips.** The control reversed almost as confidently as it made any other move.
- **Otherwise it is a less decisive version of the same policy.** It wanders a little more in argmax play, attacks less readily and takes more gas. That fits slow sharpening, not a new strategy.

**If it runs on,** compare these control evals:

| | 220M | 240M | 260M | 280M | 300M |
|---|---|---|---|---|---|
| six-tier mean | 0.248 | 0.277 | 0.283 | 0.272 | 0.266 |
| holdout | 0.213 | 0.226 | 0.241 | 0.237 | 0.233 |

The bar stays +0.03: about 0.304 for the 260–300M average, against the control's 0.274. The arm reaches each curriculum stage later, so compare eval/* only, not training stats. A win would be convincing, because it would come despite the hot learning rate. A tie would not rule capacity out.

**A clean capacity test** would rerun the arm with a lower starting learning rate, so that the brake stops firing. Try about 0.7× first (`--set learning_rate.initial=1.4e-4`, your 5e-5 floor unchanged). This is your call. It cannot share the GPU with the current run, which needs about 27 h for the full 600M at its pace (the control took 25 h). *(Declined 2026-10-11, see §9.)*

You stopped the arm at 216M (9.8 h) on 2026-10-11.

## 8. Second arm: narrow (ready 2026-10-11)

**The arm, `mortis_narrow`:** every keyed width at 2/3 of today's, rounded to a multiple of 8, with the same depth.

| | control | narrow |
|---|---|---|
| Conv stack over the grid (hard-coded) | 59,200 | 59,200 |
| Grid linear, `cnn_output_dim` | 160: 245,920 | 104: 159,848 |
| Vector MLP, `mlp_hidden_dims` | [256, 256]: 134,144 | [168, 168]: 73,248 |
| Policy head, `net_arch.pi` | [384, 384]: 307,968 | [256, 256]: 135,680 |
| Value head, `net_arch.vf` | [384, 384]: 307,968 | [256, 256]: 135,680 |
| Output layers | 8,855 | 5,911 |
| **Total** | **1,064,055** | **569,567** (0.54×) |

```bash
.venv/Scripts/python.exe scripts/train.py --set run.name=mortis_narrow --set policy.cnn_output_dim=104 --set "policy.mlp_hidden_dims=[168,168]" --set "policy.net_arch.pi=[256,256]" --set "policy.net_arch.vf=[256,256]"
```

**Verified 2026-10-11:**
- The flags build 569,567 parameters on the real config.
- `train.py --smoke` passes, and its checkpoint passes `tests/test_deployment_policy.py`.
- One real training iteration ran on the GPU at 4096 envs, and another at 8192.

**More environments are possible with either network, but they are a separate change.**
- The network holds almost no memory. What grows with the env count is the rollout buffer in RAM (4.7 KB per decision, mostly the uint8 grid) and the sim on the GPU.
- The table below comes from one real iteration each with the narrow network, eval off, measuring the process's own peaks. A real run also holds the two 1,800-env evaluators.

| | 4096 envs, batch 8192 | 8192 envs, batch 16384 |
|---|---|---|
| RAM, peak working set / peak commit | 6.6 / 10.1 GB | 11.1 / 16.4 GB |
| VRAM, max allocated / reserved | 1.8 / 3.3 GiB | 3.6 / 6.5 GiB |
| Collection speed, first iteration | 9,900 decisions/s | 14,700 decisions/s |
| One checkpoint (model + VecNormalize) | 73 MB | 139 MB |

- **RAM is the tight part, not the GPU.** The run's own peak commit grows by 6.3 GB, from 10.1 to 16.4 GB.
  - On 10-10, several things were open at once: a 4096-env wide run, BlueStacks (7.1 GB), CapCut (3.2 GB), OBS, and a 6.5 GB video export of mine. Together they brought the machine within 4.5 GB of its commit limit, and the export failed.
  - An 8192 run with the same apps open, but no export, lands at about that level.
  - Windows grows the page file on demand: the limit was 47.2 GB that day and is 37.1 GB today, with 18.6 GB free when nothing is training. So the run would probably survive, but close BlueStacks and CapCut while it trains.
- **It is faster, but by less than collection suggests.** Collection ran 1.47× faster per decision. The PPO update and the evals cost the same per decision, and they take about a third of a run. Expect roughly a fifth off the run's time. Its `time/fps` is the measurement.
- **It changes how PPO learns.** `n_steps` stays 128, so 8192 envs put twice the data into each update and make half as many updates. Scaling the batch to 16384 keeps each update's shape (64 minibatches × 6 epochs), but not the update count. A narrow run at 8192 would differ from the control in two ways at once.
- **Recommendation:** launch the narrow arm at 4096 envs, one change, as with wide2x. Test 8192 afterwards, on whichever network wins. If you want the speed now anyway, add `--set run.n_envs=8192 --set ppo.batch_size=16384`, and close BlueStacks and CapCut while it trains.

**The decision at 200M.** Same window and checks as §4, mirrored. The question is whether 0.54× the parameters loses anything.

| Narrow arm's six-tier mean, 160–200M | Reading | Then |
|---|---|---|
| 0.240 or more | No visible loss: today's network is bigger than it needs to be. | Run to 600M. |
| 0.220–0.240 | Unclear. | Run on to 300M and compare with the control's 260–300M window, 0.274. |
| below 0.220 (−0.03) | Smaller hurts: capacity matters at this size. | Stop it. |

- **Step size, mirrored.** A narrower network moves less per step at the same learning rate, so expect approx KL and clip fraction somewhat below the control's 0.028 / 0.31.
  - Much lower would mean it trained with an effectively cooler learning rate, a handicap.
  - A "no visible loss" result would then hold despite the handicap.
- **Prediction:** within 0.02 of the control at 200M, and jitter unchanged (about 12 % / 16 %). It should be only a little faster, because the conv stack and the sim dominate the time.

## 9. Narrow read at 200M (2026-10-11)

**Clearly ahead of both, by more than the study's bar for a real effect, and in the direction capacity did not predict.**
- The arm's 160–200M evals average 0.304, against the control's 0.250 (+0.054) and wide2x's 0.236. Holdout: 0.265 against 0.201.
- Every tier is higher on both map sets.
- It led at every eval from 20M on. Its 220M eval, 0.326, nearly matches the control's best, 0.333 at 580M.
- §8 predicted it would land within 0.02 of the control. That prediction was wrong.

| six-tier mean | 20M | 40M | 80M | 120M | 160M | 180M | 200M | 220M |
|---|---|---|---|---|---|---|---|---|
| narrow | 0.111 | 0.159 | 0.244 | 0.283 | 0.298 | 0.297 | 0.318 | 0.326 |
| control | 0.040 | 0.119 | 0.207 | 0.228 | 0.247 | 0.253 | 0.251 | 0.248 |
| wide2x | 0.043 | 0.065 | 0.154 | 0.199 | 0.233 | 0.234 | 0.240 | — |

| 160–200M average | narrow | control | wide2x |
|---|---|---|---|
| Holdout | 0.265 | 0.201 | 0.178 |
| Easy / medium / hard | 0.710 / 0.498 / 0.330 | 0.640 / 0.404 / 0.253 | 0.618 / 0.386 / 0.221 |
| Veteran / expert / elite | 0.158 / 0.092 / 0.038 | 0.122 / 0.058 / 0.023 | 0.116 / 0.054 / 0.020 |

**The step-size confound, mirrored, as §8 expected.**

| | narrow | control | wide2x |
|---|---|---|---|
| Approx KL / clip fraction, 150–200M | 0.017 / 0.22 | 0.028 / 0.31 | 0.037 / 0.37 |
| Updates stopped early by the KL brake, 150–200M | 0 of 95 | 0 of 95 | 84 of 95 |
| Explained variance, 150–200M | 0.49 | 0.48 | 0.49 |
| Policy entropy, 200–212M | 1.35 | 1.52 | 1.84 |
| veteran_intro / veteran / expert stage reached at | 43M / 60M / 103M | 62M / 116M / 241M | 102M / 189M / — |
| Decisions per second, 150–200M | 7,050 | 6,630 | 5,940 |
| Hours to 200M | 7.9 | 8.4 | 9.1 |

- **Network size and step size moved together in all three arms**, so the study cannot yet say which one did it. The coolest run (KL 0.017) did best, and the hottest (0.037) did worst.
- **Two explanations fit:**
  - the smaller network suits this problem better;
  - the shared learning rate is too hot for today's network as well, and narrowing cooled it.
- **Evidence for the second.** From 100M to 300M the control's approx KL sits above `target_kl` (0.025). From 100M to 450M its clip fraction stays at 0.29–0.32, and PPO usually runs at 0.1–0.2. The arm's clip fraction holds at 0.22.
- **Evidence against the second.** The arm was already far ahead at 20M (0.111 against 0.040), when the control's KL was a modest 0.015.
- **Half the critic is no worse** (EV 0.49), so the critic was not capacity-bound at today's size either.
- **It sharpened much faster.** Its entropy at 200M is where the control's was at about 500M.
- **It climbed the curriculum about twice as fast.** Since 103M it has trained against expert-stage bots, while the control was still on veteran. Its train/* stats reflect those harder opponents (train win rate 0.10, shorter episodes), so compare eval/* only.
- **It is 6 % faster:** 7.9 h to 200M.

**Movement**, probe section [5] on the three 200M checkpoints, same seeds as §2:

| | narrow | control | wide2x |
|---|---|---|---|
| Reversals, watch / train | 10 % / 18 % | 12 % / 16 % | 13 % / 20 % |
| … with nobody in view, watch / train | 15 % / 21 % | 12 % / 19 % | 17 % / 22 % |
| Top move's probability, watch / train | 0.60 / 0.57 | 0.53 / 0.53 | 0.45 / 0.45 |
| Gap between the top two moves, at a reversal / overall, watch | 0.36 / 0.42 | 0.30 / 0.33 | 0.21 / 0.26 |
| Path efficiency, median, watch / train | 0.43 / 0.46 | 0.47 / 0.45 | 0.45 / 0.45 |
| 2 s windows under 0.3 efficiency, watch / train | 30 % / 28 % | 25 % / 26 % | 34 % / 27 % |
| p(attack) with an enemy in dash reach, watch / train | 0.67 / 0.66 | 0.66 / 0.62 | 0.57 / 0.65 |
| Bot shots that hit after a reversal / otherwise, watch | 38 / 30 % | 57 / 46 % | 60 / 59 % |

- **The jitter is unchanged**, as §5 and §8 predicted. The arm reverses a little less in argmax play and a little more when sampling, within 2 points of the control either way.
- **It is the most decisive of the three.** Its reversals are confident choices too (a gap of 0.36, against 0.42 overall). So the remaining jitter is still something the reward allows, not a distinction the network cannot make.
- **Bot shots hit it less often** in watch mode: 30 % against 46 %. That rests on 46 and 13 shots, so it is a hint, not a finding.

**Next:**
- **Keep it running.** §8's table says run it to 600M, and it is running. Compare its next evals with the control's in §7: 0.277 / 0.283 / 0.272 / 0.266 at 240–300M.
- **It may soon be the best model you have.** Once it passes 0.333, deploying it needs only configs/deployment.yaml repointed; a narrow checkpoint already passed the deploy test.
- **Network size against step size stays untested, by choice.** The cooler-learning-rate rerun of the control's widths that would separate them is declined (2026-10-11): your own learning-rate experiments showed no large effect on convergence speed or final quality.
- **The next run** adds enemy memory and a privileged critic instead: CRITIC_AND_MEMORY_PLAN.md.

To watch all three side by side:

```bash
.venv/Scripts/tensorboard.exe --logdir_spec control:runs/mortis_ppo-20261007-161117/logs,wide2x:runs/mortis_wide2x-20261010-170815/logs,narrow:runs/mortis_narrow-20261011-032708/logs
```

## 10. Narrow read at 500M (2026-10-12)

**Still ahead at matched steps, by the same margin as at 200M. Level with the control's finished run, within the bar.**

Three-eval averages:

| window | narrow, six-tier / holdout | control | narrow minus control |
|---|---|---|---|
| 160–200M | 0.304 / 0.265 | 0.250 / 0.201 | +0.054 / +0.064 |
| 260–300M | 0.333 / 0.275 | 0.273 / 0.237 | +0.060 / +0.038 |
| 360–400M | 0.347 / 0.295 | 0.279 / 0.230 | +0.068 / +0.065 |
| 460–500M | 0.344 / 0.296 | 0.294 / 0.233 | +0.050 / +0.063 |

- **At matched steps, the arm led at all 25 evals from 20M to 500M.** At 460–500M every tier is higher: easy to elite 0.781 / 0.543 / 0.394 / 0.179 / 0.113 / 0.056, against 0.711 / 0.496 / 0.308 / 0.133 / 0.072 / 0.041.
- **Against the control's final result it is level.**
  - The arm's best is 0.351 at 480M, its `best_model`. The control's best, the deployed model, is 0.333 at 580M.
  - On holdout the bests are 0.305 at 400M and 0.288 at 600M.
  - Both gaps, +0.018 and +0.017, sit inside the study's ±0.03 bar. The arm reached the control's final level by about 260M.
- **It has been flat since about 360M**, at 0.345 ± 0.005.
- **The control rose late:** 0.306 at 560M, then 0.333 and 0.328 at 580–600M. So the arm's 580–600M evals settle the final comparison.
- **Train stats can't settle it.** At 450–500M both arms train against elite bots, which the arm reached at 229M and the control at 365M. Train win rate is 0.089 against 0.064; on a chart, two lines under 0.1 look alike. The rest:

  | | narrow | control |
  |---|---|---|
  | Explained variance | 0.499 | 0.492 |
  | Approx KL / clip fraction | 0.009 / 0.21 | 0.015 / 0.29 |
  | Entropy | 1.26 | 1.36 |
  | Decisions per second | 6,670 | 6,620 |

**What it means for the next run:** a bigger network bought nothing here. The narrow one learns faster and ends at least level, with 0.54× the parameters. CRITIC_AND_MEMORY_PLAN.md recommends its widths (decision 5), with this run as the matched-step baseline.
