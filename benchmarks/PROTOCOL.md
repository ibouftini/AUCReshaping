# Evaluation protocol: AUCReshaping on chest X-rays

**Status: fixed on 2026-11-06,** before the first confirmatory run. The commit is tagged and released. After that, it
changes only through a dated entry in the amendments section, made before the runs it affects. A
decision made after seeing test results is labelled exploratory.

## 1. The question

On real medical images, does choosing which positives to boost, according to the operating point, raise
sensitivity at 98% specificity beyond the same amount of boosting on random positives, and beyond a tuned
class weight?

AUCReshaping (Bhat et al., *Scientific Reports* 13, 21097, 2023) raises the loss of positives that fall
below a high-specificity threshold during fine-tuning, using $-\log(p - b)$ for those samples.

The study can show a gain, a loss, no difference larger than 1 point, or an inconclusive result, in this
setup. Its conclusions are limited to this data, resolution, model and protocol.

## 2. What was known before this protocol

This is not a blind study, and readers should weigh it accordingly.

- **The same test set was used before.** An earlier study of this library read the official NIH test list,
  with other weights, 128 px images, a 15-output model and 2 seeds. It could not separate the methods. Its
  results stay visible, and any changed conclusion will be stated as a change.
- **Earlier evaluations had flaws,** which shaped this design: a random control that differed in strength
  from the method, a synthetic laboratory that did not fine-tune, and a floor setting that was never varied.
  The repository's page on the implementation and the paper lists the choices the paper leaves open.
- **Synthetic experiments were run and read before the tuning grids were fixed.** They fine-tune every arm
  from one checkpoint, on seeds kept apart from the ones used here. Their results informed the grids in
  section 6: boosts near zero and boosts well above the earlier range were both added.
- **Validation data was used** for choosing the weights, the learning rate, the sampler and the tuned
  settings. Those runs did not load the test files.

## 3. Data

- **NIH ChestX-ray14,** 112,120 frontal images. Labels are mined from reports, and this limits the study.
- **Test:** the official list, 25,596 images. No patient is in both this list and the rest of the data.
- **Validation:** 10% of the remaining patients, with split seed 2000, giving 77,303 training and 9,221
  validation images. No patient is in both.
- **Task:** effusion, with a single-output model. Effusion is the task of the paper's Table 3, the source of the
  boosts used here. The earlier study's results on this test list had been read when it was chosen; they could
  not separate the methods.
- **Classes:** the positive class is every image without an effusion label, as in the paper (pp. 5-6), so it
  includes other findings. "No-effusion images" below means this class, not healthy patients. Images labelled
  No Finding get their own readout.
- **Label noise:** about 93 effusion images set the test threshold, and these are the ones that look least like
  effusion, so report-mining errors may be concentrated there. Mislabelled images may also stay flagged and keep
  receiving the boost. This could hide a real effect or create a false one, and the study cannot separate them.
- **Images:** resized to 224 px, as the weights' documentation asks, and scaled to $[-1024, 1024]$. The weights'
  documented preprocessing is used, not the paper's.

## 4. Model and starting weights

- **Model:** TorchXRayVision 1.5.7, DenseNet-121 at 224 px, `densenet121-res224-pc`, trained on PadChest.
  File SHA-256: `a9148ef62ae4e7a31a9bed8cef22811f39681cd61e4390398936efc6280af521`.
- **Licence:** the repository is Apache 2.0, with no separate terms on the weights. We do not redistribute
  them.
- **Why these weights:** they never saw NIH images, so positives are not already fitted, which would leave
  almost nothing to boost. Among the four eligible sets (`chex`, `mimic_nb`, `mimic_ch`, `pc`), `pc` had the
  best validation AUC and sensitivity with frozen features. `-nih`, `-all`, `resnet50-res512-all` and
  `-rsna` are excluded, because they were trained on NIH images or on a subset of them.
- **Head:** a new one-output layer, initialized by the seed.
- **Limit:** all seeds share one checkpoint, so conclusions are conditional on it, and seed spread may
  understate the uncertainty about other starting points. A second set of weights is a secondary study.

## 5. Training

Every arm shares weights, model, optimizer, learning rate, batch size, epochs and augmentation. Pairs on the
same seed share head initialization, batch order and augmentation.

- Adam, lr $10^{-4}$, weight decay $10^{-5}$, batch size 64, 10 epochs, mixed precision.
- Augmentation: horizontal flip, and intensity scaling between 0.9 and 1.1.
- No weighted sampler. A pilot compared lr $10^{-3}$, $10^{-4}$, $10^{-5}$, and $10^{-4}$ with the paper's
  sampler and class weights, for CE, on validation data, seed 1700. This setting had the best AUC, sensitivity
  and loss. One seed only.
- **Direction: the paper's.** No-effusion images are the boosted class, label 1. Specificity is measured on
  effusion images. In clinical terms, this fixes how much effusion is missed (2%) and measures how many
  no-effusion images are cleared. It differs from flagging disease at a fixed false-alarm rate.
- **Thresholds:** from a first-in, first-out window of the last 4,096 effusion scores on training data, which spans
  about half an epoch. Boosting starts after 500 such scores, about 75 batches. The sampler would shorten the
  window to about a tenth of an epoch, and the pilot judged the sampler on CE only, so this choice was not tested
  for AUCReshaping and could disadvantage it.
- Baseline check: CE validation AUC, compared with the published test range for effusion on the official list
  (0.7585 to 0.828, as tabulated by ChestNet, arXiv 1807.03058). It is not below that range. The official test
  set is a different and harder sample, so the two are not directly comparable.

## 6. Arms

| Tier | Arm | Setting |
|---|---|---|
| Primary | CE | Class-ratio weights |
| Primary | AUCReshaping, tuned | `prob`, boosts 0.4, 0.2, 0.1, 0.1 at 0.90, 0.92, 0.95, 0.98, floor tuned below |
| Primary | Weighted BCE, tuned | Class-ratio weight times a factor, tuned below |
| Primary | AUCReshaping at about 6x | The same boosts at the 6x row of the grid below |
| Primary | Mass-matched random control | At about 6x: the same extra gradient, on random positives |
| Secondary | Mass-matched control above the threshold | At about 6x: the same extra gradient, on positives above the threshold. The highest contrast |

Other secondary arms are in section 13. They run only if the budget allows, and are reported as cut otherwise.

**Why two AUCReshaping arms.** Tuning asks which setting validation data prefers. It chose a gentle one, and at a gentle
setting AUCReshaping and a random control differ too little for a targeting test to mean anything: measured on the pilot's
validation scores, about 0.3-0.5% of the gradient at 1.1x, about 3% at 1.3x, and about 30% at 6x. The control already puts
95-97% of its expected extra gradient on flagged positives, because unflagged positives hold little gradient. So the
targeting question is asked at about 6x, and the comparisons with the simple fix and the baseline use the tuned setting.
**The 6x setting was not chosen by validation, and the paper does not state it.** It is the paper's boosts at the floor an
earlier study ran.

"AUCReshaping" alone always means the `prob` form.

**Tuning.** Both tuned arms get the same strengths, measured on the pilot's validation scores as gradient
mass on the boosted class relative to BCE:

| Strength, about | AUCReshaping | Weighted BCE factor |
|---|---|---|
| 1.1x, added by the extension below | `band`, floor $r = 0.9$ | 1.1 |
| 1.3x | `band`, floor $r = 0.75$ | 1.25 |
| 3x | `band`, floor $r = 0.25$ | 3 |
| 6x | `band`, floor $r = 0.1$ (the setting an earlier study ran) | 6 |
| 15x | `sum`, floor $r = 0.05$ | 15 |

Each setting runs on seeds 1800, 1801 and 1802, with 3 CE runs to pair against, on validation data only. The
setting with the highest mean validation sensitivity at 98%, at each run's best epoch, is chosen. **If a tuned
arm picks an end of its grid, both grids are extended at that end before this protocol is fixed.**
Differences of under about a point between settings are partly chance, because validation has 957 disease
cases.

**Extension.** On the first four settings, AUCReshaping chose the gentle end of its grid, so both grids gained
one gentler setting on the same seeds. This is as far as the gentle end goes: the extra gradient is at most
about 11%, and at a floor of 1 the loss would be CE. If the new setting is chosen, the study states that
validation data preferred almost no boost.

**Consequence for reading the results.** A gentle chosen setting sits close to CE by construction, so "no difference" on
P2 and P3 would partly reflect a small boost. Contrasts S2 and S3 show the method at about 6x against the same arms.

**Chosen settings,** validation sensitivity at 98%, mean of seeds 1800-1802 at each run's best epoch, with per-seed values.
CE: 37.89 (38.0, 37.3, 38.4).

| Strength, about | AUCReshaping | Weighted BCE |
|---|---|---|
| 1.1x | 36.63 (37.2, 36.1, 36.6) | 37.15 (39.2, 34.9, 37.3) |
| 1.3x | **37.79 (37.2, 38.1, 38.0), chosen** | 37.30 (35.8, 38.5, 37.6) |
| 3x | 36.76 (37.8, 35.4, 37.0) | **39.06 (40.7, 39.1, 37.4), chosen** |
| 6x | 37.21 (36.9, 37.3, 37.5) | 37.82 (37.2, 37.6, 38.6) |
| 15x | 30.82 (32.6, 30.3, 29.5) | 36.32 (34.6, 39.2, 35.1) |

- **AUCReshaping:** `band`, floor $r = 0.75$, about 1.3x.
- **Weighted BCE:** class-ratio weight times 3, about 3x.

Neither choice is at an end of its grid. Apart from 15x, the settings of each arm differ by about a point or less, which is
within the noise of 3 seeds, so the choice is partly chance. Averaged over all 10 epochs rather than at the best one, the
settings below 15x differ by less than half a point. At 15x, AUCReshaping loses about 7 points to CE on every seed. About
half of the positives stay boosted from the first epoch to the last, under every `band` setting.

## 7. Estimands

At 98% specificity, in points, at the selected epoch:

- **P1, targeting:** AUCReshaping at about 6x minus the mass-matched random control.
- **P2, the simple fix:** tuned AUCReshaping minus tuned weighted BCE.
- **P3, the baseline:** tuned AUCReshaping minus CE.

P1 to P3 are the primary family. Every statement names the setting and its strength.

Secondary, outside the Holm family, with wording from the interval alone:

- **S1:** AUCReshaping at about 6x minus the mass-matched control above the threshold.
- **S2:** AUCReshaping at about 6x minus tuned weighted BCE.
- **S3:** AUCReshaping at about 6x minus CE.

Checks: **C1** and **C2** are each control minus CE.

**What P1 can and cannot show.** Random recipients overlap the flagged set by about 65% by count, and about 95% by gradient,
so even at 6x the control is not far from the method. A tie on P1 alone cannot show that targeting is useless, and S1 is
reported beside it. A gain on P1 counts only under the condition in section 8.

## 8. Readouts and selection

- **Primary readout:** sensitivity read from the test ROC at 98% specificity.
- **Equal prominence:** sensitivity and realised specificity at the threshold fixed on validation. An earlier
  study found the two readouts can differ by a factor of two.
- **Selection:** the epoch of highest validation sensitivity is primary. The epoch of lowest validation loss,
  the paper's rule, sits in the same table.
- **Secondary:** AUC, partial AUC up to a 2% false-alarm rate, and sensitivity on No Finding images against
  effusion images.
- **Always shown:** absolute sensitivity next to every difference, every seed as a point, the share of
  positives boosted per epoch, the recipients' overlap by count and by gradient, the extra gradient each arm
  received, how often the control hit its weight cap, each control minus CE, and the noise floor from CE repeats.
- **Only one reading makes claims.** The permitted wording applies to the test ROC readout, at the primary
  selection rule. The validation-threshold readout has its own difference and interval, and the other rule has
  its own table. Both are labelled as sensitivity analyses and reported as agreeing or disagreeing.
- **Reading a gain on targeting:** it counts only if the control is not below CE and the arms' cumulative extra
  gradient is comparable.

## 9. Claim rule

Fixed in advance, with a 95% interval and a region of $\pm 1$ point:

| Interval | Permitted wording |
|---|---|
| Entirely above +1 | gain of at least about 1 point in this setup |
| Above 0, crosses +1 | gain, size uncertain |
| Entirely inside $\pm 1$ | no difference larger than 1 point in this setup |
| Crosses 0 and a $\pm 1$ limit | inconclusive; neither a gain nor equivalence is shown |
| Below 0, crosses -1 | loss, size uncertain |
| Entirely below -1 | loss of at least about 1 point in this setup |

In the primary family, a gain or a loss must also survive Holm correction at 5%, or it reads as inconclusive.
Every wording appears with its interval. Tables may shorten a wording; public text uses it in full.

## 10. Statistics

- **Interval:** paired by seed. Each bootstrap replicate resamples patients, keeping all of a patient's images
  together, and seeds. Both arms and all seeds see the same draw.
- **Few seeds:** bootstrap intervals over few seeds can undercover. The reported interval takes the wider limit
  on each side of the bootstrap and a paired $t$ interval. In a simulation with a known answer (200 repetitions
  per row, so about $\pm 3$ points), the bootstrap alone held the answer in 90% of cases at 5, 8 and 10 seeds,
  and the reported interval in 97%, 94% and 93%, against the nominal 95%.
- **Sources:** the interval's width from seeds alone and from patients alone are both reported.
- **Holm:** the p-value of each contrast is the larger of the bootstrap's and the paired $t$ test's, because the
  bootstrap alone can undercover with few seeds. "No difference" is not a discovery, so Holm does not apply to
  it; each such claim is a two-sided 2.5% test.
- **Replicates:** 2,000 for each interval.
- **Seed count: 10,** seeds 2000 to 2009. It is set by the budget. The half-width from seeds alone is $t_{0.975, n-1} \cdot SD / \sqrt{n}$, with
  the paired SD seen in tuning. Three seeds give a rough estimate only: the true SD could be half or six times the
  value seen. At 10 seeds, a paired SD of 1.0, 1.4 and 2.1 gives 0.72, 1.00 and 1.50 points. Tuning's largest relevant
  value, chosen AUCReshaping against chosen weighted BCE, was about 2.1. Validation has 957 effusion images and test
  about 4,650, so the validation SD includes more threshold noise. Patient sampling widens the interval further. So a
  half-width of 1 point may not be reached, and "inconclusive" is a likely result on P2.
- **Failed runs:** every run counts. An infrastructure failure (a killed kernel) reruns on the same seed and is
  disclosed. A numerical failure is never rerun: it is shown, its seed leaves only the contrasts that need it, and
  each such contrast states which seeds it lost. The analysis refuses duplicate runs, and a check against the
  frozen list of planned runs finds missing ones.

## 11. Safeguards

- **Seeds:** pilot 1700 up, tuning 1800 up, synthetic laboratory 1900 up, confirmatory 2000 up.
- **Separation:** pilot and tuning runs do not load the test files, and a test pins this.
- **Coded arms:** confirmatory run records use coded arm names. The arm, the training loss, the boost statistics
  and the run times go to sealed files. The order of runs is shuffled, and the job list inside each kernel is
  encoded. The analysis is written and tested on made-up runs before any results exist, then frozen. Before
  unblinding, only completeness is checked. Unblinding happens once, and its date and commit are recorded.
- **Limits of the coding:** validation results per epoch stay open, because selection needs them. The sealed files
  are plain, and the job list is encoded, not encrypted. The author wrote the code and could decode the runs, so
  this guards against accidental choices, not deliberate ones.
- **Date:** this protocol's commit is tagged and released before the first confirmatory run. The tag keeps the
  commit and its date, even if the main branch is squashed later.
- **Interests:** the author wrote this library and wants the work to be seen.

## 12. Deviations from the paper

Page numbers are the paper's. Its code is not public, so "not stated" means not stated in the text.

| Paper | Here | Why, or what follows |
|---|---|---|
| Private data, 16,953 training images; effusion test set of 331 images, 74 with effusion (pp. 5-6) | Public NIH data; test set with about 4,650 effusion images | Public and rerunnable. The paper's 98% threshold rests on about 1.5 cases, against about 93 here. Not comparable with the paper's tables |
| Radiologist bounding boxes (p. 5) | Labels mined from reports | Section 3 states the risk |
| Fine-tuning on a subset of the pre-training data (p. 3) | Weights that never saw NIH | Otherwise positives are already fitted. The paper also tests ImageNet weights, which never saw its data (Table 3) |
| ImageNet SwAV ResNet-50 and others, 256 px, lr 0.01 (p. 8) | Chest X-ray DenseNet-121, 224 px, lr chosen on validation | Fine-tuning published weights; a secondary run checks the paper's setup |
| Histogram equalization and dynamic windowing of 10-bit DICOM (p. 8) | The weights' own preprocessing on 8-bit PNG | Windowing is not possible here. Equalization is possible, but was not used, to keep the weights' own input |
| Weighted sampler with class weights (p. 8) | Class weights only | Worse for CE on validation, on one seed. Not tested for AUCReshaping (section 5) |
| Lowest validation loss selects the epoch, "often within 10 epochs" (p. 8) | Highest validation sensitivity is primary; lowest validation loss sits in the same table; 10 epochs | The 10 epochs agree with the paper. The primary rule does not, because the target is sensitivity |
| Eq. 1, undefined at $p \le b$ (p. 4). A boost of 1 gives Table 3's highest value at 98%, 0.867 (p. 9), and Eq. 1 is then undefined for every sample, so the code cannot be literal Eq. 1 | Detached floor $r$, tuned | Sets most of the boost's strength |
| Boost values vary by task and table (pp. 9-11), set by a "piece-wise approximation" of distance (p. 4) | The boosts 0.4, 0.2, 0.1, 0.1 fixed, the floor tuned | At gentle floors the boost values barely matter |
| Several thresholds; how they combine is not stated | `band` and `sum` both in the grid | The rules differ widely in strength |
| Thresholds from the ROC curve on training samples (Fig. 2, p. 3); no method given | First-in, first-out window of recent training scores | Other strategies changed little in an earlier test |
| Boosts "modify the loss values or output logits" (p. 8), and amplify weights (p. 3) | "AUCReshaping" means Eq. 1, `prob` | Stricter than the paper's wording. `weight` and `logit` are named variants |
| Top 5 models averaged; baselines repeated 8-10 times, others 5 (p. 8) | Every run counts | Expect smaller gains than a top-5 average would give |
| Relative % changes (p. 9) | Absolute points | Relative changes at low sensitivity look large |

## 13. Secondary studies

Each runs only if the budget allows, in priority order, and is otherwise reported as cut. The primary arms take about 34.5
of the 60 GPU session hours set aside, and about 20 are already spent, so little GPU time remains.

No GPU quota needed:

1. Credit card: no momentum, repeated new splits, `band` and `sum`, weight on normal and on fraud, 98%, 99% and 99.99%,
   fresh seeds. A replication, not a primary claim. The same 492 frauds recur across splits, and its settings were read
   from test results of earlier CE runs.
2. Synthetic confirmation: fine-tuned, in the paper's regime and with the rare class boosted, fresh seeds. Its scenarios
   were chosen after seeing results, and any conclusion that reverses will be reported.

GPU, beyond the budget unless it is raised:

3. Frozen features, head only.
4. AUCReshaping and its control with the sampler or a much shorter threshold window (section 5).
5. Boosting disease.
6. The paper's setup: ImageNet SwAV at 256 px, lr 0.01, 3 seeds, descriptive.
7. A second set of weights (`mimic_ch`), with results per set.
8. The other named arms: same boosts on random positives, `weight`, `logit`, no extra gradient at the floor, focal, and
   partial AUC. The last three are not implemented yet.

## 14. Amendments

None yet.

## 15. Reproducing

```bash
uv sync --all-groups
python -m benchmarks.review.xray bundle --out DIR
python -m benchmarks.review.xray choose --results TUNING_DIR --out CHOSEN_FILE
python -m benchmarks.review.xray confirm-kernels --out DIR --weights pc --lr 1e-4 \
    --chosen CHOSEN_FILE --seeds 10 --salt-file PRIVATE_FILE
python -m benchmarks.review.xray_analysis integrity --results DIR --expected DIR/expected_runs.json
python -m benchmarks.review.xray write-codes --salt-file PRIVATE_FILE --out FILE
python -m benchmarks.review.xray_analysis analyse --results DIR --codes FILE --out FILE
pytest benchmarks/tests --no-cov -q
```

Runs execute on Kaggle kernels from `benchmarks/review/xray_prepare.py` and `benchmarks/review/xray.py`. The
data and weights are public. The weights' file hash is recorded above. Every run records the hash of the code and of its
prepared data files, and the analysis checks that they agree.
