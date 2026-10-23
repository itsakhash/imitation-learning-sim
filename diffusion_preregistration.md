# Diffusion policy study: pre-registration

**Status: draft for review. It becomes the pre-registration when it is committed, and the commit must come before any pilot or study network is trained.** Written 2026-10-22. No diffusion code exists yet.

## Question

Does a diffusion policy, written by me in plain PyTorch, improve on an ordinarily trained action-chunking network when both use the same convolutional trunk, the same data, the same number of gradient steps, the same averaging of weights (EMA) and the same number of executed targets?

## What stays fixed from the earlier studies

- The split (165 training and 41 validation episodes, split seed 0), the 50 evaluation episodes on seeds 10000 to 10049, the 300-step cap, the metrics (mean max coverage as the primary metric, mean final coverage, success with a Wilson interval), the 9-decimal rounding, and the run-twice reproducibility check (separate processes, all 50 episodes agree on steps and success and to within 1e-6 on coverage).
- The unit of analysis is the network. Intervals are percentile bootstrap intervals (10,000 resamples, the fixed-seed convention of `seed_study.py`) that resample the networks of each group independently.
- Success rate is reported but is never the headline.

## Training path

All arms train on GPU-resident data (`resident_data.py`, `resident_train.py`), checked on 2026-10-22: batches equal `ChunkDataset`'s on the real data, the epoch order equals `DataLoader`'s over 3 epochs, and in an anchor run the validation loss was identical to five digits at epoch 1 and then diverged as the original loop does against itself. The anchor's verdict rule was too lenient to be strong evidence; the path is treated as equivalent up to GPU nondeterminism, not as an exact reproduction.

Every arm trains for 370 epochs (29,970 steps) at batch 256, with Adam, a constant learning rate, no weight decay, no augmentation, and no checkpoint selection (the last step is used), except arm P below. Arms D, C and B keep an exponential moving average of the weights with decay 0.999 (an untuned choice of mine) and are evaluated with the averaged weights.

## Arms

Training seeds 0 to 9 first (block 1). Training seeds 10 to 19 (block 2, the replication) are planned now, with the same text, and are run whatever block 1 shows. Block 2 is reported separately and never pooled with block 1.

- **D, diffusion.** The convolutional trunk of `ChunkPolicyNet` layer for layer; its 2,304 features plus the agent position (scaled to [0, 1] as now) condition a denoiser: an MLP of residual blocks (width 256, 4 blocks, my choice, not tuned) whose layers are modulated by FiLM from a sinusoidal timestep embedding (64 dimensions) and the conditioning vector. The 16x2 action chunk is scaled to [-1, 1] by x = 2a - 1 with a in [0, 1] (fixed constants, nothing taken from the data). DDPM with noise (epsilon) prediction, T = 100 training steps, squared-cosine noise schedule, MSE loss over the valid chunk positions only (positions padded past an episode's end are excluded, as in `chunk_loss`). At evaluation: DDIM with 10 steps and eta 0, one sample per prediction, executing 8 of the 16 targets. The initial noise comes from a generator seeded from the episode's evaluation seed, so an evaluation can be repeated exactly. The schedule, T, the number of sampling steps and the architecture sizes come from my memory of published setups and are not tuned.
- **C, chunked control.** `ChunkPolicyNet` with the MSE `chunk_loss`, executing 8 of 16. This is the comparison for the primary hypotheses.
- **B, baseline control.** The one-step baseline network of `train_bc.py` (718,178 parameters), executing 1 target, trained on resident data with chunk length 1. A test must first show that those targets equal the baseline dataset's.
- **P, path check.** The original chunked recipe on the resident path: 30 epochs, no EMA, learning rate 1e-3, training seeds 0 to 9 (and 10 to 19 in block 2). Descriptive only.

## Learning-rate pilot

For each of D, C and B, three learning rates (1e-3, 3e-4, 1e-4) are tried once on a pilot network with training seed 99 (outside seeds 0 to 19), at the full 370 epochs. The rate with the lowest first-action error in pixels on the 41 validation episodes at the last step (EMA weights; for D, one DDIM sample per frame with a fixed seeded generator) is used for all 20 networks of that arm. Nothing else is tuned. No simulator episode on seeds 10000 to 10049 is run in the pilot; any smoke test uses seeds 20000 and above. The pilot is nine trainings of about 2 to 3 minutes each (unmeasured estimate). Its limitation: validation error against human actions penalises a multimodal policy, so it may favour rates that suit C and B over D.

## Hypotheses and verdict rules

**H1 (primary, block 1):** diffusion D compared with the chunked control C. For each of mean max coverage and mean final coverage: the difference between the group means (D minus C), with a 95% bootstrap interval over the 10 networks of each group. Verdict, exactly as in the seed study: "supported" if both intervals lie entirely above zero, "contradicted" if either lies entirely below zero, otherwise "inconclusive".

**H2 (block 2):** the same comparison and rule on the 10 new networks per arm, as a replication.

**Descriptive, no verdict rule:** D against B; C against B; coverage given back (max minus final); the number of networks ending above the do-nothing level of final coverage (Wilson intervals); the failure categories of the failed episodes by replay, with the thresholds of `failure_categories.py` unchanged; C against the original 30-epoch chunked networks (confounded by steps and EMA); P against the original chunked networks of the same seeds (it can rule out large differences only, with ten networks per side); validation curves of every arm.

## What I will and will not claim

I will report whichever way H1 and H2 come out, including a negative or inconclusive result and any divergence between blocks. I will not claim any mechanism, claim success rates, call the tuned learning-rate selection optimal, or compare with the original 30-epoch networks as if the recipe were identical.

## Implementation details fixed before any training (added 2026-10-23, after the text above was approved)

These were not specified above. They are fixed in `diffusion_core.py` and `diffusion_model.py`, which are committed before any pilot or study network is trained. All come from my memory of published setups and none is tuned.

- DDIM clips the predicted clean chunk to [-1, 1] at every step and recomputes the noise from it.
- The 10 DDIM steps are the training-step indices 99, 88, 77, 66, 55, 44, 33, 22, 11, 0 (evenly spaced and rounded); the last step ends at signal fraction 1. Training steps are drawn uniformly from 0 to 99, and the schedule's per-step noise is capped at 0.999.
- The training step and the noise are drawn on the CPU from a seeded generator, so a training run is repeatable up to GPU arithmetic.
- The denoiser encodes the image and agent position once per prediction into a 256-dimensional vector; the step index goes through a 64-dimensional sinusoidal embedding and one linear layer with SiLU; both enter only through FiLM (a scale and a shift) in each of the 4 residual blocks of width 256. Nothing is zero-initialised. The parameter count is recorded in the README from the test run.

### Training and evaluation details (added 2026-10-23, in `arm_train.py`, `diffusion_policy.py` and `evaluate_diffusion.py`, committed before any pilot or study network is trained)

- The EMA has a constant decay of 0.999, is updated after every optimiser step, has no warm-up, and starts from the initial weights. The averaged weights are what is validated, saved and evaluated; the raw weights are not kept. Arm P has no EMA.
- Validation is done every 10 epochs and at the last epoch (every epoch for P). For C and B it is the masked mean squared error and the first-action error in pixels. For D it is the denoising loss with step indices and noise from a generator seeded with 777 (the same every time), and the first-action error from one seeded DDIM sample per frame. These curves are descriptive. The only validation number used for a decision is the pilot's, at the last epoch.
- Shuffling uses a generator seeded with the training seed, as in `resident_train.py`. For D the step indices and noise of the training loss come from a second CPU generator seeded with the training seed plus 1,000,003.
- At evaluation the noise of each episode comes from a CPU generator seeded with that episode's evaluation seed, and each prediction takes the next draw. The sampling schedule used is the one stored in the checkpoint.
- Networks are named `diff_cnn` (D), `chunk_long` (C), `bc_long` (B) and `chunk_res30` (P), with `_s{k}` appended for training seed k (none for seed 0). The pilot uses the same names with a `pilot` prefix, and a checkpoint is never overwritten without an explicit flag.

## Rules during the study

- No network is added, dropped or retrained after its results are seen. A network that fails its reproducibility check is reported and stays in the study.
- If the diffusion evaluations do not reproduce on the GPU, inference for every diffusion evaluation is run on the CPU and the README says so.
- Any deviation from this text is stated in the README and, if it affects a definition, everything affected is rerun.
- Multi-hour runs are started only with the owner's permission.

## Tests written before the code

Noise-schedule properties (monotone, between 0 and 1, correct end values); the forward-process identity; DDIM sampling being deterministic given a generator; the masked loss ignoring padded positions; a deliberately broken loss or data alignment making a test fail; the conditioning actually changing the denoiser's output; the EMA update arithmetic; the baseline-target equality test.

## Cost (estimates, not measurements)

About 2 to 3 minutes of training per network, 30 networks per block (D, C, B), about an hour or more per block; P adds seconds per network. Evaluation is about a minute per run and each is run twice, so roughly an hour and a half per block single-process, about a third of that with four parallel jobs. I will time one network of each arm before planning a block.
