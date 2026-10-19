---
title: "Can a regression neural network recover $\\varphi_c$ and $\\psi$ from gravitational-wave strain?"
subtitle: "Summary of findings"
author: "Syed Ali Mohsin Bukhari"
date: "2026-09-30"
header-includes:
  - \usepackage{tcolorbox}
  - \usepackage{microtype}
  - \usepackage{amssymb}
  - \usepackage[margin=1.0in]{geometry}
---

**Manuscript**: *On the Recoverability of Coalescence Phase and Polarization Angle from Gravitational-Wave Strain: A Systematically Verified Null Result*.

Section numbers below refer to the manuscript.

## The question

This work asks whether a neural network trained by direct regression can recover two extrinsic parameters of a compact-binary signal, the coalescence phase $\varphi_c$ and the polarization angle $\psi$, which are rarely treated as regression targets.
For the dominant quadrupole mode both enter the strain doubled, so in the face-on limit only the combination $2\varphi_c \pm 2\psi$ is constrained and the individual angles cannot be told apart.
They separate only partially as the system moves toward edge-on.
Bayesian samplers and neural posterior estimators handle this naturally, because a posterior does not have to commit to a single value.
A point-estimating regression network has no such option.
We asked whether such a network can recover $\varphi_c$ and $\psi$ from two-detector strain at all, either individually or through the better-conditioned sum and difference combinations.

## What we did

The population is 25,000 training and 5,000 validation simulated events (IMRPhenomD, dominant mode only), each a 2 s window at 2048 Hz for H1 and L1.
Periodic targets are regressed on the unit circle with a circular loss, so a random guess scores exactly 1 and the angular mean absolute error of a constant prediction is $\pi/2$ for $\varphi_c$ and $\pi/4$ for $\psi$.
Chirp mass, merger time and SNR serve as positive controls; sky position is trained through the same pipeline but proves a weak control (see below).
An analytic study first confirmed the corrected degeneracy structure: a self-check that $(R, \delta)$ stays constant along $2\varphi_c + 2\psi = \text{const}$ at face-on passed to machine precision (largest standard deviations $1.16\times10^{-14}$ and $5.87\times10^{-16}$), and the preference for the well-constrained combination falls from about $1.62$–$1.65\times$ face-on to $1.04$–$1.07\times$ edge-on.

We screened five trunk architectures (temporal convolutional network, plain CNN, convolution-transformer, InceptionTime, ResNet) in seven configurations.
All seven sat at the random-guessing level on $\varphi_c$ and $\psi$ in this first round, so the screen could not discriminate on the target question.
We down-selected on the positive controls instead (validation $R^2$ on chirp mass, merger time and SNR), keeping the Tcn and the attention model.
The full analysis therefore covers four models drawn from two distinct trunks.
Two head parameterizations were compared, an independent-angle baseline and an explicit combination-space construction, and a radial magnitude penalty was applied at three strengths ($\lambda = 0.01, 0.05, 0.10$) on the Tcn models.

## What we found

The answer is a null for regression networks.
Every configuration converges to the random-guessing baseline on both angles, while the same shared features recover chirp mass ($R^2 = 0.92$–$0.96$), merger time ($0.91$–$0.93$) and SNR ($0.74$–$0.79$) well.
Two of the four models are outright mode-collapsed on $\psi$ or $\varphi_c$, with predictions concentrated in a single angular bin.
The other two are noisy and uninformative rather than collapsed.
A label-permutation bootstrap with 10,000 shuffles finds none of the eight $\varphi_c/\psi$ tests significant, and most are on the wrong side of chance.

Two secondary findings came from looking further than the main question required.
First, the attention model shows genuine partial sky-localization skill: a $12.6^\circ$ improvement over the shuffled null ($z = +23$), with error falling monotonically with SNR ($80.2^\circ$ to $74.5^\circ$), while a predictor that always guesses the mean direction scores only about $89.4^\circ$.
The leading explanation is cross-detector amplitude-ratio information, which we have not yet tested directly.
Second, the inclination head, which we had been using as a noise-floor calibration, is neither clean noise nor a small recoverable signal.
Its predictions cluster at a handful of preferred angles, a partial mode collapse whose severity tracks each model's $\varphi_c/\psi$ collapse, which points to shared training dynamics rather than recoverable physics.
The attention model also memorizes $\varphi_c/\psi$ in the training split (train–validation loss gap of about 0.44–0.47 against 0.003–0.05 for the other three models) without generalizing any of it.

## How strong is the evidence

This is the part we most want scrutinized.
We pre-registered a metric-health gate, a check that the raw head outputs have a healthy magnitude, before running the retune at higher penalty strengths.
That gate never cleared: all twelve readings across the three penalty values fail or, in one case, pass numerically but fail on inspection of the predicted-versus-true scatter plot.
The one nominal pass ($\varphi_c$ of the combination model at $\lambda = 0.10$) clears every threshold, yet its scatter plot shows two flat bands at a single angle, a literal two-point collapse present from early training; the gate measures the magnitude of the raw outputs, not their angular diversity.
We therefore filed the pre-registered path as uninterpretable, not as a null and not as counter-evidence.

The null instead rests on four checks that do not depend on the gate:

- **Prediction inspection.** Histograms of the predicted angles, to see whether predictions collapse into a few bins.
- **Gradient trace.** A code-level check that the training signal reaches the angular heads end to end.
- **Permutation bootstrap.** Whether angular error beats that of shuffled labels.
- **Stratification.** Recovery by SNR tercile and inclination band, with no band clearing a pre-set 0.10 rad materiality floor.

These were run uniformly on all four models, without excluding any on interpretability grounds.
We argue that this is a legitimate substitute, but it is a substitute.
It was assembled after the pre-registered route failed, so it carries less confirmatory weight than a fully pre-registered test would.
The gate's failure also yielded a methodological lesson we report as a contribution: passing an interpretability gate is not proof that the property it stands for holds.

## Limitations and open items

- **Narrow architecture pool.** Only two trunk families carry the analysis, and the down-selection used positive-control $R^2$ without a numeric threshold. The three dropped trunks were never run through the full battery.
- **Single seed.** Every configuration uses seed 42, so the null is established across configurations, not across seeds.
- **Reconstructed branch.** The combination model's individual-head bootstrap numbers rely on ground-truth-assisted branch reconstruction, which does not change the verdict but is disclosed.
- **Probe parked.** A perturbation-trace probe failed its own positive control on every model and is parked, not closed.
- **Unrun checks.** A synthetic-data ablation, a high-SNR ($25$–$30$) validation set or Fisher-matrix bound to separate degeneracy from a sensitivity floor, dataset regeneration from a version-controlled generator, and a joint $(\hat\varphi_c, \hat\psi)$ scatter to test for the degenerate ridge were all not run.
- **No adversarial review yet.** The manuscript states that it has not been through independent adversarial review.

## Positioning and questions for review

This does not reopen a settled question.
Neural posterior estimation (Vitamin, DINGO, DINGO-IS) already represents degenerate targets correctly by construction.
What the study documents is a practical hazard for point-estimate and hybrid pipelines still built on regression heads, and the diagnostic discipline needed to notice the failure.

Every quantitative claim traces to a repository artifact through the manuscript's claim-to-artifact appendix.
