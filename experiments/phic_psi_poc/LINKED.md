# LINKED — φc/ψ PoC Link Inventory

Mechanically generated file map of every `.md`/`.tex` file under `experiments/phic_psi_poc/` and what points to what.
This is a cross-reference audit, not a curated index — `experiment_index.md` remains the master index of what each file *is*; this file tracks whether the tree's internal links actually resolve.
Auto-generated: do not hand-edit.
Regenerate by re-running the extraction (markdown links, LaTeX `\input`/`\include`/`\includegraphics`, backtick/`\texttt{}` filename mentions, and bare filename mentions with `.md .tex .py .csv .png .pdf .yaml .yml .log .json` extensions) against the current tree and re-resolving each reference relative to (a) the referencing file's directory, (b) `experiments/phic_psi_poc/`, (c) the repo root, and (d) a repo-wide basename fallback search.

Scan scope: 64 `.md`/`.tex` files, 1016 extracted file-shaped references, 865 resolved (738 direct, 71 via unique repo-wide basename fallback, 56 ambiguous-but-real generic artifact names such as per-run `history.csv`), 6 confirmed genuinely broken (after filtering ~145 regex artifacts — glob patterns, code line-locators like `trainer.py:534`, and date-suffix fragments split off already-resolved filenames), 9 orphaned files never linked from anywhere else in the tree.

## File-by-file map

Path is relative to `experiments/phic_psi_poc/`.
"Out" = outbound file-shaped references that resolve (any target type: md/tex/py/csv/png/pdf/yaml/log/json).
"In" = number of *other* `.md`/`.tex` files in this tree that resolvably reference this file (0 = orphan, see below).

| File | Description | Out | In |
|---|---|---:|---:|
| `NOTES.md` | Running notes — running log of the whole investigation | 66 | 13 |
| `analysis_output/analysis_report_20260720_234304.md` | Prediction Analysis Report — 20260720_234304 | 0 | 10 |
| `assessment_lam0_ablation_2026-07-22.md` | Assessment — λ=0 Ablation (Run 8) and overall degeneracy status | 8 | 9 |
| `bootstrap_output/bootstrap_ang_mae_20260721_093533.md` | Bootstrap CI on ang_MAE — periodic heads | 0 | 7 |
| `cnn_attention_config_diff.md` | Section C — cnn_attention config diff and outlier investigation | 3 | 8 |
| `diagnostic_log.md` | Diagnostic Log — φc/ψ mode collapse investigation | 45 | 14 |
| `doc_update_sweep_handoff_2026-07-22.md` | Doc Update Sweep — Handoff (2026-07-22) | 14 | 2 |
| `experiment_index.md` | φc/ψ Degeneracy PoC — master experiment index | 98 | 6 |
| `experiment_summary_2026-07-22.md` | φc/ψ Degeneracy Investigation — full experiment summary | 23 | 7 |
| `inclination_loss_trace.md` | Inclination head — code-path trace | 0 | 7 |
| `inclination_output/inclination_control_stratification_20260723_140016.md` | Inclination-stratified chirp-mass MAE/R² — scalar-control check | 1 | 4 |
| `inclination_output/inclination_stratification_20260723_130630.md` | Inclination-stratified ang_MAE — all periodic heads | 0 | 5 |
| `lam005_retune_output/diagnostic_lam005_retune_20260722_142705.md` | λ=0.05 retune diagnostics — mechanical verdict | 1 | 2 |
| `lam005_retune_output/lam005_retune_report.md` | λ=0.05 retune — std_ratio stabilisation | 3 | 3 |
| `lam010_retune_output/diagnostic_lam010_retune_20260722_171025.md` | λ=0.10 retune diagnostics — mechanical verdict | 1 | 2 |
| `lam010_retune_output/lam010_retune_report.md` | λ=0.10 retune — std_ratio stabilisation | 2 | 7 |
| `lam0_ablation_output/lam0_ablation_report.md` | λ=0 ablation — circular loss drift | 1 | 4 |
| `paper/README.md` | Literature review PDFs — directory readme | 32 | 0 |
| `paper/paper1/000_abstract.tex` | Paper abstract | 0 | 1 |
| `paper/paper1/000_preamble.tex` | LaTeX preamble | 0 | 1 |
| `paper/paper1/001_introduction.tex` | Introduction section | 0 | 1 |
| `paper/paper1/002_problem-formulation.tex` | Problem formulation / targets and degeneracy hypothesis | 0 | 1 |
| `paper/paper1/003_prereq_study.tex` | Analytic prerequisite study section | 1 | 1 |
| `paper/paper1/004_methods.tex` | Dataset/methods section | 0 | 1 |
| `paper/paper1/005_diagnostics.tex` | Diagnostics section | 4 | 1 |
| `paper/paper1/006_results.tex` | Headline results section | 8 | 1 |
| `paper/paper1/007_pre-registered-lambda-retune.tex` | Pre-registered λ-retune section | 2 | 1 |
| `paper/paper1/008_discussion.tex` | Discussion section | 2 | 1 |
| `paper/paper1/009_conclusion.tex` | Conclusion section | 0 | 1 |
| `paper/paper1/010_appendix1.tex` | Appendix — claim-to-artifact traceability | 29 | 1 |
| `paper/paper1/main.tex` | Paper master file (title + `\input`/`\include` chain) | 12 | 0 |
| `perturbation_trace_output/perturbation_trace_20260723_091229.md` | Standalone multi-step perturbation trace (A.3) | 0 | 1 |
| `perturbation_trace_output/perturbation_trace_early_20260723_095357.md` | Perturbation trace — stage: early | 0 | 1 |
| `perturbation_trace_output/perturbation_trace_final_20260723_095054.md` | Perturbation trace — stage: final | 0 | 1 |
| `phic_psi_closing_punch_list.md` | φc/ψ investigation — closing punch list | 2 | 3 |
| `phic_psi_no_inclination_input_run_assessment.md` | Run 7 update assessment (no-inclination-input runs) | 2 | 1 |
| `phic_psi_run7_comparison.md` | φc/ψ PoC — Run 7 results comparison | 5 | 1 |
| `plan_iota_conditioning.md` | Plan: feed cos(ι) as model input instead of predicting it | 18 | 1 |
| `planning_files/phi_c_psi_degeneracy_poc.md` | φc/ψ degeneracy — story so far & vector-based PoC | 2 | 2 |
| `planning_files/phic_psi_implementation_plan_rev2.md` | Implementation plan, rev 2 (superseded) | 0 | 0 |
| `planning_files/phic_psi_implementation_plan_rev3.md` | Implementation plan, rev 3 (superseded) | 0 | 0 |
| `planning_files/phic_psi_implementation_plan_v4.md` | φc/ψ degeneracy handling — implementation plan (PoC), v4 | 3 | 2 |
| `poc_b_config_diff.md` | Section B — poc_b config diff and collapse investigation | 4 | 7 |
| `preregistration_lam_retune.md` | Pre-registered decision criterion — λ retune | 9 | 11 |
| `results.md` | φc/ψ Degeneracy PoC — results log | 6 | 6 |
| `reviewer_response_a3_2026-07-23.md` | Response to review — perturbation trace (A.3 closure) | 4 | 1 |
| `run7_verification_plan.md` | Run 7 / magnitude-penalty results — consolidated verification plan | 1 | 8 |
| `run7_verification_rebuttal.md` | Run 7 verification — rebuttal to Claude AI review | 8 | 7 |
| `snr_output/snr_stratification_20260721_094039.md` | SNR-stratified ang_MAE — all periodic heads | 0 | 6 |
| `std_ratio_trajectories.md` | std_ratio full trajectories — Run 7 (λ=0.01) | 3 | 9 |
| `tanh_to_linear_postmortem.md` | Tanh → linear fix — post-mortem analysis | 3 | 7 |
| `thesis/chapter_phic_psi_degeneracy.md` | Thesis chapter (Markdown twin) | 33 | 2 |
| `thesis/chapter_phic_psi_degeneracy.tex` | Thesis chapter (LaTeX twin) | 34 | 1 |
| `thesis/reviews/deepseek_ai_adversarial_review.md` | Round-1 adversarial review — DeepSeek | 0 | 0 |
| `thesis/reviews/deepseek_ai_adversarial_review_v2.md` | Round-2 adversarial review — DeepSeek | 0 | 0 |
| `thesis/reviews/deepseek_ai_adversarial_review_v4.md` | Round-4 (final) adversarial review — DeepSeek | 0 | 1 |
| `thesis/reviews/kimi_ai_adversarial_review.md` | Round-1 adversarial review — Kimi | 0 | 0 |
| `thesis/reviews/kimi_ai_adversarial_review_v2.md` | Round-2 adversarial review — Kimi | 0 | 0 |
| `thesis/reviews/kimi_ai_adversarial_review_v3.md` | Round-3 adversarial review — Kimi | 0 | 4 |
| `thesis/reviews/kimi_ai_adversarial_review_v4.md` | Round-4 adversarial review — Kimi | 0 | 1 |
| `thesis/reviews/qwen_ai_adversarial_review.md` | Round-1 adversarial review — Qwen | 0 | 0 |
| `thesis/reviews/qwen_ai_adversarial_review_v2.md` | Round-2 adversarial review — Qwen | 2 | 0 |
| `thesis/reviews/qwen_ai_adversarial_review_v4.md` | Round-4 adversarial review — Qwen | 0 | 1 |
| `thesis/reviews/v4_correction_checklist.md` | V4 adversarial review — correction checklist | 5 | 4 |

## Broken or missing references

Six confirmed genuine broken references (target does not exist anywhere in the repo, checked relative-to-file, relative-to-`experiments/phic_psi_poc/`, relative-to-repo-root, and via a repo-wide basename search).
Filtered out of this list: ~145 raw regex hits that were not genuine distinct references — LaTeX `\ref{sec:...}`/`\ref{fig:...}`/`\ref{tab:...}` (internal cross-reference labels within the same document, not file paths — see note below), glob patterns in prose (`thesis/reviews/*_v4.md`, `config_*.yaml`), source-line locators (`trainer.py:534-544`), and date-suffix fragments that were the tail of an already-resolved filename (`...2026-07-22.md` mis-tokenized as `07-22.md`).

| Referencing file | Reference text | Where |
|---|---|---|
| `phic_psi_no_inclination_input_run_assessment.md` | `analyse_phic_distributions.py` | Line 6 — "The `analyse_phic_distributions.py` script still needs to be re-run on Run 7 outputs" |
| `planning_files/phic_psi_implementation_plan_v4.md` | `phic_psi_inclination_investigation.md` | Line 4 — "Companion to `phi_c_psi_degeneracy_poc.md` and `phic_psi_inclination_investigation.md`" |
| `planning_files/phic_psi_implementation_plan_v4.md` | `joint_invert.py` | Lines 44, 253, 445 — referenced as living in the (also-missing) investigation doc |
| `planning_files/phic_psi_implementation_plan_v4.md` | `invert_test2.py` | Line 253 |
| `planning_files/phic_psi_implementation_plan_v4.md` | `degeneracy_check.py` | Line 445 |
| `thesis/reviews/qwen_ai_adversarial_review_v2.md` | `evaluation.py` | Line 33 — "As long as this matches the exact formula implemented in your `evaluation.py` script"; no file of that name exists (closest candidates are `scripts/evaluate.py` and `evaluate_poc.py`, neither named `evaluation.py`) |

Note on the four `planning_files/phic_psi_implementation_plan_v4.md` items: `phic_psi_inclination_investigation.md` appears to be the source planning document for the iota-conditioning successor work, and the three scripts (`joint_invert.py`, `invert_test2.py`, `degeneracy_check.py`) are described as living inside it — so all four missing references are one underlying gap, not four independent ones.

Note on `\ref{}`: the task brief asked to check LaTeX `\ref{}` targets as potential file references.
In this tree every `\ref{...}` and `\S\ref{...}` targets an internal label (`sec:phicpsi:*`, `tab:phicpsi:*`, `fig:phicpsi:*`) defined by `\label{}` elsewhere in the same document, not a file path — none were file references, so none appear in the table above.

## Orphaned files

Nine `.md` files exist under `experiments/phic_psi_poc/` but are never resolvably referenced (per-file link, not directory/glob mention) from any other file in the tree — not even from `experiment_index.md` or `NOTES.md`.

| File | Why it's orphaned |
|---|---|
| `paper/README.md` | Directory-local readme for `paper/`; nothing in the tree links back to it (it links out to 32 literature-review PDFs, but has no inbound link) |
| `planning_files/phic_psi_implementation_plan_rev2.md` | Superseded predecessor of `phic_psi_implementation_plan_v4.md`; no surviving doc points back to rev2 |
| `planning_files/phic_psi_implementation_plan_rev3.md` | Superseded predecessor of v4; same situation as rev2 |
| `thesis/reviews/deepseek_ai_adversarial_review.md` | Round-1 review, no-suffix. Only reachable via `experiment_index.md`'s directory-level link `thesis/reviews/` with the parenthetical "(the three review files)" — not a per-file link |
| `thesis/reviews/deepseek_ai_adversarial_review_v2.md` | Round-2 review. Only reachable via `experiment_index.md`'s glob-pattern prose mention `thesis/reviews/*_v2.md` — not a resolvable per-file link |
| `thesis/reviews/kimi_ai_adversarial_review.md` | Round-1 review; same directory-link-only situation as deepseek round 1 |
| `thesis/reviews/kimi_ai_adversarial_review_v2.md` | Round-2 review; same glob-only situation as deepseek v2 |
| `thesis/reviews/qwen_ai_adversarial_review.md` | Round-1 review; same directory-link-only situation |
| `thesis/reviews/qwen_ai_adversarial_review_v2.md` | Round-2 review; same glob-only situation |

Two different flavors of orphan here, worth distinguishing: `paper/README.md` and the two `rev2`/`rev3` planning files are **true** orphans (zero mention anywhere, superseded or standalone).
The six round-1/round-2 adversarial reviews are **soft** orphans — `experiment_index.md` narrates them in prose (directory link + glob pattern), so a human reader would find them, but no single click or exact-path reference resolves to any one of the six individually.
By contrast the round-3 Kimi review, and the round-4 reviews (all three reviewers), each do have a direct per-file link and are not orphaned.

## Recently resolved

**`planning_files/phi_c_psi_degeneracy_poc.md`** — until this session it was referenced by two files but did not exist in the repo:
- `NOTES.md` line 4: `` `phi_c_psi_degeneracy_poc.md` `` (bare filename, no path — resolves via repo-wide basename fallback to `planning_files/phi_c_psi_degeneracy_poc.md` since it's unique)
- `planning_files/phic_psi_implementation_plan_v4.md` line 3 and line 378: `` `phi_c_psi_degeneracy_poc.md` `` (same directory, resolves directly)

`git status` confirms the file is now present as an untracked addition (`?? experiments/phic_psi_poc/planning_files/phi_c_psi_degeneracy_poc.md`).
Both referencing locations now resolve correctly against the current tree — this is a **recently resolved** case, not a still-broken one.
No other file in the tree references it, so it is not orphaned (in-referenced-by count of 2, per the table above) but it also has no *inbound* link from `experiment_index.md` itself — worth a follow-up row there if the master index is meant to be exhaustive.

No other "referenced but absent, now present" cases were found in this sweep; the six items in **Broken or missing references** above remain genuinely unresolved as of this scan.
