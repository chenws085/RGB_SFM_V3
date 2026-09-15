# Experiment design: object-aligned token clustering and attention-head specialization

## 1. Claims and research questions

The study should separate representational analysis from model-performance claims.

**RQ1 — alignment:** Does object alignment make corresponding hierarchical-ViT tokens more invariant and semantically consistent under translation, scaling, and background changes?

**RQ2 — hierarchy:** At which stage/block do object and part semantics emerge, and how are they changed by `FlexiblePatchMerging`?

**RQ3 — head specialization:** Do individual heads encode distinct, label-relevant information that has unique predictive value?

**RQ4 — generalization:** Do conclusions found on controlled shapes or faces replicate on another real, part-annotated dataset?

Primary hypotheses:

- H1: oracle-mask or oracle-box alignment improves cross-transform AMI and matched-token cosine similarity over no alignment.
- H2: predicted alignment narrows the gap to the oracle condition without reducing classification accuracy materially.
- H3: at least some heads have label relevance and positive leave-one-head-out utility after correction for multiple testing.
- H4: semantic consistency changes systematically across stages and near merge boundaries.

## 2. Experimental units and leakage control

- The independent unit is the **source image/object**, not an individual token.
- Split source objects into train/validation/test before generating transformed variants. Every variant of one source remains in the same split.
- Fit scalers, PCA (if used), K-means centroids, probes, and hyperparameters on train/validation only. Report final metrics once on test.
- Use at least 3 independently trained checkpoints. For each checkpoint and clustering condition, use 20 K-means initializations.
- Keep exact source IDs, transform parameters, checkpoint hash, extraction layer, head, token coordinates, K-means seed, and alignment output in a long-format results table.

## 3. Datasets

### Tier 1: controlled paired data (debugging and causal stress test)

Start with the repository's `MultiColorShapes`, `MultiEdgeShapes`, or `MultiGrayShapes` loaders, extended to return an object mask and part/shape labels. Generate paired variants of each held-out source:

- translation: x/y offsets at 0, ±0.5, ±1, and ±2 patch widths;
- scale: 0.60, 0.80, 1.00, 1.25, and 1.50;
- background: constant, low-frequency texture, and natural-image background;
- optional rotation: −30°, −15°, 0°, 15°, and 30°.

This tier provides exact correspondences, masks, boxes, centroids, and transforms.

### Tier 2: face-part replication

Use CelebAMask-HQ masks for skin, eyes, eyebrows, nose, mouth/lips, ears, hair, and accessories. Construct the split by identity if identity metadata is available; otherwise check for near-duplicates before random splitting. Avoid making the face dataset the only evidence because faces are unusually well aligned.

### Tier 3: cross-domain replication

Use a manageable, preregistered subset of PartImageNet with common part labels and sufficient instances per category. Select categories before inspecting head-level results.

## 4. Models and checkpoints

Primary model: repository `MergingViT` with its exposed `individual_heads_output` and `pre_projection_features`.

Controls:

- same frozen MergingViT checkpoint, all alignment conditions (isolates analysis/alignment effects);
- parameter-matched non-hierarchical ViT if feasible;
- existing Swin and PVTv2 implementations as hierarchical references;
- optional self-supervised DINO features as a representation upper/reference baseline, not as a directly matched classifier.

Always report classification accuracy, balanced accuracy, and prediction consistency across paired transforms. An explanation method should not be credited for semantic stability if the underlying prediction is unstable.

## 5. Alignment conditions

Evaluate these in the stated order so each added source of error is visible:

| ID | Condition | Purpose |
|---|---|---|
| A0 | no alignment; fixed physical grid | proposal baseline |
| A1 | ground-truth bounding-box crop and rescale | oracle geometric upper bound |
| A2 | ground-truth mask token selection | oracle background-removal upper bound |
| A3 | CV mask/edge → centroid + PCA axis → affine warp | classical, model-free alignment |
| A4 | SAM mask → box/mask normalization | predicted segmentation path |
| A5 | attention-rollout centroid + scale estimate | self-alignment path |
| A6 | spatial transformer module | optional learned-alignment extension |

For A2/A4, map masks to each stage by **fractional token overlap**, not nearest-neighbor center membership alone. Predefine an inclusion threshold (for example, ≥0.5) and include a 0.25/0.75 sensitivity analysis.

For every path, save the estimated transform and an overlay. Measure alignment error against oracle centroid, box, and mask IoU so downstream failures can be attributed to alignment or representation.

## 6. Feature extraction and canonical correspondence

For each stage and block, save:

- pre-attention normalized tokens;
- attention matrices per head;
- per-head output before concatenation/projection (`individual_heads_output`);
- post-attention and post-MLP tokens;
- post-merge tokens and their receptive-field coordinates.

Maintain a token-to-input-coordinate map through every merge. Map transformed tokens to canonical object coordinates using the known/or estimated inverse transform. Only compare tokens whose receptive fields overlap the same canonical region.

Run both raw-feature and per-stage standardized-feature analyses. Normalize vectors before cosine or spherical K-means. Do not concatenate absolute x/y coordinates into the primary clustering feature; that would recreate the position-binding problem. Coordinate-augmented clustering can be a named ablation.

## 7. Clustering protocol

Primary protocol:

1. pool training tokens for one checkpoint × stage × head × alignment condition;
2. optionally reduce dimension with train-fitted PCA retaining 95% variance;
3. fit mini-batch K-means with 20 initializations;
4. freeze the scaler/PCA/centroids and assign validation/test tokens;
5. evaluate on held-out objects and their paired transforms.

Set primary `K` to the number of evaluable semantic parts in the selected dataset. Run `K ∈ {2, 4, 8, 16, 32}` only as a sensitivity analysis. Include spherical K-means and TokenCut/spectral clustering as secondary baselines.

Internal metrics are descriptive, not proof of semantics:

- Silhouette (higher);
- Davies–Bouldin (lower);
- Calinski–Harabasz (higher);
- stability across K-means initializations.

External/semantic metrics:

- AMI and ARI against part labels;
- Hungarian-matched cluster-to-part mIoU and macro F1;
- cluster purity with class-balance reporting;
- held-out linear-probe macro F1 from token/head features.

## 8. Primary alignment endpoints

For each paired original/transformed image, after canonical matching:

- **primary:** AMI between the two cluster-assignment maps using one shared, train-fitted clustering model;
- matched-token cosine similarity and normalized Euclidean distance;
- linear CKA between token matrices (image-level value);
- part-label AMI/mIoU on each variant;
- prediction consistency and confidence change;
- foreground/background leakage: fraction of selected tokens outside the oracle mask.

Analyze translation, scale, and background separately before a combined corruption score. Report effect curves versus transform magnitude, not only an average.

## 9. Head specialization: three required tests

### 9.1 Difference between heads

- pairwise AMI/ARI of cluster assignments;
- Jensen–Shannon divergence between normalized attention distributions;
- linear CKA between per-head output representations.

Low similarity establishes difference only.

### 9.2 Semantic relevance

- per-head held-out part-label AMI/mIoU;
- per-head linear-probe macro F1;
- cluster × part Cramer's V, with the null distribution obtained by shuffling part maps at the **image level** (or by a spatially valid permutation), not by treating tokens as IID.

### 9.3 Unique utility / complementarity

- leave one head out by zeroing or mean-substituting that head before output projection; record change in class accuracy and part-probe F1;
- compare best single head, all heads, and all-minus-head probes under identical cross-validation;
- call a head “specialized” only when it is stable across checkpoint seeds, semantically relevant, and adds unique held-out utility.

Use mean-substitution and cross-image shuffling controls in addition to zero ablation, because zero can create out-of-distribution activations.

## 10. Statistical analysis

- Define one primary stage or a stage-averaged endpoint before testing. Treat remaining stage/head analyses as secondary.
- Use paired image-level bootstrap confidence intervals for A1–A5 versus A0.
- Fit a mixed-effects model where feasible: metric ~ alignment × transform_type × magnitude + stage, with random intercepts for source image and checkpoint.
- For cluster/part association, use image-level permutation tests and report Cramer's V plus confidence intervals.
- Correct all head × stage × metric families with Benjamini–Hochberg FDR. A raw `p < 0.05` across dozens of heads is insufficient.
- Report effect sizes and confidence intervals even when not significant.
- Predefine failed-run handling, minimum foreground-token count, and exclusions.

## 11. VLM semantic-consistency stage

Treat this as secondary corroboration, not the primary endpoint.

- Select cluster exemplars by medoid distance from held-out samples; never hand-pick them.
- Use a fixed montage size/order, model version, prompt, decoding settings, and random seed where supported.
- Ask for a constrained label distribution over the known part vocabulary plus an “uncertain/other” option.
- Score label agreement, entropy, and consistency across prompt paraphrases.
- If free-text captions are retained, embed them with a separately fixed text encoder and compare within-cluster versus between-cluster cosine similarity.
- Avoid using the same VLM to generate captions and judge those captions as the only evidence. Add CLIPScore/image-text compatibility and a blinded human audit of a stratified subset.

## 12. Ablation matrix

Minimum ablations, performed after the primary experiment:

- positional embedding on/off or absolute versus relative position;
- foreground-only versus foreground+background tokens;
- raw versus standardized features;
- Euclidean versus spherical K-means;
- `merge_size` schedules supported by `FlexiblePatchMerging`;
- attention rollout versus Chefer relevance for A5;
- oracle mask/box versus predicted mask/box;
- cluster count K;
- pre-merge versus post-merge representations.

Do not run every combination as one huge factorial grid. Screen on Tier 1, lock the best two predicted alignment methods, then confirm on Tiers 2–3.

## 13. Execution phases and stop/go gates

### Phase 0 — instrumentation validation

- Verify tensor shapes and coordinate maps with unit tests.
- On a synthetic single-object image, confirm that a one-patch translation produces the expected canonical token correspondence.
- Gate: median oracle coordinate error ≤0.25 patch and deterministic extraction on repeated runs.

### Phase 1 — controlled alignment study

- Run A0–A3 on Tier 1 across all transform magnitudes.
- Gate: at least one oracle alignment improves primary AMI with a useful effect size; otherwise debug the token mapping before SAM/VLM work.

### Phase 2 — predicted/self alignment

- Add A4–A5, measure alignment error, and compare with oracle gaps.
- Gate: predicted method recovers a preregistered fraction of the oracle gain without material accuracy loss.

### Phase 3 — head and stage study

- Run the three-part specialization tests on the locked alignment condition.
- Produce stage × head heatmaps with uncertainty and corrected significance markers.

### Phase 4 — real-data confirmation

- Replicate locked analyses on CelebAMask-HQ and the PartImageNet subset.
- No tuning based on test-set head identities.

### Phase 5 — VLM corroboration

- Analyze medoid montages using the frozen rubric and audit protocol.

## 14. Required result tables/figures

1. Dataset/split/source-object counts and part-label balance.
2. Model/checkpoint accuracy, parameter count, and transform consistency.
3. Alignment error and runtime by A0–A5.
4. Primary AMI and matched-token similarity by transform magnitude with 95% CIs.
5. Semantic AMI/mIoU by stage, pre/post merge.
6. Head pairwise AMI, JSD, and CKA matrices.
7. Per-head semantic relevance and leave-one-out utility with FDR correction.
8. Oracle-to-predicted alignment gap across datasets.
9. VLM/human agreement and prompt-sensitivity appendix.

## 15. Suggested repository integration

- Add source IDs, transform metadata, masks, and canonical transforms to dataset outputs.
- Extend `models/MergingViT.py` hooks to save attention matrices and named pre/post-block tensors without overwriting the previous batch.
- Keep extraction separate from analysis: one immutable feature store, then reproducible clustering/statistics scripts.
- Extend `run_experiment_pipeline.py` with a config file for dataset split, alignment ID, stages/blocks, checkpoint seed, and feature-store path.
- Write tidy outputs under `results/<study_id>/`: `run_manifest.json`, `image_metrics.parquet`, `token_assignments.parquet`, `head_metrics.csv`, and figures.

