# Reference papers for the hierarchical ViT study

This folder is a curated reading set for the proposal in `plan/0909 - HackMD.pdf`.
The selection targets the proposal's two central problems:

1. fixed-grid tokens are not semantically aligned after translation, scale changes, or background changes;
2. qualitative claims about attention-head specialization need quantitative, statistically valid evidence.

All local PDFs were downloaded from open-access author, conference, or archive pages on 2026-09-15. See `manifest.csv` for source URLs, checksums, and validation results, and `references.bib` for citations.

## Recommended reading order

### A. Token semantics and transformer explanations

1. **Abnar & Zuidema (2020), Quantifying Attention Flow in Transformers** (`pdfs/01_...`)
   - Defines attention rollout and attention flow. Use it for the plan's attention-driven alignment baseline, but treat it as an attribution approximation rather than ground truth.
2. **Chefer et al. (2021), Transformer Interpretability Beyond Attention Visualization** (`pdfs/02_...`)
   - A stronger relevance-propagation comparator for rollout; useful for faithfulness checks.
3. **Caron et al. (2021), Emerging Properties in Self-Supervised Vision Transformers** (`pdfs/03_...`)
   - Establishes that ViT patch features can contain object/segmentation information and motivates token-level semantic analysis.
4. **Walmer et al. (2023), Teaching Matters** (`pdfs/04_...`)
   - Closest reference to the proposed study: measures layer-wise spatial-token, object, and part clustering purity using masks.
5. **Zeng et al. (2022), TCFormer** (`pdfs/05_...`)
   - Shows dynamic token clustering and semantic-shaped tokens in a hierarchical transformer.
6. **Wang et al. (2022), TokenCut** (`pdfs/06_...`)
   - Uses graph/spectral clustering of self-supervised transformer features for object discovery; a useful alternative to K-means.

### B. Alignment, translation, and masks

7. **Jaderberg et al. (2015), Spatial Transformer Networks** (`pdfs/07_...`)
   - Canonical differentiable learned alignment baseline (translation, scale, and affine transformation).
8. **Ding et al. (2023), Reviving Shift Equivariance in Vision Transformers** (`pdfs/08_...`)
   - Explains why patch embedding, positional encoding, and subsampling break shift equivariance; directly motivates the translation stress test.
9. **Rojas-Gomez et al. (2023), Making Vision Transformers Truly Shift-Equivariant** (`pdfs/09_...`)
   - Supplies architecture-level shift-equivariant controls and evaluation ideas.
10. **Kirillov et al. (2023), Segment Anything** (`pdfs/10_...`)
    - Reference for the proposal's predicted-mask alignment path. Use oracle masks separately so SAM errors do not confound the alignment claim.

### C. Part-label datasets

11. **He et al. (2022), PartImageNet** (`pdfs/11_...`)
    - Object-part segmentation benchmark for testing token-to-part semantics beyond faces.
12. **Lee et al. (2020), MaskGAN / CelebAMask-HQ** (`pdfs/12_...`)
    - Introduces 30,000 face images with 19 fine-grained mask classes; directly supports eye/nose/mouth hypotheses.

### D. Head diversity and redundancy

13. **Chen et al. (2022), The Principle of Diversity** (`pdfs/13_...`)
    - Studies redundancy at patch, attention-map, and weight levels in ViTs; motivates measuring more than cluster labels.
14. **Li et al. (2018), Multi-Head Attention with Disagreement Regularization** (`pdfs/14_...`)
    - Defines diversity at attended-position, subspace, and output-representation levels.
15. **Voita et al. (2019), Specialized Heads Do the Heavy Lifting** (`pdfs/15_...`)
    - Shows many heads can be pruned; motivates causal leave-one-head-out tests instead of assuming every head is specialized.

### E. Statistical and semantic evaluation

16. **Vinh et al. (2010), Information Theoretic Measures for Clusterings Comparison** (`pdfs/16_...`)
    - Explains correction for chance. Prefer AMI/ARI over raw NMI when comparing partitions with different numbers or sizes of clusters.
17. **Kim et al. (2018), TCAV** (`pdfs/17_...`)
    - A quantitative concept-sensitivity test with repeated random concepts and statistical testing; useful as an optional semantic validation route.
18. **Hessel et al. (2021), CLIPScore** (`pdfs/18_...`)
    - Grounds image-text compatibility scoring for the VLM stage and documents limitations of automatic caption evaluation.

## Key methodological correction to the proposal

Low pairwise ARI/AMI between heads means that their partitions differ. It does **not** prove that the heads are statistically independent, useful, or semantically complementary. The experiment should require all three forms of evidence:

- **difference:** pairwise AMI/ARI, attention-map Jensen-Shannon divergence, and representation CKA;
- **semantic relevance:** held-out part-label AMI/mIoU, Cramer's V with image-level permutation, or a part-label linear probe;
- **unique utility:** leave-one-head-out performance drop or incremental predictive gain beyond the other heads.

Likewise, tokens from one image are spatially correlated and cannot be treated as independent samples in a naive chi-square test. Resampling/permutation must occur at the image level, and multiple head × stage × cluster tests need false-discovery-rate correction.

## Files

- `experiment_design.md` — preregistration-style experiment structure tailored to this repository.
- `references.bib` — BibTeX entries for all downloaded papers.
- `manifest.csv` — file provenance, page-count validation, and SHA-256 checksums.
- `pdfs/` — the 18 validated PDFs.
