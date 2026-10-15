# Selected research artifacts

This directory tracks a curated snapshot of small results supporting the research
reports in `plan/`: October 5 clustering comparisons and quality evaluation,
silhouette figures and their PDF, and October 9 concept intervention results.
Summary CSVs and run manifests are included alongside the figures.

The cluster gallery includes only `stage2/spherical/cluster_00.png`, which is
referenced by the annotation coordinate audit. Other gallery links in the
cluster-meaning report require locally generated artifacts.

Feature caches, model objects, large iteration histories, most cluster galleries,
and older exploratory runs remain local. Reproduce results with the commands in
the corresponding reports. Files are explicitly allowed in the root `.gitignore`;
new outputs remain ignored until selected for publication.

October 12 corrected Caltech object-center results are in
`kmeans/caltech_corrected_center_1012/`: static and translation CSVs, per-source
coordinate/phase records, pixel checks, manifests, execution log, and the
translation figure. Training/validation/translation feature NPZs and codebook
NPZs remain local. See
[`1012_caltech_corrected_center_static_translation.md`](../plan/1012_caltech_corrected_center_static_translation.md)
for settings, results, checks, and reproduction dependencies.


October 14 aligned augmentation (`kmeans/caltech_aligned_translation_1014/`):
reports, simplified comparison tables, Mermaid links, per-image CSVs, model
manifests, translation montages and all 66 cluster-change case images. This run
uses normalized-feature sklearn Euclidean K-means.

October 15 non-aligned shifts (`kmeans/caltech_non_aligned_1015/`): true FAISS
spherical evaluation for the existing baseline and aligned-augmentation models,
with CSVs, figures, identities and manifests. The new 1–7px augmentation training
was **paused by the user after epoch 10**; config/history snapshots and
`pause_state.json` document the state. There is no final 1–7px-model evaluation.
See `plan/1015_non_aligned_translation_experiment.md` for continuation limits.

NPZ feature/codebook caches, `.pth` model weights, datasets and full overnight
logs stay local. Hashes in manifests identify execution-time source/artifacts;
not every hashed local artifact is published, and later documentation edits do
not change the recorded execution-time code snapshot.
