# Feature-Space Restoration of Sparse-View CT Embeddings

Code and results for *Feature-Space Restoration of Sparse-View CT Embeddings from
Frozen Encoders for Intracranial Hemorrhage Classification*.

A small residual network corrects the embedding a frozen encoder produces from a
sparse-view CT reconstruction, moving it toward the embedding of the 360-view
reference. The encoder is never fine-tuned and one linear probe, trained on
reference embeddings, is applied unchanged to all three representations, so any
recovered performance comes from the restored embedding rather than a retrained
classifier.

---

## Repository layout

```
src/
  extraction/          DICOM -> HU proxy, sparse-view simulation (Radon/FBP)
  extraction_3ch/      three-window input, encoder forward pass
  pooling/             pooled embedding from feature maps
  probe/               linear probe
  diffusion/           cold-diffusion restorer
  data/  utils/
  configs/             per-encoder, per-projection-count YAML
  analysis/            bootstrap, figures
  experiments/
    probe_variants/          probe refit comparison (Table 6)
    unconditioned_diffusion/ conditioning-free variant (Table 4)
  train_diffusion.py   restorer + probe training
  compare_restorers.py restorer comparison under one recipe
  collect_*.py         aggregate run outputs into CSV
  run_eval.py

results/               the numbers behind each table (see results/README.md)
```

## Requirements

Python 3.10+, PyTorch, numpy, pandas, scikit-image, pydicom, timm,
transformers. GPU needed for training; the analysis scripts are CPU-only.

## Data

RSNA 2019 Intracranial Hemorrhage Detection, from Kaggle:
<https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection>

Not redistributed here. Encoder weights are not redistributed either — they are
pulled from their original sources and carry their own licences, MedSigLIP in
particular.

## Reproducing the pipeline

```bash
# 1. extract embeddings, per encoder and projection count
python src/extraction_3ch/run_extract_encoder_mp.py --config src/configs/medsiglip_after/<cfg>.yaml

# 2. train the probe on reference embeddings, then the restorer
python src/train_diffusion.py src/configs/medsiglip_after/<cfg>.yaml

# 3. compare restorers under one data/probe/selection recipe
python src/compare_restorers.py src/configs/medsiglip_after/<cfg>.yaml --seeds 42 43 44 --save_weights

# 4. study-level paired bootstrap with Holm correction
python src/analysis/bootstrap_study.py --pred_dir preds_mlp --B 2000 --out_dir results
```

## Paper to code

| Paper | Produced by | Result file |
|---|---|---|
| Table 1 — data partition | `src/extraction/` split | — |
| Table 2 — AUROC, 5 encoders x 6 views | `collect_test_results.py` -> `analysis/bootstrap_study.py` | `results/table2_summary.csv` |
| Table 3 — per-label, MedSigLIP 6 views | `analysis/bootstrap_study.py` | `results/table3_tableS1_perlabel.csv` |
| Table 4 — restorer comparison | `compare_restorers.py`, `collect_restorers.py` | `results/table4_restorers.csv` |
| Table 5 — study-level AUROC | `analysis/bootstrap_study.py` | `results/table5_study_auroc.csv` |
| Table 6 — refit-probe baseline | `src/experiments/probe_variants/` | `results/table6_probe_matrix.csv` |
| Table S1 — per-label, all encoders | same as Table 3 | `results/table3_tableS1_perlabel.csv` |
| Table S2 — operating points | `analysis/` | `results/tableS2_sens_at_spec.csv` |
| Fig. 1 — pipeline | `analysis/` | — |
| Fig. 2 — FBP at each view count | `analysis/fig_window_channels.py` | — |
| Fig. 3 — AUROC vs views | `analysis/` | `results/table2_summary.csv` |
| Fig. 4 — t-SNE, cosine | `analysis/fig_tsne_mlp.py` | — |
| Fig. S1 — PSNR/SSIM vs views | `analysis/` | — |

## Protocol, in brief

- Reference 360 views; sparse 180, 90, 45, 23, 12, 6, each a subset of the
  reference sinogram.
- Five frozen encoders: MedSigLIP, SigLIP, DINOv2, ResNet50, DenseNet121.
- Probe: one fully connected layer, unweighted BCE, AdamW 1e-4 / 1e-4, batch 256,
  up to 200 epochs, early stopping on validation mean AUROC every 20 epochs.
- Restorer: four post-normalisation SwiGLU residual blocks of width 1024, trained
  per encoder and projection count, MSE on the embedding shift, EMA 0.999.
- Inference: paired study-level bootstrap, 3,262 test studies, 2,000 resamples,
  Holm correction across 180 label-level comparisons.
- Seeds 42, 43, 44 for the network restorers.

## Licence

Not yet set. Until a licence file is added, default copyright applies and the
code is viewable but not reusable.

## Citation

See `CITATION.cff` (added once the Zenodo DOI is issued).
