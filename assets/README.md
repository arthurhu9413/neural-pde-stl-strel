# Pre-computed assets

The files in this directory are committed so that the README and the paper-facing
figures can be rebuilt without retraining.

## Diffusion assets

- `diffusion1d_baseline_field.png`  
  Field plot for the committed baseline diffusion checkpoint.
- `diffusion1d_stl_field.png`  
  Field plot for the committed STL-regularized diffusion checkpoint.
- `diffusion1d_training_loss.png`  
  Loss comparison figure derived from committed CSV logs.
- `diffusion1d_training_robustness.png`  
  Robustness-over-training figure derived from committed logs/summaries.
- `diffusion1d_training_loss_components_stl.png`  
  STL-run loss breakdown.
- `diffusion1d_robust_vs_lambda.png`  
  Lambda sweep view derived from the committed ablation summary.

## Heat rollout asset

- `heat2d_scalar/field_xy_t.npy`  
  Committed 2D heat rollout used by the spatial monitoring example.
- `heat2d_scalar/meta.json`  
  Metadata for the committed rollout.

If any of these files are regenerated, the expected follow-up is:

1. refresh summaries if needed,
2. regenerate figures,
3. rerun tests,
4. then update prose.
