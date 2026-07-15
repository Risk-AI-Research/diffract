# Model quality

Single-number layer quality signals derived from the spectrum. See the
[catalog](catalog.md) for the exact fields.

## The metrics

- **Weighted alpha** — $\alpha_{\mathrm{PL}}\,\log_{10}\lambda_{\max}$
  (`pl_alpha_weighted`). The fitted power-law exponent $\alpha_{\mathrm{PL}}$
  scaled by the log of the largest eigenvalue, the "alpha-hat" combination of
  scale and shape {cite}`martin2021predicting`; it couples the tail exponent
  (smaller is heavier-tailed, typically better trained) with the spectral scale.
- **Randomized Wasserstein distance** — $\mathcal{W}_1(\lambda,
  \lambda^{\mathrm{rand}}) / \langle\lambda\rangle$ (`w1_rand_distance`). The
  Wasserstein-1 (earth-mover) distance $\mathcal{W}_1$ between the ESD and its
  [permutation null](spectral.md), divided by the shared mean eigenvalue
  $\langle\lambda\rangle$. It measures how far the spectrum departs from a structureless
  baseline of the same weights, and the normalisation makes the index
  dimensionless and comparable across layers of different scale.

## Conventions and pitfalls

- **Log base.** `pl_alpha_weighted` uses $\log_{10}$, matching the log-domain
  [norms](norms.md).
- **Scale invariance.** Dividing by $\bar{\lambda}$ makes `w1_rand_distance`
  invariant to rescaling the weights, so it reflects spectral *shape* rather than
  magnitude; it is $0$ for a degenerate (zero-mean) spectrum and NaN when the
  spectrum is not measurable.
- **The exponent is fitted upstream.** `pl_alpha_weighted` reuses `pl_alpha` from
  the [power-law fit](heavy_tailed.md); its reliability inherits that fit's.
