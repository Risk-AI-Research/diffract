# Norms

Matrix norms computed from the singular values $\sigma_i$ of each weight matrix,
and their model-level aggregates. See the [catalog](catalog.md) for the exact
formula, apply level, and dependencies of every field.

## Per-matrix norms

For a weight matrix $W$ with singular values $\sigma_1 \le \dots \le \sigma_k$
($k = \min(m, n)$):

- **Frobenius** — $\lVert W\rVert_F = \sqrt{\sum_i \sigma_i^2}$, the Schatten-2
  norm (`frob_norm`).
- **Nuclear** — $\lVert W\rVert_* = \sum_i \sigma_i$, the Schatten-1 (trace)
  norm (`nuclear_norm`). It is the sum of singular values; it forms a Schatten-norm family with `l2_norm`.
- **Spectral** — $\lVert W\rVert_2 = \sigma_{\max}$, the induced $L^2$
  (operator) norm (`l2_norm`).

## Model-level aggregates

These reduce a per-parameter norm over a model, summing or averaging over the
parameters $\ell$. The *log-domain* aggregates below reduce over the
**measurable** parameters only: a dead layer (norm $0$) or a diverged one (NaN)
is skipped rather than folded in, and an all-degenerate model reduces to NaN.

- **Sum of squared Frobenius norms** — $\sum_\ell \lVert W_\ell\rVert_F^2$
  (`param_norm`), a norm-based capacity measure {cite}`jiang2020`. This is a
  plain sum over every parameter, so a NaN layer propagates.
- **Log-product norms** — $\sum_\ell \log_{10}\lVert W_\ell\rVert$
  (`log_prod_frob_norm`, `log_prod_spectral_norm`). Products of layer norms
  appear in norm-based generalization bounds; the product is accumulated in the
  log domain, where it stays finite at model scale.
- **Log norms** — $\langle \log_{10}\lVert W\rVert^2\rangle$ over the model
  (`log_norm`, `log_spectral_norm`), mean-log-norm signals in the sense of
  Heavy-Tailed Self-Regularization {cite}`martin2021implicit`.

## Weighted alpha norms

The **weighted alpha norm** $\sum_i \lambda_i^{\alpha}$ (`pl_alpha_norm`,
`tpl_alpha_norm`) raises the ESD eigenvalues $\lambda_i = \sigma_i^2/N$ to the
power-law exponent $\alpha$ fitted for that layer (see
[heavy-tailed fits](heavy_tailed.md)); the catalog labels the two fits'
exponents $\alpha_{\mathrm{PL}}$ and $\alpha_{\mathrm{TPL}}$. Its model-level
mean-log form
$\langle \log_{10}\sum_i \lambda_i^{\alpha}\rangle$ (`model_pl_alpha_norm`,
`model_tpl_alpha_norm`) is the log-$\alpha$-norm, a strong test-accuracy
predictor that uses no test data {cite}`martin2021predicting`.

## Conventions and pitfalls

- **Log base.** Every log-domain norm uses $\log_{10}$, the base of the
  Heavy-Tailed Self-Regularization metrics {cite}`martin2021implicit`.
- **Squared vs. unsquared.** `log_norm` and `log_spectral_norm` take the log of
  the *squared* norm, $\langle\log_{10}\lVert W\rVert^2\rangle$; the log-product
  norms take the log of the norm itself. The catalog formula is authoritative.
- **Degenerate layers.** The measurable-only reduction lets a model with some
  empty spectra still produce a finite aggregate; a model with no measurable
  parameter produces NaN rather than a misleading $-\infty$.
