# Ranks

Continuous rank surrogates: how many singular directions carry meaningful energy.
Unlike the algebraic rank, these degrade gracefully and, where noted, are
invariant to rescaling the weights. See the [catalog](catalog.md) for exact
formulas and dependencies.

## The metrics

- **Stable rank** — $\lVert W\rVert_F^2 / \lVert W\rVert_2^2 = \sum_i (\sigma_i/\sigma_{\max})^2$ for $\sigma_{\max}>0$ (`stable_rank`).
  A scale-invariant lower bound on the rank that is robust to small singular
  values {cite}`rudelson2007`. Equal to the rank for an isometry, and small
  when energy concentrates in a few directions.
- **Effective rank** — $\exp\!\big(-\sum_i p_i \ln p_i\big)$ with
  $p_i = \sigma_i / \sum_j \sigma_j$ (`effective_rank`), the exponential of the
  spectral entropy {cite}`roy2007`. Scale invariant; interpretable as an
  effective number of active directions.
- **Hard rank** — $\#\{i : \lambda_i > \texttt{rtol}\cdot\lambda_{\max}\}$
  (`hard_rank`), a numerical rank against a threshold *relative* to the spectrum
  maximum. Its configurable tolerance (`rtol`, default `1e-5`) is relative to
  the largest eigenvalue. Very small or large weight scales can underflow or
  overflow the squared singular values used to construct the ESD.
- **MP soft rank** — $\lambda_+ / \lambda_{\max}$ (`mp_soft_rank`), the
  Marchenko-Pastur bulk edge $\lambda_+$ relative to the largest eigenvalue.
  Near $1$ when the spectrum is bulk-dominated, near $0$ when a large spike
  dominates. See [random matrix theory](rmt.md) for the bulk edge.

## Conventions and pitfalls

- **Scale invariance.** `stable_rank` and `effective_rank` are invariant to
  $W \mapsto cW$ for finite nonzero $c$ within the representable input range.
- **Normalization.** Both ranks divide the spectrum by its largest singular
  value in float64. Stable rank sums the squares of these ratios. Effective
  rank normalizes them to probabilities and uses $0\log 0 = 0$. This gives
  the ranks directly even when the Frobenius norm exceeds float64 range.
  [SciPy entropy function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.entr.html)
- **Zero and nonfinite spectra.** A zero matrix has stable rank 0. Its
  normalized probabilities are undefined. Diffract assigns it effective rank
  1 by convention. A spectrum containing NaN or infinity gives NaN for both
  ranks.
- **Inheritance.** `mp_soft_rank` is only as reliable as the Marchenko-Pastur
  fit it draws $\lambda_+$ from; read it alongside `mp_ks`.
- **Effective rank vs. entropy.** `effective_rank` is the exponential of the
  spectral (Shannon) entropy of the normalized singular values and coincides
  with the matrix-entropy measure used elsewhere in the literature.
