# Ranks

Continuous rank surrogates: how many singular directions carry meaningful energy.
Unlike the algebraic rank, these degrade gracefully and, where noted, are
invariant to rescaling the weights. See the [catalog](catalog.md) for exact
formulas and dependencies.

## The metrics

- **Stable rank** — $\lVert W\rVert_F^2 / \lVert W\rVert_2^2$ (`stable_rank`).
  A scale-invariant lower bound on the rank that is robust to small singular
  values {cite}`rudelson2007`. Equal to the rank for an isometry, and small
  when energy concentrates in a few directions.
- **Effective rank** — $\exp\!\big(-\sum_i p_i \ln p_i\big)$ with
  $p_i = \sigma_i / \sum_j \sigma_j$ (`effective_rank`), the exponential of the
  spectral entropy {cite}`roy2007`. Scale invariant; interpretable as an
  effective number of active directions.
- **Hard rank** — $\#\{i : \lambda_i > \texttt{rtol}\cdot\lambda_{\max}\}$
  (`hard_rank`), a numerical rank against a threshold *relative* to the spectrum
  maximum. The relative threshold preserves $\operatorname{rank}(cW) =
  \operatorname{rank}(W)$, following the standard numerical-rank convention of
  scaling the tolerance by the largest eigenvalue.
- **MP soft rank** — $\lambda_+ / \lambda_{\max}$ (`mp_soft_rank`), the
  Marchenko-Pastur bulk edge $\lambda_+$ relative to the largest eigenvalue.
  Near $1$ when the spectrum is bulk-dominated, near $0$ when a large spike
  dominates. See [random matrix theory](rmt.md) for the bulk edge.

## Conventions and pitfalls

- **Scale invariance.** `stable_rank` and `effective_rank` are invariant to
  $W \mapsto cW$; `hard_rank` is too, by construction, because its threshold
  scales with $\lambda_{\max}$. An absolute threshold would drift with the
  weight scale during training, which is why the relative one is the default
  (`rtol`, configurable).
- **Inheritance.** `mp_soft_rank` is only as reliable as the Marchenko-Pastur
  fit it draws $\lambda_+$ from; read it alongside `mp_ks`.
- **Effective rank vs. entropy.** `effective_rank` is the exponential of the
  spectral (Shannon) entropy of the normalized singular values and coincides
  with the matrix-entropy measure used elsewhere in the literature.
