# Metrics

Diffract's spectral metrics are computed by *kernels* over the singular values of
each weight matrix. This reference documents the full set: a generated
[catalog](catalog.md) of every field with its formula, and per-category pages
covering the mathematics, conventions, and interpretation.

```{toctree}
:maxdepth: 1

catalog
norms
ranks
spectral
heavy_tailed
rmt
alignment
quality
references
```

## Conventions

These conventions hold across every metric; the per-category pages assume them.

- **Empirical spectral distribution.** For a weight matrix $W \in
  \mathbb{R}^{m \times n}$ with singular values $\sigma_i$, the ESD eigenvalues
  are $\lambda_i = \sigma_i^2 / N$, where $N = \max(m, n)$ is `greater_dim`. This
  is the Heavy-Tailed Self-Regularization convention ($W^\top W / N$ with
  $Q = N/M \ge 1$) {cite}`martin2021implicit`. The economy SVD yields
  $M = \min(m, n)$ singular values (`lower_dim`), so the ESD has exactly $M$
  eigenvalues, $\lambda_1, \dots, \lambda_M$; a fraction over the whole spectrum
  (such as a concentration) is divided by $M$.
- **Sort order.** Singular values and ESD eigenvalues are stored in **ascending**
  order. Index $0$ is the weakest component; the top-$k$ components are the last
  $k$. Every min/max accessor and empirical CDF depends on this.
- **Aspect ratio.** $Q = N/M \ge 1$ (`aspect_ratio`).
- **Randomized null.** `weights_rand` is a uniform permutation of the entries of
  $W$: it preserves the multiset of weights and destroys correlation structure.
  `esd_rand` and the Marchenko-Pastur fit are built from it. The permutation is
  seeded (`seed=42`); `seed=-1` selects a non-deterministic draw.
- **Apply levels.** `PARAMETER` metrics are per weight matrix; `IN_MODEL`
  aggregates them over a model; `CROSS_MODEL` compares a parameter across two
  checkpoints.
- **Aggregation notation.** Angle brackets $\langle\cdot\rangle$ denote a mean;
  the averaging domain is given by context — over the model's parameters $\ell$
  for an `IN_MODEL` metric (which may also appear as an explicit $\sum_\ell$), or
  over a single matrix's eigenvalues or components otherwise.

## Reading the formulas

Each catalog formula is the exact quantity the kernel body computes, in the ESD
convention above. The log-domain model aggregates reduce only over *measurable*
parameters: a dead layer (zero norm) or a diverged one (NaN) is skipped rather
than folded in; a plain sum such as `param_norm` propagates it. Small
regularizers ($\varepsilon \sim 10^{-16}$) that guard divisions are omitted from
the displayed formulas.
