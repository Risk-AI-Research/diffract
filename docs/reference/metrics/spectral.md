# Spectral decomposition (SVD and ESD)

The singular value decomposition and the empirical spectral distribution are the
substrate for every other metric. This page fixes the conventions the rest of
the catalog assumes. See the [catalog](catalog.md) for the individual fields.

## Matrix shape

Each weight matrix $W$ is reshaped to two dimensions. Its dimensions are
$N = \max(m, n)$ (`greater_dim`) and $M = \min(m, n)$ (`lower_dim`), and the
aspect ratio is $Q = N / M \ge 1$ (`aspect_ratio`). `weights_std` is the
standard deviation of the matrix entries, the noise scale used by the
Marchenko-Pastur fit.

## Decomposition

`weights_svd` computes the economy SVD $W = U\Sigma V^\top$ and returns the left
singular vectors, the singular values, and the right singular vectors. Two
conventions are load-bearing for the whole dependency graph:

- **Non-negative, sorted ascending.** Singular values are stored as
  $\sigma_1 \le \dots \le \sigma_k$ ($k = M$). Index $0$ is the weakest
  component; the top-$k$ directions are the *last* $k$ columns. Every min/max
  accessor (e.g. `max_weights_sval`, `esd_max`) and every empirical CDF relies
  on this ordering.

## Empirical spectral distribution

The **ESD** rescales the squared singular values into eigenvalues of the
correlation matrix:

$$\lambda_i = \frac{\sigma_i^2}{N}.$$

This is the Heavy-Tailed Self-Regularization convention $W^\top W / N$ with
$Q \ge 1$ {cite}`martin2021implicit`; it differs from the random-matrix-theory
parameterisation $\lambda = M/N$. Bounds follow the same rescaling — for
example $\lambda_{\max} = \sigma_{\max}^2 / N$ relates `esd_max` to
`max_weights_sval`.

## Randomized null model

`weights_rand` is a uniform permutation of the entries of $W$: it preserves the
multiset of weights and destroys all correlation structure. Applying the same
SVD/ESD pipeline to it yields `weights_rand_svd`, `esd_rand`, and their bounds —
the structureless baseline the [Marchenko-Pastur fit](rmt.md) and
`w1_rand_distance` compare against. The permutation is seeded (`seed=42`, so
results are reproducible); `seed=-1` selects a non-deterministic draw. In the
catalog, a $(\cdot)^{\mathrm{rand}}$ superscript marks a quantity computed from
this null.

## Conventions and pitfalls

- **Sorting is a contract.** The ascending order is assumed everywhere
  downstream; a metric that reads "the largest eigenvalue" reads the last entry.
- **ESD scale.** $\lambda = \sigma^2/N$ with $N$ = `greater_dim`; quote
  $Q = N/M$, not its reciprocal.
- **CUDA path.** When a CUDA device is present and `allow_cuda` is left on, the
  decomposition runs in single precision, so its singular values carry float32
  rounding error and differ from the double-precision CPU path.
