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
singular vectors, the singular values, and the right singular vectors. The dependency graph uses the following ordering.

- **Non-negative, sorted ascending.** Singular values are stored as
  $\sigma_1 \le \dots \le \sigma_M$. Index $0$ is the weakest
  component. For $1 \le k \le M$, the top-$k$ directions are the last $k$ columns. Every min/max
  accessor (e.g. `max_weights_sval`, `esd_max`) and every empirical CDF relies
  on this ordering.

## Empirical spectral distribution

The **ESD** rescales the squared singular values into eigenvalues of the
correlation matrix:

$$\lambda_i = \frac{\sigma_i^2}{N}.$$

These are the $M$ eigenvalues of the smaller Gram matrix, $W^\top W/N$ when
$m\ge n$ and $WW^\top/N$ when $m<n$, including zeros from rank deficiency.
The aspect ratio is $Q=N/M$ in the Heavy-Tailed Self-Regularization
convention {cite}`martin2021implicit`. Its reciprocal is $\gamma=M/N$.
Bounds follow the same rescaling — for
example $\lambda_{\max} = \sigma_{\max}^2 / N$ relates `esd_max` to
`max_weights_sval`.

## Randomized null model

`weights_rand` is a uniform permutation of the entries of $W$. It preserves the
multiset of weights and randomizes their positions. Its singular
values feed `esd_rand` and its bounds. The pipeline computes the permuted matrix’s left and right singular
vectors when requested as `weights_rand_lsvs` or `weights_rand_rsvs`. The
permuted-weight spectrum is the baseline the [Marchenko-Pastur fit](rmt.md) and
`w1_rand_distance` compare against. The permutation is seeded (`seed=42`, so
results are reproducible); `seed=-1` selects a non-deterministic draw. In the
catalog, a $(\cdot)^{\mathrm{rand}}$ superscript marks a quantity computed from
this null.

## Conventions and pitfalls

- **Sorting is a contract.** The ascending order is assumed everywhere
  downstream; a metric that reads "the largest eigenvalue" reads the last entry.
- **ESD scale.** $\lambda = \sigma^2/N$ with $N$ = `greater_dim`; quote
  $Q = N/M$, not its reciprocal.
- **Small singular values.** Relative error in the weakest direction can grow
  with the condition number. A Frobenius norm can agree closely while an inverse
  moment such as $\sum_i \sigma_i^{-2}$ changes substantially.

## Precision and execution records

The default `precision="preserve"` keeps float64 weights in float64, keeps
float32 in float32, and computes float16 or bfloat16 weights in float32. The
PyTorch extractor converts bfloat16 to float32 at admission and records the
checkpoint dtype separately. The selected working dtype applies to singular
values and vectors. Set `precision="float32"` or `precision="float64"` on the
producer to request that working dtype explicitly. For example, to compute
both original and permuted spectra and vectors in double precision:

```python
for kernel in ("weights_svd", "weights_rand_svd", "weights_rand_vectors"):
    session.compute.configure_kernel(kernel, precision="float64")
session.compute.apply("weights_svals", "weights_rand_svals", "weights_rand_lsvs")
```

With `allow_cuda=True` (the default), an available NVIDIA CUDA device uses
PyTorch's cuSOLVER QR driver `gesvd`. CPU execution uses SciPy's LAPACK
divide-and-conquer driver `gesdd`. Set `allow_cuda=False` to select CPU.
The precision policy and resolved device, backend and driver participate in
cache validation. Derived scalar metrics use exact equality when comparing
outputs, unless the kernel declares its own equivalence rule. Applying a spectral field whose stored computation predates
precision records computes a spectrum under the active policy.
[PyTorch SVD drivers](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)
and [SciPy SVD drivers](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html).

Look up the computation for a specific exported weight:

```python
rows = session.results.export("weights_svals", shape="long", export_format="list")
row = rows[0]
run = session.results.describe_run(row["run_id"], parameter_uid=row["parameter_uid"])
print(run["requested"])
for entry in run["arithmetic"]:
    print(entry["checkpoint_dtype"], entry["input_dtype"], entry["working_dtype"],
          entry["output_dtype"], entry["device"], entry["driver"])
```

A batched run can contain different input dtypes. Omitting `parameter_uid`
returns all observations in the current scope. Historical values remain readable.
An uninstrumented run reports `arithmetic_status="unknown"`.
Custom kernels can return `diffract.core.compute.KernelResult` with a value
and arithmetic details. These entries carry `origin="self-reported"`.

From a source checkout, run `python scripts/check_svd_precision.py --device cuda
--csv precision.csv` on an NVIDIA host to compare original and permuted spectra
with CPU reference values. The CSV gives each input's errors and tolerances
for the smallest singular value, inverse moment and Frobenius norm.
