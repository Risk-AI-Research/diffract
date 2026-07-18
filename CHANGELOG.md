# Changelog

## 0.3.1

Hardens the session and storage layers: requests that cannot be served
raise or warn with actionable messages, the mutating verbs report what
they did, and the on-disk metadata index is schema-versioned with an
explicit upgrade path.

### Upgrading a persistent store

A `local`/`hybrid` store written by an earlier release is refused at
session start with `IncompatibleStoreError`; nothing is migrated
implicitly. Back up the database file, then upgrade in place:

```python
import diffract

diffract.upgrade_metadata_index("<path to metadata.db>")
```

The upgrade only adds the schema bookkeeping table — parameter and result
data are untouched. In-memory (`ram`) sessions are unaffected. See the
storage guide for the full schema-version contract.

### Added

- The mutating verbs return typed summaries: `compute.apply` returns an
  `ApplySummary` grouping the written fields by apply level (and naming
  skipped requests with a reason), `models.erase` and `results.erase`
  return an `EraseSummary` (erased models/fields and the number of
  parameter entries that lost data), and `compute.configure_kernel`
  returns the effective `KernelConfig`.
- Schema versioning for the SQLite metadata index (`PRAGMA user_version`):
  fresh stores are stamped at the current version and record their
  producer; `diffract.upgrade_metadata_index` applies pending migration
  steps, each in its own transaction.
- New typed errors on `diffract.session`: `ScopeValidationError`,
  `InvalidIdentifierError`, `IncompatibleStoreError`.
- `diffract.viz.styling.is_style_literal` — the public predicate behind
  the string-vs-field rule for style properties.
- A pipeline benchmark harness (`scripts/bench_pipeline.py`, `make bench`,
  the `bench` extra): times the add / apply / export stages across storage
  profiles and workload shapes on a pinned device, records peak RSS, and
  doubles as a regression gate against a baseline report.
- Documentation for the retrieval routing rule (`PARAMETER`-level fields
  come back from `export_metrics`; `IN_MODEL` and `CROSS_MODEL` fields
  from `export_aggregates`), the two-model scope of cross-model kernels,
  kernel reconfiguration semantics (stored results are keyed by field
  presence and are not invalidated by configuration changes), the
  weight-slice schema boundary, and store schema versions.

### Changed

- Applying a binary cross-model kernel (the alignment/overlap family) with
  anything other than exactly two models in scope raises
  `ScopeValidationError` up front, naming the scope and the runnable fix.
- Exports validate the requested field names: a name that neither a
  registered kernel produces nor any stored field matches raises
  `KernelNotFoundError` with did-you-mean suggestions, and a known field
  with no values in the current scope logs a warning naming the exporter
  and apply level that serve it.
- Model ids and parameter names are validated at every ingest boundary
  (ASCII letters, digits, `_`, `-`, `.`), and ingested field names reject
  the storage-unsafe characters `< > : " / \ | ? *`; violations raise
  `InvalidIdentifierError`. Merging a source session that carries invalid
  identifiers is rejected with the same error.
- Aggregate context members are stored in canonical sorted order, matching
  the order the contextual field labels already use.
- The viz wrappers (`scatter`, `box`, `violin`, `sparkline`) resolve
  string style properties uniformly: a valid Plotly literal stays a
  literal, any other string becomes a field reference, and an explicit
  `FieldRef` is the escape hatch.
- `diffract.core` does not re-export the contextual-grammar constants;
  the session-layer resolver is the single interpreter of contextual
  field labels.

### Fixed

- `models.erase` and `results.erase` operate on the active filtered scope:
  erasing from `session.filter(...)` touches only in-scope entries.
- A filtered export leaves the session-shared field cache consistent:
  subsequent whole-session erases and parameter listings see every entry
  with its real fields.
- User logging configuration survives session use; the library configures
  logging once when the container is built instead of on every namespace
  call.
- Unhandled plot ordering modes raise `ValueError` instead of silently
  falling back to the unsorted order.
- A dict-of-arrays model with non-string keys is rejected with a
  `TypeError` naming the offending key types, whether or not a deep
  learning framework is installed.

## 0.3.0

Brings the spectral kernels into line with their mathematical definitions
across scale, sign, and degenerate inputs, renames several metrics to name
the quantity they compute, and publishes a generated metrics catalog on the
docs site.

### Breaking changes

Several kernels are renamed so the field name states the quantity it computes:

| 0.2.2                | 0.3.0                    |
| -------------------- | ------------------------ |
| `l1_norm`            | `nuclear_norm`           |
| `prod_frob_norm`     | `log_prod_frob_norm`     |
| `prod_spectral_norm` | `log_prod_spectral_norm` |
| `rand_distance`      | `w1_rand_distance`       |

- `nuclear_norm` is the sum of singular values (the Schatten-1 / nuclear
  norm).
- `log_prod_frob_norm` and `log_prod_spectral_norm` accumulate in the log
  domain (the sum of the per-layer `log10` norms, i.e. the log of their
  product), so they stay finite on models with hundreds of layers.
- `w1_rand_distance` is the Wasserstein-1 (earth-mover) distance between a
  layer's spectrum and its permutation null, divided by the mean eigenvalue:
  the index is invariant to permutation and to rescaling of the weights.
- `hard_rank` takes an `rtol` config (the relative tolerance on the largest
  eigenvalue) in place of `threshold`, and counts eigenvalues above
  `rtol * lambda_max`, so the count is invariant to rescaling the weights.
- `weights_rand` takes only a `seed`; the `n_randomise_iterations` option is
  removed (repeated uniform permutations are equivalent to a single one).
- Singular-vector overlaps (`l_overlap`, `r_overlap`) are taken in absolute
  value; the option to produce signed overlaps is removed. The agreement
  metrics are therefore invariant to the arbitrary sign LAPACK assigns each
  singular vector.

Because these renames and value changes alter stored field names and outputs,
erase the affected fields and recompute before comparing results across the
boundary: `session.results.erase("<field>", ..., erase_dependent_also=True)`
(or `erase_all=True`), then re-apply the kernels.

### Added

- A generated metrics catalog on the docs site: one page per category (norms,
  ranks, spectral, heavy-tailed, RMT, alignment, model quality) with each
  kernel's produced fields, display formula, apply level, required inputs, and
  configuration, plus a references bibliography. The catalog is generated from
  the kernel registry at build time and a kernel without a formula fails the
  docs build, keeping it complete and in sync with the registry.
- Property-based tests (Hypothesis, in the `dev` extra) covering norm, rank,
  and spectral/ESD invariants: scale and permutation invariance, monotonicity,
  and NaN propagation. Coverage measurement now includes the kernel modules.

### Changed

- `log_norm` and `log_spectral_norm` average `log10(||W||^2)` over the
  measurable parameters, a monotone function of the norm.
- `mp_ks` reports the two-sided Kolmogorov-Smirnov distance between the
  empirical spectral density and the fitted Marchenko-Pastur law, with the
  model CDF conditioned on the bulk window the sample occupies.
- The Marchenko-Pastur bulk variance is estimated from the trace identity (the
  bulk eigenvalue sum over the full dimension), so eigenvalues that extend
  past the fitted bulk edge are excluded from the estimate and the fitted edge
  stays non-negative.
- The heavy-tailed concentration kernels (`pl_`, `tpl_`, `expon_`) and the RMT
  spike counters propagate NaN: a NaN spectral edge or spectrum yields NaN.
- Degenerate inputs propagate NaN: a NaN weight matrix yields all-NaN SVD
  outputs, and isometric or zero-variance inputs produce NaN norms and fit
  statistics.
- `mp_presence` is clipped to `[0, 1]`; `pl_alpha_weighted` scales the
  power-law exponent by `log10` of the largest eigenvalue.
- The results namespace honors the profile's default export format when a call
  does not name one; the `hybrid` and `sqlite` profiles default to the `dict`
  format.

### Fixed

- The floating-point environment is saved and restored around taichi
  initialization, so its flush-to-zero setting stays contained to taichi's own
  execution and does not reach numpy or scipy elsewhere in the process.
- The sqlite `:memory:` sentinel is treated as a mode rather than a filesystem
  path during index resolution.

### Security

- Cache and storage values are serialized with a typed codec that handles the
  supported value kinds explicitly and rejects the rest, avoiding
  deserialization of untrusted data (CWE-502).

## 0.2.2

### Changed

- Python 3.13 is supported (`requires-python >=3.12`); CI tests on 3.12
  and 3.13.

## 0.2.1

First release published to PyPI as `diffract-core` (the import name stays
`diffract`).

### Added

- A light core install: `pip install diffract-core` works without any deep
  learning framework, the viz stack loads lazily, optional-dependency errors
  name the extra to install, and the `all` extra bundles torch, viz, taichi,
  pandas, and polars.
- `py.typed` marker and PEP 639 license metadata; LICENSE and NOTICE ship in
  the wheel.

### Changed

- numpy 2.x is supported (`numpy >=1.26,<3`, `scipy >=1.12`). The
  Marchenko-Pastur CDF and Tracy-Widom quantiles are computed by a vendored
  implementation (closed-form MP antiderivative; Painleve II integration for
  TW), replacing the scikit-rmt dependency, whose distribution metadata
  capped numpy at 1.26 and scipy at 1.12. Parity with scikit-rmt is pinned
  by golden tests.
- powerlaw is pinned below 2.0: its 2.x line changes the Fit API the
  heavy-tailed kernels rely on.

## 0.2.0

Reorganizes the public API around namespaces, overhauls the visualization system, and
promotes parallel execution to a top-level module.

### Breaking changes

The flat `Session` API is replaced by namespaced accessors:

| 0.1.0 (flat)                        | 0.2.0 (namespaced)                            |
| ----------------------------------- | --------------------------------------------- |
| `session.add(model, ...)`           | `session.models.add(model, ...)`              |
| `session.compute("frob_norm", ...)` | `session.compute.apply("frob_norm", ...)`     |
| `session.get_results("frob_norm")`  | `session.results.export_metrics("frob_norm")` |
| `session.draw(plot=...)`            | `session.viz.draw(plot=...)`                  |
| `session.list_kernels()`            | `session.compute.list_available_kernels()`    |
| `@session.kernel()`                 | `@session.compute.kernel()`                   |

### Added

- Namespaced session API: `models`, `compute`, `results`, `viz`, `utils`.
- Visualization overhaul: composable plot base classes, a centralized `styling` layer
  (themes, palettes, property resolvers), box/violin/scatter/heatmap/sparkline plots
  with Plotly-style category ordering on categorical axes, and grid/subplot
  generation.
- Top-level `core.parallel` execution module.
- `list` export format for metrics and aggregates.
- `ParameterType` is re-exported from the package top level
  (`from diffract import ParameterType`).
- Unknown metric, kernel, and export-format errors suggest close matches; missing
  optional export dependencies name the extra to install.
- Plain dicts of NumPy arrays are accepted as models
  (`session.models.add({"encoder.weight": array})`): 2D floating-point arrays
  are extracted as dense parameters, with no deep learning framework required.
  Weight matrices loaded from `.npy`/`.npz` files or via `safetensors.numpy`
  plug straight in.
- `session.models.add` raises `SessionError` when extraction yields no parameters
  (e.g. a dict holding only 1D or integer arrays, which no handler accepts),
  naming the supported inputs.
- A smoke test executes every registered kernel end to end, and all python code blocks
  in the README run as tests in CI.
- The accelerated `diffract` fit implementation (`fit_method="diffract"`, taichi
  extra): maximum-likelihood parameter estimates with global-KS xmin selection
  exactly as in Clauset et al. (matches the `powerlaw` library on canonical
  body+tail data), a deterministic seeded bootstrap with body resampling,
  shifted-form CDFs that survive large xmin, bounded rejection sampling,
  scale-invariant fitting via exact power-of-two normalization, and strict
  input validation. Taichi initializes lazily, fields are pooled process-wide
  by padded data size, and kernels compile once per size bucket: a warm
  power-law fit at n = 5000 takes ~20 ms, and power-law fits run roughly
  10-20x faster than the `powerlaw` library at n = 10k-20k. `Fit.close`
  returns the borrowed fields to the pool. The `pl_p_value`, `tpl_p_value` and `expon_p_value` kernels
  register when the taichi extra is installed; p-values are deterministic,
  are NaN for tails smaller than 50 points, and the bootstrap plausibility
  test is documented as low-power against misspecified alternatives
  (confirmatory use, not model selection).

### Changed

- Heavy-tailed fit kernels default to `fit_method="auto"`: the accelerated
  implementation whenever the taichi extra is installed and the ESD has at
  least 100 points, the `powerlaw` library otherwise. The 100-point floor is
  statistical, not performance: the accelerated fitter only considers xmin
  candidates that leave a tail of 50 or more points (the reliability bound of
  Clauset et al.), so below ~100 points it cannot select a tail at all. On
  small ESDs (roughly 100-300 points) this constrained search yields more
  conservative estimates than the `powerlaw` library's unrestricted one, which
  happily minimizes KS over arbitrarily small tails; pass
  `fit_method="powerlaw"` for exact parity with it. `auto` also falls back to
  the `powerlaw` library when taichi is installed but fails to initialize;
  explicit `fit_method="powerlaw"`/`"diffract"` forces one implementation
  (and surfaces initialization errors).
- `weights_svd` uses economy SVD, reducing peak memory by up to `max(m, n) / min(m, n)`
  times on rectangular matrices.
- The `ram` profile logs to console only; `hybrid` logs under `.diffract/`; log
  directories are created automatically; configs without a `[logging]` section
  use the console fallback without emitting a startup warning.
- The `dev` extra includes pandas, which the README quick start uses.

### Fixed

- Aggregate filtering by context honors `require_all=False` (match at least one
  requested name) across `models.erase`, `session.filter`, and session merge.
- `ClusterBarChart` plots bins at their numeric centers, so linear and
  logarithmic x scales position bins at their true coordinates, and applies
  the theme's axes styling (frame, grid).
- `HeatmapPlot` treats its `x`/`y` axes as categorical by default, so integer
  dimension fields such as `head_id` and `layer_id` render correctly; `BoxPlot`
  and `ViolinPlot` do the same for their categorical x axis.
- Plot YAML configs can reference data fields by bare strings in style properties
  (`marker_color`, `line_color`, `jitter_color`, `marker_symbol`, `line_dash`):
  at config load, a string that is not a valid plotly literal of the property's
  kind is coerced to a field reference, deterministically and independent of
  session content (literals win on name collisions; an explicit `FieldRef` is
  the escape hatch).
- `expon_concentration` depends on `expon_esd_xmin`.

### Removed

- The accelerated path's numba requirement (the taichi extra covers
  acceleration). Without the taichi extra the p-value kernels are not
  registered.
