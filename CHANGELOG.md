# Changelog

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
