# Alignment (cross-model)

Alignment metrics compare a parameter's singular subspaces across two
checkpoints: how far the left and right singular vectors of the same layer have
moved. Each pairwise kernel uses exactly two model/checkpoint coordinates.
For two models at step zero, use
`session.filter(model_ids=[a, b], steps=[0]).compute.apply("l_agreement")`.
For one model at two steps, select its name and those two steps. Implicit comparisons require exactly two
checkpoints. Use `pairs=` to select ordered endpoints in a larger scope. See the
[Apply levels](../../guide/recipes/kernels_and_compute.md#apply-levels) guide for the
pairwise workflow and the [catalog](catalog.md) for every field.

## Which parameters were compared?

Request coverage to inspect the join before interpreting a comparison.

```python
from diffract import compare

rows, report = compare(
    source, target,
    fields=("relative_weight_change", "relative_spectrum_change"),
    return_coverage=True,
)
print(report.source_count, report.target_count)
print(report.matched, report.source_only, report.target_only)
print(report.measured["relative_weight_change"])
print(report.incompatible["relative_weight_change"])
```

The join uses stored parameter names. A renamed layer appears once on each
side under its respective name. `source_count` and `target_count` count stored
names in the selected scope. The complete admission records in
`source_extraction.items` and `target_extraction.items` include source names,
exclusions, conversion outcomes, and reasons. In a session, a checkpoint
without a recorded admission has `None` for that extraction report.

With coverage enabled, each field uses its compatible pairs. The weight-change
field requires both dimensions to match. Spectrum change requires equal
spectrum lengths. Left and right agreement also require equal row counts and
column counts, respectively. Each shape mismatch lists the name and both
matrix shapes. An empty intersection yields empty rows and zero measurements.
If a stateless comparison extracts no weights from an endpoint, the coverage
report retains the input outcomes from both sides.
Pass `required_names=("head.weight",)` to require a match for that stored name
across all requested fields.

In `compare(source, target)`, each measured parameter contributes two long rows
per field for the requested orientation, sharing one `aggregate_uid`. A session
export that includes both orientations of a symmetric field has four rows for
that measurement.
`report.measurement_counts` counts that measurement once per field.
In a session, `session.compute.apply(*fields, pairs=[(("net", 10), ("net", 20))],
return_coverage=True).pair_coverage` contains one report per ordered pair.

## Overlap

Write $s$ for the source checkpoint and $t$ for the target. For these checkpoints, `l_overlap` and `r_overlap` form the absolute overlap
matrices of the left ($U$) and right ($V$) singular-vector bases,

$$O^{L} = \lvert U_s^\top U_t\rvert, \qquad O^{R} = \lvert V_s^\top V_t\rvert.$$

The elementwise absolute value removes the arbitrary sign of each singular
vector. When singular values are tied, a rotation within their subspace can
change individual entries. Use principal angles to compare the subspaces.
Entry $O_{ij} \in [0, 1]$ is the absolute cosine of the angle between
component $i$ of the first model and component $j$ of the second.

## Summaries of the overlap

Each summary applies to both bases; write $O$ for either $O^{L}$ or $O^{R}$.

- **Diagonal agreement** — $O_{ii}$ (`l_agreement`, `r_agreement`): how well
  component $i$ overlaps the *same-index* component in the other model.
- **Best-match agreement** — $\max_j O_{ij}$ (`max_l_agreement`,
  `max_r_agreement`): the best overlap of component $i$ with *any* component,
  answering "where did this direction go" under a permutation of indices.
- **Means** — $\langle O_{ii}\rangle$ (`avg_l_agreement`, `avg_r_agreement`) and
  $\langle \max_j O_{ij}\rangle$ (`avg_max_l_agreement`, `avg_max_r_agreement`),
  the layer-level averages of the two agreements.

## Relative changes

Let $\boldsymbol{\sigma}(W)=(\sigma_1,\ldots,\sigma_M)$ be the singular values in
ascending order, and write $\boldsymbol{\sigma}_s$ and $\boldsymbol{\sigma}_t$ for the
source and target spectra.

- `relative_weight_change` measures
  $\lVert W_t-W_s\rVert_F/\lVert W_s\rVert_F$ for equal-shaped matrices.
- `relative_spectrum_change` measures
  $\lVert\boldsymbol{\sigma}_t-\boldsymbol{\sigma}_s\rVert_2/\lVert\boldsymbol{\sigma}_s\rVert_2$ for the
  sorted singular values of the two checkpoints. It responds to a change
  of scale and is invariant to orthogonal rotations of the weights.

Both quantities use the source as their denominator and return NaN when
that denominator is zero. Use the two together to distinguish spectral
drift from a change in the matrix at fixed spectrum. The default
`diffract.compare(source, target)` also includes the four per-component
agreement profiles.

For a checkpoint obtained by fine-tuning, place it first when looking for
directions with a weak match in the base checkpoint. For example,
`diffract.compare(tuned, base, fields=("max_r_agreement",))` returns the
best base match for each right singular vector of the tuned weights. Inspect
the leading components and choose a similarity threshold for the study.
[LoRA vs Full Fine-tuning (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html)

## Conventions and pitfalls

- **Read against the noise floor.** Two unrelated bases still overlap by chance.
  A single diagonal entry $O_{ii}$ is then $\sim 1/\sqrt{d}$ ($d$ the ambient
  dimension of the singular vectors), while the best-match $\max_j O_{ij}$ over
  $k$ components rises to $\sim \sqrt{2\ln k / d}$. An "overlap of $0.3$" is
  uninterpretable without the matching baseline.
- **Bulk components are background.** Away from the top singular directions the
  per-component agreement sits at $\lvert a_i\rvert \sim 1/\sqrt{d}$ — floor, not
  signal. Alignment is informative for the leading components.
- **Indexing.** Components are ordered by ascending singular value, so index $0$
  is the weakest direction and the top-$k$ are the last columns.
- **Nearly tied singular values.** Individual vectors can rotate within a
  group of nearly equal singular values even when the group still spans the
  same subspace. Use a subspace comparison to assess such groups.
- **Best-match vs. diagonal.** Both read the same absolute overlap $O$, so both
  are invariant to the SVD sign gauge. `max_*` matches greedily without enforcing
  a bijection (a maximum over each row), while the diagonal `*_agreement` compares
  like indices directly. Use `max_*` when components may have permuted between
  checkpoints.
