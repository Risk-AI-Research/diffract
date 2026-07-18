# Alignment (cross-model)

Alignment metrics compare a parameter's singular subspaces across two
checkpoints: how far the left and right singular vectors of the same layer have
moved. They are `CROSS_MODEL` kernels — each compares **exactly two models**, so
scope the session to the pair before applying,
`session.filter(model_ids=[a, b]).compute.apply("l_agreement")`; any other scope
size raises `ScopeValidationError`. See the
[Apply levels](../../guide/recipes/kernels_and_compute.md#apply-levels) guide for the
pairwise workflow and the [catalog](catalog.md) for every field.

## Overlap

For two checkpoints, `l_overlap` and `r_overlap` form the absolute overlap
matrices of the left ($U$) and right ($V$) singular-vector bases,

$$O^{L} = \lvert U_1^\top U_2\rvert, \qquad O^{R} = \lvert V_1^\top V_2\rvert.$$

The absolute value is essential: the sign of each singular vector is an arbitrary
gauge of the SVD, so only the absolute overlap is an invariant of the checkpoint
pair. Entry $O_{ij} \in [0, 1]$ is the absolute cosine of the angle between
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
- **Best-match vs. diagonal.** Both read the same absolute overlap $O$, so both
  are invariant to the SVD sign gauge. `max_*` matches greedily without enforcing
  a bijection (a maximum over each row), while the diagonal `*_agreement` compares
  like indices directly. Use `max_*` when components may have permuted between
  checkpoints.
