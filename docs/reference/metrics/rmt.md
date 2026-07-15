# Random matrix theory (Marchenko-Pastur and Tracy-Widom)

Random matrix theory supplies the null model for a structureless weight matrix:
where its eigenvalues should lie if the entries were independent noise, and how
far the largest may stray by chance. Eigenvalues beyond that null are *spikes* —
the learned signal. See the [catalog](catalog.md) for every field; the
distributions themselves are implemented in
`diffract.core.compute.extensions.rmt`.

## The Marchenko-Pastur bulk

For a matrix with independent entries of variance $\sigma^2$ and aspect ratio
$Q = N/M$, the eigenvalues of the correlation matrix fill a bulk bounded by the
Marchenko-Pastur edges {cite}`marchenko1967`

$$\lambda_\pm = \sigma^2\,\big(1 \pm 1/\sqrt{Q}\big)^2 .$$

`marchenko_pastur_fit` estimates the bulk from the permutation null
[`esd_rand`](spectral.md): it takes the noise scale from the standard deviation
of the weight entries (`weights_std`), then corrects for eigenvalues that bleed
past the edge using the trace of the randomized bulk, and returns the edges
$\lambda_+$ (`mp_esd_max`), $\lambda_-$ (`mp_esd_min`) and the fitted bulk
deviation $\sigma_{\mathrm{b}}$ (`mp_bulk_std`). `mp_sval_max` reports the same
upper edge as a singular value, $\sqrt{\lambda_+ N}$.

### Fit quality and coverage

- **`mp_ks`** is the two-sided Kolmogorov-Smirnov distance between the empirical
  eigenvalues in the bulk window and the Marchenko-Pastur CDF conditioned on that
  same window, $D = \sup_\lambda \lvert \hat{F}(\lambda) - F_{\mathrm{MP}}(\lambda)
  \rvert$, where $\hat{F}$ is the empirical CDF of the in-window eigenvalues and
  $F_{\mathrm{MP}}$ the Marchenko-Pastur CDF. It returns the sentinel $1$ when the
  bulk window is empty, signalling
  that the fit does not apply rather than a genuine statistic.
- **`mp_concentration`** is the fraction of eigenvalues inside the bulk, and
  **`mp_presence`** the bulk width as a fraction of the spectrum width.

## Spikes: the BBP transition

An eigenvalue detaches from the bulk once the underlying signal crosses the
Baik-Ben Arous-Péché threshold {cite}`baik2005`. `mp_num_spikes` counts the
eigenvalues above the bulk edge, $\#\{i : \lambda_i > \lambda_+\}$ — the standard
Heavy-Tailed Self-Regularization spike count {cite}`martin2021implicit`.

## The Tracy-Widom edge

The largest eigenvalue of pure noise does not sit exactly at $\lambda_+$: it
fluctuates around a soft edge with Tracy-Widom statistics {cite}`tracy1994`.
Following {cite}`johnstone2001`, `tw_esd_bound` centres and scales that soft
edge and places the threshold at an upper-tail quantile,

$$\lambda_{\mathrm{TW}} = \mu_{NM} + s_{NM}\,F_{\mathrm{TW}}^{-1}(1 - p),$$

where the centring $\mu_{NM}$ and scale $s_{NM}$ are the Johnstone soft-edge
constants built from $N$, $M$, and $\sigma_{\mathrm{b}}$, $F_{\mathrm{TW}}^{-1}$
is the Tracy-Widom quantile function, and $p$ is the tail probability
`p_value_threshold`. `tw_num_spikes` then counts eigenvalues above this
statistically calibrated edge, $\#\{i : \lambda_i > \lambda_{\mathrm{TW}}\}$.

## Conventions and pitfalls

- **Two spike counters, two working points.** `mp_num_spikes` counts above the
  deterministic bulk edge; `tw_num_spikes` counts above the Tracy-Widom edge,
  which accounts for finite-size fluctuations of the largest eigenvalue.
  `tw_num_spikes` is the statistically calibrated version; the two are not
  duplicates.
- **The TW threshold is not a family-wise rate.** `p_value_threshold` is the
  upper-tail probability for the largest eigenvalue of pure noise, not a
  correction over all $k$ candidates; for $k > 1$ the count is slightly liberal
  because the test is not sequential.
- **`mp_presence` bounds.** Finite-size fluctuations at the lower edge can push
  the raw bulk width above the spectrum width; the reported value is clipped to
  $[0, 1]$.
- **Partition.** In general `mp_concentration` plus the spike fraction does not
  sum to $1$: eigenvalues can also fall below the bulk.

The Marchenko-Pastur CDF and the Tracy-Widom quantile are vendored and validated
against published quantiles; their derivations are documented in the
`extensions.rmt` module docstring.
