# Heavy-tailed fits

Heavy-Tailed Self-Regularization {cite}`martin2021implicit` observes that the ESD
tail of a well-trained layer follows a power law $p(\lambda) \propto
\lambda^{-\alpha}$, and that the exponent $\alpha$ tracks layer quality. These
kernels fit that tail and summarise it. See the [catalog](catalog.md) for every
field.

## Fitting the tail

The power-law fit follows the Clauset-Shalizi-Newman procedure
{cite}`clauset2009`. For a candidate lower cutoff $x_{\min}$, the exponent is the
maximum-likelihood estimate

$$\hat{\alpha} = 1 + n_{\mathrm{tail}}\,\Big/\!\sum_{\lambda_i \ge x_{\min}}
  \ln\frac{\lambda_i}{x_{\min}},$$

where $n_{\mathrm{tail}}$ is the number of eigenvalues in the tail, and
$x_{\min}$ itself is chosen to minimise the Kolmogorov-Smirnov distance between
the empirical tail and the fitted power law. `power_law_fit` returns the
exponent, the cutoff, and the KS distance at the optimum (`pl_alpha`,
`pl_esd_xmin`, `pl_ks`).

Two contrast families are fit the same way. `truncated_power_law_fit` adds an
exponential cutoff, $p(\lambda) \propto \lambda^{-\hat{\alpha}}\,
e^{-\hat{\Lambda}\lambda}$, and `exponential_fit` is the pure light tail with
closed-form MLE $\hat{\Lambda} = 1/(\langle\lambda\rangle_{\ge x_{\min}} - x_{\min})$.

### Fit implementations

The fit kernels accept a `fit_method` argument. `"auto"` uses an accelerated
implementation when the `taichi` extra is installed and the ESD is large enough,
and the reference `powerlaw` library otherwise; `"powerlaw"` and `"diffract"`
force one path. The accelerated fitter only considers $x_{\min}$ candidates that
leave a tail of at least $50$ points, the reliability floor for a power-law tail
{cite}`clauset2009`, so on small spectra it is deliberately more conservative
than the unrestricted `powerlaw` search. Pass `fit_method="powerlaw"` for exact
parity with that library.

## Tail summaries

Concentration and presence exist for every fit, and scale for the two fits that
carry a rate ($\mathrm{TPL}$, $\mathrm{E}$; the power law has none). The catalog
labels the exponent $\alpha$, the cutoff $x_{\min}$, the rate $\Lambda$, and the
KS distance $D$ with the fit family — $\mathrm{PL}$ (power law), $\mathrm{TPL}$
(truncated power law), $\mathrm{E}$ (exponential). The tag is a superscript on
$x_{\min}$, whose subscript is occupied, and a subscript on $\alpha$, $\Lambda$,
and $D$.

- **Concentration** — $\#\{i : \lambda_i \ge x_{\min}\} / M$, the fraction of
  the $M$ eigenvalues in the fitted tail (`pl_concentration`,
  `tpl_concentration`, `expon_concentration`).
- **Presence** — $(\lambda_{\max} - x_{\min}) / (\lambda_{\max} - \lambda_{\min})$,
  the tail's share of the spectrum *width* (`pl_presence`, `tpl_presence`,
  `expon_presence`).
- **Scale** — $\lambda_{\max}\,\Lambda$, the cutoff rate scaled by the observed
  range (`tpl_scale`, `expon_scale`).

## Goodness of fit

The bootstrap p-value $p = \Pr(D^* > D_{\mathrm{obs}})$ (`pl_p_value`,
`tpl_p_value`, `expon_p_value`) draws semi-parametric synthetic data sets — the
tail from the fitted model, the body resampled from the data — and reports the
fraction whose synthetic KS distance $D^*$ strictly exceeds the observed one
$D_{\mathrm{obs}}$ {cite}`clauset2009`. These kernels register only when the
`taichi` extra is installed.

Read the p-value with its caveats:

- **It confirms plausibility; it does not select a family.** The p-value is
  computed at a fixed $x_{\min}$, which biases it upward, and it is not a
  likelihood-ratio test. A large power-law p-value does not establish a power
  law *over* a truncated power law or an exponential.
- **It needs a tail.** With fewer than $50$ tail points the p-value is NaN.

## Conventions and pitfalls

- **KS distances are not comparable across families.** `pl_ks`, `tpl_ks`, and
  `expon_ks` are each measured at that family's own $x_{\min}$ over its own tail;
  a smaller value does not by itself favour one family.
- **Presence saturates.** For a genuinely heavy-tailed layer, `pl_presence`
  clusters in $0.9$–$1.0$; distinguish such layers by `concentration` or
  $x_{\min}$ rather than presence.
- **Accelerated fitter range.** The accelerated implementation returns NaN
  parameters outside $\alpha \in (1, 8]$ rather than clamping, and its
  one-sided KS grid may differ from `powerlaw` by $O(1/n_{\mathrm{tail}})$.
