# Evaluation & metrics

A plausible-looking wind field can still be physically wrong. geo2wf reports three layers of evidence.

## Normalized image metrics

| Metric | Better | Scope |
|---|---|---|
| L1 | lower | valid target pixels in `[0,1]` |
| PSNR | higher | normalized reconstruction fidelity |
| SSIM | higher | structural similarity, masked when needed |
| saturation fraction | lower/contextual | fraction of raw diffusion output at ±1 |

These are useful for optimization diagnostics, not sufficient scientific conclusions.

## Physical and baseline metrics

Predictions are mapped back to m/s using per-channel affine parameters. Metrics include reconstruction MAE/RMSE, ERA5 MAE on common-valid pixels, and:

\[
\mathrm{MAE\ skill\ vs\ ERA5} = 1 - \frac{\mathrm{MAE}_{model}}{\mathrm{MAE}_{ERA5}}
\]

Positive skill means lower MAE than ERA5; zero ties ERA5; negative skill is worse. The residual model also logs signed bias, Huber loss, PSNR in m/s, and high-wind skill above 17 m/s.

## Probabilistic refinement metrics

Residual diffusion evaluates multiple stable samples without averaging away
their detail.

| Metric | Better | Meaning |
|---|---|---|
| `ensemble_crps_ms` | lower | calibrated scalar ensemble error in m/s |
| `ensemble_spread_ms` | contextual | mean per-pixel ensemble standard deviation |
| `ensemble_diversity_ms` | contextual | mean pairwise member difference |
| `ensemble_mean_mae_ms` | lower | MAE of the ensemble mean; expected to look smoother |
| `ensemble_best_member_mae_ms` | lower | best complete member per image, not per-pixel cherry-picking |
| `ensemble_sharpness_ratio` | near 1 | sampled versus observed gradient magnitude |
| `ensemble_log_spectrum_error` | lower | mismatch in masked log-amplitude spectra |
| `probabilistic_refinement_score` | lower | CRPS plus configured spectrum and sharpness penalties |

## Storm-centric structure

All storm metrics use only observed target pixels and geographic distance from
the IBTrACS center. The same center metadata creates the normalized
`distance_to_ibtracs_center` condition channel. The center is not a target or
output of either neural network.

| Metric | Definition |
|---|---|
| `eye_mae_ms` | MAE inside 25 km |
| `eye_mean_wind_error_ms` | absolute error in eye-mean wind |
| `inner_core_mae_ms` | MAE inside 100 km |
| `radial_profile_mae_ms` | mean error between 10 km radial-bin profiles out to 200 km |
| `rmw_error_km` | radius-of-maximum-wind difference from radial profiles |
| `eye_to_eyewall_contrast_error_ms` | error in peak radial wind minus eye mean |
| `eye_center_displacement_km` | distance between smoothed predicted and target wind minima |
| `high_wind_mae_ms` | MAE where target wind is at least 17 m/s |

## Eye-center displacement

`eye_center_displacement_km` answers a deliberately narrow question:

> When the observed SAR field contains a sufficiently clear and well-sampled
> low-wind eye, how far is the minimum of the reconstructed wind field from the
> minimum of the observed wind field?

It does **not** measure the prediction against the IBTrACS point directly.
IBTrACS supplies a trustworthy geographic anchor and constrains the region in
which the algorithm is allowed to look. The two locations actually compared
are inferred from the target and predicted wind fields.

### 1. Metadata establishes the geographic frame

For each sample, the exported manifest carries an IBTrACS center

\[
\mathbf{c}=(\phi_c,\lambda_c)
\]

in latitude and longitude. The target GeoTIFF supplies bounds

\[
(\lambda_L,\lambda_R,\phi_B,\phi_T)
\]

for a raster of height \(H\) and width \(W\). Evaluation reconstructs the
geographic coordinate of pixel center \((i,j)\):

\[
\phi_i
=
\phi_T-\frac{i+\tfrac12}{H}(\phi_T-\phi_B),
\qquad
\lambda_j
=
\lambda_L+\frac{j+\tfrac12}{W}(\lambda_R-\lambda_L).
\]

The half-pixel terms matter: raster values describe cells, so evaluation uses
cell centers rather than their outer edges.

Longitude difference is wrapped into \([-180^\circ,180^\circ)\),

\[
\Delta\lambda_j
=
\left((\lambda_j-\lambda_c+180^\circ)\bmod 360^\circ\right)-180^\circ,
\]

which prevents a storm near the dateline from appearing almost \(360^\circ\)
away. A local east/north coordinate system in kilometres is then

\[
n_{ij}
=
R\,\operatorname{rad}(\phi_i-\phi_c),
\qquad
e_{ij}
=
R\,\operatorname{rad}(\Delta\lambda_j)\cos(\operatorname{rad}\phi_c),
\]

\[
r_{ij}=\sqrt{e_{ij}^2+n_{ij}^2},
\qquad R=6371\ \mathrm{km}.
\]

This is the local equirectangular approximation. It is appropriate for these
tropical-cyclone diagnostics because the eye search is restricted to 100 km
and the other radial metrics stop at 200 km.

### 2. Smoothing makes the minimum less pixel-sensitive

Let \(X_{ij}\) be either the target or predicted physical wind speed in m/s,
and let \(M_{ij}\in\{0,1\}\) mark pixels where the SAR target and both fields
are finite. For the 3×3 neighbourhood \(\mathcal N_{ij}\), evaluation computes
a masked mean

\[
\widetilde X_{ij}
=
\frac{
  \sum_{(u,v)\in\mathcal N_{ij}}M_{uv}X_{uv}
}{
  \sum_{(u,v)\in\mathcal N_{ij}}M_{uv}
}.
\]

A smoothed pixel is eligible only when at least 8 of its 9 neighbours are
valid. Prediction and target therefore use the same well-supported observed
area. The smoothing is not a learned operation and does not change the model
output; it is used only to locate robust minima for this diagnostic.

### 3. Locate the observed and reconstructed eyes

Define the search disk and reference ring around IBTrACS:

\[
\mathcal S=\{(i,j):r_{ij}\le100\ \mathrm{km}\},
\]

\[
\mathcal A=\{(i,j):20\ \mathrm{km}\le r_{ij}<60\ \mathrm{km}\}.
\]

Among eligible pixels in \(\mathcal S\), the target-eye and predicted-eye
indices are

\[
p_{\mathrm{target}}
=
\underset{p\in\mathcal S}{\arg\min}\;
\widetilde y_p,
\qquad
p_{\mathrm{pred}}
=
\underset{p\in\mathcal S}{\arg\min}\;
\widetilde{\hat y}_p.
\]

This definition uses the physical structure of a mature cyclone: a relatively
low-wind eye surrounded by stronger winds. It does not claim that the global
minimum anywhere in a wind map is a storm center; the spatial and quality
gates below are essential parts of the definition.

### 4. Measure displacement in kilometres

The per-sample diagnostic is the distance between those two minima in the same
local coordinate frame:

\[
d_{\mathrm{eye}}
=
\sqrt{
  \left(e_{p_{\mathrm{pred}}}-e_{p_{\mathrm{target}}}\right)^2
  +
  \left(n_{p_{\mathrm{pred}}}-n_{p_{\mathrm{target}}}\right)^2
}.
\]

Consequently, a value of 30 km means that the reconstructed low-wind eye is
30 km from the observed SAR low-wind eye. It does not mean that either eye is
30 km from IBTrACS. The implementation currently retains only this scalar
distance; it does not log the inferred eye latitude/longitude or displacement
direction.

### 5. Reject cases without defensible eye evidence

The metric is available for a sample only when all of the following hold.

**Coverage.** At least 80% of pixels in both \(\mathcal S\) and
\(\mathcal A\) are observed and valid:

\[
\frac{\sum_{p\in\mathcal R}M_p}{|\mathcal R|}\ge0.8,
\qquad \mathcal R\in\{\mathcal S,\mathcal A\}.
\]

**Storm strength.** The observed mean wind in the 20–60 km reference ring is
at least 17 m/s:

\[
\mu_{\mathcal A}
=
\frac{\sum_{p\in\mathcal A}M_p y_p}
     {\sum_{p\in\mathcal A}M_p}
\ge17\ \mathrm{m\,s^{-1}}.
\]

**Track proximity.** The observed minimum is plausibly close to the track
center:

\[
r_{p_{\mathrm{target}}}\le50\ \mathrm{km}.
\]

**Eye contrast.** The observed eye has at least 5 m/s contrast against the
reference ring:

\[
\mu_{\mathcal A}-\widetilde y_{p_{\mathrm{target}}}
\ge5\ \mathrm{m\,s^{-1}}.
\]

**Geometric support.** The search disk, reference ring, and at least one
8-of-9-supported smoothing location must exist.

These gates prevent incomplete SAR swaths, weak storms, flat wind fields, and
unrelated outer minima from being assigned a falsely precise eye location.
They deliberately gate on evidence in the observed target. There is currently
no separate predicted-eye contrast threshold; within an accepted sample, the
predicted location is simply the lowest eligible predicted value inside
100 km.

### 6. Availability, aggregation, and model use

If any gate fails, the metric is **unavailable**, not zero. Across an epoch,
the logged result is therefore

\[
\overline d_{\mathrm{eye}}
=
\frac{1}{N_{\mathrm{available}}}
 \sum_{k\in\mathrm{available}}d_{\mathrm{eye},k}.
\]

Sums and available-sample counts are reduced across distributed ranks before
forming this mean. Report \(N_{\mathrm{available}}\) when comparing experiments:
two means computed from very different accepted subsets are not equally
informative.

Neither model receives \((\phi_c,\lambda_c)\) as input. They reconstruct a
complete wind-speed field from imagery and optional ERA5 context; evaluation
then derives the two eye locations from that field. The displacement is:

- logged during validation and test;
- excluded from the training loss;
- excluded from `eye_structure_score` and checkpoint selection; and
- not returned as a center-coordinate prediction.

The deterministic ERA5-residual model evaluates storm metrics over every
validation and test batch. Diffusion test evaluation also covers every test
batch, but diffusion validation computes reconstruction-based storm metrics
only for `validation.reconstruction_batches`, because each reverse diffusion
sample is expensive. That coverage difference must be recorded when comparing
validation results.

## Exact epoch aggregation

Pixel metrics accumulate sums and counts, then reduce across distributed ranks before forming means. Storm metrics accumulate per-sample metric sums and available counts. Unavailable metrics are omitted instead of being treated as zero.

## Recommended comparison table

For a serious model comparison, report at least:

1. observed-pixel MAE/RMSE in m/s;
2. ERA5 MAE and skill on the same common-valid pixels;
3. high-wind MAE;
4. eye, inner-core, radial-profile, and RMW error with sample counts;
5. qualitative reconstructions using fixed latents; and
6. split policy, number of storms, normalization, and sampling settings.

!!! caution "The checked-in split policy"
    Presets combine the test split into training. Do not label resulting test metrics as held-out generalization unless `include_test_in_train` was disabled before training.
