# laplace-gaussian_processes

A [Laplace](https://github.com/mlatinov/laplace) library of Gaussian process building blocks for Stan — twenty covariance kernels, marginal and latent GP models, posterior prediction, and helpers for hierarchical (per-group) GPs. Import it into any `.laplace` model and call it with namespaced calls (`gaussian_process::function_name(...)`).

Like all Laplace libraries, `gaussian_process` compiles down to plain, readable Stan functions. Nothing about how you use it hides what actually ends up in your `.stan` file.

## How it's organised

A Gaussian process model has three separate pieces, and the library keeps them separate:

1. **A kernel** $k(x, x')$ describes how correlated the function is at two inputs. Every kernel function returns the full covariance matrix $K_{ij} = k(x_i, x_j)$.
2. **A model builder** turns $K$ into something you can put in a model: either the marginal likelihood (Gaussian data, $f$ integrated out) or a non-centred latent function (any likelihood).
3. **Prediction** conditions on the fit to give the function at new inputs, in `generated quantities`.

Because kernels are just matrices, you combine them with ordinary Stan arithmetic, and the model builders never need to know which kernel you chose:

```stan
// smooth trend + a seasonal pattern that drifts slowly
matrix[N, N] K = gaussian_process::rbf_cov(t, alpha_trend, rho_trend)
               + gaussian_process::local_periodic_cov(t, alpha_seas, rho_seas, period, rho_decay);
```

A **sum** (`+`) means independent components added together. A **product** (`.*`, elementwise) means one component modulates another.

## Kernels

Every kernel comes in two forms: `<name>_cov(x, ...)` returns the $N \times N$ matrix for one set of inputs, and `<name>_cross_cov(x1, x2, ...)` returns the $N_1 \times N_2$ matrix between two sets, which is what prediction needs. Both take the same hyperparameters in the same order: inputs, amplitude, length-scale(s), then extras.

In the formulas, $r = \lvert x - x' \rvert$ is the distance between inputs and $\tau = x - x'$ the signed difference. $\alpha$ is always the marginal standard deviation (for stationary kernels, $k(x, x) = \alpha^2$) and $\rho$ a length-scale: small $\rho$ gives wiggly functions, large $\rho$ smooth ones.

### Stationary kernels

These depend only on the distance between inputs, so the function behaves the same everywhere.

| Kernel | $k(x, x')$ | Arguments after `x` | Notes |
| --- | --- | --- | --- |
| `rbf` | $\alpha^2 \exp\left(-\frac{r^2}{2\rho^2}\right)$ | `alpha, rho` | Squared exponential. Infinitely smooth; wraps `gp_exp_quad_cov` |
| `matern12` | $\alpha^2 e^{-r/\rho}$ | `alpha, rho` | Rough, nowhere differentiable; wraps `gp_exponential_cov` |
| `matern32` | $\alpha^2 \left(1 + \frac{\sqrt{3} r}{\rho}\right) e^{-\sqrt{3} r/\rho}$ | `alpha, rho` | Once differentiable; wraps `gp_matern32_cov` |
| `matern52` | $\alpha^2 \left(1 + \frac{\sqrt{5} r}{\rho} + \frac{5 r^2}{3\rho^2}\right) e^{-\sqrt{5} r/\rho}$ | `alpha, rho` | Twice differentiable, a good general default; wraps `gp_matern52_cov` |
| `ornstein_uhlenbeck` | $\alpha^2 e^{-r/\rho}$ | `alpha, rho` | Identical to `matern12`, under its process name |
| `rational_quad` | $\alpha^2 \left(1 + \frac{r^2}{2 a \rho^2}\right)^{-a}$ | `alpha, rho, a` | Mixture of RBF kernels over many length-scales; becomes `rbf` as $a \to \infty$ |
| `periodic` | $\alpha^2 \exp\left(-\frac{2\sin^2(\pi r / p)}{\rho^2}\right)$ | `alpha, rho, p` | Repeats exactly every period `p`; wraps `gp_periodic_cov` |
| `local_periodic` | periodic $\times$ RBF envelope | `alpha, rho, p, rho_decay` | Seasonal shape that drifts, at a speed set by `rho_decay` |
| `rq_periodic` | periodic $\times$ RQ envelope | `alpha, rho, p, rho_rq, a` | Seasonal shape that drifts at several speeds |
| `damped_periodic` | $\alpha^2 \cos\left(\frac{2\pi\tau}{p}\right) e^{-r/\rho}$ | `alpha, rho, p` | Oscillation whose correlation decays with distance |
| `cosine` | $\alpha^2 \cos\left(\frac{2\pi\tau}{p}\right)$ | `alpha, p` | A single pure sinusoid (rank 2) |
| `spectral_mixture` | $\sum\_{q} w\_q \, e^{-2\pi^2 \tau^2 v\_q} \cos(2\pi\tau\mu\_q)$ | `w, mu, v` (vectors, length Q) | Gaussian mixture in frequency space (Wilson & Adams, 2013); `mu` are frequencies (1 / period) |
| `white` | $\sigma^2 \, \mathbf{1}[x = x']$ | `sigma` | Independent noise at each point; the cross form is all zeros |
| `constant` | $\sigma^2$ | `sigma` | A random offset shared by every point (rank 1) |

### Non-stationary kernels

These depend on the input values themselves, not just their distance.

| Kernel | $k(x, x')$ | Arguments after `x` | Notes |
| --- | --- | --- | --- |
| `linear` | $\sigma\_b^2 + \sigma\_v^2 (x - c)(x' - c)$ | `sigma_b, sigma_v, c` | Bayesian linear regression; `c` is where the variance is smallest (rank 2) |
| `polynomial` | $\left(\sigma\_v^2 \, x x' + c\right)^{d}$ | `sigma_v, c, d` | Polynomials of integer degree `d`; here `c >= 0` is an offset, not a centre |
| `arccosine` | $1 - \theta / \pi$, with $\theta$ the angle between $x$ and $x'$ | none | Order-0 arc-cosine kernel (Cho & Saul, 2009); takes `array[] vector` inputs |
| `wiener` | $\sigma^2 \min(x, x')$ | `sigma` | Brownian motion started at 0; times must be $\geq 0$ |
| `brownian_bridge` | $\sigma^2 \left(\min(x, x') - \frac{x x'}{T}\right)$ | `sigma, T` | Brownian motion pinned to 0 at times 0 and `T` |
| `integrated_ornstein_uhlenbeck` | $\frac{\alpha^2}{2\theta^3}\left(2\theta \min(s,t) + e^{-\theta s} + e^{-\theta t} - 1 - e^{-\theta \lvert s-t \rvert}\right)$ | `alpha, theta` | Integral of an OU process (Taylor, Cumberland & Sy, 1994); `theta` is the mean-reversion rate |

### Choosing a smoothness

`matern12`, `matern32`, `matern52` and `rbf` are one family at increasing smoothness (RBF is the Matérn limit $\nu \to \infty$). Picking among them is really picking how smooth you believe the underlying function is. RBF is the classic default but is often too smooth for real data, which makes it overconfident between observations; `matern52` is a safer starting point.

The length-scale $\rho$ means roughly, but not exactly, the same thing across the family, so a prior tuned for one is a reasonable start for the others. It does **not** carry over to `periodic`, where $\rho$ measures wiggliness within one period.

## What's included

Besides the kernels, the library has three groups of functions.

**Model builders**, used in `model` or `transformed parameters`:

| Function | Returns | Use it when |
| --- | --- | --- |
| `marginal_normal_lpdf(y \| mu, K, sigma)` | Log density of $y \sim \mathcal{N}(\mu, K + \sigma^2 I)$ | The likelihood is Gaussian. $f$ is integrated out analytically, so it isn't sampled at all |
| `latent(K, z, delta)` | $f = \operatorname{chol}(K + \delta I)\, z$ | Any other likelihood (Poisson, Bernoulli, ...), or per-group GPs. Give `z ~ std_normal()` |

**Prediction**, used in `generated quantities`:

| Function | Returns |
| --- | --- |
| `predict_marginal_mean(y, K, K_cross, sigma)` | Posterior mean of $f$ at the new points (no randomness) |
| `predict_marginal_rng(y, K, K_cross, K_star, sigma, delta)` | One posterior draw of $f$ at the new points, for the marginal model |
| `predict_latent_rng(f, K, K_cross, K_star, delta)` | One conditional draw of $f$ at the new points, for the latent model |

All three use the standard Gaussian conditioning result, computed with triangular solves rather than matrix inverses:

$$
f_* \mid y \sim \mathcal{N}\left(K_\times^\top K_y^{-1} y,\; K_* - K_\times^\top K_y^{-1} K_\times\right),
$$

where $K_y$ is `K` plus $\sigma^2 I$ (marginal) or $\delta I$ (latent), $K_\times$ is `K_cross` ($N \times N_{new}$) and $K_*$ is `K_star` ($N_{new} \times N_{new}$).

**Grouping helpers**, used once in `transformed data`, to fit a separate GP per group without scanning every row inside the model:

| Function | Returns |
| --- | --- |
| `group_sizes(id, J)` | Number of observations in each of the `J` groups |
| `group_starts(sizes)` | Where each group begins in the group-sorted order |
| `group_rows(ord, starts, sizes, j)` | The original row indices of group `j`, where `ord = sort_indices_asc(id)` |

Every function carries `@brief`, `@param`, `@return`, `@math`, and (where useful) `@example` documentation, so you can read it from the terminal without leaving your model:

```
laplace doc gaussian_process::matern52_cov
```

## Installation

`gaussian_process` is distributed as a git-hosted Laplace library — there's no published registry entry yet, so it's added by pointing `laplace` (or `cmdlaplacer`, if you're working from R) directly at the repository. The package lives in the repository's `laplace/` subdirectory, so pass it as the subdir.

### Via the `laplace` CLI

From inside a Laplace project (a directory with its own `laplace.toml`):

```
laplace add gaussian_process --git https://github.com/mlatinov/laplace-gaussian_processes- --tag 0.1.0 --subdir laplace
```

### Via R (`cmdlaplacer`)

```r
library(cmdlaplacer)

laplace_install_git(
  "gaussian_process",
  "https://github.com/mlatinov/laplace-gaussian_processes-",
  tag = "0.1.0",
  subdir = "laplace"
)
```

Either way, this pins the dependency in your project's `laplace.toml`/`laplace.lock` at tag `0.1.0`. Check the [tags](https://github.com/mlatinov/laplace-gaussian_processes-/tags) for newer versions as they become available.

## Usage

Import the library in a `library { }` block and call its functions with the `gaussian_process::` namespace prefix.

### GP regression with prediction

A Matérn 5/2 GP fitted with the marginal likelihood, with posterior draws of the function and of new observations at `x_new`:

```stan
library {
    import gaussian_process
}

data {
  int<lower=1> N;
  array[N] real x;                     // inputs
  vector[N] y;                         // observations (centred, so the GP mean is 0)
  int<lower=1> N_new;
  array[N_new] real x_new;             // prediction points
}

parameters {
  real<lower=0> alpha;                 // marginal standard deviation of f
  real<lower=0> rho;                   // length-scale
  real<lower=0> sigma;                 // observation noise
}

model {
  matrix[N, N] K = gaussian_process::matern52_cov(x, alpha, rho);

  alpha ~ std_normal();
  rho   ~ inv_gamma(5, 5);
  sigma ~ std_normal();

  target += gaussian_process::marginal_normal_lpdf(y | rep_vector(0, N), K, sigma);
}

generated quantities {
  vector[N_new] f_new = gaussian_process::predict_marginal_rng(
    y,
    gaussian_process::matern52_cov(x, alpha, rho),                 // K
    gaussian_process::matern52_cross_cov(x, x_new, alpha, rho),    // K_cross
    gaussian_process::matern52_cov(x_new, alpha, rho),             // K_star
    sigma, 1e-9
  );
  array[N_new] real y_new = normal_rng(f_new, sigma);
}
```

Switching kernels means changing `matern52` in all four places, for example to `rbf` or `matern32`. Keep them in sync: the kernel in `generated quantities` must match the one in `model`.

### Composite kernel: trend plus seasonality

A long-term trend plus a seasonal pattern whose shape drifts over time, for a series with a known period:

```stan
library {
    import gaussian_process
}

data {
  int<lower=1> N;
  array[N] real t;                     // time
  vector[N] y;                         // centred series
  real<lower=0> period;                // known period, e.g. 12 for monthly data
}

parameters {
  real<lower=0> alpha_trend;
  real<lower=0> rho_trend;
  real<lower=0> alpha_seas;
  real<lower=0> rho_seas;              // wiggliness within one period
  real<lower=0> rho_decay;             // how fast the seasonal shape drifts
  real<lower=0> sigma;
}

model {
  matrix[N, N] K = gaussian_process::rbf_cov(t, alpha_trend, rho_trend)
                 + gaussian_process::local_periodic_cov(t, alpha_seas, rho_seas, period, rho_decay);

  alpha_trend ~ std_normal();
  rho_trend   ~ inv_gamma(5, 50);      // long, slow trend
  alpha_seas  ~ std_normal();
  rho_seas    ~ lognormal(0, 0.5);
  rho_decay   ~ inv_gamma(5, 50);
  sigma       ~ std_normal();

  target += gaussian_process::marginal_normal_lpdf(y | rep_vector(0, N), K, sigma);
}
```

Any valid kernels can be summed this way, e.g. `linear_cov(...) + matern52_cov(...)` for a straight-line trend with smooth deviations.

### Latent GP with a Poisson likelihood

When the data aren't Gaussian, $f$ can't be integrated out, so it is sampled through the non-centred parameterization $f = L z$:

```stan
library {
    import gaussian_process
}

data {
  int<lower=1> N;
  array[N] real x;
  array[N] int<lower=0> y;             // counts
  int<lower=1> N_new;
  array[N_new] real x_new;
}

transformed data {
  real delta = 1e-9;
}

parameters {
  real mu;                             // baseline log rate
  real<lower=0> alpha;
  real<lower=0> rho;
  vector[N] z;                         // non-centred latent GP
}

transformed parameters {
  vector[N] f = gaussian_process::latent(gaussian_process::rbf_cov(x, alpha, rho), z, delta);
}

model {
  mu    ~ normal(0, 2);
  alpha ~ std_normal();
  rho   ~ inv_gamma(5, 5);
  z     ~ std_normal();

  y ~ poisson_log(mu + f);
}

generated quantities {
  vector[N_new] f_new = gaussian_process::predict_latent_rng(
    f,
    gaussian_process::rbf_cov(x, alpha, rho),
    gaussian_process::rbf_cross_cov(x, x_new, alpha, rho),
    gaussian_process::rbf_cov(x_new, alpha, rho),
    delta
  );
  array[N_new] int y_new = poisson_log_rng(mu + f_new);
}
```

The same pattern works for any likelihood: swap `poisson_log` for `bernoulli_logit`, `neg_binomial_2_log`, and so on.

### Hierarchical GP: one curve per group

A separate GP for each group, with length-scales and amplitudes partially pooled across groups. The group bookkeeping depends only on data, so it runs once in `transformed data`:

```stan
library {
    import gaussian_process
}

data {
  int<lower=1> N;
  int<lower=1> J;                      // number of groups
  array[N] int<lower=1, upper=J> id;   // group of each observation
  array[N] real x;
  vector[N] y;
}

transformed data {
  real delta = 1e-9;
  array[J] int sizes  = gaussian_process::group_sizes(id, J);
  array[J] int starts = gaussian_process::group_starts(sizes);
  array[N] int ord    = sort_indices_asc(id);
}

parameters {
  real b0;
  real<lower=0> sigma;

  // Partially pooled GP hyperparameters, one pair per group (non-centred)
  real mean_log_rho;
  real<lower=0> tau_log_rho;
  vector[J] z_rho;
  real mean_log_alpha;
  real<lower=0> tau_log_alpha;
  vector[J] z_alpha;

  vector[N] z;                         // latent GP innovations
}

transformed parameters {
  vector[J] rho   = exp(mean_log_rho + tau_log_rho * z_rho);
  vector[J] alpha = exp(mean_log_alpha + tau_log_alpha * z_alpha);

  vector[N] f;
  for (j in 1:J) {
    array[sizes[j]] int idx = gaussian_process::group_rows(ord, starts, sizes, j);
    f[idx] = gaussian_process::latent(
      gaussian_process::rbf_cov(x[idx], alpha[j], rho[j]), z[idx], delta
    );
  }
}

model {
  b0 ~ normal(0, 2);
  sigma ~ std_normal();
  mean_log_rho ~ normal(0, 1);
  tau_log_rho ~ normal(0, 0.5);
  z_rho ~ std_normal();
  mean_log_alpha ~ normal(0, 1);
  tau_log_alpha ~ normal(0, 0.5);
  z_alpha ~ std_normal();
  z ~ std_normal();

  y ~ normal(b0 + f, sigma);
}
```

Each group's values are gathered with `x[idx]`, turned into a GP, and written back to their original rows with `f[idx] = ...`, so the data don't need to be sorted by group.

### From R

With `cmdlaplacer`, the `.laplace` file compiles straight to a `cmdstanr` model, and the generated `.stan` file stays on disk next to it:

```r
library(cmdlaplacer)

mod <- laplace_model("gp_regression.laplace")
fit <- mod$sample(
  data = list(N = length(x), x = x, y = y - mean(y),
              N_new = length(x_new), x_new = x_new)
)

fit$draws("f_new")
```

## Things to know

- **Use `target +=`, not `~`.** Laplace rewrites `gaussian_process::func(` calls, so write `target += gaussian_process::marginal_normal_lpdf(y | ...)`. The `y ~ gaussian_process::marginal_normal(...)` form won't resolve.
- **Multiply kernels with `.*`, never `*`.** Between two matrices, `*` is a matrix product, which is not a kernel product and gives a wrong, usually invalid, covariance.
- **In a product, only one amplitude is identified.** $\alpha_1 \alpha_2$ is all the data can see, so set every amplitude but one to `1.0`, as `local_periodic` and `rq_periodic` already do. Otherwise the sampler drifts along a ridge where one amplitude grows as the other shrinks.
- **Prediction needs the kernel written three times** — `K`, `K_cross`, and `K_star` — with the same kernel and hyperparameters as in `model`. The library can't check this for you, since Stan can't pass functions as arguments.
- **Don't store `K` in `transformed parameters`.** Everything there is written to the output on every draw, and an $N \times N$ matrix quickly makes enormous CSV files. Build it in `model` and again in `generated quantities`; recomputing it is cheap by comparison.
- **The GP mean is assumed zero in prediction.** If your model has a mean, pass `y - mu` to the prediction functions and add the mean at the new points back afterwards (or centre `y` beforehand, as in the examples).
- **`predict_*_rng` returns the noise-free function.** For new observations, add the likelihood noise yourself, e.g. `normal_rng(f_new, sigma)`.
- **Some kernels need jitter.** `cosine`, `linear`, `polynomial`, and `constant` are low-rank, and `wiener`, `brownian_bridge`, and `integrated_ornstein_uhlenbeck` are zero at time 0, so their matrices are singular. Used alone in a latent GP they need a `delta` large enough for the Cholesky decomposition to succeed, typically `1e-6` to `1e-9` relative to the kernel's scale. In the marginal model, $\sigma^2 I$ already takes care of this.
- **`white_cross_cov` is all zeros**, which assumes the prediction points are different observations from the training points.
- **Inputs are one-dimensional** (`array[] real`), except `arccosine`, which needs `array[] vector` because it is degenerate in 1D.
- **Exact GPs scale as $O(N^3)$.** Each gradient evaluation needs a Cholesky decomposition of an $N \times N$ matrix, which is practical up to a few thousand observations. A Hilbert-space approximation (HSGP) for larger data is planned.

## License

See [LICENSE](https://github.com/mlatinov/laplace-gaussian_processes-/blob/main/LICENSE).