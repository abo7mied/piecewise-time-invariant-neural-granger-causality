# Piecewise Time-invariant Neural Granger Causality Detection

## Abstract

We study Granger-causal graph recovery in multivariate time series with timevarying
Granger-causality, in particular assuming piecewise time-invariance. We
utilize a two-step approach: to estimate the regime boundary and then to recover the
causal graphs in each regime through already established time-invariant methods.
The main contribution is a normalized local cross-regime prediction score: for each
candidate boundary, we compare cross-regime prediction error to same-regime
prediction error. Thus, the detector is causality-based rather than distributionbased,
as it is based on discrepancy in predictive power rather than in marginal
mean or covariance. After estimating the boundary, we split the sequence based
on the estimated boundary, and we fit local Granger-causal models: linear VAR
and neural autoregressive component-wise MLP. On controlled nonlinear VAR
experiments, the boundary score substantially outperforms mean, covariance, meanplus-
covariance, midpoint, and random baselines, achieving much lower median
boundary error and higher Success@100 over repeated runs. On a two-regime
generalized Lorenz system, the boundary score accurately localizes the regime
change. The downstream graph recovery results are more modest: local cMLP
improves mean F1 and mean normalized SHD compared with a single global cMLP,
suggesting that the split is useful, but neural graph recovery remains sensitive to
training variability.
