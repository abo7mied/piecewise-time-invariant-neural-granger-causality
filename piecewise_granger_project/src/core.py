"""Core implementation for piecewise time-invariant Granger-causality experiments.

Refactored from the project notebook. Importing this module has no side effects.
"""

import numpy as np


def validate_adjacency(A, allow_self=False):
    """
    Validate a binary directed adjacency matrix.

    Convention:
        A[i, j] = 1 means x_j Granger-causes / predicts x_i.
    """
    A = np.asarray(A, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')
    if not np.all((A == 0) | (A == 1)):
        raise ValueError('A must be binary.')
    if not allow_self and np.any(np.diag(A) != 0):
        raise ValueError('Self-edges are not allowed. Set allow_self=True if desired.')
    return A


def make_fixed_indegree_graphs(d, q=2, disjoint=True, seed=None):
    """
    Create two directed graphs A1 and A2 where each variable has exactly q
    non-self parents.

    Convention:
        A[i, j] = 1 means x_j predicts x_i.

    If disjoint=True, then for every target variable i, the non-self parent
    sets in A1 and A2 are disjoint.
    """
    rng = np.random.default_rng(seed)
    if q <= 0:
        raise ValueError('q must be positive.')
    if disjoint and 2 * q > d - 1:
        raise ValueError(f'Cannot make disjoint parent sets with d={d}, q={q}. Need 2*q <= d-1.')
    if not disjoint and q > d - 1:
        raise ValueError(f'q must be at most d-1={d - 1}.')
    A1 = np.zeros((d, d), dtype=int)
    A2 = np.zeros((d, d), dtype=int)
    for i in range(d):
        possible_parents = [j for j in range(d) if j != i]
        rng.shuffle(possible_parents)
        parents_1 = possible_parents[:q]
        if disjoint:
            parents_2 = possible_parents[q:2 * q]
        else:
            parents_2 = rng.choice(possible_parents, size=q, replace=False)
        A1[i, parents_1] = 1
        A2[i, parents_2] = 1
    return (A1, A2)


def initialize_nonlinear_var_weights(A, coefficient_scale=0.25, balanced_signs=True, seed=None):
    """
    Initialize coefficient matrix W supported exactly on adjacency A.

    W[i, j] is nonzero only when A[i, j] = 1.
    """
    rng = np.random.default_rng(seed)
    A = validate_adjacency(A, allow_self=False)
    d = A.shape[0]
    W = np.zeros((d, d), dtype=float)
    for i in range(d):
        parents = np.where(A[i] == 1)[0]
        q_i = len(parents)
        if q_i == 0:
            continue
        if balanced_signs:
            signs = rng.choice([-1.0, 1.0], size=q_i)
            magnitudes = coefficient_scale * np.ones(q_i)
            W[i, parents] = signs * magnitudes
        else:
            W[i, parents] = rng.normal(loc=0.0, scale=coefficient_scale, size=q_i)
    return W


def standardize_per_variable(X, eps=1e-08):
    """
    Standardize each variable to mean 0 and standard deviation 1.
    """
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + eps)


def match_marginal_mean_std(X_source, X_target, eps=1e-08):
    """
    Match per-variable mean and standard deviation of X_source to X_target.

    This removes easy marginal mean/variance differences while preserving the
    temporal ordering and directed dependency support up to per-variable affine
    rescaling.
    """
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    source_mean = X_source.mean(axis=0, keepdims=True)
    source_std = X_source.std(axis=0, keepdims=True)
    target_mean = X_target.mean(axis=0, keepdims=True)
    target_std = X_target.std(axis=0, keepdims=True)
    X_matched = (X_source - source_mean) / (source_std + eps)
    X_matched = X_matched * target_std + target_mean
    stats = {'source_mean_before': source_mean.ravel(), 'source_std_before': source_std.ravel(), 'target_mean': target_mean.ravel(), 'target_std': target_std.ravel(), 'source_mean_after': X_matched.mean(axis=0), 'source_std_after': X_matched.std(axis=0)}
    return (X_matched, stats)


def simulate_nonlinear_var(A, T=1000, p=1, burn_in=500, rho=0.5, coefficient_scale=0.25, noise_std=0.5, nonlinearity='tanh', W=None, x0=None, weight_seed=42, noise_seed=123, standardize=False, return_metadata=True):
    """
    Simulate a stable nonlinear VAR process with a controlled Granger graph.

    Model for p=1:

        x[t, i]
        =
        rho * x[t-1, i]
        +
        (1 / sqrt(q_i)) * sum_{j in Pa(i)} W[i, j] phi(x[t-1, j])
        +
        eps[t, i]

    where:
        A[i, j] = 1 means x_j -> x_i.

    The normalization by sqrt(q_i) helps keep variances comparable when each
    row has the same number of parents.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)
        Directed causal graph.

    T : int
        Number of returned time points.

    p : int
        Lag order. Currently p=1 is implemented cleanly.

    burn_in : int
        Number of initial samples discarded.

    rho : float
        Self-memory coefficient.

    coefficient_scale : float
        Magnitude scale for parent effects.

    noise_std : float
        Innovation noise standard deviation.

    nonlinearity : {"tanh", "sin", "linear"}
        Nonlinear parent transformation.

    W : np.ndarray or None
        Optional coefficient matrix supported on A.

    standardize : bool
        If True, standardize each variable after simulation.

    Returns
    -------
    If return_metadata=False:
        X : np.ndarray, shape (T, d)

    If return_metadata=True:
        dict with X, A, W, params.
    """
    A = validate_adjacency(A, allow_self=False)
    d = A.shape[0]
    if p != 1:
        raise NotImplementedError('This clean version currently supports p=1.')
    if T <= 0:
        raise ValueError('T must be positive.')
    if burn_in < 0:
        raise ValueError('burn_in must be nonnegative.')
    if noise_std < 0:
        raise ValueError('noise_std must be nonnegative.')
    if nonlinearity == 'tanh':
        phi = np.tanh
    elif nonlinearity == 'sin':
        phi = np.sin
    elif nonlinearity == 'linear':
        phi = lambda z: z
    else:
        raise ValueError("nonlinearity must be one of: 'tanh', 'sin', 'linear'.")
    if W is None:
        W = initialize_nonlinear_var_weights(A=A, coefficient_scale=coefficient_scale, balanced_signs=True, seed=weight_seed)
    else:
        W = np.asarray(W, dtype=float)
        if W.shape != A.shape:
            raise ValueError(f'W must have shape {A.shape}, got {W.shape}.')
        if np.any((A == 0) & (np.abs(W) > 1e-12)):
            raise ValueError('W has nonzero entries outside the support of A.')
    rng = np.random.default_rng(noise_seed)
    total_T = burn_in + T
    X_full = np.zeros((total_T, d), dtype=float)
    if x0 is None:
        X_full[0] = rng.normal(loc=0.0, scale=noise_std, size=d)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (d,):
            raise ValueError(f'x0 must have shape ({d},), got {x0.shape}.')
        X_full[0] = x0.copy()
    parent_counts = A.sum(axis=1)
    normalizers = np.sqrt(np.maximum(parent_counts, 1))
    for t in range(1, total_T):
        x_prev = X_full[t - 1]
        parent_signal = W @ phi(x_prev)
        parent_signal = parent_signal / normalizers
        eps = rng.normal(loc=0.0, scale=noise_std, size=d)
        X_full[t] = rho * x_prev + parent_signal + eps
    X = X_full[burn_in:]
    if standardize:
        X = standardize_per_variable(X)
    if not return_metadata:
        return X
    params = {'T': T, 'p': p, 'burn_in': burn_in, 'rho': rho, 'coefficient_scale': coefficient_scale, 'noise_std': noise_std, 'nonlinearity': nonlinearity, 'weight_seed': weight_seed, 'noise_seed': noise_seed, 'standardize': standardize}
    return {'X': X, 'A': A, 'W': W, 'params': params}


def simulate_two_regime_nonlinear_var(A1, A2, T1=1000, T2=1000, p=1, burn_in=1000, rho=0.5, coefficient_scale=0.25, noise_std=0.5, nonlinearity='tanh', weight_seed_1=42, weight_seed_2=43, noise_seed_1=123, noise_seed_2=456, same_weight_values=True, match_marginal_moments=True, standardize_global=True, return_metadata=True):
    """
    Simulate a two-regime nonlinear VAR process with different causal graphs.

    The two regimes are generated with:
        - different adjacency matrices A1 and A2,
        - the same rho,
        - the same noise_std,
        - the same number of parents per variable if A1 and A2 are constructed
          that way,
        - optional marginal mean/std matching.

    This is designed so that the causal graph changes while per-variable means
    and variances are approximately stationary across regimes.

    Returns
    -------
    result : dict
        {
            "X": full series, shape (T1 + T2, d),
            "tau": T1,
            "A1": regime 1 graph,
            "A2": regime 2 graph,
            "W1": regime 1 weights,
            "W2": regime 2 weights,
            "moment_matching_stats": diagnostics,
            "config": parameters
        }
    """
    A1 = validate_adjacency(A1, allow_self=False)
    A2 = validate_adjacency(A2, allow_self=False)
    if A1.shape != A2.shape:
        raise ValueError('A1 and A2 must have the same shape.')
    d = A1.shape[0]
    if same_weight_values:
        rng = np.random.default_rng(weight_seed_1)
        W1 = np.zeros((d, d), dtype=float)
        W2 = np.zeros((d, d), dtype=float)
        for i in range(d):
            parents_1 = np.where(A1[i] == 1)[0]
            parents_2 = np.where(A2[i] == 1)[0]
            if len(parents_1) != len(parents_2):
                raise ValueError('same_weight_values=True requires each row of A1 and A2 to have the same number of parents.')
            q_i = len(parents_1)
            if q_i == 0:
                continue
            signs = rng.choice([-1.0, 1.0], size=q_i)
            coeffs = coefficient_scale * signs
            W1[i, parents_1] = coeffs
            W2[i, parents_2] = coeffs
    else:
        W1 = initialize_nonlinear_var_weights(A=A1, coefficient_scale=coefficient_scale, balanced_signs=True, seed=weight_seed_1)
        W2 = initialize_nonlinear_var_weights(A=A2, coefficient_scale=coefficient_scale, balanced_signs=True, seed=weight_seed_2)
    result_1 = simulate_nonlinear_var(A=A1, T=T1, p=p, burn_in=burn_in, rho=rho, coefficient_scale=coefficient_scale, noise_std=noise_std, nonlinearity=nonlinearity, W=W1, weight_seed=weight_seed_1, noise_seed=noise_seed_1, standardize=False, return_metadata=True)
    result_2 = simulate_nonlinear_var(A=A2, T=T2, p=p, burn_in=burn_in, rho=rho, coefficient_scale=coefficient_scale, noise_std=noise_std, nonlinearity=nonlinearity, W=W2, weight_seed=weight_seed_2, noise_seed=noise_seed_2, standardize=False, return_metadata=True)
    X1 = result_1['X']
    X2 = result_2['X']
    moment_matching_stats = None
    if match_marginal_moments:
        X2, moment_matching_stats = match_marginal_mean_std(X_source=X2, X_target=X1)
    X = np.vstack([X1, X2])
    if standardize_global:
        X = standardize_per_variable(X)
    tau = T1
    if not return_metadata:
        return X
    config = {'T1': T1, 'T2': T2, 'tau': tau, 'p': p, 'burn_in': burn_in, 'rho': rho, 'coefficient_scale': coefficient_scale, 'noise_std': noise_std, 'nonlinearity': nonlinearity, 'weight_seed_1': weight_seed_1, 'weight_seed_2': weight_seed_2, 'noise_seed_1': noise_seed_1, 'noise_seed_2': noise_seed_2, 'same_weight_values': same_weight_values, 'match_marginal_moments': match_marginal_moments, 'standardize_global': standardize_global}
    return {'X': X, 'tau': tau, 'A1': A1, 'A2': A2, 'W1': W1, 'W2': W2, 'moment_matching_stats': moment_matching_stats, 'config': config}


import numpy as np


import matplotlib.pyplot as plt


def plot_two_regime_result(result, color_by_change_point=True, show_change_line=True, show_regime_labels=True, variable_names=None, title='Two-Regime Generalized Lorenz Time Series', xlabel='Time', ylabel='Value', figsize=(14, 6), linewidth=1.0, alpha=0.85, before_color='tab:blue', after_color='tab:orange', change_line_color='black', change_line_style='--', grid=True, title_y=0.98, regime_label_y=1.01):
    """
    Visualize the output of simulate_two_regime_generalized_lorenz.

    Expected input
    --------------
    result : dict
        Dictionary returned by simulate_two_regime_generalized_lorenz.
        Must contain:
            result["X"]   : np.ndarray, shape (T, k)
            result["tau"] : int

    Plot behavior
    -------------
    If color_by_change_point=True:
        - X[:tau] is plotted in blue.
        - X[tau:] is plotted in orange.

    If color_by_change_point=False:
        - Each variable is plotted across the full time axis with its own label.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis objects.
    """
    if 'X' not in result:
        raise KeyError("result must contain key 'X'.")
    if 'tau' not in result:
        raise KeyError("result must contain key 'tau'.")
    X = np.asarray(result['X'], dtype=float)
    tau = int(result['tau'])
    if X.ndim != 2:
        raise ValueError(f"result['X'] must have shape (T, k), got {X.shape}.")
    T, k = X.shape
    if tau <= 0 or tau >= T:
        raise ValueError(f'tau must satisfy 0 < tau < T, got tau={tau}, T={T}.')
    if variable_names is None:
        variable_names = [f'x{i}' for i in range(k)]
    if len(variable_names) != k:
        raise ValueError(f'variable_names must have length {k}, got {len(variable_names)}.')
    time = np.arange(T)
    fig, ax = plt.subplots(figsize=figsize)
    if color_by_change_point:
        for j in range(k):
            ax.plot(time[:tau], X[:tau, j], color=before_color, linewidth=linewidth, alpha=alpha, label='Regime 1' if j == 0 else None)
            ax.plot(time[tau:], X[tau:, j], color=after_color, linewidth=linewidth, alpha=alpha, label='Regime 2' if j == 0 else None)
    else:
        for j in range(k):
            ax.plot(time, X[:, j], linewidth=linewidth, alpha=alpha, label=variable_names[j])
    if show_change_line:
        ax.axvline(tau, color=change_line_color, linestyle=change_line_style, linewidth=1.5, label=f'True change point: tau = {tau}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.suptitle(title, y=title_y, fontsize=14)
    if show_regime_labels:
        ax.text(tau / 2, regime_label_y, 'Regime 1', ha='center', va='bottom', fontsize=11, transform=ax.get_xaxis_transform())
        ax.text(tau + (T - tau) / 2, regime_label_y, 'Regime 2', ha='center', va='bottom', fontsize=11, transform=ax.get_xaxis_transform())
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return (fig, ax)


import numpy as np


def make_lagged_autoregressive_pairs(X, lag_order, reverse=False, flatten_lags=True, return_time_indices=True):
    """
    Convert a multivariate time series into lagged autoregressive
    input-output pairs.

    Forward model:
        (x_{t-p}, ..., x_{t-1}) -> x_t

    Backward model:
        The input series is first reversed, then the same lagged construction
        is applied.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Multivariate time series.

    lag_order : int
        Number of past time steps used as input.

    reverse : bool
        If True, reverse the time series before constructing lagged examples.
        This is used for the backward autoregressive model.

    flatten_lags : bool
        If True, each input has shape (lag_order * d,).
        If False, each input has shape (lag_order, d).

    return_time_indices : bool
        If True, also return the original time index associated with each target.

    Returns
    -------
    X_lagged : np.ndarray
        If flatten_lags=True:
            shape (T - lag_order, lag_order * d)
        If flatten_lags=False:
            shape (T - lag_order, lag_order, d)

    y : np.ndarray, shape (T - lag_order, d)
        Target values.

    time_indices : np.ndarray, shape (T - lag_order,)
        Original time index of each target.
        Returned only if return_time_indices=True.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must have shape (T, d), got {X.shape}.')
    T, d = X.shape
    if lag_order <= 0:
        raise ValueError('lag_order must be positive.')
    if lag_order >= T:
        raise ValueError(f'lag_order must be smaller than T. Got lag_order={lag_order}, T={T}.')
    if reverse:
        X_work = X[::-1].copy()
        original_time = np.arange(T - 1, -1, -1)
    else:
        X_work = X
        original_time = np.arange(T)
    inputs = []
    targets = []
    target_times = []
    for t in range(lag_order, T):
        lag_block = X_work[t - lag_order:t]
        target = X_work[t]
        if flatten_lags:
            lag_block = lag_block.reshape(-1)
        inputs.append(lag_block)
        targets.append(target)
        target_times.append(original_time[t])
    X_lagged = np.asarray(inputs)
    y = np.asarray(targets)
    target_times = np.asarray(target_times)
    if return_time_indices:
        return (X_lagged, y, target_times)
    return (X_lagged, y)


import numpy as np


def fit_var_ridge_numpy(X_lagged, y, alpha=1.0, fit_intercept=True):
    """
    Fit a linear VAR-style autoregressive predictor using pure NumPy.

    Model:
        y ≈ X_lagged @ W + b

    Parameters
    ----------
    X_lagged : np.ndarray, shape (n_samples, p * d)
        Flattened lagged autoregressive inputs.

    y : np.ndarray, shape (n_samples, d)
        Targets.

    alpha : float
        Ridge regularization strength.
        alpha=0.0 gives ordinary least squares.

    fit_intercept : bool
        Whether to include an intercept term.

    Returns
    -------
    model : dict
        Dictionary containing fitted weights and settings.
    """
    X_lagged = np.asarray(X_lagged, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_lagged.ndim != 2:
        raise ValueError(f'X_lagged must be 2D, got shape {X_lagged.shape}.')
    if y.ndim != 2:
        raise ValueError(f'y must be 2D, got shape {y.shape}.')
    if X_lagged.shape[0] != y.shape[0]:
        raise ValueError(f'X_lagged and y must have same number of samples. Got {X_lagged.shape[0]} and {y.shape[0]}.')
    n_samples, input_dim = X_lagged.shape
    if fit_intercept:
        X_design = np.hstack([np.ones((n_samples, 1)), X_lagged])
    else:
        X_design = X_lagged
    n_features = X_design.shape[1]
    ridge_matrix = alpha * np.eye(n_features)
    if fit_intercept:
        ridge_matrix[0, 0] = 0.0
    lhs = X_design.T @ X_design + ridge_matrix
    rhs = X_design.T @ y
    beta = np.linalg.solve(lhs, rhs)
    if fit_intercept:
        intercept = beta[0]
        weights = beta[1:]
    else:
        intercept = np.zeros(y.shape[1])
        weights = beta
    return {'weights': weights, 'intercept': intercept, 'alpha': alpha, 'fit_intercept': fit_intercept}


def predict_var_ridge_numpy(model, X_lagged):
    """
    Predict using a fitted pure NumPy VAR/Ridge model.
    """
    X_lagged = np.asarray(X_lagged, dtype=float)
    return X_lagged @ model['weights'] + model['intercept']


def evaluate_var_ridge_numpy(model, X_lagged, y, model_name='VAR/Ridge model'):
    """
    Evaluate a fitted pure NumPy VAR/Ridge model.
    """
    y = np.asarray(y, dtype=float)
    y_pred = predict_var_ridge_numpy(model, X_lagged)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    signal_var = np.var(y)
    nmse = mse / (signal_var + 1e-12)
    per_variable_mse = np.mean(residuals ** 2, axis=0)
    per_variable_var = np.var(y, axis=0)
    per_variable_nmse = per_variable_mse / (per_variable_var + 1e-12)
    print(f'{model_name}')
    print('-' * len(model_name))
    print(f'MSE:    {mse:.6f}')
    print(f'RMSE:   {rmse:.6f}')
    print(f'MAE:    {mae:.6f}')
    print(f'NMSE:   {nmse:.6f}')
    print('Per-variable NMSE:')
    print(per_variable_nmse)
    return {'y_pred': y_pred, 'residuals': residuals, 'mse': mse, 'rmse': rmse, 'mae': mae, 'nmse': nmse, 'per_variable_mse': per_variable_mse, 'per_variable_nmse': per_variable_nmse}


import numpy as np


import matplotlib.pyplot as plt


def squared_residual_norms(residuals):
    """
    Convert vector residuals into squared L2 residuals.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_samples, d)

    Returns
    -------
    err : np.ndarray, shape (n_samples,)
        err[i] = ||residuals[i]||_2^2
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError(f'residuals must have shape (n_samples, d), got {residuals.shape}.')
    return np.sum(residuals ** 2, axis=1)


def align_residuals_to_original_time(residuals, time_indices, T):
    """
    Align residuals from lagged examples back to the original time axis.

    Parameters
    ----------
    residuals : np.ndarray, shape (n_samples, d)
        Prediction residuals.

    time_indices : np.ndarray, shape (n_samples,)
        Original target time corresponding to each residual.

    T : int
        Total length of the original time series.

    Returns
    -------
    err_by_time : np.ndarray, shape (T,)
        Squared residual norm at each original time.
        Times without available predictions are filled with np.nan.
    """
    err = squared_residual_norms(residuals)
    time_indices = np.asarray(time_indices, dtype=int)
    if len(time_indices) != len(err):
        raise ValueError('time_indices and residuals must have the same number of samples.')
    err_by_time = np.full(T, np.nan, dtype=float)
    err_by_time[time_indices] = err
    return err_by_time


def moving_average_nan(x, window):
    """
    Moving average that handles NaNs by ignoring them.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    window : int

    Returns
    -------
    smoothed : np.ndarray, shape (T,)
    """
    x = np.asarray(x, dtype=float)
    if window is None or window <= 1:
        return x.copy()
    valid = np.isfinite(x).astype(float)
    x_filled = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(x_filled, kernel, mode='same')
    denominator = np.convolve(valid, kernel, mode='same')
    return numerator / np.maximum(denominator, 1e-12)


def compute_boundary_score(fwd_residuals, fwd_time_indices, bwd_residuals, bwd_time_indices, T, h, lag_order, candidates=None, score_mode='absolute', smooth_window=None, normalize_by_local_variance=False, X=None):
    """
    Compute the residual-asymmetry boundary score.

    Score:
        s(t) =
        |
            mean_{u=t-h+1}^{t}     ||x_u - xhat_fwd_u||^2
            -
            mean_{u=t+1}^{t+h}     ||x_u - xhat_bwd_u||^2
        |

    Parameters
    ----------
    fwd_residuals : np.ndarray, shape (n_fwd, d)
        Forward model residuals.

    fwd_time_indices : np.ndarray, shape (n_fwd,)
        Original target times for forward residuals.

    bwd_residuals : np.ndarray, shape (n_bwd, d)
        Backward model residuals.

    bwd_time_indices : np.ndarray, shape (n_bwd,)
        Original target times for backward residuals.

    T : int
        Length of original time series.

    h : int
        Window size on each side of candidate boundary.

    lag_order : int
        Autoregressive lag order p. Used to keep candidates away from edges.

    candidates : np.ndarray or None
        Candidate boundary times. If None, uses valid range h+p <= t <= T-h-p.

    score_mode : {"absolute", "squared"}
        Whether to use absolute difference or squared difference.

    smooth_window : int or None
        Optional moving-average smoothing window for s(t).

    normalize_by_local_variance : bool
        If True, normalize each side's residual average by local signal variance.
        Requires X.

    X : np.ndarray or None, shape (T, d)
        Original time series. Required if normalize_by_local_variance=True.

    Returns
    -------
    result : dict
        {
            "candidates": candidates,
            "scores": scores,
            "scores_raw": scores_raw,
            "tau_hat": tau_hat,
            "fwd_err_by_time": fwd_err_by_time,
            "bwd_err_by_time": bwd_err_by_time,
        }
    """
    if h <= 0:
        raise ValueError('h must be positive.')
    if lag_order <= 0:
        raise ValueError('lag_order must be positive.')
    if score_mode not in {'absolute', 'squared'}:
        raise ValueError("score_mode must be either 'absolute' or 'squared'.")
    if normalize_by_local_variance and X is None:
        raise ValueError('X must be provided when normalize_by_local_variance=True.')
    fwd_err_by_time = align_residuals_to_original_time(residuals=fwd_residuals, time_indices=fwd_time_indices, T=T)
    bwd_err_by_time = align_residuals_to_original_time(residuals=bwd_residuals, time_indices=bwd_time_indices, T=T)
    if candidates is None:
        start = h + lag_order
        end = T - h - lag_order
        if start > end:
            raise ValueError(f'No valid candidates. Need T > 2 * (h + lag_order). Got T={T}, h={h}, lag_order={lag_order}.')
        candidates = np.arange(start, end + 1)
    else:
        candidates = np.asarray(candidates, dtype=int)
    scores_raw = np.full(len(candidates), np.nan, dtype=float)
    if normalize_by_local_variance:
        X = np.asarray(X, dtype=float)
        if X.shape[0] != T:
            raise ValueError(f'X must have length T={T}, got {X.shape[0]}.')
    for idx, t in enumerate(candidates):
        left_slice = slice(t - h + 1, t + 1)
        right_slice = slice(t + 1, t + h + 1)
        left_err = fwd_err_by_time[left_slice]
        right_err = bwd_err_by_time[right_slice]
        if np.any(~np.isfinite(left_err)) or np.any(~np.isfinite(right_err)):
            continue
        left_score = np.mean(left_err)
        right_score = np.mean(right_err)
        if normalize_by_local_variance:
            left_var = np.var(X[left_slice])
            right_var = np.var(X[right_slice])
            left_score = left_score / (left_var + 1e-12)
            right_score = right_score / (right_var + 1e-12)
        diff = left_score - right_score
        if score_mode == 'absolute':
            scores_raw[idx] = abs(diff)
        else:
            scores_raw[idx] = diff ** 2
    scores = moving_average_nan(scores_raw, smooth_window)
    if np.all(~np.isfinite(scores)):
        raise ValueError('All boundary scores are NaN. Check h, lag_order, and residual alignment.')
    tau_hat = int(candidates[np.nanargmax(scores)])
    return {'candidates': candidates, 'scores': scores, 'scores_raw': scores_raw, 'tau_hat': tau_hat, 'fwd_err_by_time': fwd_err_by_time, 'bwd_err_by_time': bwd_err_by_time}


def plot_boundary_score(score_result, true_tau=None, title='Residual-Asymmetry Boundary Score', figsize=(12, 4)):
    """
    Plot boundary score over candidate times.
    """
    candidates = score_result['candidates']
    scores = score_result['scores']
    tau_hat = score_result['tau_hat']
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(candidates, scores, linewidth=1.5, label='Boundary score')
    ax.axvline(tau_hat, color='tab:red', linestyle='--', linewidth=1.5, label=f'Estimated tau = {tau_hat}')
    if true_tau is not None:
        ax.axvline(true_tau, color='black', linestyle=':', linewidth=2.0, label=f'True tau = {true_tau}')
    ax.set_title(title)
    ax.set_xlabel('Candidate boundary time')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax)


import numpy as np


import matplotlib.pyplot as plt


def valid_local_cross_regime_candidates(T, lag_order, h, train_window, step=1):
    """
    Construct valid candidate boundaries for the local cross-regime method.

    A candidate t is valid if all four windows are available:

        left training window:   target times t-train_window+1, ..., t
        right test window:      target times t+1, ..., t+h
        right training window:  target times t+1, ..., t+train_window
        left test window:       target times t-h+1, ..., t

    Parameters
    ----------
    T : int
        Length of the time series.

    lag_order : int
        Autoregressive lag order p.

    h : int
        Evaluation window size.

    train_window : int
        Number of target-time examples used to train each local model.

    step : int
        Candidate grid spacing.

    Returns
    -------
    candidates : np.ndarray
        Valid candidate boundary times.
    """
    if lag_order <= 0:
        raise ValueError('lag_order must be positive.')
    if h <= 0:
        raise ValueError('h must be positive.')
    if train_window <= 0:
        raise ValueError('train_window must be positive.')
    if step <= 0:
        raise ValueError('step must be positive.')
    start = lag_order + train_window + h
    end = T - lag_order - train_window - h
    if start > end:
        raise ValueError('No valid candidate boundaries. Try reducing lag_order, h, or train_window.')
    return np.arange(start, end + 1, step, dtype=int)


def mean_squared_prediction_error(model, X_test, y_test):
    """
    Compute mean squared vector prediction error for a fitted VAR/Ridge model.

    Error is averaged over samples:

        mean_i ||y_i - yhat_i||_2^2
    """
    y_pred = predict_var_ridge_numpy(model, X_test)
    residuals = y_test - y_pred
    return np.mean(np.sum(residuals ** 2, axis=1))


def compute_local_cross_regime_boundary_score_normalized(X, lag_order=20, h=100, train_window=1000, val_window=None, alpha=1.0, fit_intercept=True, candidates=None, step=10, smooth_window=None, eps=1e-08, score_mode='ratio_sum', verbose=False):
    """
    Compute a normalized local cross-regime boundary score.

    For each candidate t:

        Left model:
            Train on left local training window.
            Test across boundary on right test window: E_LR.
            Test within left side on held-out left validation window: E_LL.

        Right model:
            Train on right local training window.
            Test across boundary on left test window: E_RL.
            Test within right side on held-out right validation window: E_RR.

    Normalized score:

        s(t) = E_LR / (E_LL + eps) + E_RL / (E_RR + eps)

    This asks whether crossing candidate t is harder than predicting within
    the same side.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must have shape (T, d), got {X.shape}.')
    T, d = X.shape
    if val_window is None:
        val_window = h
    if score_mode not in {'ratio_sum', 'ratio_minus_one', 'log_ratio'}:
        raise ValueError("score_mode must be one of: 'ratio_sum', 'ratio_minus_one', 'log_ratio'.")
    if candidates is None:
        start = lag_order + train_window + val_window + h
        end = T - lag_order - train_window - val_window - h
        if start > end:
            raise ValueError('No valid candidate boundaries. Try reducing lag_order, h, train_window, or val_window.')
        candidates = np.arange(start, end + 1, step, dtype=int)
    else:
        candidates = np.asarray(candidates, dtype=int)
    X_fwd, y_fwd, t_fwd = make_lagged_autoregressive_pairs(X, lag_order=lag_order, reverse=False, flatten_lags=True, return_time_indices=True)
    X_bwd, y_bwd, t_bwd = make_lagged_autoregressive_pairs(X, lag_order=lag_order, reverse=True, flatten_lags=True, return_time_indices=True)
    scores = np.full(len(candidates), np.nan, dtype=float)
    scores_raw = np.full(len(candidates), np.nan, dtype=float)
    E_LR_arr = np.full(len(candidates), np.nan, dtype=float)
    E_LL_arr = np.full(len(candidates), np.nan, dtype=float)
    E_RL_arr = np.full(len(candidates), np.nan, dtype=float)
    E_RR_arr = np.full(len(candidates), np.nan, dtype=float)
    for idx, t in enumerate(candidates):
        if verbose and idx % 25 == 0:
            print(f'Scoring candidate {idx + 1}/{len(candidates)}: t={t}')
        fwd_train_mask = (t_fwd >= t - val_window - train_window + 1) & (t_fwd <= t - val_window)
        fwd_within_mask = (t_fwd >= t - val_window + 1) & (t_fwd <= t)
        fwd_cross_mask = (t_fwd >= t + 1) & (t_fwd <= t + h)
        bwd_train_mask = (t_bwd >= t + val_window + 1) & (t_bwd <= t + val_window + train_window)
        bwd_within_mask = (t_bwd >= t + 1) & (t_bwd <= t + val_window)
        bwd_cross_mask = (t_bwd >= t - h + 1) & (t_bwd <= t)
        if fwd_train_mask.sum() == 0 or fwd_within_mask.sum() == 0 or fwd_cross_mask.sum() == 0 or (bwd_train_mask.sum() == 0) or (bwd_within_mask.sum() == 0) or (bwd_cross_mask.sum() == 0):
            continue
        fwd_model = fit_var_ridge_numpy(X_lagged=X_fwd[fwd_train_mask], y=y_fwd[fwd_train_mask], alpha=alpha, fit_intercept=fit_intercept)
        bwd_model = fit_var_ridge_numpy(X_lagged=X_bwd[bwd_train_mask], y=y_bwd[bwd_train_mask], alpha=alpha, fit_intercept=fit_intercept)
        E_LR = mean_squared_prediction_error(model=fwd_model, X_test=X_fwd[fwd_cross_mask], y_test=y_fwd[fwd_cross_mask])
        E_LL = mean_squared_prediction_error(model=fwd_model, X_test=X_fwd[fwd_within_mask], y_test=y_fwd[fwd_within_mask])
        E_RL = mean_squared_prediction_error(model=bwd_model, X_test=X_bwd[bwd_cross_mask], y_test=y_bwd[bwd_cross_mask])
        E_RR = mean_squared_prediction_error(model=bwd_model, X_test=X_bwd[bwd_within_mask], y_test=y_bwd[bwd_within_mask])
        E_LR_arr[idx] = E_LR
        E_LL_arr[idx] = E_LL
        E_RL_arr[idx] = E_RL
        E_RR_arr[idx] = E_RR
        if score_mode == 'ratio_sum':
            score = E_LR / (E_LL + eps) + E_RL / (E_RR + eps)
        elif score_mode == 'ratio_minus_one':
            score = E_LR / (E_LL + eps) - 1.0 + (E_RL / (E_RR + eps) - 1.0)
        elif score_mode == 'log_ratio':
            score = np.log((E_LR + eps) / (E_LL + eps)) + np.log((E_RL + eps) / (E_RR + eps))
        scores_raw[idx] = score
    scores = scores_raw.copy()
    if smooth_window is not None and smooth_window > 1:
        scores = moving_average_nan(scores, smooth_window)
    if np.all(~np.isfinite(scores)):
        raise ValueError('All scores are NaN. Check candidates, h, lag_order, train_window, and val_window.')
    tau_hat = int(candidates[np.nanargmax(scores)])
    return {'candidates': candidates, 'scores': scores, 'scores_raw': scores_raw, 'tau_hat': tau_hat, 'E_left_to_right': E_LR_arr, 'E_left_to_left': E_LL_arr, 'E_right_to_left': E_RL_arr, 'E_right_to_right': E_RR_arr, 'lag_order': lag_order, 'h': h, 'train_window': train_window, 'val_window': val_window, 'alpha': alpha, 'step': step, 'score_mode': score_mode}


def plot_cross_regime_boundary_score(score_result, true_tau=None, title='Local Candidate-Specific Cross-Regime Boundary Score', figsize=(12, 4)):
    """
    Plot a candidate-specific cross-regime boundary score.
    """
    candidates = score_result['candidates']
    scores = score_result['scores']
    tau_hat = score_result['tau_hat']
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(candidates, scores, linewidth=1.5, label='Cross-regime score')
    ax.axvline(tau_hat, color='tab:red', linestyle='--', linewidth=1.5, label=f'Estimated tau = {tau_hat}')
    if true_tau is not None:
        ax.axvline(true_tau, color='black', linestyle=':', linewidth=2.0, label=f'True tau = {true_tau}')
    ax.set_title(title)
    ax.set_xlabel('Candidate boundary time')
    ax.set_ylabel('Cross-regime prediction error')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax)


import numpy as np


def midpoint_boundary_baseline(T, candidates=None):
    """
    Midpoint boundary baseline.

    Predicts the valid candidate closest to T / 2.

    Parameters
    ----------
    T : int
        Length of the time series.

    candidates : np.ndarray or None
        Valid candidate boundary times. If provided, the midpoint prediction
        is projected onto the nearest valid candidate.

    Returns
    -------
    tau_hat : int
        Predicted change point.
    """
    midpoint = T // 2
    if candidates is None:
        return midpoint
    candidates = np.asarray(candidates, dtype=int)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError('candidates must be a nonempty 1D array.')
    tau_hat = candidates[np.argmin(np.abs(candidates - midpoint))]
    return int(tau_hat)


def random_boundary_baseline(candidates, seed=None):
    """
    Random valid boundary baseline.

    Samples tau_hat uniformly from the valid candidate boundary times.

    Parameters
    ----------
    candidates : np.ndarray
        Valid candidate boundary times.

    seed : int or None
        Random seed.

    Returns
    -------
    tau_hat : int
        Randomly sampled candidate boundary.
    """
    candidates = np.asarray(candidates, dtype=int)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError('candidates must be a nonempty 1D array.')
    rng = np.random.default_rng(seed)
    tau_hat = rng.choice(candidates)
    return int(tau_hat)


def boundary_error_metrics(tau_hat, true_tau, T, tolerance=0):
    """
    Compute boundary-localization error metrics.

    Parameters
    ----------
    tau_hat : int
        Estimated change point.

    true_tau : int
        True change point.

    T : int
        Length of the time series.

    tolerance : int
        Optional tolerance. If abs(tau_hat - true_tau) <= tolerance,
        tolerance_error is reported as 0.

    Returns
    -------
    metrics : dict
        Boundary error metrics.
    """
    abs_error = abs(int(tau_hat) - int(true_tau))
    normalized_abs_error = abs_error / T
    tolerance_error = 0 if abs_error <= tolerance else normalized_abs_error
    return {'tau_hat': int(tau_hat), 'true_tau': int(true_tau), 'absolute_error': abs_error, 'normalized_absolute_error': normalized_abs_error, 'tolerance': tolerance, 'tolerance_error': tolerance_error}


import numpy as np


import matplotlib.pyplot as plt


def compute_window_distribution_score(X, h, lag_order=0, candidates=None, metric='mean', smooth_window=None):
    """
    Compute a distributional change score by comparing left and right windows
    around each candidate boundary.

    For each candidate t:

        left window  = X[t-h+1 : t+1]
        right window = X[t+1   : t+h+1]

    Then score(t) compares the two windows using a simple distributional
    statistic.

    Parameters
    ----------
    X : np.ndarray, shape (T, d)
        Multivariate time series.

    h : int
        Window size on each side of the candidate boundary.

    lag_order : int
        Optional lag order used only to keep candidates away from edges.
        For consistency with the residual-asymmetry method, use the same p.

    candidates : np.ndarray or None
        Candidate boundary times. If None, uses:
            h + lag_order <= t <= T - h - lag_order

    metric : {"mean", "covariance", "mean_covariance"}
        Distributional comparison metric.

        "mean":
            ||mean(left) - mean(right)||_2^2

        "covariance":
            ||Cov(left) - Cov(right)||_F^2

        "mean_covariance":
            ||mean(left) - mean(right)||_2^2
            + ||Cov(left) - Cov(right)||_F^2

    smooth_window : int or None
        Optional moving-average smoothing window for the score.

    Returns
    -------
    result : dict
        {
            "candidates": candidates,
            "scores": scores,
            "scores_raw": scores_raw,
            "tau_hat": tau_hat,
            "metric": metric,
            "h": h,
        }
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must have shape (T, d), got {X.shape}.')
    T, d = X.shape
    if h <= 0:
        raise ValueError('h must be positive.')
    if lag_order < 0:
        raise ValueError('lag_order must be nonnegative.')
    if metric not in {'mean', 'covariance', 'mean_covariance'}:
        raise ValueError("metric must be one of: 'mean', 'covariance', 'mean_covariance'.")
    if candidates is None:
        start = h + lag_order
        end = T - h - lag_order
        if start > end:
            raise ValueError(f'No valid candidates. Need T > 2 * (h + lag_order). Got T={T}, h={h}, lag_order={lag_order}.')
        candidates = np.arange(start, end + 1)
    else:
        candidates = np.asarray(candidates, dtype=int)
    scores_raw = np.zeros(len(candidates), dtype=float)
    for idx, t in enumerate(candidates):
        left = X[t - h + 1:t + 1]
        right = X[t + 1:t + h + 1]
        left_mean = left.mean(axis=0)
        right_mean = right.mean(axis=0)
        mean_score = np.sum((left_mean - right_mean) ** 2)
        if metric == 'mean':
            scores_raw[idx] = mean_score
            continue
        left_cov = np.cov(left, rowvar=False)
        right_cov = np.cov(right, rowvar=False)
        cov_score = np.sum((left_cov - right_cov) ** 2)
        if metric == 'covariance':
            scores_raw[idx] = cov_score
        elif metric == 'mean_covariance':
            scores_raw[idx] = mean_score + cov_score
    scores = moving_average_nan(scores_raw, smooth_window)
    tau_hat = int(candidates[np.nanargmax(scores)])
    return {'candidates': candidates, 'scores': scores, 'scores_raw': scores_raw, 'tau_hat': tau_hat, 'metric': metric, 'h': h}


def plot_distribution_score(score_result, true_tau=None, title=None, figsize=(12, 4)):
    """
    Plot a window-based distributional boundary score.
    """
    candidates = score_result['candidates']
    scores = score_result['scores']
    tau_hat = score_result['tau_hat']
    metric = score_result.get('metric', 'distributional')
    if title is None:
        title = f'Window Distributional Boundary Score ({metric})'
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(candidates, scores, linewidth=1.5, label=f'{metric} score')
    ax.axvline(tau_hat, color='tab:red', linestyle='--', linewidth=1.5, label=f'Estimated tau = {tau_hat}')
    if true_tau is not None:
        ax.axvline(true_tau, color='black', linestyle=':', linewidth=2.0, label=f'True tau = {true_tau}')
    ax.set_title(title)
    ax.set_xlabel('Candidate boundary time')
    ax.set_ylabel('Distributional score')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax)


import numpy as np


import matplotlib.pyplot as plt


import csv


def sample_true_boundary(T, min_boundary_distance, rng):
    """
    Sample a true boundary tau away from the edges.

    We require:
        min_boundary_distance <= tau <= T - min_boundary_distance
    """
    low = min_boundary_distance
    high = T - min_boundary_distance
    if low >= high:
        raise ValueError(f'No valid boundary range. Got T={T}, min_boundary_distance={min_boundary_distance}.')
    return int(rng.integers(low, high + 1))


def absolute_error(tau_hat, tau_true):
    return abs(int(tau_hat) - int(tau_true))


def normalized_error(tau_hat, tau_true, T):
    return absolute_error(tau_hat, tau_true) / T


def run_one_replicate(replicate_id, base_seed=123, T=12000, min_boundary_distance=None, d=6, q=2, generator_burn_in=1000, generator_p=1, rho=0.5, coefficient_scale=0.25, noise_std=0.5, nonlinearity='tanh', detector_lag_order=20, h=100, train_window=800, val_window=150, alpha=10.0, step=50, smooth_window=5, score_mode='log_ratio', distribution_smooth_window=11):
    """
    Run one replicate of the full boundary-detection experiment.

    Assumes the following functions already exist:
        - make_fixed_indegree_graphs
        - simulate_two_regime_nonlinear_var
        - compute_local_cross_regime_boundary_score_normalized
        - compute_window_distribution_score
        - midpoint_boundary_baseline
        - random_boundary_baseline
    """
    rng = np.random.default_rng(base_seed + replicate_id)
    if min_boundary_distance is None:
        min_boundary_distance = detector_lag_order + train_window + val_window + h + 100
    tau_true = sample_true_boundary(T=T, min_boundary_distance=min_boundary_distance, rng=rng)
    T1 = tau_true
    T2 = T - tau_true
    graph_seed = int(rng.integers(0, 2 ** 32 - 1))
    weight_seed = int(rng.integers(0, 2 ** 32 - 1))
    noise_seed_1 = int(rng.integers(0, 2 ** 32 - 1))
    noise_seed_2 = int(rng.integers(0, 2 ** 32 - 1))
    random_seed = int(rng.integers(0, 2 ** 32 - 1))
    A1, A2 = make_fixed_indegree_graphs(d=d, q=q, disjoint=True, seed=graph_seed)
    sim_result = simulate_two_regime_nonlinear_var(A1=A1, A2=A2, T1=T1, T2=T2, p=generator_p, burn_in=generator_burn_in, rho=rho, coefficient_scale=coefficient_scale, noise_std=noise_std, nonlinearity=nonlinearity, weight_seed_1=weight_seed, weight_seed_2=weight_seed, noise_seed_1=noise_seed_1, noise_seed_2=noise_seed_2, same_weight_values=True, match_marginal_moments=True, standardize_global=True, return_metadata=True)
    X = sim_result['X']
    tau_true = sim_result['tau']
    T = X.shape[0]
    ours_score = compute_local_cross_regime_boundary_score_normalized(X=X, lag_order=detector_lag_order, h=h, train_window=train_window, val_window=val_window, alpha=alpha, fit_intercept=True, step=step, smooth_window=smooth_window, score_mode=score_mode, verbose=False)
    candidates = ours_score['candidates']
    tau_ours = ours_score['tau_hat']
    mean_score = compute_window_distribution_score(X=X, h=h, lag_order=detector_lag_order, candidates=candidates, metric='mean', smooth_window=distribution_smooth_window)
    cov_score = compute_window_distribution_score(X=X, h=h, lag_order=detector_lag_order, candidates=candidates, metric='covariance', smooth_window=distribution_smooth_window)
    mean_cov_score = compute_window_distribution_score(X=X, h=h, lag_order=detector_lag_order, candidates=candidates, metric='mean_covariance', smooth_window=distribution_smooth_window)
    tau_mid = midpoint_boundary_baseline(T=T, candidates=candidates)
    tau_rand = random_boundary_baseline(candidates=candidates, seed=random_seed)
    tau_hats = {'ours': tau_ours, 'mean': mean_score['tau_hat'], 'covariance': cov_score['tau_hat'], 'mean_covariance': mean_cov_score['tau_hat'], 'midpoint': tau_mid, 'random': tau_rand}
    row = {'replicate_id': replicate_id, 'T': T, 'tau_true': tau_true, 'T1': T1, 'T2': T2, 'graph_seed': graph_seed, 'weight_seed': weight_seed, 'noise_seed_1': noise_seed_1, 'noise_seed_2': noise_seed_2}
    for method, tau_hat in tau_hats.items():
        row[f'{method}_tau_hat'] = int(tau_hat)
        row[f'{method}_abs_error'] = absolute_error(tau_hat, tau_true)
        row[f'{method}_norm_error'] = normalized_error(tau_hat, tau_true, T)
    return row


def run_full_experiment(n_replicates=10, base_seed=123, save_every=None, save_path='boundary_experiment_partial.csv', **kwargs):
    """
    Run repeated boundary-detection experiments.

    Returns
    -------
    rows : list[dict]
        One dictionary per replicate.
    """
    rows = []
    for r in range(n_replicates):
        print(f'Running replicate {r + 1}/{n_replicates}')
        row = run_one_replicate(replicate_id=r, base_seed=base_seed, **kwargs)
        rows.append(row)
        if save_every is not None and (r + 1) % save_every == 0:
            save_rows_to_csv(rows, save_path)
            print(f'Saved partial results to {save_path}')
    return rows


def save_rows_to_csv(rows, path):
    """
    Save list-of-dictionaries results to CSV without pandas.
    """
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows, max_rows=None):
    """
    Pretty-print a list of dictionaries without pandas.
    """
    if max_rows is not None:
        rows = rows[:max_rows]
    for i, row in enumerate(rows):
        print(f'Row {i}:')
        for k, v in row.items():
            print(f'  {k}: {v}')
        print()


def summarize_methods(rows, methods=('ours', 'mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), tolerances=(50, 100, 250, 500, 1000)):
    """
    Summarize absolute error and success rates for all methods.

    Returns
    -------
    summary_rows : list[dict]
    """
    summary_rows = []
    for method in methods:
        errors = np.array([row[f'{method}_abs_error'] for row in rows], dtype=float)
        norm_errors = np.array([row[f'{method}_norm_error'] for row in rows], dtype=float)
        summary = {'method': method, 'mean_abs_error': float(errors.mean()), 'median_abs_error': float(np.median(errors)), 'std_abs_error': float(errors.std(ddof=1)) if len(errors) > 1 else 0.0, 'min_abs_error': float(errors.min()), 'max_abs_error': float(errors.max()), 'mean_norm_error': float(norm_errors.mean()), 'median_norm_error': float(np.median(norm_errors))}
        for tol in tolerances:
            summary[f'success_at_{tol}'] = float(np.mean(errors <= tol))
        summary_rows.append(summary)
    return summary_rows


def bootstrap_mean_ci(values, n_bootstrap=2000, seed=123, ci=95):
    """
    Bootstrap confidence interval for the mean.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    alpha = (100 - ci) / 2
    low, high = np.percentile(boot_means, [alpha, 100 - alpha])
    return (float(low), float(high))


def compare_ours_to_baselines(rows, baselines=('mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), margin=None, n_bootstrap=2000, seed=123):
    """
    Compare our method to each baseline using paired errors.
    """
    comparison_rows = []
    ours_errors = np.array([row['ours_abs_error'] for row in rows], dtype=float)
    if margin is None:
        margin = int(0.005 * rows[0]['T'])
    for baseline in baselines:
        base_errors = np.array([row[f'{baseline}_abs_error'] for row in rows], dtype=float)
        improvement = base_errors - ours_errors
        win = ours_errors < base_errors
        tie = ours_errors == base_errors
        loss = ours_errors > base_errors
        margin_win = ours_errors + margin < base_errors
        ci_low, ci_high = bootstrap_mean_ci(improvement, n_bootstrap=n_bootstrap, seed=seed, ci=95)
        comparison_rows.append({'baseline': baseline, 'ours_mean_error': float(ours_errors.mean()), 'baseline_mean_error': float(base_errors.mean()), 'mean_improvement': float(improvement.mean()), 'ci_95_low': ci_low, 'ci_95_high': ci_high, 'win_rate': float(win.mean()), 'tie_rate': float(tie.mean()), 'loss_rate': float(loss.mean()), f'margin_win_rate_m={margin}': float(margin_win.mean()), 'significant_mean_improvement': bool(ci_low > 0)})
    return comparison_rows


def plot_error_boxplot(rows, methods=('ours', 'mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), title='Boundary Localization Error Across Replicates'):
    """
    Plot absolute boundary error by method.
    """
    data = [np.array([row[f'{method}_abs_error'] for row in rows], dtype=float) for method in methods]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel('Absolute boundary error')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return (fig, ax)


def compute_best_method_proportions(rows, methods=('ours', 'mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), error_type='abs', split_ties=True):
    """
    Compute the proportion of replicates in which each method is best.

    Parameters
    ----------
    rows : list[dict]
        Experimental results.

    methods : tuple[str]
        Methods to compare.

    error_type : str
        Either "abs" or "norm".

    split_ties : bool
        If True, tied best methods share credit equally.
        If False, each tied best method gets full credit.

    Returns
    -------
    proportion_rows : list[dict]
        One row per method with win counts and proportions.
    """
    if error_type not in {'abs', 'norm'}:
        raise ValueError("error_type must be 'abs' or 'norm'.")
    n_replicates = len(rows)
    if n_replicates == 0:
        raise ValueError('rows is empty.')
    win_scores = {method: 0.0 for method in methods}
    suffix = 'abs_error' if error_type == 'abs' else 'norm_error'
    for row in rows:
        method_errors = {method: row[f'{method}_{suffix}'] for method in methods}
        best_error = min(method_errors.values())
        best_methods = [method for method, err in method_errors.items() if np.isclose(err, best_error)]
        if split_ties:
            credit = 1.0 / len(best_methods)
            for method in best_methods:
                win_scores[method] += credit
        else:
            for method in best_methods:
                win_scores[method] += 1.0
    proportion_rows = []
    for method in methods:
        proportion_rows.append({'method': method, 'best_score_total': float(win_scores[method]), 'proportion_best': float(win_scores[method] / n_replicates)})
    return proportion_rows


def plot_best_method_proportions(rows, methods=('ours', 'mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), error_type='abs', split_ties=True, title=None, figsize=(10, 6)):
    """
    Plot a single bar chart showing the proportion of replicates
    in which each method was best.
    """
    proportion_rows = compute_best_method_proportions(rows=rows, methods=methods, error_type=error_type, split_ties=split_ties)
    labels = [row['method'] for row in proportion_rows]
    proportions = [row['proportion_best'] for row in proportion_rows]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, proportions)
    if title is None:
        tie_text = 'Tie-split' if split_ties else 'Ties counted fully'
        title = f'Proportion of Replicates Where Each Method Was Best ({tie_text})'
    ax.set_title(title)
    ax.set_ylabel('Proportion of replicates best')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, p in zip(bars, proportions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return (fig, ax, proportion_rows)


def compute_best_baseline_per_replicate(rows, baseline_methods=('mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), error_type='abs'):
    """
    For each replicate, find the best baseline method and its error.

    Returns
    -------
    best_rows : list[dict]
        One row per replicate with:
            replicate_id
            ours_error
            best_baseline_error
            best_baseline_method
            ours_minus_best_baseline
    """
    if error_type not in {'abs', 'norm'}:
        raise ValueError("error_type must be 'abs' or 'norm'.")
    suffix = 'abs_error' if error_type == 'abs' else 'norm_error'
    best_rows = []
    for row in sorted(rows, key=lambda r: r['replicate_id']):
        baseline_errors = {method: row[f'{method}_{suffix}'] for method in baseline_methods}
        best_baseline_method = min(baseline_errors, key=baseline_errors.get)
        best_baseline_error = baseline_errors[best_baseline_method]
        ours_error = row[f'ours_{suffix}']
        best_rows.append({'replicate_id': row['replicate_id'], 'ours_error': float(ours_error), 'best_baseline_error': float(best_baseline_error), 'best_baseline_method': best_baseline_method, 'ours_minus_best_baseline': float(ours_error - best_baseline_error)})
    return best_rows


def plot_ours_vs_best_baseline_error_curve(rows, baseline_methods=('mean', 'covariance', 'mean_covariance', 'midpoint', 'random'), error_type='abs', title=None, xlabel='Replicate', ylabel=None, marker='o', linewidth=2.0, figsize=(12, 6), annotate_best_baseline=False):
    """
    Plot our method's error curve against the best baseline error
    in each replicate.

    The best baseline is selected separately for each replicate.
    """
    best_rows = compute_best_baseline_per_replicate(rows=rows, baseline_methods=baseline_methods, error_type=error_type)
    replicate_ids = [row['replicate_id'] for row in best_rows]
    ours_errors = [row['ours_error'] for row in best_rows]
    best_baseline_errors = [row['best_baseline_error'] for row in best_rows]
    if title is None:
        title = 'Ours vs Best Baseline Error per Replicate'
    if ylabel is None:
        ylabel = 'Absolute boundary error' if error_type == 'abs' else 'Normalized absolute boundary error'
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(replicate_ids, ours_errors, marker=marker, linewidth=linewidth, label='ours')
    ax.plot(replicate_ids, best_baseline_errors, marker=marker, linewidth=linewidth, label='best baseline per replicate')
    if annotate_best_baseline:
        for row in best_rows:
            ax.text(row['replicate_id'], row['best_baseline_error'], row['best_baseline_method'], fontsize=8, rotation=45, ha='left', va='bottom')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax, best_rows)


import numpy as np


def validate_adjacency(A, k=None, allow_self=True):
    """
    Validate a binary adjacency matrix.

    Convention:
        A[i, j] = 1 means x_j influences dx_i/dt.

    Parameters
    ----------
    A : np.ndarray, shape (k, k)
        Binary adjacency matrix.

    k : int or None
        Expected number of variables.

    allow_self : bool
        Whether self-edges are allowed.

    Returns
    -------
    A : np.ndarray, shape (k, k)
        Validated integer adjacency matrix.
    """
    A = np.asarray(A, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix.')
    if k is not None and A.shape != (k, k):
        raise ValueError(f'A must have shape ({k}, {k}), got {A.shape}.')
    if not np.all((A == 0) | (A == 1)):
        raise ValueError('A must be binary, containing only 0s and 1s.')
    if not allow_self and np.any(np.diag(A) != 0):
        raise ValueError('Self-edges are not allowed, but A has nonzero diagonal entries.')
    return A


def make_random_adjacency(k, edge_prob=0.3, include_self=False, seed=None):
    """
    Create a random directed adjacency matrix.

    Convention:
        A[i, j] = 1 means x_j influences x_i.

    Parameters
    ----------
    k : int
        Number of variables.

    edge_prob : float
        Probability of each directed edge.

    include_self : bool
        Whether to include self-edges.

    seed : int or None
        Random seed.

    Returns
    -------
    A : np.ndarray, shape (k, k)
        Binary directed adjacency matrix.
    """
    rng = np.random.default_rng(seed)
    A = rng.binomial(1, edge_prob, size=(k, k)).astype(int)
    if not include_self:
        np.fill_diagonal(A, 0)
    return A


def classical_lorenz96_adjacency(k, include_self=True):
    """
    Return the classical Lorenz-96 structural graph.

    Classical Lorenz-96:
        dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F

    So x_i depends on:
        x_i, x_{i-1}, x_{i+1}, x_{i-2}

    Convention:
        A[i, j] = 1 means x_j influences x_i.

    Parameters
    ----------
    k : int
        Number of variables.

    include_self : bool
        Whether to include self-dependence.

    Returns
    -------
    A : np.ndarray, shape (k, k)
        Binary adjacency matrix.
    """
    A = np.zeros((k, k), dtype=int)
    for i in range(k):
        if include_self:
            A[i, i] = 1
        A[i, (i - 1) % k] = 1
        A[i, (i + 1) % k] = 1
        A[i, (i - 2) % k] = 1
    return A


def initialize_generalized_lorenz_params(A, F=8.0, linear_scale=0.2, nonlinear_scale=1.0, pairwise_scale=0.1, damping=1.0, seed=None):
    """
    Initialize parameters for the generalized Lorenz-style system.

    The model is:

        dx_i/dt =
            - damping_i * x_i
            + F_i
            + sum_j A[i,j] * W_linear[i,j] * x_j
            + sum_j A[i,j] * W_nonlinear[i,j] * tanh(x_j)
            + pairwise nonlinear interactions among parents of i

    Parameters
    ----------
    A : np.ndarray, shape (k, k)
        Causal adjacency matrix.

    F : float or np.ndarray, shape (k,)
        Forcing term.

    linear_scale : float
        Scale of linear parent effects.

    nonlinear_scale : float
        Scale of nonlinear tanh parent effects.

    pairwise_scale : float
        Scale of parent-parent nonlinear interactions.

    damping : float or np.ndarray, shape (k,)
        Damping coefficient.

    seed : int or None
        Random seed.

    Returns
    -------
    params : dict
        Model parameters.
    """
    rng = np.random.default_rng(seed)
    A = validate_adjacency(A, allow_self=True)
    k = A.shape[0]
    if np.isscalar(F):
        F_vec = float(F) * np.ones(k)
    else:
        F_vec = np.asarray(F, dtype=float)
        if F_vec.shape != (k,):
            raise ValueError(f'F must be scalar or shape ({k},).')
    if np.isscalar(damping):
        damping_vec = float(damping) * np.ones(k)
    else:
        damping_vec = np.asarray(damping, dtype=float)
        if damping_vec.shape != (k,):
            raise ValueError(f'damping must be scalar or shape ({k},).')
    W_linear = rng.normal(loc=0.0, scale=linear_scale, size=(k, k)) * A
    W_nonlinear = rng.normal(loc=0.0, scale=nonlinear_scale, size=(k, k)) * A
    C_pairwise = np.zeros((k, k, k), dtype=float)
    for i in range(k):
        parents = np.where(A[i] == 1)[0]
        if len(parents) > 1:
            for j in parents:
                for l in parents:
                    if j != l:
                        C_pairwise[i, j, l] = rng.normal(loc=0.0, scale=pairwise_scale)
    params = {'A': A, 'F': F_vec, 'damping': damping_vec, 'W_linear': W_linear, 'W_nonlinear': W_nonlinear, 'C_pairwise': C_pairwise, 'linear_scale': linear_scale, 'nonlinear_scale': nonlinear_scale, 'pairwise_scale': pairwise_scale, 'seed': seed}
    return params


def generalized_lorenz_rhs(x, params):
    """
    Compute dx/dt for the generalized Lorenz-style model.

    Convention:
        A[i, j] = 1 means x_j is allowed to influence dx_i/dt.

    Model:
        dx_i/dt =
            - damping_i * x_i
            + F_i
            + sum_j A[i,j] W_linear[i,j] x_j
            + sum_j A[i,j] W_nonlinear[i,j] tanh(x_j)
            + sum_{j,l in Pa(i), j != l} C[i,j,l] x_j x_l

    Parameters
    ----------
    x : np.ndarray, shape (k,)
        Current state.

    params : dict
        Parameters returned by initialize_generalized_lorenz_params.

    Returns
    -------
    dxdt : np.ndarray, shape (k,)
        Time derivative.
    """
    x = np.asarray(x, dtype=float)
    A = params['A']
    F = params['F']
    damping = params['damping']
    W_linear = params['W_linear']
    W_nonlinear = params['W_nonlinear']
    C_pairwise = params['C_pairwise']
    k = A.shape[0]
    if x.shape != (k,):
        raise ValueError(f'x must have shape ({k},), got {x.shape}.')
    linear_part = W_linear @ x
    nonlinear_part = W_nonlinear @ np.tanh(x)
    pairwise_part = np.zeros(k, dtype=float)
    for i in range(k):
        parents = np.where(A[i] == 1)[0]
        for j in parents:
            for l in parents:
                if j != l:
                    pairwise_part[i] += C_pairwise[i, j, l] * x[j] * x[l]
    dxdt = -damping * x + F + linear_part + nonlinear_part + pairwise_part
    return dxdt


def generalized_lorenz_step(x, params, dt=0.01, method='rk4'):
    """
    Advance the generalized Lorenz-style system by one step.

    Parameters
    ----------
    x : np.ndarray, shape (k,)
        Current state.

    params : dict
        Model parameters.

    dt : float
        Integration step size.

    method : {"rk4", "euler"}
        Numerical integration method.

    Returns
    -------
    x_next : np.ndarray, shape (k,)
        Next state.
    """
    if dt <= 0:
        raise ValueError('dt must be positive.')
    if method == 'euler':
        return x + dt * generalized_lorenz_rhs(x, params)
    elif method == 'rk4':
        k1 = generalized_lorenz_rhs(x, params)
        k2 = generalized_lorenz_rhs(x + 0.5 * dt * k1, params)
        k3 = generalized_lorenz_rhs(x + 0.5 * dt * k2, params)
        k4 = generalized_lorenz_rhs(x + dt * k3, params)
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        raise ValueError("method must be either 'rk4' or 'euler'.")


def make_initial_condition(k, F=8.0, mode='near_forcing', random_scale=0.1, perturbation_scale=0.01, seed=None):
    """
    Generate an initial condition.

    Parameters
    ----------
    k : int
        Number of variables.

    F : float
        Baseline forcing value.

    mode : {"near_forcing", "random_normal", "random_uniform", "zeros"}
        Initialization mode.

    random_scale : float
        Random noise scale.

    perturbation_scale : float
        Extra perturbation added to first coordinate in near_forcing mode.

    seed : int or None
        Random seed.

    Returns
    -------
    x0 : np.ndarray, shape (k,)
        Initial state.
    """
    rng = np.random.default_rng(seed)
    if mode == 'near_forcing':
        x0 = F * np.ones(k)
        x0 += rng.uniform(-random_scale, random_scale, size=k)
        x0[0] += perturbation_scale
    elif mode == 'random_normal':
        x0 = F + random_scale * rng.standard_normal(size=k)
    elif mode == 'random_uniform':
        x0 = F + rng.uniform(-random_scale, random_scale, size=k)
    elif mode == 'zeros':
        x0 = np.zeros(k)
    else:
        raise ValueError("mode must be one of: 'near_forcing', 'random_normal', 'random_uniform', 'zeros'.")
    return x0.astype(float)


def standardize_series(X, eps=1e-08):
    """
    Standardize each variable to mean 0 and standard deviation 1.

    Parameters
    ----------
    X : np.ndarray, shape (T, k)
        Time series.

    eps : float
        Numerical stability value.

    Returns
    -------
    X_std : np.ndarray, shape (T, k)
        Standardized time series.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def rescale_and_center_to_range(X, target_range=300.0, eps=1e-08):
    """
    Center the series and rescale it to a target global range.

    Parameters
    ----------
    X : np.ndarray, shape (T, k)
        Time series.

    target_range : float
        Desired approximate max-min range.

    eps : float
        Numerical stability value.

    Returns
    -------
    X_scaled : np.ndarray, shape (T, k)
        Centered and rescaled series.
    """
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)
    current_range = X_centered.max() - X_centered.min()
    return X_centered * (target_range / (current_range + eps))


def simulate_generalized_lorenz(A, T=1000, dt=0.01, burn_in=0, F=8.0, damping=1.0, linear_scale=0.2, nonlinear_scale=1.0, pairwise_scale=0.1, x0=None, initial_mode='near_forcing', random_scale=0.1, perturbation_scale=0.01, method='rk4', noise_std=0.0, standardize=False, rescale_to_range=None, param_seed=None, initial_seed=None, noise_seed=None, return_metadata=True):
    """
    Simulate a generalized Lorenz-style system with fully controlled causal graph.

    Parameters
    ----------
    A : np.ndarray, shape (k, k)
        Directed causal adjacency matrix.
        A[i, j] = 1 means x_j influences x_i.

    T : int
        Number of returned time points.

    dt : float
        Integration step size.

    burn_in : int
        Number of initial steps to discard.

    F : float or np.ndarray, shape (k,)
        Forcing term.

    damping : float or np.ndarray, shape (k,)
        Damping coefficient.

    linear_scale : float
        Scale of random linear parent effects.

    nonlinear_scale : float
        Scale of random nonlinear parent effects.

    pairwise_scale : float
        Scale of random pairwise parent interactions.

    x0 : np.ndarray or None
        Optional initial state.

    initial_mode : str
        Initialization mode if x0 is None.

    random_scale : float
        Random initialization scale.

    perturbation_scale : float
        Perturbation to first coordinate in near_forcing mode.

    method : {"rk4", "euler"}
        Numerical integration method.

    noise_std : float
        Standard deviation of additive Gaussian observation noise.

    standardize : bool
        Whether to standardize variables.

    rescale_to_range : float or None
        Optional global rescaling target.

    param_seed : int or None
        Seed for model weights.

    initial_seed : int or None
        Seed for initial condition.

    noise_seed : int or None
        Seed for observation noise.

    return_metadata : bool
        Whether to return parameters and graph.

    Returns
    -------
    If return_metadata is False:
        X : np.ndarray, shape (T, k)

    If return_metadata is True:
        result : dict
            {
                "X": X,
                "A": A,
                "params": params,
                "x0": x0
            }
    """
    A = validate_adjacency(A, allow_self=True)
    k = A.shape[0]
    if T <= 0:
        raise ValueError('T must be positive.')
    if burn_in < 0:
        raise ValueError('burn_in must be nonnegative.')
    if dt <= 0:
        raise ValueError('dt must be positive.')
    if noise_std < 0:
        raise ValueError('noise_std must be nonnegative.')
    if standardize and rescale_to_range is not None:
        raise ValueError('Use either standardize=True or rescale_to_range, not both.')
    params = initialize_generalized_lorenz_params(A=A, F=F, linear_scale=linear_scale, nonlinear_scale=nonlinear_scale, pairwise_scale=pairwise_scale, damping=damping, seed=param_seed)
    if x0 is None:
        if np.isscalar(F):
            F_for_init = float(F)
        else:
            F_for_init = float(np.mean(F))
        x = make_initial_condition(k=k, F=F_for_init, mode=initial_mode, random_scale=random_scale, perturbation_scale=perturbation_scale, seed=initial_seed)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (k,):
            raise ValueError(f'x0 must have shape ({k},), got {x.shape}.')
    original_x0 = x.copy()
    total_steps = burn_in + T - 1
    X_full = np.zeros((total_steps + 1, k), dtype=float)
    X_full[0] = x.copy()
    for t in range(1, total_steps + 1):
        x = generalized_lorenz_step(x=x, params=params, dt=dt, method=method)
        X_full[t] = x.copy()
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f'Simulation became unstable at step {t}. Try reducing dt, nonlinear_scale, pairwise_scale, or F.')
    X = X_full[burn_in:]
    if noise_std > 0:
        rng_noise = np.random.default_rng(noise_seed)
        X = X + rng_noise.normal(loc=0.0, scale=noise_std, size=X.shape)
    if standardize:
        X = standardize_series(X)
    if rescale_to_range is not None:
        X = rescale_and_center_to_range(X, target_range=rescale_to_range)
    if not return_metadata:
        return X
    return {'X': X, 'A': A, 'params': params, 'x0': original_x0}


def generate_generalized_lorenz_dataset(N, A, T=1000, dt=0.01, burn_in=0, F=8.0, damping=1.0, linear_scale=0.2, nonlinear_scale=1.0, pairwise_scale=0.1, initial_mode='near_forcing', random_scale=0.1, perturbation_scale=0.01, method='rk4', noise_std=0.0, standardize=False, rescale_to_range=None, seed=None, same_dynamics_for_all=True, return_metadata=True):
    """
    Generate many trajectories from a generalized Lorenz-style system.

    Parameters
    ----------
    N : int
        Number of trajectories.

    A : np.ndarray, shape (k, k)
        Directed causal graph.

    same_dynamics_for_all : bool
        If True, all trajectories use the same sampled weights.
        If False, each trajectory gets new weights with the same graph.

    Returns
    -------
    If return_metadata is False:
        X_all : np.ndarray, shape (N, T, k)

    If return_metadata is True:
        result : dict
            {
                "X": X_all,
                "A": A,
                "params": params or list_of_params
            }
    """
    if N <= 0:
        raise ValueError('N must be positive.')
    A = validate_adjacency(A, allow_self=True)
    k = A.shape[0]
    rng = np.random.default_rng(seed)
    X_all = np.zeros((N, T, k), dtype=float)
    params_list = []
    if same_dynamics_for_all:
        shared_param_seed = int(rng.integers(0, 2 ** 32 - 1))
    else:
        shared_param_seed = None
    for i in range(N):
        if same_dynamics_for_all:
            param_seed = shared_param_seed
        else:
            param_seed = int(rng.integers(0, 2 ** 32 - 1))
        initial_seed = int(rng.integers(0, 2 ** 32 - 1))
        noise_seed = int(rng.integers(0, 2 ** 32 - 1))
        result_i = simulate_generalized_lorenz(A=A, T=T, dt=dt, burn_in=burn_in, F=F, damping=damping, linear_scale=linear_scale, nonlinear_scale=nonlinear_scale, pairwise_scale=pairwise_scale, x0=None, initial_mode=initial_mode, random_scale=random_scale, perturbation_scale=perturbation_scale, method=method, noise_std=noise_std, standardize=standardize, rescale_to_range=rescale_to_range, param_seed=param_seed, initial_seed=initial_seed, noise_seed=noise_seed, return_metadata=True)
        X_all[i] = result_i['X']
        params_list.append(result_i['params'])
    if not return_metadata:
        return X_all
    if same_dynamics_for_all:
        params_out = params_list[0]
    else:
        params_out = params_list
    return {'X': X_all, 'A': A, 'params': params_out}


import numpy as np


def simulate_two_regime_generalized_lorenz(A1, A2, T1, T2, dt=0.001, burn_in=500, F=10, damping=1.0, linear_scale=0.1, nonlinear_scale=0.05, pairwise_scale=0.0, x0=None, initial_mode='near_forcing', random_scale=0.05, perturbation_scale=0.005, method='rk4', noise_std=0.001, standardize=True, rescale_to_range=None, param_seed_1=42, param_seed_2=42, initial_seed=123, noise_seed=999, return_metadata=True):
    """
    Simulate a continuous two-regime generalized Lorenz-style time series.

    Regime 1 uses adjacency matrix A1 for T1 returned time points.
    Regime 2 uses adjacency matrix A2 for T2 returned time points.
    The second regime starts from the final state of the first regime.

    Convention:
        A[i, j] = 1 means variable x_j directly influences the dynamics of x_i.

    Returns
    -------
    If return_metadata is False:
        X : np.ndarray, shape (T1 + T2, k)

    If return_metadata is True:
        result : dict with keys:
            X, tau, A1, A2, params_1, params_2, x0, config
    """
    A1 = validate_adjacency(A1, allow_self=True)
    A2 = validate_adjacency(A2, allow_self=True)
    if A1.shape != A2.shape:
        raise ValueError(f'A1 and A2 must have the same shape, got {A1.shape} and {A2.shape}.')
    k = A1.shape[0]
    if T1 <= 0:
        raise ValueError('T1 must be positive.')
    if T2 <= 0:
        raise ValueError('T2 must be positive.')
    if burn_in < 0:
        raise ValueError('burn_in must be nonnegative.')
    if dt <= 0:
        raise ValueError('dt must be positive.')
    if noise_std < 0:
        raise ValueError('noise_std must be nonnegative.')
    if standardize and rescale_to_range is not None:
        raise ValueError('Use either standardize=True or rescale_to_range, not both.')
    params_1 = initialize_generalized_lorenz_params(A=A1, F=F, damping=damping, linear_scale=linear_scale, nonlinear_scale=nonlinear_scale, pairwise_scale=pairwise_scale, seed=param_seed_1)
    params_2 = initialize_generalized_lorenz_params(A=A2, F=F, damping=damping, linear_scale=linear_scale, nonlinear_scale=nonlinear_scale, pairwise_scale=pairwise_scale, seed=param_seed_2)
    if x0 is None:
        F_for_init = float(F) if np.isscalar(F) else float(np.mean(F))
        x = make_initial_condition(k=k, F=F_for_init, mode=initial_mode, random_scale=random_scale, perturbation_scale=perturbation_scale, seed=initial_seed)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (k,):
            raise ValueError(f'x0 must have shape ({k},), got {x.shape}.')
    original_x0 = x.copy()
    for step in range(burn_in):
        x = generalized_lorenz_step(x=x, params=params_1, dt=dt, method=method)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f'Simulation became unstable during burn-in at step {step}.')
    X1 = np.zeros((T1, k), dtype=float)
    X1[0] = x.copy()
    for t in range(1, T1):
        x = generalized_lorenz_step(x=x, params=params_1, dt=dt, method=method)
        X1[t] = x.copy()
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f'Regime 1 became unstable at recorded step {t}.')
    X2 = np.zeros((T2, k), dtype=float)
    for t in range(T2):
        x = generalized_lorenz_step(x=x, params=params_2, dt=dt, method=method)
        X2[t] = x.copy()
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f'Regime 2 became unstable at recorded step {t}.')
    X = np.vstack([X1, X2])
    if noise_std > 0:
        rng_noise = np.random.default_rng(noise_seed)
        X = X + rng_noise.normal(loc=0.0, scale=noise_std, size=X.shape)
    if standardize:
        X = standardize_series(X)
    if rescale_to_range is not None:
        X = rescale_and_center_to_range(X, target_range=rescale_to_range)
    tau = T1
    if not return_metadata:
        return X
    config = {'T1': T1, 'T2': T2, 'tau': tau, 'dt': dt, 'burn_in': burn_in, 'F': F, 'damping': damping, 'linear_scale': linear_scale, 'nonlinear_scale': nonlinear_scale, 'pairwise_scale': pairwise_scale, 'initial_mode': initial_mode, 'random_scale': random_scale, 'perturbation_scale': perturbation_scale, 'method': method, 'noise_std': noise_std, 'standardize': standardize, 'rescale_to_range': rescale_to_range, 'param_seed_1': param_seed_1, 'param_seed_2': param_seed_2, 'initial_seed': initial_seed, 'noise_seed': noise_seed}
    return {'X': X, 'tau': tau, 'A1': A1, 'A2': A2, 'params_1': params_1, 'params_2': params_2, 'x0': original_x0, 'config': config}


import numpy as np


import matplotlib.pyplot as plt


def create_lagged_var_source_grouped(data, p):
    """
    Create lagged autoregressive examples using the same source-grouped
    convention as the cMLP notebook:

        data[t-p:t].T.reshape(-1)

    Therefore the flattened input is grouped by source variable:

        [x0(t-p), ..., x0(t-1),
         x1(t-p), ..., x1(t-1),
         ...
         xd-1(t-p), ..., xd-1(t-1)]

    Returns
    -------
    X_lagged : np.ndarray, shape (T-p, d*p)
    Y : np.ndarray, shape (T-p, d)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f'data must have shape (T, d), got {data.shape}.')
    T, d = data.shape
    if p <= 0:
        raise ValueError('p must be positive.')
    if p >= T:
        raise ValueError(f'p must be smaller than T. Got p={p}, T={T}.')
    X_lagged = []
    Y = []
    for t in range(p, T):
        X_lagged.append(data[t - p:t].T.reshape(-1))
        Y.append(data[t])
    return (np.asarray(X_lagged, dtype=float), np.asarray(Y, dtype=float))


def fit_var_ridge_segment(data, p=6, alpha=1.0, fit_intercept=True, standardize_segment=True):
    """
    Fit multi-output VAR/Ridge on one segment.

    Model:
        x_t = B z_t + b

    where z_t is the source-grouped lag vector.

    Coefficient convention:
        coef[source, target, lag]

    Therefore:
        scores[source, target] = ||coef[source, target, :]||_2
    """
    data = np.asarray(data, dtype=float)
    if standardize_segment:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-08
        data_work = (data - mu) / sigma
    else:
        data_work = data
    X_lagged, Y = create_lagged_var_source_grouped(data_work, p)
    n, input_dim = X_lagged.shape
    d = Y.shape[1]
    if input_dim != d * p:
        raise ValueError(f'Expected input_dim={d * p}, got {input_dim}.')
    if fit_intercept:
        X_design = np.column_stack([np.ones(n), X_lagged])
    else:
        X_design = X_lagged
    ridge_matrix = np.eye(X_design.shape[1])
    if fit_intercept:
        ridge_matrix[0, 0] = 0.0
    W = np.linalg.solve(X_design.T @ X_design + alpha * ridge_matrix, X_design.T @ Y)
    if fit_intercept:
        intercept = W[0]
        coef_flat = W[1:]
    else:
        intercept = np.zeros(d)
        coef_flat = W
    coef_source_lag_target = coef_flat.reshape(d, p, d)
    coef_source_target_lag = np.transpose(coef_source_lag_target, axes=(0, 2, 1))
    return {'intercept': intercept, 'coef_source_target_lag': coef_source_target_lag, 'p': p, 'alpha': alpha, 'fit_intercept': fit_intercept, 'standardize_segment': standardize_segment}


def var_scores_from_model(var_model, remove_self_edges=True):
    """
    Convert VAR/Ridge coefficients into source-to-target Granger scores.

    Convention:
        scores[source, target]
    """
    coef = var_model['coef_source_target_lag']
    scores = np.linalg.norm(coef, ord=2, axis=2)
    if remove_self_edges:
        np.fill_diagonal(scores, 0.0)
    return scores


def var_scores_to_adjacency(scores, threshold=None, quantile=0.75, top_k_per_target=None, remove_self_edges=True):
    """
    Convert source-to-target VAR scores to binary adjacency.

    Convention:
        A_hat[source, target] = 1 means source -> target.
    """
    scores = np.asarray(scores, dtype=float)
    d = scores.shape[0]
    A_hat = np.zeros_like(scores, dtype=int)
    if top_k_per_target is not None:
        for target in range(d):
            col = scores[:, target].copy()
            if remove_self_edges:
                col[target] = -np.inf
            k = min(top_k_per_target, d - int(remove_self_edges))
            if k <= 0:
                continue
            selected_sources = np.argsort(col)[-k:]
            A_hat[selected_sources, target] = 1
    else:
        if threshold is None:
            if remove_self_edges:
                valid_mask = ~np.eye(d, dtype=bool)
                valid_scores = scores[valid_mask]
            else:
                valid_scores = scores.reshape(-1)
            threshold = np.quantile(valid_scores, quantile)
        A_hat = (scores >= threshold).astype(int)
        if remove_self_edges:
            np.fill_diagonal(A_hat, 0)
    return A_hat


def fit_var_granger_segment_with_adjacency(data, p=6, alpha=1.0, fit_intercept=True, standardize_segment=True, threshold=None, quantile=0.75, top_k_per_target=2, remove_self_edges=True):
    """
    Fit VAR/Ridge Granger model on one segment and extract adjacency.

    Output convention:
        A_hat[source, target] = 1.
    """
    model = fit_var_ridge_segment(data=data, p=p, alpha=alpha, fit_intercept=fit_intercept, standardize_segment=standardize_segment)
    scores = var_scores_from_model(model, remove_self_edges=remove_self_edges)
    A_hat = var_scores_to_adjacency(scores=scores, threshold=threshold, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=remove_self_edges)
    return {'model': model, 'scores': scores, 'A_hat': A_hat, 'threshold': threshold, 'quantile': quantile, 'top_k_per_target': top_k_per_target}


def fit_local_var_granger_after_boundary(X, tau_hat, p=6, alpha=1.0, fit_intercept=True, standardize_segment=True, threshold=None, quantile=0.75, top_k_per_target=2, min_segment_length=None):
    """
    Split X at tau_hat and fit VAR/Ridge Granger models separately
    on the two estimated regimes.

    Output convention:
        A_hat[source, target] = 1.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'X must have shape (T, d), got {X.shape}.')
    T, d = X.shape
    tau_hat = int(tau_hat)
    if min_segment_length is None:
        min_segment_length = max(5 * p + 100, 200)
    if tau_hat < min_segment_length:
        raise ValueError(f'Left segment too short: tau_hat={tau_hat}, min_segment_length={min_segment_length}.')
    if T - tau_hat < min_segment_length:
        raise ValueError(f'Right segment too short: T-tau_hat={T - tau_hat}, min_segment_length={min_segment_length}.')
    X_left = X[:tau_hat]
    X_right = X[tau_hat:]
    var_left = fit_var_granger_segment_with_adjacency(data=X_left, p=p, alpha=alpha, fit_intercept=fit_intercept, standardize_segment=standardize_segment, threshold=threshold, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True)
    var_right = fit_var_granger_segment_with_adjacency(data=X_right, p=p, alpha=alpha, fit_intercept=fit_intercept, standardize_segment=standardize_segment, threshold=threshold, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True)
    return {'tau_hat': tau_hat, 'X_left': X_left, 'X_right': X_right, 'var_left': var_left, 'var_right': var_right}


def graph_metrics_source_target(A_true, A_hat, remove_self_edges=True):
    """
    Precision/recall/F1 for graph recovery.

    Convention:
        A[source, target] = 1.
    """
    A_true = np.asarray(A_true, dtype=int).copy()
    A_hat = np.asarray(A_hat, dtype=int).copy()
    if A_true.shape != A_hat.shape:
        raise ValueError(f'Shape mismatch: {A_true.shape} vs {A_hat.shape}.')
    if remove_self_edges:
        np.fill_diagonal(A_true, 0)
        np.fill_diagonal(A_hat, 0)
    true_edges = A_true.astype(bool)
    pred_edges = A_hat.astype(bool)
    tp = np.sum(true_edges & pred_edges)
    fp = np.sum(~true_edges & pred_edges)
    fn = np.sum(true_edges & ~pred_edges)
    tn = np.sum(~true_edges & ~pred_edges)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def normalized_shd_directed(A_true, A_hat, remove_self_edges=True):
    """
    Compute normalized Structural Hamming Distance for directed graphs.

    Convention:
        A[source, target] = 1 means source -> target.

    nSHD(A, A_hat) =
        (# additions + # deletions + # reversals) / (d * (d - 1))

    A reversal is counted as one error, not one deletion plus one addition.
    """
    A_true = np.asarray(A_true, dtype=int).copy()
    A_hat = np.asarray(A_hat, dtype=int).copy()
    if A_true.shape != A_hat.shape:
        raise ValueError(f'Shape mismatch: {A_true.shape} vs {A_hat.shape}.')
    d = A_true.shape[0]
    if remove_self_edges:
        np.fill_diagonal(A_true, 0)
        np.fill_diagonal(A_hat, 0)
    additions = 0
    deletions = 0
    reversals = 0
    for i in range(d):
        for j in range(i + 1, d):
            true_ij = A_true[i, j]
            true_ji = A_true[j, i]
            pred_ij = A_hat[i, j]
            pred_ji = A_hat[j, i]
            if true_ij == 1 and true_ji == 0 and (pred_ij == 0) and (pred_ji == 1):
                reversals += 1
            elif true_ij == 0 and true_ji == 1 and (pred_ij == 1) and (pred_ji == 0):
                reversals += 1
            else:
                deletions += int(true_ij == 1 and pred_ij == 0)
                deletions += int(true_ji == 1 and pred_ji == 0)
                additions += int(true_ij == 0 and pred_ij == 1)
                additions += int(true_ji == 0 and pred_ji == 1)
    shd = additions + deletions + reversals
    nshd = shd / (d * (d - 1))
    return {'additions': int(additions), 'deletions': int(deletions), 'reversals': int(reversals), 'shd': int(shd), 'nshd': float(nshd)}


def plot_adjacency_source_target(A, title='Adjacency matrix', variable_names=None, figsize=(5, 4)):
    """
    Plot adjacency matrix with convention:

        rows    = source variables
        columns = target variables
    """
    A = np.asarray(A)
    d = A.shape[0]
    if variable_names is None:
        variable_names = [f'x{j}' for j in range(d)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target variable')
    ax.set_ylabel('Source variable')
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels(variable_names)
    ax.set_yticklabels(variable_names)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return (fig, ax)


def plot_var_score_histogram(scores, title='VAR/Ridge source-group norm histogram', remove_self_edges=True, bins=20, threshold=None, quantile=None, top_k_per_target=None, figsize=(7, 4)):
    """
    Plot histogram of VAR/Ridge source-to-target lag coefficient norms.

    scores[source, target] measures the lag coefficient norm for source -> target.
    """
    scores = np.asarray(scores, dtype=float)
    d = scores.shape[0]
    if remove_self_edges:
        mask = ~np.eye(d, dtype=bool)
        values = scores[mask]
    else:
        values = scores.reshape(-1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, edgecolor='black', alpha=0.75)
    if threshold is not None:
        ax.axvline(threshold, linestyle='--', linewidth=2, label=f'threshold = {threshold:.4g}')
    if quantile is not None:
        q_value = np.quantile(values, quantile)
        ax.axvline(q_value, linestyle=':', linewidth=2, label=f'quantile {quantile:.2f} = {q_value:.4g}')
    ax.set_title(title)
    ax.set_xlabel('VAR/Ridge lag coefficient L2 norm')
    ax.set_ylabel('Count')
    ax.grid(True, axis='y', alpha=0.3)
    if top_k_per_target is not None:
        ax.text(0.02, 0.95, f'Adjacency uses top_k_per_target={top_k_per_target}', transform=ax.transAxes, ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.15))
    if threshold is not None or quantile is not None:
        ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax)


import numpy as np


import matplotlib.pyplot as plt


import torch


import torch.nn as nn


from torch.utils.data import DataLoader, TensorDataset


def create_lagged_cmlp(data, p):
    """
    Create lagged autoregressive examples using the same convention
    as your MLP-Granger notebook:

        data[t-p:t].T.reshape(-1)

    Therefore the flattened input is grouped by source variable:

        [x0(t-p), ..., x0(t-1),
         x1(t-p), ..., x1(t-1),
         ...
         xd-1(t-p), ..., xd-1(t-1)]
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f'data must have shape (T, d), got {data.shape}.')
    T, d = data.shape
    if p <= 0:
        raise ValueError('p must be positive.')
    if p >= T:
        raise ValueError(f'p must be smaller than T. Got p={p}, T={T}.')
    X_lagged = []
    Y = []
    for t in range(p, T):
        X_lagged.append(data[t - p:t].T.reshape(-1))
        Y.append(data[t])
    return (np.asarray(X_lagged, dtype=np.float32), np.asarray(Y, dtype=np.float32))


class ComponentMLP(nn.Module):
    """
    One target-specific MLP.

    One ComponentMLP is trained per target variable.
    """

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x).squeeze(-1)


def get_groups_for_component_mlp(model, d, p):
    """
    Define first-layer source-variable groups.

    Each source variable j corresponds to first-layer input columns:

        j*p : (j+1)*p
    """
    first_layer = model.hidden_layers[0]
    W = first_layer.weight
    groups = []
    for source in range(d):
        start = source * p
        end = (source + 1) * p
        groups.append({'param': W, 'source': source, 'start': start, 'end': end})
    return groups


class ProximalGroupLasso:
    """
    Proximal group-lasso optimizer wrapper.

    Step:
        1. Gradient step.
        2. Group soft-thresholding on first-layer source groups.
    """

    def __init__(self, parameters, lr, group_lambda, groups):
        self.parameters = list(parameters)
        self.lr = lr
        self.group_lambda = group_lambda
        self.groups = groups

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param -= self.lr * param.grad
            shrink = self.lr * self.group_lambda
            for group_info in self.groups:
                W = group_info['param']
                start = group_info['start']
                end = group_info['end']
                group = W[:, start:end]
                norm = torch.norm(group, p=2)
                if norm > 0:
                    scale = torch.clamp(1.0 - shrink / norm, min=0.0)
                    group.mul_(scale)


def hierarchical_group_lasso(model, d, p, lambda_hgl=0.001):
    """
    Optional group-lasso penalty over first-layer source groups.
    """
    first_layer = model.hidden_layers[0]
    W = first_layer.weight
    penalty = 0.0
    for source in range(d):
        start = source * p
        end = (source + 1) * p
        group = W[:, start:end]
        penalty = penalty + torch.norm(group, p=2)
    return lambda_hgl * penalty


def prune_cmlp_first_layer_groups(model, d, p, group_prune_epsilon=0.0001):
    """
    Post-training pruning of first-layer source-variable groups.

    If

        ||W[:, source*p:(source+1)*p]||_F < group_prune_epsilon,

    then the whole source-variable group is set to zero.
    """
    if group_prune_epsilon is None:
        return
    first_layer = model.hidden_layers[0]
    with torch.no_grad():
        W = first_layer.weight
        for source in range(d):
            start = source * p
            end = (source + 1) * p
            group = W[:, start:end]
            group_norm = torch.norm(group, p='fro')
            if group_norm < group_prune_epsilon:
                group.zero_()


def prune_all_cmlp_models(models, d, p, group_prune_epsilon=0.0001):
    """
    Apply post-training group pruning to all target-specific cMLP models.
    """
    if group_prune_epsilon is None:
        return
    for model in models:
        prune_cmlp_first_layer_groups(model=model, d=d, p=p, group_prune_epsilon=group_prune_epsilon)


def fit_cmlp_granger_segment(data, p=4, hidden_dims=(32, 16), group_lambda=0.1, lr=0.03, n_epochs=5, batch_size=64, use_hgl_penalty=False, hgl_lambda=0.001, standardize_segment=False, group_prune_epsilon=0.0001, device=None, verbose=False):
    """
    Fit cMLP-style Granger model on one segment.

    Training:
        - one ComponentMLP per target variable
        - loss is MSE
        - sparsity from ProximalGroupLasso
        - optional HGL penalty
        - post-training first-layer group pruning

    Convention:
        scores[source, target]
        A_hat[source, target] = 1 means source -> target
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.asarray(data, dtype=np.float32)
    if standardize_segment:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, keepdims=True) + 1e-08
        data_work = (data - mu) / sigma
    else:
        data_work = data
    X_lagged, Y = create_lagged_cmlp(data_work, p)
    n, input_dim = X_lagged.shape
    d = Y.shape[1]
    X_tensor = torch.from_numpy(X_lagged)
    Y_tensor = torch.from_numpy(Y)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    models = []
    optimizers = []
    for target in range(d):
        model = ComponentMLP(input_dim=input_dim, hidden_dims=list(hidden_dims)).to(device)
        groups = get_groups_for_component_mlp(model, d=d, p=p)
        optimizer = ProximalGroupLasso(model.parameters(), lr=lr, group_lambda=group_lambda, groups=groups)
        models.append(model)
        optimizers.append(optimizer)
    criterion = nn.MSELoss()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for Xb, Yb in loader:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            for target in range(d):
                model = models[target]
                optimizer = optimizers[target]
                model.train()
                optimizer.zero_grad()
                pred = model(Xb)
                mse_loss = criterion(pred, Yb[:, target])
                if use_hgl_penalty:
                    penalty = hierarchical_group_lasso(model, d=d, p=p, lambda_hgl=hgl_lambda)
                    loss = mse_loss + penalty
                else:
                    loss = mse_loss
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
        if verbose:
            print(f'epoch {epoch + 1}/{n_epochs}, total_component_loss={epoch_loss:.6f}')
    prune_all_cmlp_models(models=models, d=d, p=p, group_prune_epsilon=group_prune_epsilon)
    scores = np.zeros((d, d), dtype=float)
    for target, model in enumerate(models):
        first_layer = model.hidden_layers[0]
        W = first_layer.weight.detach().cpu().numpy()
        for source in range(d):
            start = source * p
            end = (source + 1) * p
            group = W[:, start:end]
            scores[source, target] = np.linalg.norm(group, ord='fro')
    return {'models': models, 'scores': scores, 'p': p, 'hidden_dims': hidden_dims, 'group_lambda': group_lambda, 'lr': lr, 'n_epochs': n_epochs, 'batch_size': batch_size, 'standardize_segment': standardize_segment, 'group_prune_epsilon': group_prune_epsilon, 'device': str(device)}


def cmlp_scores_to_adjacency(scores, epsilon=None, quantile=0.75, top_k_per_target=None, remove_self_edges=True):
    """
    Convert cMLP source-to-target group norms to binary adjacency.

    Convention:
        A_hat[source, target] = 1 means source -> target.
    """
    scores = np.asarray(scores, dtype=float)
    d = scores.shape[0]
    A_hat = np.zeros_like(scores, dtype=int)
    if top_k_per_target is not None:
        for target in range(d):
            col = scores[:, target].copy()
            if remove_self_edges:
                col[target] = -np.inf
            k = min(top_k_per_target, d - int(remove_self_edges))
            if k <= 0:
                continue
            selected_sources = np.argsort(col)[-k:]
            A_hat[selected_sources, target] = 1
    else:
        if epsilon is None:
            if remove_self_edges:
                valid_mask = ~np.eye(d, dtype=bool)
                valid_scores = scores[valid_mask]
            else:
                valid_scores = scores.reshape(-1)
            epsilon = np.quantile(valid_scores, quantile)
        A_hat = (scores >= epsilon).astype(int)
        if remove_self_edges:
            np.fill_diagonal(A_hat, 0)
    return A_hat


def fit_cmlp_granger_segment_with_adjacency(data, p=4, hidden_dims=(32, 16), group_lambda=0.1, lr=0.03, n_epochs=5, batch_size=64, use_hgl_penalty=False, hgl_lambda=0.001, standardize_segment=False, group_prune_epsilon=0.0001, epsilon=None, quantile=0.75, top_k_per_target=2, remove_self_edges=True, device=None, verbose=False):
    """
    Fit cMLP Granger model on one segment and extract binary adjacency.
    """
    result = fit_cmlp_granger_segment(data=data, p=p, hidden_dims=hidden_dims, group_lambda=group_lambda, lr=lr, n_epochs=n_epochs, batch_size=batch_size, use_hgl_penalty=use_hgl_penalty, hgl_lambda=hgl_lambda, standardize_segment=standardize_segment, group_prune_epsilon=group_prune_epsilon, device=device, verbose=verbose)
    scores = result['scores']
    A_hat = cmlp_scores_to_adjacency(scores=scores, epsilon=epsilon, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=remove_self_edges)
    result['A_hat'] = A_hat
    result['epsilon'] = epsilon
    result['quantile'] = quantile
    result['top_k_per_target'] = top_k_per_target
    return result


def fit_local_cmlp_granger_after_boundary(X, tau_hat, p=4, hidden_dims=(32, 16), group_lambda=0.1, lr=0.03, n_epochs=5, batch_size=64, use_hgl_penalty=False, hgl_lambda=0.001, standardize_segment=False, group_prune_epsilon=0.0001, epsilon=None, quantile=0.75, top_k_per_target=2, min_segment_length=None, device=None, verbose=False):
    """
    Split X at tau_hat and fit cMLP Granger models separately
    on the two estimated regimes.

    Output convention:
        A_hat[source, target] = 1.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f'X must have shape (T, d), got {X.shape}.')
    T, d = X.shape
    tau_hat = int(tau_hat)
    if min_segment_length is None:
        min_segment_length = max(5 * p + 100, 200)
    if tau_hat < min_segment_length:
        raise ValueError(f'Left segment too short: tau_hat={tau_hat}, min_segment_length={min_segment_length}.')
    if T - tau_hat < min_segment_length:
        raise ValueError(f'Right segment too short: T-tau_hat={T - tau_hat}, min_segment_length={min_segment_length}.')
    X_left = X[:tau_hat]
    X_right = X[tau_hat:]
    cmlp_left = fit_cmlp_granger_segment_with_adjacency(data=X_left, p=p, hidden_dims=hidden_dims, group_lambda=group_lambda, lr=lr, n_epochs=n_epochs, batch_size=batch_size, use_hgl_penalty=use_hgl_penalty, hgl_lambda=hgl_lambda, standardize_segment=standardize_segment, group_prune_epsilon=group_prune_epsilon, epsilon=epsilon, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True, device=device, verbose=verbose)
    cmlp_right = fit_cmlp_granger_segment_with_adjacency(data=X_right, p=p, hidden_dims=hidden_dims, group_lambda=group_lambda, lr=lr, n_epochs=n_epochs, batch_size=batch_size, use_hgl_penalty=use_hgl_penalty, hgl_lambda=hgl_lambda, standardize_segment=standardize_segment, group_prune_epsilon=group_prune_epsilon, epsilon=epsilon, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True, device=device, verbose=verbose)
    return {'tau_hat': tau_hat, 'X_left': X_left, 'X_right': X_right, 'cmlp_left': cmlp_left, 'cmlp_right': cmlp_right}


def graph_metrics_source_target(A_true, A_hat, remove_self_edges=True):
    """
    Precision/recall/F1 for graph recovery.

    Convention:
        A[source, target] = 1.
    """
    A_true = np.asarray(A_true, dtype=int).copy()
    A_hat = np.asarray(A_hat, dtype=int).copy()
    if A_true.shape != A_hat.shape:
        raise ValueError(f'Shape mismatch: {A_true.shape} vs {A_hat.shape}.')
    if remove_self_edges:
        np.fill_diagonal(A_true, 0)
        np.fill_diagonal(A_hat, 0)
    true_edges = A_true.astype(bool)
    pred_edges = A_hat.astype(bool)
    tp = np.sum(true_edges & pred_edges)
    fp = np.sum(~true_edges & pred_edges)
    fn = np.sum(true_edges & ~pred_edges)
    tn = np.sum(~true_edges & ~pred_edges)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def normalized_shd_directed(A_true, A_hat, remove_self_edges=True):
    """
    Compute normalized Structural Hamming Distance for directed graphs.

    Convention:
        A[source, target] = 1 means source -> target.

    nSHD(A, A_hat) =
        (# additions + # deletions + # reversals) / (d * (d - 1))

    A reversal is counted as one error, not one deletion plus one addition.
    """
    A_true = np.asarray(A_true, dtype=int).copy()
    A_hat = np.asarray(A_hat, dtype=int).copy()
    if A_true.shape != A_hat.shape:
        raise ValueError(f'Shape mismatch: {A_true.shape} vs {A_hat.shape}.')
    d = A_true.shape[0]
    if remove_self_edges:
        np.fill_diagonal(A_true, 0)
        np.fill_diagonal(A_hat, 0)
    additions = 0
    deletions = 0
    reversals = 0
    for i in range(d):
        for j in range(i + 1, d):
            true_ij = A_true[i, j]
            true_ji = A_true[j, i]
            pred_ij = A_hat[i, j]
            pred_ji = A_hat[j, i]
            if true_ij == 1 and true_ji == 0 and (pred_ij == 0) and (pred_ji == 1):
                reversals += 1
            elif true_ij == 0 and true_ji == 1 and (pred_ij == 1) and (pred_ji == 0):
                reversals += 1
            else:
                deletions += int(true_ij == 1 and pred_ij == 0)
                deletions += int(true_ji == 1 and pred_ji == 0)
                additions += int(true_ij == 0 and pred_ij == 1)
                additions += int(true_ji == 0 and pred_ji == 1)
    shd = additions + deletions + reversals
    nshd = shd / (d * (d - 1))
    return {'additions': int(additions), 'deletions': int(deletions), 'reversals': int(reversals), 'shd': int(shd), 'nshd': float(nshd)}


def plot_adjacency_source_target(A, title='Adjacency matrix', variable_names=None, figsize=(5, 4)):
    """
    Plot adjacency matrix with convention:

        rows    = source variables
        columns = target variables
    """
    A = np.asarray(A)
    d = A.shape[0]
    if variable_names is None:
        variable_names = [f'x{j}' for j in range(d)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target variable')
    ax.set_ylabel('Source variable')
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d))
    ax.set_xticklabels(variable_names)
    ax.set_yticklabels(variable_names)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return (fig, ax)


def plot_cmlp_score_histogram(scores, title='cMLP source-group norm histogram', remove_self_edges=True, bins=20, epsilon=None, quantile=None, top_k_per_target=None, figsize=(7, 4)):
    """
    Plot histogram of cMLP source-to-target first-layer group norms.

    scores[source, target] measures the learned group norm for source -> target.
    """
    scores = np.asarray(scores, dtype=float)
    d = scores.shape[0]
    if remove_self_edges:
        mask = ~np.eye(d, dtype=bool)
        values = scores[mask]
    else:
        values = scores.reshape(-1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, edgecolor='black', alpha=0.75)
    if epsilon is not None:
        ax.axvline(epsilon, linestyle='--', linewidth=2, label=f'epsilon = {epsilon:.4g}')
    if quantile is not None:
        q_value = np.quantile(values, quantile)
        ax.axvline(q_value, linestyle=':', linewidth=2, label=f'quantile {quantile:.2f} = {q_value:.4g}')
    ax.set_title(title)
    ax.set_xlabel('First-layer source-group Frobenius norm')
    ax.set_ylabel('Count')
    ax.grid(True, axis='y', alpha=0.3)
    if top_k_per_target is not None:
        ax.text(0.02, 0.95, f'Adjacency uses top_k_per_target={top_k_per_target}', transform=ax.transAxes, ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.15))
    if epsilon is not None or quantile is not None:
        ax.legend(loc='best')
    plt.tight_layout()
    return (fig, ax)


def fit_global_var_granger_baseline(X, p=6, alpha=1.0, fit_intercept=True, standardize_segment=False, top_k_per_target=2, threshold=None, quantile=0.75):
    """
    Fit one VAR/Ridge Granger model on the entire time series,
    ignoring the regime change.

    Output convention:
        A_hat[source, target] = 1.
    """
    return fit_var_granger_segment_with_adjacency(data=X, p=p, alpha=alpha, fit_intercept=fit_intercept, standardize_segment=standardize_segment, threshold=threshold, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True)


def fit_global_cmlp_granger_baseline(X, p=6, hidden_dims=(32, 16), group_lambda=0.1, lr=0.03, n_epochs=5, batch_size=64, use_hgl_penalty=False, hgl_lambda=0.001, standardize_segment=False, group_prune_epsilon=0.0001, top_k_per_target=2, epsilon=None, quantile=0.75, device=None, verbose=False):
    """
    Fit one cMLP Granger model on the entire time series,
    ignoring the regime change.

    Output convention:
        A_hat[source, target] = 1.
    """
    return fit_cmlp_granger_segment_with_adjacency(data=X, p=p, hidden_dims=hidden_dims, group_lambda=group_lambda, lr=lr, n_epochs=n_epochs, batch_size=batch_size, use_hgl_penalty=use_hgl_penalty, hgl_lambda=hgl_lambda, standardize_segment=standardize_segment, group_prune_epsilon=group_prune_epsilon, epsilon=epsilon, quantile=quantile, top_k_per_target=top_k_per_target, remove_self_edges=True, device=device, verbose=verbose)


def compare_global_vs_local_graphs(global_result, local_left_A, local_right_A, A1_true_source_target, A2_true_source_target, name='Model'):
    """
    Compare a single-regime global graph against both true regime graphs,
    and compare local split graphs against their corresponding regimes.
    """
    A_global = global_result['A_hat']
    global_vs_A1_metrics = graph_metrics_source_target(A1_true_source_target, A_global)
    global_vs_A2_metrics = graph_metrics_source_target(A2_true_source_target, A_global)
    local_vs_A1_metrics = graph_metrics_source_target(A1_true_source_target, local_left_A)
    local_vs_A2_metrics = graph_metrics_source_target(A2_true_source_target, local_right_A)
    global_vs_A1_nshd = normalized_shd_directed(A1_true_source_target, A_global)
    global_vs_A2_nshd = normalized_shd_directed(A2_true_source_target, A_global)
    local_vs_A1_nshd = normalized_shd_directed(A1_true_source_target, local_left_A)
    local_vs_A2_nshd = normalized_shd_directed(A2_true_source_target, local_right_A)
    summary = {'model': name, 'global_vs_regime1_f1': global_vs_A1_metrics['f1'], 'global_vs_regime2_f1': global_vs_A2_metrics['f1'], 'global_mean_f1': 0.5 * (global_vs_A1_metrics['f1'] + global_vs_A2_metrics['f1']), 'local_vs_regime1_f1': local_vs_A1_metrics['f1'], 'local_vs_regime2_f1': local_vs_A2_metrics['f1'], 'local_mean_f1': 0.5 * (local_vs_A1_metrics['f1'] + local_vs_A2_metrics['f1']), 'global_vs_regime1_nshd': global_vs_A1_nshd['nshd'], 'global_vs_regime2_nshd': global_vs_A2_nshd['nshd'], 'global_mean_nshd': 0.5 * (global_vs_A1_nshd['nshd'] + global_vs_A2_nshd['nshd']), 'local_vs_regime1_nshd': local_vs_A1_nshd['nshd'], 'local_vs_regime2_nshd': local_vs_A2_nshd['nshd'], 'local_mean_nshd': 0.5 * (local_vs_A1_nshd['nshd'] + local_vs_A2_nshd['nshd'])}
    print('\n' + '=' * 80)
    print(f'{name}: single-regime global baseline vs time-varying split')
    print('=' * 80)
    print('\nGlobal graph vs true regime 1:')
    print(global_vs_A1_metrics)
    print(global_vs_A1_nshd)
    print('\nGlobal graph vs true regime 2:')
    print(global_vs_A2_metrics)
    print(global_vs_A2_nshd)
    print('\nLocal split graph vs true regime 1:')
    print(local_vs_A1_metrics)
    print(local_vs_A1_nshd)
    print('\nLocal split graph vs true regime 2:')
    print(local_vs_A2_metrics)
    print(local_vs_A2_nshd)
    print('\nCompact summary:')
    print(summary)
    return {'A_global': A_global, 'global_vs_A1_metrics': global_vs_A1_metrics, 'global_vs_A2_metrics': global_vs_A2_metrics, 'local_vs_A1_metrics': local_vs_A1_metrics, 'local_vs_A2_metrics': local_vs_A2_metrics, 'global_vs_A1_nshd': global_vs_A1_nshd, 'global_vs_A2_nshd': global_vs_A2_nshd, 'local_vs_A1_nshd': local_vs_A1_nshd, 'local_vs_A2_nshd': local_vs_A2_nshd, 'summary': summary}
