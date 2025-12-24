pip install scipy
import numpy as np
from scipy.optimize import minimize

def max_sharpe_weights(mu_daily, cov_daily, rf_daily=0.0, long_only=True):
    """
    Maximize Sharpe ratio subject to sum(w)=1 and (optionally) w>=0.
    Returns optimal weights.
    """
    n = len(mu_daily)

    def neg_sharpe(w):
        port_mu = w @ mu_daily
        port_var = w @ cov_daily @ w
        port_sigma = np.sqrt(port_var) if port_var > 0 else 1e-12
        sharpe = (port_mu - rf_daily) / port_sigma
        return -sharpe

    # constraints: sum(w)=1
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # bounds
    if long_only:
        bounds = [(0.0, 1.0)] * n
    else:
        bounds = [(-1.0, 1.0)] * n

    w0 = np.ones(n) / n

    res = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 2000}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return res.x


def portfolio_stats(mu_daily, cov_daily, w, rf_daily=0.0):
    port_mu = float(w @ mu_daily)
    port_sigma = float(np.sqrt(w @ cov_daily @ w))
    sharpe = (port_mu - rf_daily) / (port_sigma if port_sigma > 0 else 1e-12)
    return port_mu, port_sigma, sharpe
