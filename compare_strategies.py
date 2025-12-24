import numpy as np
import matplotlib.pyplot as plt
from optimize_weights import max_sharpe_weights, portfolio_stats

np.random.seed(42)

# ----- Inputs (same as your model) -----
mu = np.array([0.0006, 0.0002])     # stock, bond (daily)
sigma = np.array([0.012, 0.004])    # daily vol

rho = -0.2
corr = np.array([[1.0, rho],
                 [rho, 1.0]])
cov = np.outer(sigma, sigma) * corr

days = 252
start_value = 100
n_sims = 5000

rf_daily = 0.0  # you can set e.g. 0.0001 if you want

# ----- Strategies -----
w_7030 = np.array([0.7, 0.3])
w_equal = np.array([0.5, 0.5])
w_opt = max_sharpe_weights(mu, cov, rf_daily=rf_daily, long_only=True)

# ----- Correlated Monte Carlo (same engine) -----
L = np.linalg.cholesky(cov)
Z = np.random.normal(size=(n_sims, days, 2))
asset_returns = mu + (Z @ L.T)  # correlated simple returns

def run_portfolio(w):
    port_rets = asset_returns @ w
    paths = start_value * np.cumprod(1 + port_rets, axis=1)
    final_vals = paths[:, -1]
    r_1y = final_vals / start_value - 1
    var95 = np.percentile(r_1y, 5)
    cvar95 = r_1y[r_1y <= var95].mean()
    prob_loss = np.mean(final_vals < start_value)
    return paths, final_vals, r_1y, prob_loss, var95, cvar95

# Run
paths_7030, final_7030, r7030, pl7030, var7030, cvar7030 = run_portfolio(w_7030)
paths_eq,   final_eq,   req,   pleq,   vareq,   cvareq   = run_portfolio(w_equal)
paths_opt,  final_opt,  ropt,  plopt,  varopt,  cvaropt  = run_portfolio(w_opt)

# ----- Print weights + theoretical Sharpe proxy -----
mu7030, sig7030, sh7030 = portfolio_stats(mu, cov, w_7030, rf_daily)
mueq,   sigeq,   sheq   = portfolio_stats(mu, cov, w_equal, rf_daily)
muopt,  sigopt,  shopt  = portfolio_stats(mu, cov, w_opt, rf_daily)

print("\nWeights:")
print(f"70/30:      {w_7030}")
print(f"Equal:      {w_equal}")
print(f"Optimized:  {np.round(w_opt, 4)}")

print("\nTheoretical (daily) stats from mu/cov:")
print(f"70/30: mu={mu7030:.6f}, sigma={sig7030:.6f}, Sharpe={sh7030:.3f}")
print(f"Equal: mu={mueq:.6f}, sigma={sigeq:.6f}, Sharpe={sheq:.3f}")
print(f"Opt:   mu={muopt:.6f}, sigma={sigopt:.6f}, Sharpe={shopt:.3f}")

print("\nMonte Carlo (1Y) risk metrics:")
print("Strategy   P(Loss)   MeanFinal   VaR95(1Y)   CVaR95(1Y)")
print(f"70/30     {pl7030:7.2%}   {final_7030.mean():9.2f}   {var7030:9.2%}   {cvar7030:10.2%}")
print(f"Equal     {pleq:7.2%}   {final_eq.mean():9.2f}   {vareq:9.2%}   {cvareq:10.2%}")
print(f"Opt       {plopt:7.2%}   {final_opt.mean():9.2f}   {varopt:9.2%}   {cvaropt:10.2%}")

# ----- Quick plot: final value distributions -----
plt.hist(final_7030, bins=50, alpha=0.6, label="70/30")
plt.hist(final_eq, bins=50, alpha=0.6, label="Equal")
plt.hist(final_opt, bins=50, alpha=0.6, label="Optimized")
plt.title("Final Portfolio Value Distribution (1Y)")
plt.xlabel("Final Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
