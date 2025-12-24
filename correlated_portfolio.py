import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

mu = np.array([0.0006, 0.0002])     # stock, bond
sigma = np.array([0.012, 0.004])

days = 252
start_value = 100
n_sims = 2000

w = np.array([0.7, 0.3])            # weights

rho = -0.2                          # correlation (try -0.4, 0, +0.2 later)
corr = np.array([[1.0, rho],
                 [rho, 1.0]])

cov = np.outer(sigma, sigma) * corr
L = np.linalg.cholesky(cov)

Z = np.random.normal(size=(n_sims, days, 2))
asset_returns = mu + (Z @ L.T)      # correlated simple returns

portfolio_returns = asset_returns @ w
portfolio_paths = start_value * np.cumprod(1 + portfolio_returns, axis=1)

# Plot a sample of paths
for i in range(200):
    plt.plot(portfolio_paths[i], linewidth=0.6)

plt.title(f"Monte Carlo: 70/30 Stock–Bond (rho={rho})")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.show()

final_vals = portfolio_paths[:, -1]
one_year_returns = final_vals / start_value - 1

prob_loss = np.mean(final_vals < start_value)

var_95 = np.percentile(one_year_returns, 5)
cvar_95 = one_year_returns[one_year_returns <= var_95].mean()

print(f"P(ending below ${start_value}): {prob_loss*100:.1f}%")
print(f"Average final value: ${final_vals.mean():.2f}")
print(f"VaR 95% (1Y return): {var_95*100:.2f}%")
print(f"CVaR 95% (1Y return): {cvar_95*100:.2f}%")
for rho in [-0.4, -0.2, 0.0, 0.2]:
    corr = np.array([[1.0, rho],[rho, 1.0]])
    cov = np.outer(sigma, sigma) * corr
    L = np.linalg.cholesky(cov)

    Z = np.random.normal(size=(n_sims, days, 2))
    asset_returns = mu + (Z @ L.T)
    port_rets = asset_returns @ w
    paths = start_value * np.cumprod(1 + port_rets, axis=1)

    final_vals = paths[:, -1]
    r = final_vals / start_value - 1
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()

    print(f"rho={rho:+.1f} | VaR95={var_95*100:6.2f}% | CVaR95={cvar_95*100:6.2f}%")
