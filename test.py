import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Daily assumptions for two assets
# Asset 1: Stock
mu_stock = 0.0006
sigma_stock = 0.012

# Asset 2: Bond
mu_bond = 0.0002
sigma_bond = 0.004

days = 252
start_value = 100
n_sims = 500

# Portfolio weights (must sum to 1)
w_stock = 0.7
w_bond = 0.3

# Simulate daily returns
stock_returns = np.random.normal(mu_stock, sigma_stock, size=(n_sims, days))
bond_returns = np.random.normal(mu_bond, sigma_bond, size=(n_sims, days))

# Portfolio daily returns
portfolio_returns = w_stock * stock_returns + w_bond * bond_returns

# Portfolio value paths
portfolio_paths = start_value * np.exp(np.cumsum(portfolio_returns, axis=1))

# Plot paths
for i in range(n_sims):
    plt.plot(portfolio_paths[i], linewidth=0.6)

plt.title("Monte Carlo Simulation: 70/30 Stock–Bond Portfolio")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.show()

# Risk metrics
final_vals = portfolio_paths[:, -1]
prob_loss = np.mean(final_vals < start_value)

print(f"Probability of ending below ${start_value}: {prob_loss*100:.1f}%")
print(f"Average final portfolio value: ${final_vals.mean():.2f}")
# Value at Risk (VaR) at 95% confidence
returns = final_vals / start_value - 1
var_95 = np.percentile(returns, 5)

print(f"95% Value at Risk (VaR): {var_95*100:.2f}%")
