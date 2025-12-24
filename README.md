# Monte Carlo Portfolio Risk & Optimization

This project implements a Monte Carlo simulation framework to evaluate the
risk–return profile of a diversified investment portfolio. The model simulates
thousands of possible market scenarios to quantify downside risk and assess the
impact of diversification and portfolio construction decisions.

---

## Overview

A two-asset portfolio consisting of equities and bonds is simulated over a
one-year horizon using daily return dynamics. Asset returns are modeled jointly
using a covariance matrix to capture correlation effects. Portfolio performance
is evaluated across simulated paths using standard risk metrics employed in
financial risk management.

---

## Methodology

- Simulated thousands of one-year price paths using Monte Carlo methods
- Modeled stock and bond returns with specified means, volatilities, and
  correlation structure
- Constructed and analyzed multiple portfolio strategies, including:
  - Fixed-weight (70/30 stock–bond) portfolio
  - Equal-weight portfolio
  - Optimized portfolio (maximum Sharpe ratio, long-only)
- Computed portfolio value paths and evaluated risk metrics based on simulated
  outcomes

---

## Risk Metrics

For each portfolio strategy, the following measures were estimated from the
simulated return distribution:

- Expected portfolio value
- Probability of capital loss
- Value at Risk (VaR) at the 95% confidence level
- Conditional Value at Risk (CVaR / Expected Shortfall)

---

## Results (Representative)

- Expected final portfolio value: ~$114
- Probability of loss: ~19%
- 95% Value at Risk (VaR): ~−9.4%
- Diversification via negatively correlated assets reduced tail risk relative
  to single-asset exposure

(Exact results vary with simulation parameters and correlation assumptions.)

---

## Key Insight

Monte Carlo simulation highlights how portfolio diversification and correlation
structure significantly influence downside risk. While expected returns remain
stable across strategies, tail risk metrics such as VaR and CVaR are highly
sensitive to asset allocation and dependence assumptions.

---

## Tools & Technologies

- Python
- NumPy
- SciPy (optimization)
- Matplotlib

---

## Future Extensions

- Bootstrap or heavy-tailed return distributions
- Stress testing under adverse market regimes
- Multi-asset portfolio expansion
- Transaction costs and leverage constraints
