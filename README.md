# Monte Carlo Portfolio Risk Simulation

This project uses Monte Carlo simulation to analyze the risk and return of a diversified investment portfolio.

## Overview
A two-asset portfolio (stocks and bonds) is simulated over a one-year horizon using normally distributed daily returns. The simulation estimates expected return, probability of loss, and Value at Risk (VaR).

## Methodology
- Simulated 500 one-year price paths using daily return assumptions
- Constructed a 70/30 stock–bond portfolio
- Measured:
  - Expected final portfolio value
  - Probability of ending below the initial investment
  - 95% Value at Risk (VaR)

## Results
- Expected final value: ~$114
- Probability of loss: ~19%
- 95% VaR: ~−9.4%

## Tools
- Python
- NumPy
- Matplotlib

## Key Insight
Diversification reduced downside risk while preserving expected return, demonstrating the risk–return tradeoff in portfolio construction.
