# Market Risk & Value at Risk (VaR) Modeling Project

## Overview
This project implements a complete **Market Risk Management Framework** using real financial data.  
It estimates **Value at Risk (VaR)** using three different approaches and compares their performance:

- Parametric VaR (Variance-Covariance)
- Historical Simulation VaR
- Monte Carlo Simulation VaR

The project uses real market data from the **S&P 500 index**.

---

## Objectives

- Measure market risk using statistical models  
- Compare different VaR methodologies  
- Simulate portfolio risk under uncertainty  
- Build a quant-style risk analysis framework  

---

##  Data Source

- Yahoo Finance (`yfinance`)
- Asset: S&P 500 Index (^GSPC)

---

## Methodology

### 1. Parametric VaR
Assumes normal distribution of returns.

### 2. Historical VaR
Based on empirical return distribution.

### 3. Monte Carlo VaR
Simulates thousands of random return paths.

---

##  Key Concepts Used

- Log Returns
- Standard Deviation (Volatility)
- Normal Distribution
- Monte Carlo Simulation
- Risk Metrics

---

## Results

The model outputs:
- VaR at 95% confidence level
- Comparison across all methods
- Visual risk distribution charts

---
