
import numpy as np
import matplotlib.pyplot as plt

def cal_simulation(rf, r_risky, sigma_risky, A, max_weight=1.5, show_plot=True):
    '''
    Capital Allocation Line simulation and optimal allocation calculator.

    Parameters:
    rf : float
        Risk-free rate (decimal, e.g., 0.03 for 3%)
    r_risky : float
        Expected return of risky asset (decimal)
    sigma_risky : float
        Volatility of risky asset (decimal)
    A : float
        Risk aversion coefficient
    max_weight : float
        Maximum risky asset weight for plotting (default 1.5 allows leverage)
    show_plot : bool
        Whether to display CAL plot (default True)
    
    Returns:
    dict containing optimal allocation metrics
    '''

    # --- Optimal Weight ---
    w_opt = (r_risky - rf) / (A * sigma_risky**2)

    # --- Generate CAL ---
    weights = np.linspace(0, max_weight, 200)
    portfolio_returns = weights * r_risky + (1 - weights) * rf
    portfolio_risk = weights * sigma_risky

    # --- Optimal Portfolio ---
    opt_return = w_opt * r_risky + (1 - w_opt) * rf
    opt_risk = w_opt * sigma_risky

    # --- Plot ---
    if show_plot:
        plt.figure()
        plt.plot(portfolio_risk * 100, portfolio_returns * 100, label="Capital Allocation Line")
        plt.scatter([0], [rf * 100], label="Risk-Free Asset")
        plt.scatter([sigma_risky * 100], [r_risky * 100], label="Risky Asset")
        plt.scatter([opt_risk * 100], [opt_return * 100], label="Optimal Portfolio")

        plt.xlabel("Portfolio Risk (Volatility %)")
        plt.ylabel("Expected Return (%)")
        plt.title("Capital Allocation Line Simulation")
        plt.legend()
        plt.grid(True)
        plt.show()

    results = {
        "optimal_risky_weight": w_opt,
        "optimal_riskfree_weight": 1 - w_opt,
        "optimal_return": opt_return,
        "optimal_risk": opt_risk
    }

    return results


if __name__ == "__main__":
    # Example usage
    results = cal_simulation(
        rf=0.03,
        r_risky=0.08,
        sigma_risky=0.12,
        A=4
    )

    print("Optimal Risky Allocation:", round(results["optimal_risky_weight"]*100, 2), "%")
    print("Optimal Risk-Free Allocation:", round(results["optimal_riskfree_weight"]*100, 2), "%")
    print("Optimal Expected Return:", round(results["optimal_return"]*100, 2), "%")
    print("Optimal Risk:", round(results["optimal_risk"]*100, 2), "%")
