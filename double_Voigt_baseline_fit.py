import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from os import listdir

# Define the Voigt model with a linear baseline (optional quadratic baseline for more flexibility)
def double_voigt_with_baseline(x, amp1, cen1, sigma1, gamma1, amp2, cen2, sigma2, gamma2, a0, a1, a2=0):
    voigt1 = amp1 * voigt_profile(x - cen1, sigma1, gamma1)
    voigt2 = amp2 * voigt_profile(x - cen2, sigma2, gamma2)
    baseline = a0 + a1 * x + a2 * x**2
    return voigt1 + voigt2 + baseline

# 讀取數據
files = listdir('.\data')

def preduce_figure(file):
    # Load data
    data = np.loadtxt(f".\data\{file}")
    x_data = data[:, 0]
    y_data = data[:, 1]

    # Initial guesses (add a linear baseline a0 and a1, and optional a2 for quadratic)
    initial_guess = [
        max(y_data), np.mean(x_data) - np.std(x_data)*1.5, np.std(x_data), np.std(x_data) / 2,  # First Voigt
        max(y_data) / 2, np.mean(x_data) + np.std(x_data)*1.5, np.std(x_data), np.std(x_data) / 2,  # Second Voigt
        min(y_data), 0, 0  # Linear and optional quadratic baseline terms
    ]

    # Parameter bounds for constraint-based fitting
    bounds = (
        [0, x_data.min(), 0, 0, 0, x_data.min(), 0, 0, -np.inf, -np.inf, -np.inf],  # Lower bounds
        [np.inf, x_data.max(), np.inf, np.inf, np.inf, x_data.max(), np.inf, np.inf, np.inf, np.inf, np.inf]  # Upper bounds
    )

    try:
        params, covariance = curve_fit(double_voigt_with_baseline, x_data, y_data, p0=initial_guess, bounds=bounds)
    except ValueError as e:
        print(f"Error fitting {file}: {e}")
        return

    # Extract fitted parameters
    [fitted_amp1, fitted_cen1, fitted_sigma1, fitted_gamma1,
     fitted_amp2, fitted_cen2, fitted_sigma2, fitted_gamma2,
     fitted_a0, fitted_a1, fitted_a2] = params
    print(f"{file}:")
    print(f"    First Voigt - Amplitude: {fitted_amp1}, Center: {fitted_cen1}, Sigma: {fitted_sigma1}, Gamma: {fitted_gamma1}")
    print(f"    Second Voigt - Amplitude: {fitted_amp2}, Center: {fitted_cen2}, Sigma: {fitted_sigma2}, Gamma: {fitted_gamma2}")
    print(f"    Baseline - Constant (a0): {fitted_a0}, Linear (a1): {fitted_a1}, Quadratic (a2): {fitted_a2}")

    # Calculate R^2
    y_pred = double_voigt_with_baseline(x_data, *params)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    print(f"    r_sq = {r_squared}")

    # Plotting
    plt.figure()
    plt.plot(x_data, y_data, 'b-', label='Data')
    plt.plot(x_data, double_voigt_with_baseline(x_data, *params), 'r--', label='Double Voigt with Baseline Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(file)
    plt.savefig(f'./image/two_Voigt_with_baseline/{file}.png')
    # plt.show()

for file in files:
    preduce_figure(file)
