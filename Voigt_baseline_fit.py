import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from os import listdir

# Define a single Voigt model with a polynomial baseline
def single_voigt_with_baseline(x, amplitude, center, sigma, gamma, a0, a1, a2=0):
    voigt = amplitude * voigt_profile(x - center, sigma, gamma)
    baseline = a0 + a1 * x + a2 * x**2  # Linear + optional quadratic term for flexibility
    return voigt + baseline

# 讀取數據
files = listdir('.\data')

def preduce_figure(file):
    # Load data
    data = np.loadtxt(f".\data\{file}")
    x_data = data[:, 0]
    y_data = data[:, 1]

    # Initial guesses
    initial_guess = [
        max(y_data), np.mean(x_data), np.std(x_data), np.std(x_data) / 2,  # Voigt parameters
        min(y_data), 0, 0  # Baseline terms (a0, a1, and a2 if quadratic is needed)
    ]

    # Bounds for constraint fitting
    bounds = (
        [0, x_data.min(), 0, 0, -np.inf, -np.inf, -np.inf],  # Lower bounds
        [np.inf, x_data.max(), np.inf, np.inf, np.inf, np.inf, np.inf]  # Upper bounds
    )

    # Fit the model to data
    params, covariance = curve_fit(single_voigt_with_baseline, x_data, y_data, p0=initial_guess, bounds=bounds)

    # Extract fitted parameters
    fitted_amplitude, fitted_center, fitted_sigma, fitted_gamma, fitted_a0, fitted_a1, fitted_a2 = params

    print(f"{file}:")
    print(f"    Amplitude: {fitted_amplitude}")
    print(f"    Center: {fitted_center}")
    print(f"    Gaussian width (sigma): {fitted_sigma}")
    print(f"    Lorentzian width (gamma): {fitted_gamma}")
    print(f"    Baseline - Constant (a0): {fitted_a0}, Linear (a1): {fitted_a1}, Quadratic (a2): {fitted_a2}")

    # Calculate R^2
    y_pred = single_voigt_with_baseline(x_data, *params)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    print(f"    r_sq = {r_squared}")

    # Plotting
    plt.figure()
    plt.plot(x_data, y_data, 'x', label='Data')
    plt.plot(np.linspace(max(x_data), min(x_data), 100), single_voigt_with_baseline(np.linspace(max(x_data), min(x_data), 100), *params), 'r--', label=f'fitting_curve, R^2={r_squared:.3f}')
    plt.xlabel('theta (degree)')
    plt.ylabel('eff amp')
    plt.legend()
    plt.title(file)
    plt.savefig(f'./image/baseline/{file}.png')
    # plt.show()

for file in files:
    preduce_figure(file)
