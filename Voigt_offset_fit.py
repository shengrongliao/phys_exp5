import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from os import listdir

# 定義 Voigt 分布函數
def voigt(x, amplitude, center, sigma, gamma, offset):
    return amplitude * voigt_profile(x - center, sigma, gamma) + offset

# 讀取數據
files = listdir('.\data')

def preduce_figure(file):
    data = np.loadtxt(f".\data\{file}")
    x_data = data[:, 0]
    y_data = data[:, 1]
    # 使用 curve_fit 進行 Voigt 分布擬合
    initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), np.std(x_data) / 2, 1]  # 初始猜測值：amplitude, center, sigma, gamma
    params, covariance = curve_fit(voigt, x_data, y_data, p0=initial_guess, maxfev = 10000000)

    # 擬合參數
    fitted_amplitude, fitted_center, fitted_sigma, fitted_gamma, fitted_offset = params
    func = f"y = {fitted_amplitude:.4f} * voigt_profile(x - μ={fitted_center:.4f}, σ={fitted_sigma:.4f}, γ={fitted_gamma:.4f}) + {fitted_offset:.4f}"
    print(file + "\n    " + func)

    # Calculate R^2
    y_pred = voigt(x_data, *params)
    residuals = y_data - y_pred
    ss_res = np.sum(residuals**2)  # Sum of squares of residuals
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    print(f"    r_sq = {r_squared}")

    # 繪圖
    plt.figure()
    plt.plot(x_data, y_data, 'x', label='data')
    plt.plot(np.linspace(max(x_data), min(x_data), 100), voigt(np.linspace(max(x_data), min(x_data), 100), *params), 'r--', label=f'fitting_curve, R^2={r_squared:.3f}')
    plt.xlabel('theta (degree)')
    plt.ylabel('eff amp')
    plt.legend()
    plt.title(file)
    plt.savefig(f'./image/offset/{file}.png')
    # plt.show()

for file in files:
    preduce_figure(file)
