import numpy as np
import matplotlib.pyplot as plt
from os import listdir

# 讀取數據
files = listdir('.\data')

def preduce_figure(file):
    data = np.loadtxt(f".\data\{file}")
    x_data = data[:, 0]
    y_data = data[:, 1]

    # 繪圖
    plt.figure()
    plt.plot(x_data, y_data, 'x', label='data')
    plt.xlabel('theta (degree)')
    plt.ylabel('eff amp')
    plt.legend()
    plt.title(file)
    plt.savefig(f'./image/data/{file}.png')
    # plt.show()

for file in files:
    preduce_figure(file)
