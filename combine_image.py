import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir


target_floder = 'two_Voigt_with_baseline'


files = listdir('./image/' + target_floder)
# 設定圖片檔案的路徑
files = ['./image/' + target_floder + '/' + i for i in files]
print(files)
# 建立子圖：2 行 3 列的網格
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 遍歷檔案路徑和子圖位置
for i, (file, ax) in enumerate(zip(files, axes.flat)):
    # 讀取圖片
    img = mpimg.imread(file)
    # 在子圖中顯示圖片
    ax.imshow(img)
    ax.axis('off')  # 隱藏軸

# 調整子圖間距
plt.tight_layout()
# 保存合併後的圖片
plt.savefig('./image/combined_'+ target_floder +'.png')
# plt.show()
