"""
1バッチ分の入力画像と教師画像を確認して img/check_teachers.png に保存
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os

from config import train_loader
from data_utils import get_fixed_related_images_batch

N_SHOW = 8   # 表示するサンプル数

data, labels = next(iter(train_loader))
data = data.view(-1, 784)
teachers = get_fixed_related_images_batch(labels)   # (B, 3, 784)

fig, axes = plt.subplots(N_SHOW, 4, figsize=(10, N_SHOW * 2.2))

col_titles = ['Input', 'Teacher\n(digit-1)', 'Teacher\n(same)', 'Teacher\n(digit+1)']
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9)

for i in range(N_SHOW):
    digit = labels[i].item()
    axes[i, 0].imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
    axes[i, 0].set_ylabel(f'label={digit}', fontsize=8, rotation=0, labelpad=35)
    axes[i, 0].axis('off')
    for j in range(3):
        axes[i, j + 1].imshow(teachers[i, j].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[i, j + 1].axis('off')

plt.suptitle('Input vs Sekitori Teachers (1 batch)', fontsize=12)
plt.tight_layout()

os.makedirs('img', exist_ok=True)
plt.savefig('img/check_teachers.png', dpi=120)
plt.close()
print("Saved: img/check_teachers.png")
