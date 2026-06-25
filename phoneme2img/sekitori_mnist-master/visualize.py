import os
import datetime
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_utils import get_related_digits, fixed_label_images

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUN_DATE = datetime.date.today().strftime("%Y-%m-%d")
_IMG_ROOT = "img"

def set_img_root(name):
    global _IMG_ROOT
    _IMG_ROOT = name

def _imgpath(subdir, filename):
    path = os.path.join(_BASE, _IMG_ROOT, _RUN_DATE, subdir)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)

DIGIT_COLORS = plt.cm.tab10.colors   # 10色、全グラフで共通

# エポック間で軸を固定するPCAキャッシュ
_pca_sekitori = None   # plot_latent_sekitori / plot_latent_close_teacher 共用
_pca_all = None        # plot_latent_space_all_digits / plot_latent_monitor 共用



def plot_latent_space_all_digits(mu, z_samples, labels, epoch):
    """
    mnist_test4.png: PCAで主成分分析した潜在空間を表示
    各数字について代表μとその周囲の100個のzを色分け
    """
    mu_np = mu.detach().cpu().numpy()
    z_samples_np = z_samples.permute(1, 0, 2).detach().cpu().numpy()
    labels_np = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    cmap = plt.get_cmap('tab10')
    global _pca_all
    # mu_np = mu.detach().cpu().numpy()
    # z_samples_np = z_samples.permute(1, 0, 2).detach().cpu().numpy()
    # labels_np = labels.cpu().numpy()

    selected_digits, selected_mu, selected_z = [], [], []
    for digit in range(10):
        idxs = np.where(labels_np == digit)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        selected_digits.append(digit)
        selected_mu.append(mu_np[idx])
        selected_z.append(z_samples_np[idx])

    if not selected_digits:
        return

    n_per_digit = z_samples_np[0].shape[0]

    target_arr = np.vstack(selected_mu)
    z_arr = np.vstack(selected_z)
    all_points = np.vstack([target_arr, z_arr])

    if _pca_all is None:
        _pca_all = PCA(n_components=2)
        all_2d = _pca_all.fit_transform(all_points)
    else:
        all_2d = _pca_all.transform(all_points)

    target_2d = all_2d[: target_arr.shape[0]]
    z_2d = all_2d[target_arr.shape[0] :]

    plt.figure(figsize=(12, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-8.0, 8.0])
    plt.ylim([-8.0, 8.0])

    start = 0
    for i, digit in enumerate(selected_digits):
        color = DIGIT_COLORS[digit % 10]
        end = start + n_per_digit
        plt.scatter(z_2d[start:end, 0], z_2d[start:end, 1],
                    color=color, alpha=1.0, s=15, label=f'Digit {digit}')
        plt.scatter(target_2d[i, 0], target_2d[i, 1],
                    color=color, marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        start = end

    plt.title(f"Latent Space All Digits PCA (epoch: {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_all_digits", f"epoch_{epoch:04d}.png"))
    plt.close()


def plot_latent_monitor(model, monitor_images, monitor_labels, device, epoch, num_z=20):
    global _pca_all
    model.eval()

    with torch.no_grad():
        mu, logvar = model.encode(monitor_images.to(device))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_z, mu.size(0), mu.size(1), device=device)
        z = (mu + eps * std).permute(1, 0, 2)                     # (N, num_z, latent_dim)

    mu_np = mu.cpu().numpy()
    z_np = z.cpu().numpy()
    labels_np = monitor_labels.numpy()
    n = mu_np.shape[0]

    all_points = np.vstack([mu_np, z_np.reshape(-1, mu_np.shape[1])])
    if _pca_all is None:
        _pca_all = PCA(n_components=2)
        all_2d = _pca_all.fit_transform(all_points)
    else:
        all_2d = _pca_all.transform(all_points)

    mu_2d = all_2d[:n]
    z_2d = all_2d[n:].reshape(n, num_z, 2)

    plt.figure(figsize=(12, 10))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-8.0, 8.0])
    plt.ylim([-8.0, 8.0])

    plotted = set()
    for i in range(n):
        digit = int(labels_np[i])
        color = DIGIT_COLORS[digit % 10]
        label_str = f'Digit {digit}' if digit not in plotted else None
        if label_str:
            plotted.add(digit)
        plt.scatter(z_2d[i, :, 0], z_2d[i, :, 1],
                    color=color, alpha=1.0, s=15, label=label_str)
        plt.scatter(mu_2d[i, 0], mu_2d[i, 1],
                    color=color, marker='X', s=150,
                    edgecolors='black', linewidths=1.5, zorder=10)

    plt.title(f"Fixed Sample Monitor (Epoch {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_monitor", f"epoch_{epoch:04d}.png"))
    plt.close()


def plot_latent_sekitori(z_single, selected_labels, epoch, track_digit=None):
    z_np = z_single.detach().cpu().numpy()
    z_2d = PCA(n_components=2).fit_transform(z_np)

    colors = ['red', 'green', 'blue']
    if track_digit is not None:
        related = get_related_digits(track_digit)
        label_names = [f'Digit {related[0]} (digit-1)', f'Digit {related[1]} (same)', f'Digit {related[2]} (digit+1)']
    else:
        label_names = ['digit-1', 'same', 'digit+1']

    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-8.0, 8.0])
    plt.ylim([-8.0, 8.0])

    plotted = set()
    for i in range(z_2d.shape[0]):
        t = max(0, min(2, int(selected_labels[i])))
        label_str = label_names[t] if t not in plotted else None
        if label_str:
            plotted.add(t)
        plt.scatter(z_2d[i, 0], z_2d[i, 1],
                    c=colors[t], alpha=1.0, s=30, label=label_str)

    digit_str = f" [Digit {track_digit} Fixed Sample]" if track_digit is not None else ""
    plt.title(f"Latent Sekitori Assignment{digit_str} (Epoch {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_sekitori_target", f"epoch_{epoch:04d}.png"))
    plt.close()


def plot_latent_close_teacher(z_single, close_labels, epoch, track_digit=None):
    """各zサンプルを最も近い教師（最小MSE）の色で表示"""
    global _pca_sekitori
    z_np = z_single.detach().cpu().numpy()
    if _pca_sekitori is None:
        _pca_sekitori = PCA(n_components=2)
        z_2d = _pca_sekitori.fit_transform(z_np)
    else:
        z_2d = _pca_sekitori.transform(z_np)

    colors = ['red', 'green', 'blue']
    if track_digit is not None:
        related = get_related_digits(track_digit)
        label_names = [f'Digit {related[0]} (digit-1)', f'Digit {related[1]} (same)', f'Digit {related[2]} (digit+1)']
    else:
        label_names = ['digit-1', 'same', 'digit+1']

    plt.figure(figsize=(8, 8))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([-8.0, 8.0])
    plt.ylim([-8.0, 8.0])

    plotted = set()
    for i in range(z_2d.shape[0]):
        t = max(0, min(2, int(close_labels[i])))
        label_str = label_names[t] if t not in plotted else None
        if label_str:
            plotted.add(t)
        plt.scatter(z_2d[i, 0], z_2d[i, 1],
                    c=colors[t], alpha=1.0, s=30, label=label_str)

    digit_str = f" [Digit {track_digit} Fixed Sample]" if track_digit is not None else ""
    plt.title(f"Latent Close Teacher{digit_str} (Epoch {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_sekitori_mse", f"epoch_{epoch:04d}.png"))
    plt.close()


def plot_teacher_loss_bar(loss_mean_target_np, worst_idx_np, epoch, track_digit=None):
    """教師ごとの平均lossを棒グラフで表示。最悪教師を赤でハイライト"""
    losses = loss_mean_target_np[0]   # (T,)
    worst_t = int(worst_idx_np[0])
    T = len(losses)

    if track_digit is not None:
        related = get_related_digits(track_digit)
        xlabels = [f'Digit {related[i]}' for i in range(T)]
    else:
        xlabels = [f'Teacher {i}' for i in range(T)]

    colors = ['red' if i == worst_t else 'lightgray' for i in range(T)]

    plt.figure(figsize=(6, 4))
    plt.bar(range(T), losses, color=colors, edgecolor='black', width=0.6)
    plt.xticks(range(T), xlabels)
    plt.xlabel('Teacher')
    plt.ylabel('Mean MSE Loss')
    plt.ylim(0, max(losses.max() * 1.3, 0.1))
    digit_str = f' [Digit {track_digit}]' if track_digit is not None else ''
    plt.title(f'Teacher Loss{digit_str} (Epoch {epoch})  worst={xlabels[worst_t]}')
    plt.tight_layout()
    plt.savefig(_imgpath('teacher_loss_bar', f'epoch_{epoch:04d}.png'))
    plt.close()


def plot_sorted_loss_bar(sorted_losses_np, n_worst, epoch, track_digit=None):
    """割り当て後のlossを降順に並べた棒グラフ。使用したworst n_worst本を赤でハイライト"""
    N = len(sorted_losses_np)
    colors = ['red' if i < n_worst else 'lightgray' for i in range(N)]

    plt.figure(figsize=(12, 4))
    plt.bar(range(N), sorted_losses_np, color=colors, edgecolor='none', width=1.0)
    plt.xlabel('Sorted sample index (descending loss)')
    plt.ylabel('Loss value')
    digit_str = f' [Digit {track_digit}]' if track_digit is not None else ''
    plt.title(f'Sorted Sekitori Losses{digit_str} (Epoch {epoch})  used={n_worst}/{N}')
    plt.ylim(0, max(sorted_losses_np.max() * 1.2, 0.1))
    plt.tight_layout()
    plt.savefig(_imgpath('sorted_loss_bar', f'epoch_{epoch:04d}.png'))
    plt.close()


def plot_grad_histogram(grad_norms, epoch, track_digit=None):
    """モジュールごとの勾配L2ノルムを棒グラフで表示"""
    groups = {'encoder': 0.0, 'fc_mu': 0.0, 'fc_logvar': 0.0, 'decoder': 0.0}
    for name, norm in grad_norms.items():
        for key in groups:
            if name.startswith(key):
                groups[key] += norm ** 2
    group_norms = {k: v ** 0.5 for k, v in groups.items()}

    labels = list(group_norms.keys())
    values = list(group_norms.values())

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(labels)), values, color='steelblue', edgecolor='black', width=0.6)
    plt.xticks(range(len(labels)), labels)
    plt.xlabel('Module')
    plt.ylabel('Gradient L2 Norm')
    digit_str = f' [Digit {track_digit}]' if track_digit is not None else ''
    plt.title(f'Gradient Norms{digit_str} (Epoch {epoch})')
    plt.tight_layout()
    plt.savefig(_imgpath('grad_histogram', f'epoch_{epoch:04d}.png'))
    plt.close()


def plot_batch_teachers(data, teachers, labels, epoch, batch_idx, n_show=8):
    """
    バッチ内の先頭 n_show サンプルについて、入力画像と教師3枚を横並びで保存
    data:     (B, 784) Tensor
    teachers: (B, 3, 784) Tensor
    labels:   (B,) Tensor
    """
    n_show = min(n_show, data.size(0))
    fig, axes = plt.subplots(n_show, 4, figsize=(10, n_show * 2.2))
    if n_show == 1:
        axes = axes[None, :]

    col_titles = ['Input', 'Teacher\n(digit-1)', 'Teacher\n(same)', 'Teacher\n(digit+1)']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i in range(n_show):
        digit = labels[i].item()
        axes[i, 0].imshow(data[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[i, 0].set_ylabel(f'label={digit}', fontsize=8, rotation=0, labelpad=35)
        axes[i, 0].axis('off')
        for j in range(3):
            axes[i, j + 1].imshow(teachers[i, j].cpu().numpy().reshape(28, 28), cmap='gray')
            axes[i, j + 1].axis('off')

    plt.suptitle(f'Sekitori Teachers — Epoch {epoch} Batch {batch_idx}', fontsize=11)
    plt.tight_layout()
    plt.savefig(_imgpath('batch_teachers', f'epoch_{epoch:04d}_batch_{batch_idx:04d}.png'))
    plt.close()


def plot_sekitori_teachers(input_img, teacher_imgs, epoch, track_digit=None):
    """
    入力画像1枚 + 席取りlossで使われた教師3枚を横並びで保存
    input_img:   (1, 784) Tensor
    teacher_imgs: (1, 3, 784) Tensor
    """
    related = get_related_digits(track_digit) if track_digit is not None else [None, None, None]
    titles = [
        f'Input\n(digit {track_digit})',
        f'Teacher 0\n(digit {related[0]})',
        f'Teacher 1\n(digit {related[1]})',
        f'Teacher 2\n(digit {related[2]})',
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    axes[0].imshow(input_img.view(28, 28).cpu().numpy(), cmap='gray')
    axes[0].set_title(titles[0], fontsize=9)
    axes[0].axis('off')

    for i in range(3):
        axes[i + 1].imshow(teacher_imgs[0, i].view(28, 28).cpu().numpy(), cmap='gray')
        axes[i + 1].set_title(titles[i + 1], fontsize=9)
        axes[i + 1].axis('off')

    plt.suptitle(f'Sekitori Teachers (Epoch {epoch})', fontsize=11)
    plt.tight_layout()
    plt.savefig(_imgpath('sekitori_teachers', f'epoch_{epoch:04d}.png'))
    plt.close()


def plot_teacher_images(fixed_label_images, epoch):
    """固定教師画像（digit 0〜9 各1枚）を横に並べて保存。毎エポック同じ画像であることを確認するため"""
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.5))
    for digit in range(10):
        img = fixed_label_images[digit].cpu().numpy().reshape(28, 28)
        axes[digit].imshow(img, cmap='gray')
        axes[digit].set_title(f'{digit}', fontsize=11)
        axes[digit].axis('off')
    plt.suptitle(f'Fixed Teacher Images (Epoch {epoch})', fontsize=12)
    plt.tight_layout()
    plt.savefig(_imgpath('teachers', f'epoch_{epoch:04d}.png'))
    plt.close()


def generate_digit_variations(model, device, epoch, target_digit=1, num_variants=10):
    """固定教師画像の target_digit を encode し、同じ mu/logvar から num_variants 回サンプリングして生成"""
    model.eval()
    src_image = fixed_label_images[target_digit].unsqueeze(0).to(device)  # (1, 784)

    with torch.no_grad():
        mu, logvar = model.encode(src_image)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_variants, mu.size(1), device=device)
        z_samples = mu + eps * std                        # (num_variants, latent_dim)
        generated = model.decode(z_samples).cpu()         # (num_variants, 784)

    total = 1 + num_variants
    fig, axes = plt.subplots(1, total, figsize=(total * 1.5, 2.5))

    axes[0].imshow(src_image.view(28, 28).cpu(), cmap='gray')
    axes[0].set_title('Input', fontsize=9)
    axes[0].axis('off')

    for i in range(num_variants):
        axes[i + 1].imshow(generated[i].view(28, 28), cmap='gray')
        axes[i + 1].set_title(f'Output {i+1}', fontsize=9)
        axes[i + 1].axis('off')

    plt.suptitle(f"Epoch {epoch} - Digit {target_digit} ({num_variants} samples from same input)", fontsize=11)
    plt.tight_layout()
    plt.savefig(_imgpath(f"digit_variations_{target_digit}", f"epoch_{epoch:04d}.png"))
    plt.close()
