import os
import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_utils_5 import get_related_digits, fixed_label_images

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUN_DATE = datetime.date.today().strftime("%Y-%m-%d")
_IMG_ROOT = "img"
DIGIT_COLORS = plt.cm.tab10.colors


def set_img_root(name):
    global _IMG_ROOT
    _IMG_ROOT = name


def _imgpath(subdir, filename):
    path = os.path.join(_BASE, _IMG_ROOT, subdir)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def plot_latent_sekitori(z, selected_labels, epoch, track_digit):
    z_np = z.detach().cpu().numpy()
    related_digits = get_related_digits(track_digit)
    colors = ["red", "green", "blue", "cyan", "magenta"]
    labels = [str(d) for d in related_digits]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)

    plt.figure(figsize=(8, 8))
    for rel in range(len(related_digits)):
        idx = np.where(selected_labels == rel)[0]
        if idx.size == 0:
            continue
        plt.scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            color=colors[rel],
            alpha=0.6,
            s=30,
            label=f"target={labels[rel]}",
        )
    plt.title(f"Latent Sekitori (track={track_digit}) epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_sekitori_target", f"epoch_{epoch:03d}.png"))
    plt.close()


def plot_latent_close_teacher(z, close_labels, epoch, track_digit):
    z_np = z.detach().cpu().numpy()
    related_digits = get_related_digits(track_digit)
    colors = ["red", "green", "blue", "cyan", "magenta"]
    labels = [str(d) for d in related_digits]

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_np)

    plt.figure(figsize=(8, 8))
    for rel in range(len(related_digits)):
        idx = np.where(np.array(close_labels) == rel)[0]
        if idx.size == 0:
            continue
        plt.scatter(
            z_2d[idx, 0],
            z_2d[idx, 1],
            color=colors[rel],
            alpha=0.6,
            s=30,
            label=f"closest={labels[rel]}",
        )
    plt.title(f"Latent Close Teacher (track={track_digit}) epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_sekitori_mse", f"epoch_{epoch:03d}.png"))
    plt.close()


def plot_sekitori_teachers(track_image, related_tr, epoch, track_digit):
    related_digits = get_related_digits(track_digit)
    images = related_tr.squeeze(0).detach().cpu().numpy()
    plt.figure(figsize=(15, 3))
    for i, digit in enumerate(related_digits):
        plt.subplot(1, len(related_digits), i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f"{digit}")
        plt.axis("off")
    plt.suptitle(f"Track {track_digit} related teachers epoch {epoch}")
    plt.tight_layout()
    plt.savefig(_imgpath("sekitori_teachers", f"epoch_{epoch:03d}.png"))
    plt.close()


def plot_latent_space_all_digits(mu, z_samples, labels, epoch):
    mu_np = mu.detach().cpu().numpy()
    if isinstance(z_samples, torch.Tensor):
        z_np = z_samples.permute(1, 0, 2).detach().cpu().numpy()
    else:
        z_np = np.asarray(z_samples)
    labels_np = np.asarray(labels)

    batch, n_sample, dim = z_np.shape
    flat_z = z_np.reshape(batch * n_sample, dim)

    pca = PCA(n_components=2)
    flat_2d = pca.fit_transform(flat_z)
    z_2d = flat_2d.reshape(batch, n_sample, 2)
    mu_2d = pca.transform(mu_np)

    plt.figure(figsize=(10, 10))
    for i, digit in enumerate(labels_np):
        color = DIGIT_COLORS[int(digit) % len(DIGIT_COLORS)]
        plt.scatter(
            z_2d[i, :, 0],
            z_2d[i, :, 1],
            color=[color],
            alpha=0.4,
            s=20,
            label=f"{digit}",
        )
        plt.scatter(
            mu_2d[i, 0],
            mu_2d[i, 1],
            color=color,
            edgecolor="black",
            s=120,
            marker="x",
        )

    plt.title(f"Latent Space All Digits epoch {epoch}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_imgpath("latent_all_digits", f"epoch_{epoch:03d}.png"))
    plt.close()


def plot_teacher_images(fixed_label_images, epoch):
    digits = sorted(fixed_label_images.keys())
    plt.figure(figsize=(12, 2))
    for i, d in enumerate(digits):
        plt.subplot(1, len(digits), i + 1)
        plt.imshow(fixed_label_images[d].detach().cpu().numpy().reshape(28, 28), cmap="gray")
        plt.title(str(d))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(_imgpath("teachers", f"epoch_{epoch:03d}.png"))
    plt.close()


def plot_batch_teachers(data, related_images_batch, labels, epoch, batch_idx):
    sample_img = data[0].detach().cpu().numpy().reshape(28, 28)
    related = related_images_batch[0].detach().cpu().numpy()
    plt.figure(figsize=(12, 3))
    plt.subplot(1, len(related) + 1, 1)
    plt.imshow(sample_img, cmap="gray")
    plt.title("input")
    plt.axis("off")
    for i in range(len(related)):
        plt.subplot(1, len(related) + 1, i + 2)
        plt.imshow(related[i].reshape(28, 28), cmap="gray")
        plt.title(f"rel{i}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(_imgpath("batch_teachers", f"epoch_{epoch:03d}_batch_{batch_idx:03d}.png"))
    plt.close()


def generate_digit_variations(model, device, epoch, target_digit, num_variants=10):
    z = torch.randn(num_variants, model.z_dim, device=device)
    with torch.no_grad():
        variants = model.decode(z).detach().cpu().numpy()

    plt.figure(figsize=(num_variants * 1.5, 2))
    for i in range(num_variants):
        plt.subplot(1, num_variants, i + 1)
        plt.imshow(variants[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(_imgpath("digit_variations_5", f"epoch_{epoch:03d}_digit_{target_digit}.png"))
    plt.close()