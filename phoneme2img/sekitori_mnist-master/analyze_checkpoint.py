import argparse
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from config import device, full_dataset, latent_dim as default_latent_dim
from model import VAE


BASE = os.path.dirname(os.path.abspath(__file__))
DIGIT_COLORS = plt.cm.tab10.colors


def _load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    z_dim = ckpt.get("config", {}).get("latent_dim", default_latent_dim)
    model = VAE(z_dim=z_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, z_dim


def _sample_images_per_digit(per_digit):
    buckets = defaultdict(list)
    targets = full_dataset.targets
    for idx, label in enumerate(targets.tolist()):
        if len(buckets[label]) < per_digit:
            img = full_dataset.data[idx].float().view(784) / 255.0
            buckets[label].append(img)
        if all(len(buckets[d]) >= per_digit for d in range(10)):
            break

    images, labels = [], []
    for digit in range(10):
        stack = torch.stack(buckets[digit], dim=0)
        images.append(stack)
        labels.extend([digit] * stack.size(0))
    return torch.cat(images, dim=0), torch.tensor(labels)


def _fit_pca(points, max_points=10000):
    if points.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        fit_points = points[idx]
    else:
        fit_points = points
    pca = PCA(n_components=2)
    pca.fit(fit_points)
    return pca


def _padded_limits(xy, pad_ratio=0.08):
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * pad_ratio
    return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])


def _plot_latent_mu(mu_np, labels_np, out_path, pca, axis_limits):
    xy = pca.transform(mu_np)

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        mask = labels_np == digit
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=12,
            alpha=0.75,
            color=DIGIT_COLORS[digit],
            label=str(digit),
        )
    ax = plt.gca()
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_aspect("equal", adjustable="box")
    plt.title("Encoded mu distribution by digit")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_latent_samples(z_np, labels_np, out_path, pca, axis_limits):
    flat_z = z_np.reshape(-1, z_np.shape[-1])
    repeated_labels = np.repeat(labels_np, z_np.shape[1])
    xy = pca.transform(flat_z)

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        mask = repeated_labels == digit
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=5,
            alpha=0.28,
            color=DIGIT_COLORS[digit],
            label=str(digit),
        )
    ax = plt.gca()
    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_aspect("equal", adjustable="box")
    plt.title("Sampled latent z distribution by input digit (mu PCA axes)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_generated_grid(src_images, generated, out_path, samples_per_digit):
    fig, axes = plt.subplots(10, samples_per_digit + 1, figsize=((samples_per_digit + 1) * 1.2, 12))
    for digit in range(10):
        axes[digit, 0].imshow(src_images[digit].view(28, 28).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[digit, 0].set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=12)
        axes[digit, 0].set_title("input" if digit == 0 else "", fontsize=8)
        axes[digit, 0].axis("off")
        for j in range(samples_per_digit):
            axes[digit, j + 1].imshow(generated[digit, j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[digit, j + 1].set_title(f"{j + 1}" if digit == 0 else "", fontsize=8)
            axes[digit, j + 1].axis("off")
    plt.suptitle("Generated samples from each digit encoder distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_generated_pixel_pca(real_np, real_labels, gen_np, gen_labels, out_path):
    points = np.vstack([real_np, gen_np])
    pca = _fit_pca(points)
    real_xy = pca.transform(real_np)
    gen_xy = pca.transform(gen_np)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, xy, labels, title, marker_size, alpha in [
        (axes[0], real_xy, real_labels, "Real MNIST images PCA", 10, 0.65),
        (axes[1], gen_xy, gen_labels, "Generated images PCA", 10, 0.65),
    ]:
        for digit in range(10):
            mask = labels == digit
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=marker_size,
                alpha=alpha,
                color=DIGIT_COLORS[digit],
                label=str(digit),
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def analyze(checkpoint_path, out_dir, per_digit, z_samples, grid_samples):
    os.makedirs(out_dir, exist_ok=True)
    model, ckpt, z_dim = _load_model(checkpoint_path)
    images, labels = _sample_images_per_digit(per_digit)
    images = images.to(device)

    with torch.no_grad():
        mu, logvar = model.encode(images)
        z = model.reparameterize(mu, logvar, num_samples=z_samples).permute(1, 0, 2)
        decoded = model.decode(z.reshape(-1, z_dim)).view(images.size(0), z_samples, 784)

    labels_np = labels.numpy()
    mu_np = mu.cpu().numpy()
    z_np = z.cpu().numpy()
    decoded_np = decoded.cpu().numpy()

    epoch = ckpt.get("epoch", "unknown")
    latent_pca = _fit_pca(mu_np)
    latent_axis_limits = _padded_limits(latent_pca.transform(mu_np))
    _plot_latent_mu(
        mu_np,
        labels_np,
        os.path.join(out_dir, f"epoch_{epoch}_latent_mu_pca.png"),
        latent_pca,
        latent_axis_limits,
    )
    _plot_latent_samples(
        z_np,
        labels_np,
        os.path.join(out_dir, f"epoch_{epoch}_latent_z_samples_pca.png"),
        latent_pca,
        latent_axis_limits,
    )

    src_images = []
    grid_generated = []
    for digit in range(10):
        first = int(np.where(labels_np == digit)[0][0])
        src_images.append(images[first].cpu())
        grid_generated.append(decoded_np[first, :grid_samples])
    _plot_generated_grid(
        src_images,
        np.stack(grid_generated, axis=0),
        os.path.join(out_dir, f"epoch_{epoch}_generated_grid.png"),
        grid_samples,
    )

    gen_flat = decoded_np.reshape(-1, 784)
    gen_labels = np.repeat(labels_np, z_samples)
    _plot_generated_pixel_pca(
        images.cpu().numpy(),
        labels_np,
        gen_flat,
        gen_labels,
        os.path.join(out_dir, f"epoch_{epoch}_generated_pixel_pca.png"),
    )

    metric_text = ckpt.get("metrics", {})
    summary_path = os.path.join(out_dir, f"epoch_{epoch}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"epoch: {epoch}\n")
        f.write(f"latent_dim: {z_dim}\n")
        f.write(f"per_digit: {per_digit}\n")
        f.write(f"z_samples: {z_samples}\n")
        for key, value in metric_text.items():
            f.write(f"{key}: {value}\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize latent and generated distributions from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(BASE, "checkpoints", "random_teacher", "2026-07-01", "epoch_0200.pt"),
    )
    parser.add_argument("--out-dir", default=os.path.join(BASE, "analysis", "random_teacher", "epoch_0200"))
    parser.add_argument("--per-digit", type=int, default=200)
    parser.add_argument("--z-samples", type=int, default=20)
    parser.add_argument("--grid-samples", type=int, default=12)
    args = parser.parse_args()

    out_dir = analyze(args.checkpoint, args.out_dir, args.per_digit, args.z_samples, args.grid_samples)
    print(f"Saved analysis images to: {out_dir}")


if __name__ == "__main__":
    main()
