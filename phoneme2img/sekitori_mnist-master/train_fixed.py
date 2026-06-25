"""
席取りLoss — 教師固定版
各数字の先頭1枚を全バッチ・全エポックで固定教師として使用
"""

import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import (
    device, latent_dim, epochs, lr, num_samples, beta,
    train_loader, val_loader,
    monitor_images, monitor_labels,
    TRACK_DIGIT,
)
from model import VAE
from loss import sekitori_loss_sum
from data_utils import get_fixed_related_images_batch as sample_related_images_batch, fixed_label_images
from visualize import (
    plot_latent_sekitori,
    plot_latent_close_teacher,
    plot_latent_space_all_digits,
    generate_digit_variations,
    plot_teacher_images,
    plot_sekitori_teachers,
    plot_batch_teachers,
)
import visualize
visualize.set_img_root("img_fixed")


def build_selected_labels(assignment, n):
    labels = np.zeros(n, dtype=int)
    for t in range(assignment.shape[0]):
        for pred_idx in assignment[t]:
            labels[int(pred_idx)] = t
    return labels


model = VAE(z_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
writer = SummaryWriter(log_dir="runs/vae_sekitori_fixed")

print(f"[Fixed Teachers] Training VAE on {device} | num_samples={num_samples} | track_digit={TRACK_DIGIT}")

_track_mask = (monitor_labels == TRACK_DIGIT)
track_image = monitor_images[_track_mask][0:1]   # (1, 784)
track_images = torch.stack([fixed_label_images[d] for d in range(10)]).to(device)  # (10, 784)
track_labels = torch.arange(10, device=device)


for epoch in range(1, epochs + 1):
    # ===== Training =====
    model.train()
    train_mse = 0
    train_kld = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)

        optimizer.zero_grad()

        mu, logvar = model.encode(data)
        z_n = model.reparameterize(mu, logvar, num_samples=num_samples)
        recon_n = model.decode(z_n.view(-1, latent_dim)).view(num_samples, -1, 784)

        related_images_batch = sample_related_images_batch(labels)
        pred = recon_n.permute(1, 0, 2)
        target_imgs = related_images_batch

        # plot_batch_teachers(data, related_images_batch, labels, epoch, batch_idx)

        losses_mean, _, _, _, _ = sekitori_loss_sum(pred, target_imgs)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)

        loss = losses_mean + KLD * beta
        loss.backward()
        optimizer.step()

        train_mse += losses_mean.item()
        train_kld += KLD.item()

        n_batches = len(train_loader)
        print(
            f"\rEpoch {epoch:2d}/{epochs} | Batch {batch_idx+1:3d}/{n_batches} | "
            f"MSE: {losses_mean.item():.4f}  KLD: {KLD.item():.4f}",
            end="", flush=True,
        )

    # ===== Validation =====
    model.eval()
    val_mse = 0
    val_kld = 0
    with torch.no_grad():
        for data_v, labels_v in val_loader:
            data_v = data_v.to(device).view(-1, 784)
            mu_v, logvar_v = model.encode(data_v)
            z_v = model.reparameterize(mu_v, logvar_v, num_samples=num_samples)
            recon_v = model.decode(z_v.view(-1, latent_dim)).view(num_samples, -1, 784)

            related_v = sample_related_images_batch(labels_v)
            pred_v = recon_v.permute(1, 0, 2)
            losses_v, *_ = sekitori_loss_sum(pred_v, related_v)

            KLD_v = -0.5 * torch.sum(1 + logvar_v - mu_v.pow(2) - logvar_v.exp()) / data_v.size(0)
            val_mse += losses_v.item()
            val_kld += KLD_v.item()

    n_train = len(train_loader)
    n_val   = len(val_loader)

    writer.add_scalars('MSE',   {'train': train_mse / n_train, 'val': val_mse / n_val},   epoch)
    writer.add_scalars('KLD',   {'train': train_kld / n_train, 'val': val_kld / n_val},   epoch)
    writer.add_scalars('Total', {
        'train': (train_mse + train_kld * beta) / n_train,
        'val':   (val_mse   + val_kld   * beta) / n_val,
    }, epoch)

    print(
        f"\nEpoch {epoch:2d}/{epochs} | "
        f"Train MSE: {train_mse/n_train:.4f}  KLD: {train_kld/n_train:.4f} | "
        f"Val   MSE: {val_mse/n_val:.4f}  KLD: {val_kld/n_val:.4f}"
    )

    # ===== 固定サンプルで潜在空間・教師割り当てを計算 =====
    with torch.no_grad():
        mu_tr, logvar_tr = model.encode(track_image.to(device))
        z_tr = model.reparameterize(mu_tr, logvar_tr, num_samples=num_samples)
        recon_tr = model.decode(z_tr.view(-1, latent_dim)).view(num_samples, 1, 784)
        pred_tr = recon_tr.permute(1, 0, 2)
        related_tr = sample_related_images_batch(torch.tensor([TRACK_DIGIT]))
        _, _, _, indices_close_tr, indices_tr = sekitori_loss_sum(pred_tr, related_tr)
        track_selected = build_selected_labels(indices_tr[0], num_samples)
        track_close = indices_close_tr[0]
        track_z = z_tr[:, 0, :].detach()

    if epoch <= 10:
        # ===== Epoch-end visualization =====
        plot_latent_sekitori(track_z, track_selected, epoch, TRACK_DIGIT) #latent_sekitori_target、 5,4,6の潜在空間
        plot_latent_close_teacher(track_z, track_close, epoch, TRACK_DIGIT) #latent_sekitori_mse、 各zサンプルを最も近い教師（最小MSE）の色で表示
        plot_sekitori_teachers(track_image, related_tr, epoch, TRACK_DIGIT) #sekitori_teachers、入力画像と教師に使われた画像3枚を表示
        generate_digit_variations(model, device, epoch, target_digit=5, num_variants=10) #digit_valiations_5、 5,4,6の画像を10枚生成
        generate_digit_variations(model, device, epoch, target_digit=8, num_variants=10) #digit_valiations_8、 8,7,9の画像を10枚生成
        plot_teacher_images(fixed_label_images, epoch) #teachers、教師画像をエポックごとに表示
    if epoch % 10 == 0:
        # ===== Epoch-end visualization =====
        plot_latent_sekitori(track_z, track_selected, epoch, TRACK_DIGIT) #latent_sekitori_target、 5,4,6の潜在空間
        plot_latent_close_teacher(track_z, track_close, epoch, TRACK_DIGIT) #latent_sekitori_mse、 各zサンプルを最も近い教師（最小MSE）の色で表示
        plot_sekitori_teachers(track_image, related_tr, epoch, TRACK_DIGIT) #sekitori_teachers、入力画像と教師に使われた画像3枚を表示
        generate_digit_variations(model, device, epoch, target_digit=5, num_variants=10) #digit_valiations_5、 5,4,6の画像を10枚生成
        generate_digit_variations(model, device, epoch, target_digit=8, num_variants=10) #digit_valiations_8、 8,7,9の画像を10枚生成
        plot_teacher_images(fixed_label_images, epoch) #teachers、教師画像をエポックごとに表示

        # ===== 固定サンプルで潜在空間・教師割り当てを計算 =====
    with torch.no_grad():
        mu_tr, logvar_tr = model.encode(track_images)
        z_tr = model.reparameterize(mu_tr, logvar_tr, num_samples=num_samples)
        recon_tr = model.decode(z_tr.view(-1, latent_dim)).view(num_samples, -1, 784)
        pred_tr = recon_tr.permute(1, 0, 2)
        related_tr = sample_related_images_batch(track_labels).to(device)

        _, _, _, indices_close_tr, indices_tr = sekitori_loss_sum(pred_tr, related_tr)
        track_selected = np.stack(
            [build_selected_labels(indices_tr[b], num_samples) for b in range(10)],
            axis=0
        )
        track_z = z_tr.detach()

    if epoch <= 10:
        # ===== Epoch-end visualization =====
        # plot_latent_space_all_digits(mu, z_n, labels, epoch) #latent_all_digits、 潜在空間
        plot_latent_space_all_digits(mu_tr, track_z, track_labels.cpu().numpy(), epoch)

    if epoch % 10 == 0:
        # ===== Epoch-end visualization =====
        # plot_latent_space_all_digits(mu, z_n, labels, epoch) #latent_all_digits、 潜在空間
        plot_latent_space_all_digits(mu_tr, track_z, track_labels.cpu().numpy(), epoch)

writer.close()
print("Training Complete!")
