import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
from sklearn.decomposition import PCA

# ==========================================
# 1. 設定
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64  # エリート選択を各データで行うため、バッチサイズは控えめに
latent_dim = 20
epochs = 20
lr = 1e-3

transform = transforms.ToTensor()
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ラベル別インデックスと画像を事前作成
label_indices = {d: torch.where(train_dataset.targets == d)[0] for d in range(10)}
label_images = {
    d: train_dataset.data[label_indices[d]].float().view(-1, 784) / 255.0
    for d in range(10)
}

# ==========================================
# 2. VAEモデルの定義
# ==========================================
class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        # エンコーダー: Conv層を追加
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, z_dim)
        self.fc_logvar = nn.Linear(400, z_dim)
        
        # デコーダー: Conv層を追加
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 64*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28))  # 入力形状を (batch, 1, 28, 28) に変更
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar, num_samples=1):
        # 指定個数(100個)サンプリングできるように拡張
        std = torch.exp(0.5 * logvar)
        # epsの形状: (num_samples, batch_size, z_dim)
        eps = torch.randn(num_samples, mu.size(0), mu.size(1)).to(device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 784)  # 出力形状を (batch, 784) に戻す


model = VAE(z_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)



# ==========================================
# 3. 可視化関数の定義
# ==========================================

# グローバル変数: 各数字の最初の出現を記録
digit_first_occurrence = {d: None for d in range(10)}
digit_colors = plt.cm.tab10.colors

def update_digit_first_occurrence(labels, mu, z_100):
    """バッチ内で新しく出現した数字のμとzを記録"""
    for digit in labels.unique().cpu().tolist():
        if digit_first_occurrence[digit] is not None:
            continue
        idx = (labels == digit).nonzero(as_tuple=True)[0][0]
        digit_first_occurrence[digit] = {
            "mu": mu[idx:idx+1].detach().cpu(),
            "z": z_100[:, idx, :].detach().cpu()
        }


# def compute_related_digit_loss(recon_current, related_images):
#     """
#     recon_current: (100, 784)
#     related_images: (3, 784), get_related_digits() による順序は
#       [digit-1, digit, digit+1]
#     上位34個は same digit、次の33個は +1、残り33個は -1 の損失を使う
#     """
#     diff = recon_current.unsqueeze(1) - related_images.unsqueeze(0)  # (100, 3, 784)
#     mse = torch.sum(diff * diff, dim=2)  # (100, 3)

#     same_mse = mse[:, 1]      # same digit
#     plus_mse = mse[:, 2]      # digit + 1
#     minus_mse = mse[:, 0]     # digit - 1

#     order = torch.argsort(same_mse)  # same digit に近い順
#     top34 = order[:34]
#     mid33 = order[34:67]
#     last33 = order[67:]

#     loss_same = same_mse[top34].mean()
#     loss_plus = plus_mse[mid33].mean()
#     loss_minus = minus_mse[last33].mean()

#     return (loss_same + loss_plus + loss_minus) / 3.0


def compute_related_digit_loss(recon_current, related_images):
    """
    recon_current: (100, 784)
    related_images: (3, 784)
    """
    # 100個の生成画像 × 3つの関連画像 = 300個のMSE
    diff = recon_current.unsqueeze(1) - related_images.unsqueeze(0)  # (100, 3, 784)
    mse = torch.sum(diff * diff, dim=2)  # (100, 3)
    mse_flat = mse.view(-1)  # (300,) ← ここがポイント1
    
    # 300個の中から最小の100個を選ぶ
    top100 = torch.topk(mse_flat, k=100, largest=False).values
    loss = top100.mean()  # ← ここがポイント2
    return loss

def plot_latent_space(mu, z_samples, epoch, label):
    """
    入力数字に関連する数字群（±1と本体の3個）
    の潜在空間を同時にプロット。点群が3つ表示される
    """
    related_digits = get_related_digits(label)
    
    plt.figure(figsize=(12, 10))
    plt.xlim([-5.0, 5.0])
    plt.ylim([-5.0, 5.0])
    plt.gca().set_aspect('equal', adjustable='box')
    
    # 関連する各数字についてプロット
    for digit in related_digits:
        if digit_first_occurrence[digit] is None:
            continue
        
        color = digit_colors[digit % len(digit_colors)]
        z_plot = digit_first_occurrence[digit]["z"].numpy()  # (100, latent_dim)
        mu_plot = digit_first_occurrence[digit]["mu"].numpy()  # (1, latent_dim)
        
        # 点群（100個のサンプル）
        plt.scatter(z_plot[:, 0], z_plot[:, 1], 
                   alpha=0.4, s=10, c=[color], label=f'z Samples (Digit {digit})')
        # 中心点μ
        plt.scatter(mu_plot[0, 0], mu_plot[0, 1], 
                   c=[color], marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=10)
        # 数字ラベルを追加
        plt.text(mu_plot[0, 0] + 0.15, mu_plot[0, 1] + 0.15, 
                str(digit), color=color, fontsize=12, fontweight='bold')
    
    plt.title(f"Latent Space (Epoch: {epoch}) | Input Digit: {label} (Related: {related_digits})")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(f"mnist_test.png")
    plt.savefig(save_path)
    plt.close()

def select_top100_recon_labels(recon_current, related_images):
    diff = recon_current.unsqueeze(1) - related_images.unsqueeze(0)  # (100, 3, 784)
    mse = torch.sum(diff * diff, dim=2)  # (100, 3)
    flat = mse.view(-1)
    topk = torch.topk(flat, k=100, largest=False)
    return (topk.indices // 100).cpu().numpy()

def plot_output_pca(recon_100, target_img, related_labels, epoch):
    """
    出力空間のPCA: 選ばれた100個の生成画像を関連数字毎に色分け、ターゲット画像を▲で描画
    """
    if len(related_labels) != recon_100.size(0):
        related_labels = np.zeros(recon_100.size(0), dtype=int)

    pca = PCA(n_components=2)
    recon_plot = recon_100.detach().cpu().numpy()
    target_plot = target_img.detach().cpu().numpy().reshape(1, -1)

    combined_data = np.vstack([recon_plot, target_plot])
    pca_res = pca.fit_transform(combined_data)

    recon_pca = pca_res[: recon_plot.shape[0]]
    target_pca = pca_res[recon_plot.shape[0]]

    plt.figure(figsize=(12, 10))
    plt.gca().set_aspect('equal', adjustable='box')

    colors = ['red', 'green', 'blue']
    plotted = set()
    for i in range(recon_pca.shape[0]):
        rel = int(related_labels[i])
        label_str = f'Related {rel}' if rel not in plotted else None
        if label_str:
            plotted.add(rel)
        plt.scatter(recon_pca[i, 0], recon_pca[i, 1],
                    c=colors[rel], alpha=0.5, s=30, label=label_str)

    plt.scatter(target_pca[0], target_pca[1],
                c='black', marker='^', s=200, label='Target Image')

    plt.title(f"Output PCA with Related Colors (epoch: {epoch})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(f"mnist_test2.png"))
    plt.close()


def plot_latent_space_all_digits(mu, z_samples, labels, epoch):
    """
    mnist_test4.png: 各データのmuとその周囲のz(100個)を色分けして表示
    """
    plt.figure(figsize=(12, 10))
    plt.xlim([-5.0, 5.0])
    plt.ylim([-5.0, 5.0])
    plt.gca().set_aspect('equal', adjustable='box')
    
    cmap = plt.get_cmap('tab10')
    mu_np = mu.detach().cpu().numpy()
    # z_samplesの形状を (batch_size, 100, latent_dim) に変換して扱いやすくする
    z_samples_np = z_samples.permute(1, 0, 2).detach().cpu().numpy()
    labels_np = labels.cpu().numpy()

# 0から9までの各数字について1つずつデータを探す
    for digit in range(10):
        # バッチの中から、現在のdigit（0, 1, 2...）に一致するインデックスを探す
        indices = np.where(labels_np == digit)[0]
        
        if len(indices) > 0:
            # 見つかった場合、その中の最初のデータ（代表1個）を使用
            target_idx = indices[0]
            color = cmap(digit)
            
            # 代表データのサンプリング点100個をプロット (雲のように表示)
            plt.scatter(z_samples_np[target_idx, :, 0], z_samples_np[target_idx, :, 1], 
                        color=color, alpha=0.9, s=10, edgecolors='none', label=f'Digit {digit} (z)')
            
            # 代表データの中心μをプロット (バツ印)
            plt.scatter(mu_np[target_idx, 0], mu_np[target_idx, 1], 
                        color=color, marker='X', s=200, 
                        edgecolors='black', linewidths=1.5, zorder=10)
    # 凡例用のダミープロット (各クラス1つずつ)
    for digit in range(10):
        plt.scatter([], [], color=cmap(digit), label=f'Digit {digit}')

    plt.title(f"Latent Space All Digits (epoch: {epoch})")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig("mnist_test4.png")
    plt.close()

# ==========================================
# 4. 「1」のバリエーションを生成して可視化
# ==========================================
def generate_digit_variations(epoch, target_digit=1, num_variants=10):
    model.eval()
    latents = []
    with torch.no_grad():
        for data, labels in train_loader:
            target_mask = (labels == target_digit)
            if target_mask.any():
                mu, _ = model.encode(data[target_mask].to(device))
                latents.append(mu)
            if len(latents) > 20: break
            
    all_latents = torch.cat(latents, dim=0)
    mean_z = all_latents.mean(dim=0, keepdim=True)
    std_z = all_latents.std(dim=0, keepdim=True)

    # 特定した「1」の領域からランダムにサンプリング
    z_samples = mean_z + torch.randn(num_variants, latent_dim).to(device) * std_z
    with torch.no_grad():
        generated = model.decode(z_samples).cpu()

    plt.figure(figsize=(15, 3))
    for i in range(num_variants):
        plt.subplot(1, num_variants, i+1)
        plt.imshow(generated[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch} - Elite-Learned Variants of Digit {target_digit}")
    save_path = os.path.join(f"mnist_test3.png")
    plt.savefig(save_path)
    plt.close()


def get_related_digits(digit):
    """
    入力数字から関連する数字セットを返す
    入力がdigitのとき、digit-1, digit, digit+1 (modulo 10)の3枚を返す
    例: 0 -> [9, 0, 1], 5 -> [4, 5, 6]
    """
    return [(digit - 1) % 10, digit, (digit + 1) % 10]

def sample_related_images(related_digits):
    """
    関連数字からランダムに1枚ずつ画像をサンプリング
    """
    images = []
    for d in related_digits:
        idx = torch.randint(len(label_images[d]), (1,)).item()
        images.append(label_images[d][idx])
    return torch.stack(images).to(device)

# ==========================================
# 5. 学習ループ (1対多学習)
# ==========================================
print(f"Training VAE with 1-to-Many relationship on {device}...")

# ...existing code...
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0

    last_recon_100 = None
    last_target_img = None
    last_selected_labels = None

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()

        mu, logvar = model.encode(data)
        z_100 = model.reparameterize(mu, logvar, num_samples=100)
        update_digit_first_occurrence(labels, mu, z_100)
        recon_100 = model.decode(z_100.view(-1, latent_dim)).view(100, -1, 784)

        total_recon_loss = 0
        for batch_data_idx in range(data.size(0)):
            input_digit = labels[batch_data_idx].item()
            related_digits = get_related_digits(input_digit)

            related_images_tensor = sample_related_images(related_digits)
            recon_current = recon_100[:, batch_data_idx, :]

            if batch_data_idx == 0:
                last_selected_labels = select_top100_recon_labels(recon_current, related_images_tensor)
            loss_current = compute_related_digit_loss(recon_current, related_images_tensor)
            total_recon_loss += loss_current

        last_recon_100 = recon_100[:, 0, :].detach()
        last_target_img = data[0:1, :].detach()

        recon_loss = total_recon_loss / data.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
        loss = recon_loss + KLD 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if last_selected_labels is not None:
        plot_output_pca(last_recon_100, last_target_img, last_selected_labels, epoch)

    plot_latent_space(mu, z_100, batch_idx, labels[0].item())
    generate_digit_variations(epoch, target_digit=8, num_variants=10)
    plot_latent_space_all_digits(mu, z_100, labels, epoch)
    print(f'Epoch {epoch}, Avg Loss: {train_loss / len(train_loader):.4f}')

print("Training Complete!")