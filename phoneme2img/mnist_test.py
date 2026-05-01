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
epochs = 100
lr = 1e-3

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

def plot_latent_space(mu, z_samples, epoch, label):
    """図1: 潜在空間における中心点μとサンプリング点zの可視化"""
    # 最初のデータのみを抽出
    z_plot = z_samples[:, 0, :].detach().cpu().numpy()  # (100, latent_dim)
    mu_plot = mu[0:1].detach().cpu().numpy()  # (1, latent_dim)

    plt.figure(figsize=(12, 10))
    plt.xlim([-5.0, 5.0])
    plt.ylim([-5.0, 5.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(z_plot[:, 0], z_plot[:, 1], alpha=0.9, s=10, c='red', label='z Samples (n=100)')
    plt.scatter(mu_plot[0, 0], mu_plot[0, 1], c='blue', marker='X', s=200, label='mu')
    plt.title(f"Latent Space (Batch: {epoch}) | Digit: {label}")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(f"mnist_test.png")
    plt.savefig(save_path)
    plt.close()


def plot_output_pca(recon_100, target_img, top10_idx, bottom90_idx, epoch):
    """図2: 出力空間における教師データ(▲)と生成データ(●)のPCA投影"""
    pca = PCA(n_components=2)
    recon_plot = recon_100.detach().cpu().numpy()
    target_plot = target_img.detach().cpu().numpy().reshape(1, -1)

    # 100個の生成画像と1個の教師画像を結合してPCA
    combined_data = np.vstack([recon_plot, target_plot])
    pca_res = pca.fit_transform(combined_data)
    
    recon_pca = pca_res[:100]
    target_pca = pca_res[100]

    plt.figure(figsize=(12, 10))
    plt.xlim([-5.0, 5.0])
    plt.ylim([-5.0, 5.0])
    plt.gca().set_aspect('equal', adjustable='box')
    # 下位90個 (赤●)
    plt.scatter(recon_pca[bottom90_idx, 0], recon_pca[bottom90_idx, 1], 
                c='red', label='Bottom 90 (High Error)', alpha=0.5, s=30)
    # 上位10個 (緑●)
    plt.scatter(recon_pca[top10_idx, 0], recon_pca[top10_idx, 1], 
                c='green', label='Top 10 (Low Error)', alpha=0.8, s=50)
    # 教師ベクトル (青▲)
    plt.scatter(target_pca[0], target_pca[1], c='blue', marker='^', s=200, label='Target Label')

    plt.title(f"Output Pixel Space PCA (epoch: {epoch})")
    plt.xlabel(f"PC1 (Contribution: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC1 (Contribution: {pca.explained_variance_ratio_[0]:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(f"mnist_test2.png")
    plt.savefig(save_path)
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
def generate_digit_variations(epoch, target_digit=0, num_variants=10):
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


# ==========================================
# 5. 学習ループ (上位10個選択アルゴリズム)
# ==========================================
print(f"Training VAE with Elite-Selection (Top 10/100) on {device}...")

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        
        # 1. 潜在変数の分布(μ, σ)を推定
        mu, logvar = model.encode(data)
        
        # 2. 各データに対して100個のベクトルzをサンプリング
        # z_100の形状: (100, batch_size, latent_dim)
        z_100 = model.reparameterize(mu, logvar, num_samples=100)

        # 3. 100個すべてデコードして出力を得る
        # recon_100の形状: (100, batch_size, 784)
        recon_100 = model.decode(z_100.view(-1, latent_dim)).view(100, -1, 784)

        # 4. 各サンプルの誤差(MSE)を計算し、上位10個を特定
        # 誤差計算: (100, batch_size)
        target = data.unsqueeze(0) # (1, batch_size, 784)
        mses = torch.sum((recon_100 - target)**2, dim=2) 
        
        # 各データ（バッチ内）ごとに、100個の中から誤差の小さい順にソート
        _, top_indices = torch.sort(mses, dim=0) # (100, batch_size)
        top10_indices = top_indices[:10, :] # 上位10個のインデックス
        bottom90_indices = top_indices[10:, :] # 下位90個のインデックス
        
        # 5. 上位10個の誤差のみを平均して損失とする
        # batch内の各データについて上位10個のMSEを抽出
        elite_mses = torch.gather(mses, 0, top10_indices) # (10, batch_size)
        recon_loss = elite_mses.mean()
        
        # KLD (正則化項)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
        
        loss = recon_loss + KLD * 0.1 # KLDの重みは適宜調整
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    #     if batch_idx % 10 == 0:
    #         # 図1: 潜在空間の表示
    #         plot_latent_space(mu, z_100, batch_idx, labels[0].item())
    # #     # 図2: 出力空間のPCA表示
    # # plot_output_pca(recon_100.view(-1, 784), data.view(-1,784), top10_indices.cpu(), bottom90_indices.cpu(), batch_idx)
    #         plot_output_pca(
    #             recon_100[:, 0, :],              # 最初のデータの100個のサンプル
    #             data[0:1, :],                    # 最初のデータの教師画像
    #             top10_indices[:10, 0].cpu(),     # 最初のデータの上位10個のインデックス
    #             bottom90_indices[:, 0].cpu(),    # 最初のデータの下位90個のインデックス
    #             f"Epoch {epoch} - Batch {batch_idx} (Digit: {labels[0].item()})"
    #         )
    #         plot_latent_space_all_digits(mu, z_100, labels, epoch)
    
    generate_digit_variations(epoch, target_digit=0,num_variants=10)
    print(f'Epoch {epoch}, Avg Loss: {train_loss / len(train_loader):.4f}')




# # ==========================================
# # 4. 学習ループ
# # ==========================================
# print("Starting Training with Elite-Sample Strategy...")

# for epoch in range(1, epochs + 1):
#     model.train()
#     train_loss = 0
    
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         data = data.to(device).view(-1, 784)
#         optimizer.zero_grad()
        
#         # エンコードとKLD計算
#         mu, logvar = model.encode(data)
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
#         # --- 独自アルゴリズム: 各ステップで最初の1データに対して100個サンプリング ---
#         single_mu = mu[0:1]
#         single_logvar = logvar[0:1]
#         z_100 = model.reparameterize(single_mu, single_logvar, num_samples=100).squeeze(1)
        
#         # 100個デコード
#         recon_100 = model.decode(z_100)
#         target_img = data[0]
        
#         # 誤差(MSE)計算
#         mses = torch.sum((recon_100 - target_img)**2, dim=1)
#         sorted_indices = torch.argsort(mses)
#         top10_indices = sorted_indices[:10]
#         bottom90_indices = sorted_indices[10:]
        
#         # 上位10個のみを学習対象にする
#         loss = mses[top10_indices].mean() + KLD / batch_size
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#         # --- 各エポックの最初のみ可視化関数を呼び出す ---
#         if batch_idx % 50 == 0:
#             # 図1: 潜在空間の表示
#             plot_latent_space(single_mu, z_100, batch_idx, labels[0].item())
#             # 図2: 出力空間のPCA表示
#             plot_output_pca(recon_100, target_img, top10_indices.cpu(), bottom90_indices.cpu(), batch_idx)

#     print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader):.4f}')

# print("Training Complete!")

# #pca_transformする！