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

def sample_related_images(related_digits):
    images = []
    for d in related_digits:
        idx = torch.randint(len(label_images[d]), (1,)).item()
        images.append(label_images[d][idx])
    return torch.stack(images).to(device)

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

# def plot_latent_space(mu, z_samples, epoch, label):
#     """
#     入力数字に関連する数字群（偶数なら0,2,4,6,8、奇数なら1,3,5,7,9）
#     の潜在空間を同時にプロット。点群が5つ表示される
#     """
#     related_digits = get_related_digits(label)
    
#     plt.figure(figsize=(12, 10))
#     plt.xlim([-5.0, 5.0])
#     plt.ylim([-5.0, 5.0])
#     plt.gca().set_aspect('equal', adjustable='box')
    
#     # 関連する各数字についてプロット
#     for digit in related_digits:
#         if digit_first_occurrence[digit] is None:
#             continue
        
#         color = digit_colors[digit % len(digit_colors)]
#         z_plot = digit_first_occurrence[digit]["z"].numpy()  # (100, latent_dim)
#         mu_plot = digit_first_occurrence[digit]["mu"].numpy()  # (1, latent_dim)
        
#         # 点群（100個のサンプル）
#         plt.scatter(z_plot[:, 0], z_plot[:, 1], 
#                    alpha=0.4, s=10, c=[color], label=f'z Samples (Digit {digit})')
#         # 中心点μ
#         plt.scatter(mu_plot[0, 0], mu_plot[0, 1], 
#                    c=[color], marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=10)
#         # 数字ラベルを追加
#         plt.text(mu_plot[0, 0] + 0.15, mu_plot[0, 1] + 0.15, 
#                 str(digit), color=color, fontsize=12, fontweight='bold')
    
#     digit_type = "Even" if label % 2 == 0 else "Odd"
#     plt.title(f"Latent Space (Batch: {epoch}) | Input Digit: {label} ({digit_type})")
#     plt.xlabel("z1")
#     plt.ylabel("z2")
#     plt.legend(loc='upper right', fontsize='small')
#     plt.grid(True, alpha=0.3)
#     save_path = os.path.join(f"mnist_test.png")
#     plt.savefig(save_path)
#     plt.close()

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


# def get_related_digits(digit):
#     """
#     入力数字から関連する数字セットを返す
#     偶数入力: 0, 2, 4, 6, 8
#     奇数入力: 1, 3, 5, 7, 9
#     """
#     if digit % 2 == 0:
#         return [0, 2, 4, 6, 8]
#     else:
#         return [1, 3, 5, 7, 9]

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

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        
        # 1. 潜在変数の分布(μ, σ)を推定（入力画像に対して）
        mu, logvar = model.encode(data)
        
        # 2. 各データに対して100個のベクトルzをサンプリング
        # z_100 = model.reparameterize(mu, logvar, num_samples=100)
        
        z_100 = model.reparameterize(mu, logvar, num_samples=100)
        
        # 各数字の最初の出現を記録
        update_digit_first_occurrence(labels, mu, z_100)

        # 3. 100個すべてデコードして出力を得る
        recon_100 = model.decode(z_100.view(-1, latent_dim)).view(100, -1, 784)

        # 4. バッチ内の各データに対して、関連数字の教師データを集める
        total_recon_loss = 0
        
        for batch_data_idx in range(data.size(0)):
            # input_digit = labels[batch_data_idx].item()
            # related_digits = get_related_digits(input_digit)
            
            # # 訓練データから関連数字の画像を取得
            # related_images = []
            # for related_digit in related_digits:
            #     # 訓練セット全体から関連数字のデータを取得（簡易版）
            #     for train_data, train_labels in train_loader:
            #         mask = train_labels == related_digit
            #         if mask.any():
            #             related_images.append(train_data[mask][0].to(device).view(-1))
            #             break
            
            # if len(related_images) == len(related_digits):
            #     related_images_tensor = torch.stack(related_images)  # (num_related, 784)
                
            #     # 現在のデータのz_100に対する再構成と、関連数字との誤差を計算
            #     recon_current = recon_100[:, batch_data_idx, :]  # (100, 784)
                
            #     # 関連数字に対する誤差（平均）
            #     mse_to_related = 0
            #     for related_img in related_images_tensor:
            #         mse_to_related += torch.sum((recon_current - related_img)**2, dim=1)  # (100,)
                
            #     mse_to_related /= len(related_images_tensor)
                
            #     # 上位10個を選択
            #     _, top_indices = torch.topk(mse_to_related, k=10, largest=False)
            #     elite_loss = mse_to_related[top_indices].mean()
            #     total_recon_loss += elite_loss

            input_digit = labels[batch_data_idx].item()
            related_digits = get_related_digits(input_digit)

            related_images_tensor = sample_related_images(related_digits)

            # 現在のデータのz_100に対する再構成と、関連数字との誤差を計算
            recon_current = recon_100[:, batch_data_idx, :]  # (100, 784)

            # 関連数字に対する誤差（平均）
            mse_to_related = 0
            for related_img in related_images_tensor:
                mse_to_related += torch.sum((recon_current - related_img)**2, dim=1)  # (100,)

            mse_to_related /= len(related_images_tensor)

            # 上位10個を選択
            _, top_indices = torch.topk(mse_to_related, k=10, largest=False)
            elite_loss = mse_to_related[top_indices].mean()
            total_recon_loss += elite_loss
        
        recon_loss = total_recon_loss / data.size(0)
        
        # KLD (正則化項)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
        
        loss = recon_loss + KLD * 0.1
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # if batch_idx % 10 == 0:
            # 可視化（最初のデータのみ）
    plot_latent_space(mu, z_100, batch_idx, labels[0].item())
    generate_digit_variations(epoch, target_digit=0, num_variants=10)
    plot_latent_space_all_digits(mu, z_100, labels, epoch)
    print(f'Epoch {epoch}, Avg Loss: {train_loss / len(train_loader):.4f}')

print("Training Complete!")

# # ==========================================
# # 5. 学習ループ (上位10個選択アルゴリズム)
# # ==========================================
# print(f"Training VAE with Elite-Selection (Top 10/100) on {device}...")

# for epoch in range(1, epochs + 1):
#     model.train()
#     train_loss = 0
    
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         data = data.to(device).view(-1, 784)
#         optimizer.zero_grad()
        
#         # 1. 潜在変数の分布(μ, σ)を推定
#         mu, logvar = model.encode(data)
        
#         # 2. 各データに対して100個のベクトルzをサンプリング
#         # z_100の形状: (100, batch_size, latent_dim)
#         z_100 = model.reparameterize(mu, logvar, num_samples=100)

#         # 3. 100個すべてデコードして出力を得る
#         # recon_100の形状: (100, batch_size, 784)
#         recon_100 = model.decode(z_100.view(-1, latent_dim)).view(100, -1, 784)

#         # 4. 各サンプルの誤差(MSE)を計算し、上位10個を特定
#         # 誤差計算: (100, batch_size)
#         target = data.unsqueeze(0) # (1, batch_size, 784)
#         mses = torch.sum((recon_100 - target)**2, dim=2) 
        
#         # 各データ（バッチ内）ごとに、100個の中から誤差の小さい順にソート
#         _, top_indices = torch.sort(mses, dim=0) # (100, batch_size)
#         top10_indices = top_indices[:10, :] # 上位10個のインデックス
#         bottom90_indices = top_indices[10:, :] # 下位90個のインデックス
        
#         # 5. 上位10個の誤差のみを平均して損失とする
#         # batch内の各データについて上位10個のMSEを抽出
#         elite_mses = torch.gather(mses, 0, top10_indices) # (10, batch_size)
#         recon_loss = elite_mses.mean()
        
#         # KLD (正則化項)
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
        
#         loss = recon_loss + KLD * 0.1 # KLDの重みは適宜調整
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
        
#         # if batch_idx % 10 == 0:
#             # 図1: 潜在空間の表示
#     plot_latent_space(mu, z_100, batch_idx, labels[0].item())
#         # 図2: 出力空間のPCA表示
#         # plot_output_pca(recon_100.view(-1, 784), data.view(-1,784), top10_indices.cpu(), bottom90_indices.cpu(), batch_idx)
#     # plot_output_pca(
#     #     recon_100[:, 0, :],              # 最初のデータの100個のサンプル
#     #     data[0:1, :],                    # 最初のデータの教師画像
#     #     top10_indices[:10, 0].cpu(),     # 最初のデータの上位10個のインデックス
#     #     bottom90_indices[:, 0].cpu(),    # 最初のデータの下位90個のインデックス
#     #     f"Epoch {epoch} - Batch {batch_idx} (Digit: {labels[0].item()})"
#     # )

#     generate_digit_variations(epoch, target_digit=8,num_variants=10)
#     plot_latent_space_all_digits(mu, z_100, labels, epoch)
#     print(f'Epoch {epoch}, Avg Loss: {train_loss / len(train_loader):.4f}')
