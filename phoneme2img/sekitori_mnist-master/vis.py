import matplotlib
matplotlib.use("Agg") #GUIを使わない、windowを生成せず保存だけ行う
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

def plot_loss_histogram(loss_mean_target, worst_idx, prefix="", modelname="", sample_id="", epoch=1):
    """
    loss_mean_target: torch.Tensor, shape [T]
    worst_idx: torch.Tensor, shape: 0 ~ T の値
    save_path: 画像保存パス (Noneなら表示)
    """
    losses = loss_mean_target[0] # [T]
    worst_t = worst_idx[0]
    
    T = len(losses)

    plt.figure(figsize=(6,4))
    for i in range(T):
        color = "red" if i == worst_t else "lightgray"
        plt.bar(
            i, losses[i],
            width=1,
            facecolor=color,      # 塗りつぶし色
            edgecolor="black"     # 枠線の色
        )

    plt.xticks(np.arange(T))
    plt.xlabel("Teacher index")
    plt.ylabel("Mean MSE loss")
    plt.ylim(0, 3)
    plt.title(f"Worst teacher = {worst_t}")
    os.makedirs(f"histgram/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path = f"histgram/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_sorted_loss_histogram(losses_tensor, top_k, prefix="", modelname="", sample_id="", epoch=1):
    """
    losses_tensor : torch.Tensor [1, T, K]
    top_k         : 上位何個を赤にするか（例：36 = 30%）
    """
    # 4. ヒストグラムではなく「並び順そのままの棒グラフ」
    plt.figure(figsize=(10,4))

    N = len(losses_tensor)
    x = np.arange(N)

    # 5. 色指定（上位 top_k を赤、それ以降は灰）
    colors = ["red" if i < top_k else "lightgray" for i in range(N)]

    plt.bar(
        x, losses_tensor,
        width=1,
        color=colors,
        edgecolor="black"
    )

    plt.xlabel("Sorted sample index")
    plt.ylabel("Loss value")
    plt.title(f"Top-{top_k} losses highlighted (red)")
    plt.ylim(0, 3)

    # 保存
    os.makedirs(f"worst_histgram_sorted/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path = f"worst_histgram_sorted/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return losses_tensor

def visualize_pca_dual(Z, indices1, indices2, prefix="", modelname="", sample_id="", assigned_indices=[], epoch=1):
    #どの勾配(全部の勾配を使わない)を使ったかのPCA と そもそもどの教師に近いのか？のPCA
    if not isinstance(Z, np.ndarray):
        Z = Z.cpu().numpy()
    N, D = Z.shape

    # PCAで2次元に圧縮
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    n_classes = 12
    cmap = plt.get_cmap("tab20", n_classes)
    label_texts = [f"{30*i}°" for i in range(n_classes)]

    # ========== 1枚目 ==========
    #assigned_indices   [T, K]

    plt.figure(figsize=(8,6))

    # predを黒で描画
    plt.scatter(Z_pca[:,0], Z_pca[:,1],  c="black", marker="o", label="Pred")

    # worst teacher に割り当てられた pred を赤で強調
    
    worst_pred_idx = assigned_indices[indices1]
    #print(worst_pred_idx)
    plt.scatter(Z_pca[worst_pred_idx,0], Z_pca[worst_pred_idx,1],
                c="red", marker="o", label=f"Worst teacher")

    plt.legend()

    # 凡例を手動で作成（データに無くても全クラス表示される）
    patches = [mpatches.Patch(color=cmap(i), label=label_texts[i]) for i in range(n_classes)]
    plt.legend(handles=patches, title="Teacher index",
               loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of Z - use loss (epoch {epoch:03d})")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.tight_layout()

    os.makedirs(f"PCA_use/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path1 = f"PCA_use/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    #print(f"PCA plot saved as {save_path1}")

    # ========== 2枚目 ==========
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=indices2, cmap=cmap, vmin=0, vmax= n_classes-1, s=20)

    patches = [mpatches.Patch(color=cmap(i), label=label_texts[i]) for i in range(n_classes)]
    plt.legend(handles=patches, title="Teacher index",
               loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of Z - min loss (epoch {epoch:03d})")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.tight_layout()

    os.makedirs(f"PCA_min/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path2 = f"PCA_min/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.close()
    #print(f"PCA plot saved as {save_path2}")


def visualize_pca_dual_mean(Z, indices1, indices2, prefix="", modelname="", sample_id="", assigned_indices=[], epoch=1):

    if not isinstance(Z, np.ndarray):
        Z = Z.cpu().numpy()
    N, D = Z.shape

    # PCAで2次元に圧縮
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Z)

    n_classes = 12
    cmap = plt.get_cmap("tab20", n_classes)
    label_texts = [f"{30*i}°" for i in range(n_classes)]

    plt.figure(figsize=(8,6))
    assigned_indices = assigned_indices[0]
    # 教師ごとに色分けして描画
    for t in range(n_classes):
        idxs = assigned_indices[t]   # shape (K,)
        if len(idxs) > 0:
            plt.scatter(Z_pca[idxs,0], Z_pca[idxs,1],
                        c=[cmap(t)], marker="o", label=label_texts[t], alpha=0.8, s=20)

    # 凡例を手動で作成（必ず全クラス表示）
    patches = [mpatches.Patch(color=cmap(i), label=label_texts[i]) for i in range(n_classes)]
    plt.legend(handles=patches, title="Teacher index",
               loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of Z - assigned teacher groups (epoch {epoch:03d})")
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.tight_layout()

    os.makedirs(f"PCA_use/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path1 = f"PCA_use/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"
    plt.savefig(save_path1, dpi=300, bbox_inches="tight")
    plt.close()
    #print(f"PCA plot saved as {save_path1}")

    # ========== 2枚目 ==========
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=indices2, cmap=cmap, vmin=0,  vmax= n_classes-1, s=20)

    patches = [mpatches.Patch(color=cmap(i), label=label_texts[i]) for i in range(n_classes)]
    plt.legend(handles=patches, title="Teacher index",
               loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of Z - min loss (epoch {epoch:03d})")
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.tight_layout()

    os.makedirs(f"PCA_min/{prefix}/{modelname}/{sample_id}", exist_ok=True)
    save_path2 = f"PCA_min/{prefix}/{modelname}/{sample_id}/epoch_{epoch:03d}.png"
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    plt.close()
    #print(f"PCA plot saved as {save_path2}")


def all_sample_pca(all_sample_z, all_indices, outf="", filename="zl", epoch=0):
    all_sample_z = np.concatenate(all_sample_z, axis=0) # (B*2) * 16
    all_indices = np.array(all_indices) # (B*2) * 1
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(all_sample_z)
    plt.figure(figsize=(8,6))
    label = ["basket", "bottle", "jug", "kattle", "mug", "pan", "pc", "pot", "vase"]
    cmap = plt.get_cmap("tab10", len(label))

    scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=all_indices, cmap=cmap, vmin=0,  vmax= len(label)-1, s=20)

    patches = [mpatches.Patch(color=cmap(i), label=label[i]) for i in range(len(label))]
    plt.legend(handles=patches, title="Teacher index",
               loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"PCA of all Z - all samples (epoch {epoch:03d})")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.tight_layout()
    os.makedirs(f"PCA_all_samples/{outf}/{filename}", exist_ok=True)
    save_path = f"PCA_all_samples/{outf}/{filename}/epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_grad_bottle(grad_l_all=[], grad_r_all=[], idx_l=[], idx_r=[], outf ="", epoch=1):
    """gradを可視化"""

    T, K = idx_l.shape
    teacher_grad_l = [[] for _ in range(T)]
    teacher_grad_r = [[] for _ in range(T)]
    for t in range(T):
        for k in range(K):
            pred_idx_l = int(idx_l[t, k]) #idx 
            pred_idx_r = int(idx_r[t, k])
            grad_l = grad_l_all[pred_idx_l]  # grad (23,3)
            grad_r = grad_r_all[pred_idx_r]
            teacher_grad_l[t].append(grad_l.detach().cpu())
            teacher_grad_r[t].append(grad_r.detach().cpu())
    # Tensor
    for t in range(T):
        teacher_grad_l[t] = torch.stack(teacher_grad_l[t], dim=0)  # (K,23,3)
        teacher_grad_r[t] = torch.stack(teacher_grad_r[t], dim=0)
    # 教師ごとの L2 ノルム
    grad_l_stats = torch.norm(torch.stack(teacher_grad_l, dim=0), dim=(2,3)).mean(dim=1).numpy()
    grad_r_stats = torch.norm(torch.stack(teacher_grad_r, dim=0), dim=(2,3)).mean(dim=1).numpy()

    # 左手
    plt.figure(figsize=(8,5))
    plt.bar(range(T), grad_l_stats, alpha=0.8)
    plt.xlabel("Teacher index")
    plt.ylabel("L2 Norm of Gradients (Left hand)")
    plt.title("Per-teacher Gradient L2 Norm (Left Hand)")
    plt.tight_layout()
    plt.savefig(f"grad_histgram/{outf}/bottle/hand_l/epoch_{epoch}.png")
    plt.close()

    # 右手
    plt.figure(figsize=(8,5))
    plt.bar(range(T), grad_r_stats, alpha=0.8)
    plt.xlabel("Teacher index")
    plt.ylabel("L2 Norm of Gradients (Right hand)")
    plt.title("Per-teacher Gradient L2 Norm (Right Hand)")
    plt.tight_layout()
    plt.savefig(f"grad_histgram/{outf}/bottle/hand_r/epoch_{epoch}.png")
    plt.close()

def visualize_grad_else(grad_l_all=[], grad_r_all=[], class_name="", outf ="", epoch=1):

    """gradを可視化"""
    grad_l_mean = torch.norm(grad_l_all, dim=(1,2)).mean().detach().cpu().numpy()
    grad_r_mean = torch.norm(grad_r_all, dim=(1,2)).mean().detach().cpu().numpy()

    # 左手
    plt.figure(figsize=(6,6))
    plt.bar([0], [grad_l_mean], alpha=0.8)
    plt.xticks([0], [class_name])
    plt.ylabel("L2 Norm of Gradients (Left hand)")
    plt.title("Left Hand Gradient L2 Norm")
    plt.tight_layout()
    plt.savefig(f"grad_histgram/{outf}/{class_name}/hand_l/epoch_{epoch}.png")
    plt.close()

    # 右手
    plt.figure(figsize=(6,6))
    plt.bar([0], [grad_r_mean], alpha=0.8)
    plt.xticks([0], [class_name])
    plt.ylabel("L2 Norm of Gradients (Right hand)")
    plt.title("Right Hand Gradient L2 Norm")
    plt.tight_layout()
    plt.savefig(f"grad_histgram/{outf}/{class_name}/hand_r/epoch_{epoch}.png")
    plt.close()