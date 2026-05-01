#カテゴライズが難しい関数はここに
import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import japanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from collections import defaultdict # データをグループ化するために使用


# グラム行列計算
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (h * w)
    return gram


def tensorFromSentence( lang, sentence,EOS_token,device): #sentenceをインデックス番号に変換したtensor配列にする 
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ] #sentenceの音素をインデックス番号に変換してリストにする
    indexes.append( EOS_token )
    

    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)


def select_top_k_outputs(target, mu_p,log_var_p,top_k=10):
    # 各 mu_p[i] と target 間の MSE を各バッチごとに計算
    # nn.MSELoss(reduction='none') は各要素ごとの損失を返すので、mean(dim=(1,2))でバッチ内の全要素の平均を取る
    losses = torch.stack([
        nn.MSELoss(reduction='none')(out, target).mean(dim=(1, 2))
        for out in mu_p
    ], dim=0)  # shape: (num_samples, batch_size)

    # 各バッチごとに MSE が最小の上位 top_k のインデックスを取得
    best_indices = torch.argsort(losses, dim=0)[:top_k]  # shape: (top_k, batch_size)

    # 各バッチで、上位 top_k の mu_p の出力を選択
    batch_indices = torch.arange(mu_p.shape[1]).unsqueeze(0).expand(top_k, -1)  # shape: (top_k, batch_size)
    best_outputs = mu_p[best_indices, batch_indices].to(dtype=torch.bfloat16)  # shape: (top_k, batch_size, 77, 1024)
    best_log_var_p = log_var_p[best_indices, batch_indices].to(dtype=torch.bfloat16).requires_grad_(True)     # shape: (top_k, batch_size, 77, 1024)
    # target を top_k 個に複製（先頭に次元を追加して expand）
    expanded_target = target.unsqueeze(0).expand(top_k, -1, -1, -1)  # shape: (top_k, batch_size, 77, 1024)
    
    return best_outputs, expanded_target,best_log_var_p

# def select_top_k_outputs(target, mu_p, log_var_p, top_k=10):
#     # 1. 各サンプルとtarget間のMSEを計算 (shape: num_samples, batch_size)
#     losses = torch.stack([
#         nn.MSELoss(reduction='none')(out, target).mean(dim=(1, 2))
#         for out in mu_p
#     ], dim=0)

#     # 2. 誤差が小さい順にインデックスを取得 (shape: top_k, batch_size)
#     best_indices = torch.argsort(losses, dim=0)[:top_k]

#     # 3. 上位 top_k の出力を抽出
#     batch_indices = torch.arange(mu_p.shape[1]).unsqueeze(0).expand(top_k, -1)
    
#     # 抽出時点の shape: (top_k, batch_size, 77, 1024)
#     top_k_mu = mu_p[best_indices, batch_indices]
#     top_k_log_var = log_var_p[best_indices, batch_indices]

#     # --- ここで平均化処理を行う ---
#     # dim=0（top_kの次元）方向に平均を取る
#     # 結果の shape: (batch_size, 77, 1024)
#     best_output_mean = top_k_mu.mean(dim=0).to(dtype=torch.bfloat16).requires_grad_(True)
#     best_log_var_mean = top_k_log_var.mean(dim=0).to(dtype=torch.bfloat16).requires_grad_(True)

#     return best_output_mean, target, best_log_var_mean


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # VAEの出力: mu_p.shape = (num_samples, batch_size, 77, 1024)
# num_samples は VAEに設定した値 (例: 10, 20 など)

def select_random_output(mu_p, log_var_p, target):
    num_samples = mu_p.size(0)
    batch_size = mu_p.size(1)
    
    # ----------------------------------------------------
    # 1. 各バッチごとに、ランダムなインデックスを一つ選択 (0から num_samples-1)
    # ----------------------------------------------------
    # shape: (batch_size,)
    random_indices = torch.randint(0, num_samples, (batch_size,), device=mu_p.device)
    
    # ----------------------------------------------------
    # 2. 選択したインデックスを使って、mu_pとlog_var_pから対応するサンプルを取得
    # ----------------------------------------------------
    # torch.arange(batch_size) は各バッチのインデックス
    batch_indices = torch.arange(batch_size, device=mu_p.device)
    
    # サンプル取得 (mu_p[random_indices, batch_indices])
    # [random_indices]はnum_samples軸でインデックスを指定し、[batch_indices]はbatch_size軸でインデックスを指定します。
    # shape: (batch_size, 77, 1024)
    best_outputs = mu_p[random_indices, batch_indices].to(dtype=torch.bfloat16)
    best_log_var_p = log_var_p[random_indices, batch_indices].to(dtype=torch.bfloat16)

    # ----------------------------------------------------
    # 3. target（my_hidden）は複製せずにそのまま返すか、unsqueezeする
    # ----------------------------------------------------
    # shape: (batch_size, 77, 1024)
    expanded_target = target
    
    # ⚠️ GradScaler対応: 以前の提案通り、GradScalerを使う場合は型を整える
    # この部分でbfloat16への変換とrequires_grad_(True)は、
    # 学習ループ外では不要、autocast内で処理するなら安全
    
    return best_outputs, expanded_target, best_log_var_p

# 実際の使用例
# best_outputs, my_hidden2, log_var_p2 = select_random_output(mu_p, log_var_p, my_hidden)



def draw_pca_plot3(epoch, nums, all_labels,
                  all_img_features, target=None, pca_model=None, limits=None, dir="ono_img_ver11", mode="train"):
    """
    all_labels           : [1400] ラベル文字列
    all_img_features     : [1400, 128] my_hidden（予測）
    target  : [1400, 128] IMG_HIDDEN（教師）
    all_ono_features     : [1400, 128] phoneme_hidden
    """
    
# --- PCAの計算は全データで行う（これでサンプルの0件エラーを回避） ---
    X_img = np.array(all_img_features)
    X_img = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8)

    if pca_model is None:
        pca_model = PCA(n_components=2)
        X_img_pca = pca_model.fit_transform(X_img)
    else:
        X_img_pca = pca_model.transform(X_img)

    # --- ここから追加：表示範囲の限定 ---
    # ラベルでソートした時の700〜799番目のインデックスを集合(set)として保持
    if mode == "amiami":
        target_indices = set(np.argsort(all_labels)[0:100])
    elif mode == "shimashima":
        target_indices = set(np.argsort(all_labels)[700:800])
    else:
        # "trainimage" やそれ以外のモードの時は、全インデックス（1400個）を対象にする
        target_indices = set(range(len(all_labels)))

    # 凡例に余計なラベルが出ないよう、対象範囲に存在するラベル名だけを抽出
    unique_labels = sorted(set([all_labels[idx] for idx in target_indices]))
    # ----------------------------------

    plt.figure(figsize=(12, 10))

    if limits:
        plt.xlim(limits[0])
        plt.ylim(limits[1])
        plt.gca().set_aspect('equal', adjustable='box') # 1:1の比率にする

    unique_labels = sorted(set(all_labels))
    # 色の準備 (タブローカラー20色などを使う)
    colors = ["red","yellow", "gray","silver","rosybrown","firebrick",
            "darksalmon","sienna","sandybrown","tan",
                "gold","olivedrab","chartreuse","palegreen",
                "darkgreen","lightseagreen","paleturquoise",
                "deepskyblue","blue","pink","orange","crimson",
                "mediumvioletred","plum","darkorchid","mediumpurple",
                "chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]
    
    for i, label in enumerate(unique_labels):
        # ★ここを修正：現在のラベル名と一致し、かつ target_indices に含まれるものだけ抽出
        indices = [j for j, l in enumerate(all_labels) if (l == label and j in target_indices)]
        
        if not indices: continue # 万が一空の場合はスキップ
        
        img_points = X_img_pca[indices]

        # --- 教師データ (all_IMG_HIDDEN) がある場合のみ描画 ---
        if target is not None:
            X_gt = np.array(target)
            X_gt = X_gt / (np.linalg.norm(X_gt, axis=1, keepdims=True) + 1e-8)
            X_gt_pca = pca_model.transform(X_gt)
            gt_points = X_gt_pca[indices] # indicesが絞り込まれているので、ここも自動で絞られます

            # 教師データの散布図 (△)
            plt.scatter(gt_points[:, 0], gt_points[:, 1], marker='^', s=50,
                        color=colors[i], alpha=0.9)

            # 対応線
            for j in range(len(indices)):
                plt.plot([gt_points[j, 0], img_points[j, 0]],
                         [gt_points[j, 1], img_points[j, 1]],
                         color=colors[i], linewidth=0.4, alpha=0.8)

        plt.title(f"Latent space Mapping\n△:my_hidden | 〇:best_outputs(VAE)\n(Epoch{epoch+1}train)")

        # my_hidden（予測）〇
        plt.scatter(
            img_points[:, 0], img_points[:, 1],
            marker='o', s=20,
            color=colors[i], alpha=0.9
        )

    


    # グラフの装飾
    plt.xlabel(f"PC1 (Contribution: {pca_model.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Contribution: {pca_model.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)
    
    # 凡例 (画像のみ表示)
    # 重複を除くための処理
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=colors[i], markersize=10) for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    # save_path = os.path.join(f"figure/{nums}/train/ono_img_ver10", f"epoch{epoch+1}_trainimagehiddenPCA.png")
    save_path = os.path.join(f"figure/{nums}/train/{dir}", f"epoch{epoch+1}_{mode}hiddenPCA.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # フォルダがない場合のエラー防止
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved: {save_path}")


def draw_pca_plot_vae_samples(epoch, nums, idx, amiami_features, target_feature, 
                             pca_model=None, limits=None, dir="ono_img_ver11", mode="amiami",ono="あみあみ"):
    """
    amiami_features : [100, 128] VAEからサンプリングされた100個の特徴量
    target_feature  : [1, 128]  元となった画像'woven_001.png'の教師特徴量
    """
    
    # --- 1. VAEサンプリング100個のPCA変換 ---
    X_vae = np.array(amiami_features)
    # ノルム正規化（モデルの学習に合わせる）
    X_vae = X_vae / (np.linalg.norm(X_vae, axis=1, keepdims=True) + 1e-8)

    # pca_model が None の場合、その場で新しく作成して fit する
    if pca_model is None:
        pca_model = PCA(n_components=2)
        X_vae_pca = pca_model.fit_transform(X_vae)  # 学習と変換を同時に行う
    else:
        X_vae_pca = pca_model.transform(X_vae)

    # --- 2. 教師データ（1個）のPCA変換 ---
    X_gt = np.array(target_feature)
    if X_gt.ndim == 1: X_gt = X_gt.reshape(1, -1) # 1次元なら2次元にする
    X_gt = X_gt / (np.linalg.norm(X_gt, axis=1, keepdims=True) + 1e-8)
    X_gt_pca = pca_model.transform(X_gt)

    plt.figure(figsize=(10, 8))

    if limits is not None:
        plt.xlim(limits[0])
        plt.ylim(limits[1])
        plt.gca().set_aspect('equal', adjustable='box')

    # --- 3. 描画 ---

    if mode == "KL1e2_std0.2_z":

        # zサンプリング点 (〇) - 500個
        plt.scatter(
            X_vae_pca[:, 0], X_vae_pca[:, 1],
            marker='o', s=20, color='red', alpha=0.9, label='z Samples (n=500)'
        )

        # 教師データ (△) - 1個
        plt.scatter(
            X_gt_pca[:, 0], X_gt_pca[:, 1],
            marker='^', s=50, color='blue',  
            label='mu'
        )

        # # 対応線 (各サンプルから教師へ線を引く)
        # for i in range(len(X_vae_pca)):
        #     plt.plot(
        #         [X_gt_pca[0, 0], X_vae_pca[i, 0]],
        #         [X_gt_pca[0, 1], X_vae_pca[i, 1]],
        #         color='gray', linewidth=0.2, alpha=0.4
        #     )

        # --- 4. グラフ装飾 ---
        plt.title(f"z Sampling Diversity\nMode: {mode} | Epoch: {epoch+1} | Batch: {idx}\nOnomatope: {ono}")

    else:
                # VAEサンプリング点 (〇) - 100個
        plt.scatter(
            X_vae_pca[:, 0], X_vae_pca[:, 1],
            marker='o', s=20, color='red', alpha=0.9, label='VAE Samples (n=100)'
        )

        # 教師データ (△) - 1個
        plt.scatter(
            X_gt_pca[:, 0], X_gt_pca[:, 1],
            marker='^', s=50, color='blue',  
            label='Target'
        )

        # # 対応線 (各サンプルから教師へ線を引く)
        # for i in range(len(X_vae_pca)):
        #     plt.plot(
        #         [X_gt_pca[0, 0], X_vae_pca[i, 0]],
        #         [X_gt_pca[0, 1], X_vae_pca[i, 1]],
        #         color='gray', linewidth=0.2, alpha=0.4
        #     )

        # --- 4. グラフ装飾 ---
        plt.title(f"Latent space: VAE Sampling Diversity\nMode: {mode} | Epoch: {epoch+1} | Batch: {idx}\nOnomatope: {ono}")

    plt.xlabel(f"PC1 (Contribution: {pca_model.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Contribution: {pca_model.explained_variance_ratio_[1]:.2f})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')


    save_path = os.path.join(f"figure/{nums}/train/{dir}", f"{mode}_VAEsamplingPCA.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    # print(f"PCA plot saved: {save_path}")

def select_subset_from_combined(combined_numpy, n_labels=14, n_samples_per_label=100, n_subset=10): #2400サンプルの78848次元の共分散行列の計算は重たいので、数サンプルピックする関数
    """
    combined_numpy (2400, 78848) の配列から、
    画像特徴(先頭1400) と 音素特徴(後ろ1400) を、
    ラベルごとに n_subset 個ずつ抜き出した新たな配列を作成する。

    ・画像特徴は先頭1400行 (14ラベル x 100個)
    ・音素特徴は後ろ1400行 (14ラベル x 100個)
    ・各ラベルごとに先頭 n_subset 行を抜き出す（本例では 10）

    Args:
        combined_numpy: shape (2400, 78848) の配列
        n_labels: ラベルの総数 (デフォルト14)
        n_samples_per_label: 各ラベルのデータ数 (デフォルト100)
        n_subset: 1ラベルあたり何個抜き出すか (デフォルト10)

    Returns:
        sub_combined: shape ((n_labels*n_subset*2), 78848) の配列
                      (画像特徴 + 音素特徴でラベルごと各 n_subset 個)
    """
    # 画像特徴: 先頭 1400 行 (14ラベル x 100個 = 1400)
    # 音素特徴: 後ろ 1400 行 (14ラベル x 100個 = 1400)
    # 合計 2400 行（最初の 1400 が画像、次の 1400 が音素）
    
    # 出力用リスト
    selected_image_features = []
    selected_phoneme_features = []

    # 画像特徴をラベル毎に n_subset 個抜き出し
    for label_idx in range(n_labels):
        start_idx = label_idx * n_samples_per_label
        end_idx = start_idx + n_subset
        selected_image_features.append(combined_numpy[start_idx:end_idx])

    # 音素特徴をラベル毎に n_subset 個抜き出し
    # 画像特徴1400行をオフセットとして足す
    offset = n_labels * n_samples_per_label  # 14*100 = 1400
    for label_idx in range(n_labels):
        start_idx = offset + label_idx * n_samples_per_label
        end_idx = start_idx + n_subset
        selected_phoneme_features.append(combined_numpy[start_idx:end_idx])

    # listを結合して縦方向に積む
    sub_combined = np.vstack(selected_image_features + selected_phoneme_features)
    return sub_combined

def draw_vae_ellipse_plot(epoch, nums, all_labels, all_mus):
    """
    オノマトペ一つにつき一つの楕円（計14個）を描画する関数。
    この楕円は、そのオノマトペに対応する100個の mu ベクトルがPCA空間上に
    射影された点群を包含する範囲（±3σ）を表します。
    """

    # データをNumpy配列に変換
    X_mu = np.array(all_mus)          # (1400, 78848)
    
    # 1. mu を基準に PCA を学習・変換
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_mu) # (1400, 2)
    
    # プロットの準備
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # ラベルごとにデータをグループ化
    data_by_label = defaultdict(list)
    for label, pca_point in zip(all_labels, X_pca):
        data_by_label[label].append(pca_point)

    unique_labels = sorted(list(data_by_label.keys()))
    colors = ["red","yellow", "gray","silver","rosybrown","firebrick",
            "darksalmon","sienna","sandybrown","tan",
                "gold","olivedrab","chartreuse","palegreen",
                "darkgreen","lightseagreen","paleturquoise",
                "deepskyblue","blue","pink","orange","crimson",
                "mediumvioletred","plum","darkorchid","mediumpurple",
                "chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]

    for i, label in enumerate(unique_labels):
        color = colors[i]
        
        # 現在のラベルに対応する全てのPCA点群 (100, 2)
        points = np.array(data_by_label[label]) 
        
        # 1. 中心の計算 (100個の mu_p ベクトルの平均)
        center_x, center_y = np.mean(points, axis=0)
        
        # 2. 共分散行列の計算 (2x2)
        # np.cov(points, rowvar=False) は、列を特徴量(X, Y)として共分散を計算
        cov_matrix = np.cov(points, rowvar=False)
        
        # 3. 固有値分解で楕円のパラメータを計算
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 固有値が負になる計算誤差対策
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # 4. 幅、高さ、回転角の計算
        
        # 描画したいのは 3 sigma の範囲
        # Ellipseの引数は「直径(幅・高さ)」なので、半径(3σ) * 2
        # np.sqrt(eigenvalues) が 標準偏差 sigma
        width = 2 * 3 * np.sqrt(eigenvalues[0])
        height = 2 * 3 * np.sqrt(eigenvalues[1])
        
        # 回転角 (ラジアン -> 度)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # 5. 楕円の描画 (枠線はクラス色で、薄く塗る)
        ellipse = Ellipse(xy=(center_x, center_y), width=width, height=height, angle=angle,
                          edgecolor=color, facecolor=color, alpha=0.1, linewidth=3)
        ax.add_patch(ellipse)

        # 中心のバツ印描画
        ax.scatter(center_x, center_y, marker='x', color=color, s=150, linewidths=3, 
                   label=label, zorder=10)

        # テキストラベルをオノマトペの位置に表示
        plt.text(center_x, center_y, label, 
                 fontsize=9, fontweight='bold', color='black', alpha=0.8)

    # グラフ装飾
    plt.title(f"VAE Latent Space Grouped Ellipses (Epoch {epoch+1})\nEllipses cover $\pm3\sigma$ range of 100 samples", fontsize=16)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)
    
    # 凡例の整理
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path = os.path.join(f"figure/{nums}/train/VAE_ver3", f"epoch{epoch+1}_VAEtrainimagehiddenPCA.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grouped VAE Ellipse plot saved: {save_path}")
