#カテゴライズが難しい関数はここに
import torch
import torch.nn as nn
import random
import torch
import torch.nn as nn
import os
import numpy as np
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


import torch
import numpy as np
import random

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


def draw_pca_plot(epoch, nums, all_labels, all_img_features, all_ono_features):
    """
    all_labels: リスト [1400個のラベル文字列]
    all_img_features: リスト [1400個のnumpy配列(128次元)]
    all_ono_features: リスト [1400個のnumpy配列(128次元)]
    """

    # データをNumpy配列に変換
    X_img = np.array(all_img_features) # (1400, 128)
    X_ono = np.array(all_ono_features) # (1400, 128)
    
    # 画像とオノマトペを結合してPCAを学習させる（同じ空間に射影するため）
    X_concat = np.concatenate([X_img, X_ono], axis=0)
    
    # PCAで2次元に圧縮
    pca = PCA(n_components=2)
    pca.fit(X_concat)
    
    # それぞれ変換
    X_img_pca = pca.transform(X_img)
    X_ono_pca = pca.transform(X_ono)
    
    # プロットの準備
    plt.figure(figsize=(12, 10))
    
    # ユニークなラベルを取得 (14種類)
    unique_labels = sorted(list(set(all_labels)))
    
    # 色の準備 (タブローカラー20色などを使う)
    colors = ["red","yellow", "gray","silver","rosybrown","firebrick",
            "darksalmon","sienna","sandybrown","tan",
                "gold","olivedrab","chartreuse","palegreen",
                "darkgreen","lightseagreen","paleturquoise",
                "deepskyblue","blue","pink","orange","crimson",
                "mediumvioletred","plum","darkorchid","mediumpurple",
                "chocolate","peru","yellow","y","aqua","lightsteelblue","linen","teal"]

    for i, label in enumerate(unique_labels):
        # 現在のラベルに対応するインデックスを取得
        indices = [idx for idx, x in enumerate(all_labels) if x == label]
        
        # 1. 画像のプロット (丸印 'o')
        # そのラベルに対応する画像群を取り出す
        img_points = X_img_pca[indices]
        plt.scatter(img_points[:, 0], img_points[:, 1], 
                    color=colors[i], marker='o', alpha=0.6, s=30, 
                    label=label if epoch == 0 else "") # 凡例が多すぎないように調整

        # 2. オノマトペのプロット (バツ印 'x')
        # そのラベルに対応するオノマトペ群を取り出す
        ono_points = X_ono_pca[indices]
        
        # 学習中はオノマトペベクトルも微妙に動くが、代表点(平均)を一つ描画する形が見やすい
        # 全部の点を描画したい場合は下のmeanをとらずにscatterしてください
        ono_center = np.mean(ono_points, axis=0)
        
        plt.scatter(ono_center[0], ono_center[1], 
                    color=colors[i], marker='x', s=200, linewidths=3, edgecolors='black')

        # テキストラベルをオノマトペの位置に表示
        plt.text(ono_center[0], ono_center[1], label, 
                 fontsize=9, fontweight='bold', color='black', alpha=0.8)

    # グラフの装飾
    plt.title(f"ENCODER_hidden and hidden\nJoint Latent Space (Epoch {epoch+1})", fontsize=16)
    plt.xlabel(f"PC1 (Contribution: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Contribution: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)
    
    # 凡例 (画像のみ表示)
    # 重複を除くための処理
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=colors[i], markersize=10) for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(f"figure/{nums}/train/ono_img_ver4", f"epoch{epoch+1}_a_trainimagehiddenPCA.png")
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved: {save_path}")


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def draw_pca_plot2(epoch, nums, all_labels,
                  all_img_features, all_IMG_HIDDEN, all_ono_features):
    """
    all_labels           : [1400] ラベル文字列
    all_img_features     : [1400, 128] my_hidden（予測）
    all_IMG_HIDDEN  : [1400, 128] IMG_HIDDEN（教師）
    all_ono_features     : [1400, 128] phoneme_hidden
    """

    # ===============================
    # numpy化
    # ===============================
    X_img = np.array(all_img_features)        # (1400, 128)
    X_gt  = np.array(all_IMG_HIDDEN)     # (1400, 128)
    X_ono = np.array(all_ono_features)        # (1400, 128)

    # ===============================
    # PCAは必ず「全部まとめて1回」
    # ===============================
    X_concat = np.concatenate([X_img, X_gt, X_ono], axis=0)

    X_img = l2_normalize(X_img)
    X_gt  = l2_normalize(X_gt)
    X_ono = l2_normalize(X_ono)


    pca = PCA(n_components=2)
    pca.fit(X_concat)

    X_img_pca = pca.transform(X_img)
    X_gt_pca  = pca.transform(X_gt)
    X_ono_pca = pca.transform(X_ono)

    # ===============================
    # 描画準備
    # ===============================
    plt.figure(figsize=(12, 10))

    unique_labels = sorted(list(set(all_labels)))

    colors = [
        "red","yellow","gray","silver","rosybrown","firebrick",
        "darksalmon","sienna","sandybrown","tan",
        "gold","olivedrab","chartreuse","palegreen"
    ]

    # ===============================
    # ラベルごとに描画
    # ===============================
    for i, label in enumerate(unique_labels):
        indices = [idx for idx, x in enumerate(all_labels) if x == label]

        # -------- my_hidden（予測）
        img_points = X_img_pca[indices]
        plt.scatter(
            img_points[:, 0], img_points[:, 1],
            color=colors[i], marker='o',
            alpha=0.4, s=25,
            label=label if epoch == 0 else ""
        )

        # -------- IMG_HIDDEN（教師）
        gt_points = X_gt_pca[indices]
        plt.scatter(
            gt_points[:, 0], gt_points[:, 1],
            color=colors[i], marker='^',
            alpha=0.4, s=25
        )

        # -------- 対応する点を線で結ぶ
        for j in range(len(indices)):
            plt.plot(
                [img_points[j, 0], gt_points[j, 0]],
                [img_points[j, 1], gt_points[j, 1]],
                color=colors[i],
                linewidth=0.3,
                alpha=0.3
            )

        # -------- phoneme（14点のみ：平均）
        ono_points = X_ono_pca[indices]
        ono_center = np.mean(ono_points, axis=0)

        plt.scatter(
            ono_center[0], ono_center[1],
            color=colors[i], marker='x',
            s=220, linewidths=3, edgecolors='black', zorder=10
        )

        plt.text(
            ono_center[0], ono_center[1],
            label,
            fontsize=10, fontweight='bold',
            color='black', alpha=0.9
        )

    # ===============================
    # 装飾
    # ===============================
    plt.title(
        f"Joint PCA Space (Epoch {epoch+1})\n"
        "circle: my_hidden / triangle: IMG_HIDDEN / x: phoneme",
        fontsize=15
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', label='my_hidden',
               markerfacecolor='gray', markersize=8),
        Line2D([0],[0], marker='^', color='w', label='IMG_HIDDEN',
               markerfacecolor='gray', markersize=8),
        Line2D([0],[0], marker='x', color='black', label='phoneme',
               markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(
        f"figure/{nums}/train/ono_img_ver4",
        f"epoch{epoch+1}_a_trainimagehiddePCA.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"PCA plot saved: {save_path}")


def draw_pca_plot3(epoch, nums, all_labels,
                  all_img_features, all_IMG_HIDDEN, all_ono_features):
    """
    all_labels           : [1400] ラベル文字列
    all_img_features     : [1400, 128] my_hidden（予測）
    all_IMG_HIDDEN  : [1400, 128] IMG_HIDDEN（教師）
    all_ono_features     : [1400, 128] phoneme_hidden
    """
    # IMG_HIDDENのみでPCA空間を定義
    X_gt = np.array(all_IMG_HIDDEN)   # (1400, 128)

    # 念のため正規化（重要）
    X_gt = X_gt / (np.linalg.norm(X_gt, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=2)
    pca.fit(X_gt)

    X_gt_pca = pca.transform(X_gt)

    X_img = np.array(all_img_features)
    X_img = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8)

    X_img_pca = pca.transform(X_img)

    plt.figure(figsize=(12, 10))


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
        indices = [j for j, l in enumerate(all_labels) if l == label]

        gt_points  = X_gt_pca[indices]
        img_points = X_img_pca[indices]

        # IMG_HIDDEN（教師）▽
        plt.scatter(
            gt_points[:, 0], gt_points[:, 1],
            marker='^', s=50,
            color=colors[i], 
            label=label, alpha=0.9
        )

        # my_hidden（予測）〇
        plt.scatter(
            img_points[:, 0], img_points[:, 1],
            marker='o', s=20,
            color=colors[i], alpha=0.9
        )

        # 対応線
        for j in range(len(indices)):
            plt.plot(
                [gt_points[j, 0], img_points[j, 0]],
                [gt_points[j, 1], img_points[j, 1]],
                color=colors[i],
                linewidth=0.4,
                alpha=0.8
            )

    # グラフの装飾
    plt.title(f"IMG_HIDDEN and my_hidden\nJoint Latent Space (Epoch {epoch+1} train)\n△:IMG_HIDDEN | 〇:my_hidden", fontsize=16)
    plt.xlabel(f"PC1 (Contribution: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Contribution: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)
    
    # 凡例 (画像のみ表示)
    # 重複を除くための処理
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=colors[i], markersize=10) for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(f"figure/{nums}/train/ono_img_ver6", f"epoch{epoch+1}_trainimagehiddenPCA.png")
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved: {save_path}")

def draw_valid_pca_plot3(epoch, nums, all_labels,
                  all_img_features, all_IMG_HIDDEN, all_ono_features):

    # IMG_HIDDENのみでPCA空間を定義
    X_gt = np.array(all_IMG_HIDDEN)   # (1400, 128)

    # 念のため正規化（重要）
    X_gt = X_gt / (np.linalg.norm(X_gt, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=2)
    pca.fit(X_gt)

    X_gt_pca = pca.transform(X_gt)

    X_img = np.array(all_img_features)
    X_img = X_img / (np.linalg.norm(X_img, axis=1, keepdims=True) + 1e-8)

    X_img_pca = pca.transform(X_img)

    plt.figure(figsize=(12, 10))


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
        indices = [j for j, l in enumerate(all_labels) if l == label]

        gt_points  = X_gt_pca[indices]
        img_points = X_img_pca[indices]

        # IMG_HIDDEN（教師）▽
        plt.scatter(
            gt_points[:, 0], gt_points[:, 1],
            marker='^', s=50,
            color=colors[i], 
            label=label, alpha=0.9
        )

        # my_hidden（予測）〇
        plt.scatter(
            img_points[:, 0], img_points[:, 1],
            marker='o', s=20,
            color=colors[i], alpha=0.9
        )

        # 対応線
        for j in range(len(indices)):
            plt.plot(
                [gt_points[j, 0], img_points[j, 0]],
                [gt_points[j, 1], img_points[j, 1]],
                color=colors[i],
                linewidth=0.4,
                alpha=0.8
            )

    # グラフの装飾
    plt.title(f"IMG_HIDDEN and my_hidden\nJoint Latent Space (Epoch {epoch+1} Valid)\n△:IMG_HIDDEN | 〇:my_hidden", fontsize=16)
    plt.xlabel(f"PC1 (Contribution: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PC2 (Contribution: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)
    
    # 凡例 (画像のみ表示)
    # 重複を除くための処理
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=colors[i], markersize=10) for i, label in enumerate(unique_labels)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(f"figure/{nums}/train/ono_img_ver6", f"epoch{epoch+1}_validimagehiddenPCA.png")
    plt.savefig(save_path)
    plt.close()
    print(f"PCA plot saved: {save_path}")



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
