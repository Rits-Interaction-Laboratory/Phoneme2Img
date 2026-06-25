import torch
import numpy as np

def compute_related_digit_loss(recon_current, related_images):
    """
    recon_current: (100, 784)
    related_images: (3, 784) -> [digit-1, digit, digit+1]
    席取り方式: 各recon vectorを最もMSEが小さい related image に割り当てる
    割り当て数の上限: same=34, minus1=33, plus1=33
    """
    diff = recon_current.unsqueeze(1) - related_images.unsqueeze(0)  # (100, 3, 784)
    mse = torch.sum(diff * diff, dim=2)  # (100, 3)

    orders = [torch.argsort(mse[:, r]) for r in range(3)]
    quotas = [33, 34, 33]
    ptr = [0, 0, 0]
    selected_losses = []
    selected_recons = torch.zeros(recon_current.size(0), dtype=torch.bool, device=recon_current.device)
    inf = float('inf')

    def current_value(r):
        while ptr[r] < 100 and selected_recons[orders[r][ptr[r]]]:
            ptr[r] += 1
        if ptr[r] >= 100:
            return inf
        return mse[orders[r][ptr[r]], r].item()

    while len(selected_losses) < 100:
        best_rel = None
        best_val = inf
        for r in range(3):
            if quotas[r] <= 0:
                continue
            val = current_value(r)
            if val < best_val:
                best_val = val
                best_rel = r
        if best_rel is None:
            break

        recon_idx = int(orders[best_rel][ptr[best_rel]])
        if selected_recons[recon_idx]:
            ptr[best_rel] += 1
            continue

        selected_losses.append(mse[recon_idx, best_rel])
        selected_recons[recon_idx] = True
        quotas[best_rel] -= 1
        ptr[best_rel] += 1

    return torch.stack(selected_losses).mean()


def select_top100_recon_labels(recon_current, related_images):
    """
    席取りLossで選ばれた100個のベクトルに対応する関連数字ラベルを返す
    戻り値: shape (100,), 値は 0/1/2 (digit-1 / same / digit+1)
    """
    diff = recon_current.unsqueeze(1) - related_images.unsqueeze(0)  # (100, 3, 784)
    mse = torch.sum(diff * diff, dim=2)  # (100, 3)
    mse_flat = mse.view(-1)  # (300,)

    sorted_indices = torch.argsort(mse_flat)
    quotas = [33, 34, 33]
    selected_labels = []
    selected_recons = torch.zeros(recon_current.size(0), dtype=torch.bool, device=recon_current.device)

    for idx in sorted_indices:
        idx_int = int(idx)
        recon_idx = idx_int // 3
        rel_idx = idx_int % 3

        if selected_recons[recon_idx]:
            continue
        if quotas[rel_idx] <= 0:
            continue

        selected_labels.append(rel_idx)
        selected_recons[recon_idx] = True
        quotas[rel_idx] -= 1

        if len(selected_labels) == recon_current.size(0):
            break

    while len(selected_labels) < recon_current.size(0):
        selected_labels.append(0)

    return np.array(selected_labels[: recon_current.size(0)])


def sekitori_loss_sum(pred, target):
    """
    pred:   [B, N, D]  # サンプリングされた予測リスト
    target: [B, T, D]  # 教師データ (T個)

    各教師(T個)に対して、重複なくK個のpredを割り当てる。
    戻り値:
        losses:  torch.Tensor, [B, T, K] 割り当てられたloss (autograd対応)
        indices: np.ndarray,  [B, T, K] 割り当てられたpredのインデックス
        loss_mean_target: torch.Tensor, [B, T] 各ターゲットの平均loss
        indices_min_loss: list of np.ndarray, 各バッチのpredごとに最小lossの教師index
    """
    B, N, _ = pred.shape
    T = target.shape[1]

    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    all_losses = []
    all_indices = []
    indices_min_loss = []

    # loss_matrix (B, N, T) を一括計算: ループ不要
    diff_all = pred.unsqueeze(2) - target.unsqueeze(1)   # (B, N, T, D)
    loss_matrix_all = (diff_all ** 2).sum(dim=3)         # (B, N, T)

    for b in range(B):
        loss_matrix = loss_matrix_all[b]  # (N, T) — 参照のみ、再計算なし

        # 2. predごとに最小の教師を記録（どの教師に近いか） 
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        # 3. predを各教師に割り当てる
        sorted_gst = torch.argsort(min_loss)  # 最小loss順にpredを処理
        sel_list = [[] for _ in range(T)] # 各教師に割り当てられたlossを格納
        sel_indices_list = [[] for _ in range(T)]  # index
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g]) # このpredに対する教師の優先順位
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])  # 勾配追跡可能
                    sel_indices_list[ti].append(g.item())   # pred index
                    counts[ti] += 1
                    break
            if torch.all(counts == K): # 全員割り当て終わったら終了
                break

        # 4. Tensorに変換
        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])      # [T, K]
        chosen_indices = np.array(sel_indices_list)                          # [T, K]

        all_losses.append(chosen_losses) 
        all_indices.append(chosen_indices)

    # 5. バッチ方向にstack
    losses_tensor = torch.stack(all_losses, dim=0)     # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)   # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0) # list of length B * N 

    # 6. 各教師の平均lossと最悪ターゲット
    loss_mean_target = torch.mean(losses_tensor, dim=2)    # [B, T] ターゲットごとの平均値 # dim 2?
    losses_worst, worst_idx = torch.max(loss_mean_target, dim=1)  # [B] 最悪の教師
    losses_mean = torch.mean(loss_mean_target)  # [B] 各ターゲットの平均
    # meanではなくsumの可能性あり
    return losses_mean, worst_idx.detach().cpu().numpy(), loss_mean_target.detach().cpu().numpy(), indices_loss_min, indices_use_loss


def sekitori_loss_worst_percent(pred, target, worst_percent=0.3):
    """
    sekitori assignment後、割り当てlossの上位worst_percent（高loss側）のみで学習
    pred:          [B, N, D]
    target:        [B, T, D]
    worst_percent: float 0.0~1.0 (例: 0.3 = 30%)

    戻り値:
        loss_worst:       scalar Tensor (autograd対応)
        worst_idx:        np [B]        最悪の教師index
        loss_mean_target: np [B, T]     教師ごとの平均loss
        indices_loss_min: np [B, N]     predごとの最小loss教師
        indices_use_loss: np [B, T, K]  割り当てindex
        sorted_flat_np:   np [T*K]      バッチ先頭、降順ソート済みflat losses
        n_worst:          int           使用したworst sample数
    """
    B, N, _ = pred.shape
    T = target.shape[1]
    assert N % T == 0, "predの数が教師で割り切れません"
    K = N // T

    diff_all = pred.unsqueeze(2) - target.unsqueeze(1)   # (B, N, T, D)
    loss_matrix_all = (diff_all ** 2).sum(dim=3)         # (B, N, T)

    all_losses = []
    all_indices = []
    indices_min_loss = []

    for b in range(B):
        loss_matrix = loss_matrix_all[b]
        min_loss, min_loss_idx = torch.min(loss_matrix, dim=1)
        indices_min_loss.append(min_loss_idx.detach().cpu().numpy())

        sorted_gst = torch.argsort(min_loss)
        sel_list = [[] for _ in range(T)]
        sel_indices_list = [[] for _ in range(T)]
        counts = torch.zeros(T, dtype=torch.long)

        for g in sorted_gst:
            prefs = torch.argsort(loss_matrix[g])
            for t in prefs:
                ti = t.item()
                if counts[ti] < K:
                    sel_list[ti].append(loss_matrix[g, t])
                    sel_indices_list[ti].append(g.item())
                    counts[ti] += 1
                    break
            if torch.all(counts == K):
                break

        chosen_losses = torch.stack([torch.stack(v) for v in sel_list])  # [T, K]
        chosen_indices = np.array(sel_indices_list)
        all_losses.append(chosen_losses)
        all_indices.append(chosen_indices)

    losses_tensor = torch.stack(all_losses, dim=0)      # [B, T, K]
    indices_use_loss = np.stack(all_indices, axis=0)    # [B, T, K]
    indices_loss_min = np.stack(indices_min_loss, axis=0)  # [B, N]

    loss_mean_target = torch.mean(losses_tensor, dim=2)  # [B, T]
    _, worst_idx = torch.max(loss_mean_target, dim=1)    # [B]

    n_worst = max(1, int(T * K * worst_percent))
    worst_loss_per_batch = []
    sorted_flat_np = None
    for b in range(B):
        flat = losses_tensor[b].reshape(-1)              # (T*K,)
        sorted_vals, _ = torch.sort(flat, descending=True)
        if b == 0:
            sorted_flat_np = sorted_vals.detach().cpu().numpy()
        worst_loss_per_batch.append(sorted_vals[:n_worst].mean())

    loss_worst = torch.stack(worst_loss_per_batch).mean()

    return (loss_worst,
            worst_idx.detach().cpu().numpy(),
            loss_mean_target.detach().cpu().numpy(),
            indices_loss_min,
            indices_use_loss,
            sorted_flat_np,
            n_worst)
