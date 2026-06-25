import torch
import numpy as np


def sekitori_loss_sum(pred, target_imgs):
    """
    pred: (batch, num_samples, dim)
    target_imgs: (batch, num_rel, dim)
    """
    batch, n_pred, _ = pred.shape
    n_rel = target_imgs.shape[1]

    diff = pred.unsqueeze(2) - target_imgs.unsqueeze(1)
    mse = torch.sum(diff * diff, dim=3)

    batch_losses = []
    batch_loss_per_item = []
    batch_selected_labels = []
    batch_closest = []
    batch_assignments = []

    for b in range(batch):
        mse_b = mse[b]
        closest = torch.argmin(mse_b, dim=1)
        batch_closest.append(closest.cpu().numpy())

        orders = [torch.argsort(mse_b[:, r]) for r in range(n_rel)]
        quotas = [n_pred // n_rel] * n_rel
        for i in range(n_pred % n_rel):
            quotas[i] += 1

        selected = torch.zeros(n_pred, dtype=torch.bool, device=pred.device)
        per_rel = [[] for _ in range(n_rel)]
        ptrs = [0] * n_rel
        selected_losses = []

        while sum(len(lst) for lst in per_rel) < n_pred:
            best_val = float("inf")
            best_rel = None
            best_idx = None

            for r in range(n_rel):
                if quotas[r] <= 0:
                    continue
                while ptrs[r] < n_pred and selected[int(orders[r][ptrs[r]])]:
                    ptrs[r] += 1
                if ptrs[r] >= n_pred:
                    continue

                idx = int(orders[r][ptrs[r]])
                val = mse_b[idx, r].item()
                if val < best_val:
                    best_val = val
                    best_rel = r
                    best_idx = idx

            if best_rel is None:
                break

            selected[best_idx] = True
            per_rel[best_rel].append(best_idx)
            selected_losses.append(mse_b[best_idx, best_rel])
            quotas[best_rel] -= 1
            ptrs[best_rel] += 1

        if len(selected_losses) == 0:
            batch_loss = torch.tensor(0.0, device=pred.device)
        else:
            batch_loss = torch.stack(selected_losses).mean()

        labels_b = torch.full((n_pred,), -1, dtype=torch.long, device=pred.device)
        for r, idx_list in enumerate(per_rel):
            for idx in idx_list:
                labels_b[idx] = r

        batch_losses.append(batch_loss)
        batch_loss_per_item.append(batch_loss)
        batch_selected_labels.append(labels_b)
        batch_assignments.append([np.array(idx_list, dtype=np.int32) for idx_list in per_rel])

    batch_losses_t = torch.stack(batch_losses)
    return batch_losses_t.mean(), torch.stack(batch_loss_per_item), torch.stack(batch_selected_labels), batch_closest, batch_assignments