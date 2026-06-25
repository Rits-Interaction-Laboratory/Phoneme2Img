import torch
from config import label_images, device


def get_related_digits(digit):
    """digit-1, digit, digit+1 (mod 10) の3つを返す"""
    return [(digit - 1) % 10, digit, (digit + 1) % 10]


# 各数字の先頭1枚を固定教師として使用
fixed_label_images = {d: label_images[d][0] for d in range(10)}


def get_fixed_related_images_batch(labels):
    """
    固定教師版: 数字ごとに1枚だけ決まった画像を教師として返す
    shape: (batch, 3, 784)
    """
    batch_related = []
    for digit in labels.tolist():
        related_digits = get_related_digits(int(digit))
        imgs = torch.stack([fixed_label_images[d] for d in related_digits])
        batch_related.append(imgs)
    return torch.stack(batch_related, dim=0)


def sample_related_images(related_digits):
    """関連数字からランダムに1枚ずつ画像をサンプリング"""
    images = []
    for d in related_digits:
        idx = torch.randint(len(label_images[d]), (1,)).item()
        images.append(label_images[d][idx])
    return torch.stack(images).to(device)


def sample_related_images_batch(labels):
    """
    バッチ内の各サンプルについて related_images をまとめて返す
    shape: (batch, 3, 784)
    """
    batch_related = []
    for digit in labels.tolist():
        related_digits = get_related_digits(int(digit))
        imgs = torch.stack([
            label_images[d][torch.randint(len(label_images[d]), (1,), device=device).item()]
            for d in related_digits
        ])
        batch_related.append(imgs)
    return torch.stack(batch_related, dim=0)
