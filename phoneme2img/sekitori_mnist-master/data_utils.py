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


def sample_related_images_batch(labels, inputs=None):
    """
    バッチ内の各サンプルについて related_images をまとめて返す
    inputsを渡した場合、same digitの教師は入力画像そのものを避けてランダム選択する
    shape: (batch, 3, 784)
    """
    batch_related = []
    if inputs is not None:
        inputs = inputs.to(device).view(-1, 784)

    for i, digit in enumerate(labels.tolist()):
        related_digits = get_related_digits(int(digit))
        images = []
        for d in related_digits:
            candidates = label_images[d]
            if inputs is not None and d == int(digit):
                same_as_input = torch.all(candidates == inputs[i], dim=1)
                candidates = candidates[~same_as_input]
                if candidates.size(0) == 0:
                    candidates = label_images[d]

            idx = torch.randint(candidates.size(0), (1,), device=device).item()
            images.append(candidates[idx])

        imgs = torch.stack(images)
        batch_related.append(imgs)
    return torch.stack(batch_related, dim=0)
