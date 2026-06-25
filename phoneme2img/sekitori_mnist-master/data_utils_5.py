import torch
from config import label_images, device


def get_related_digits(digit):
    """
    偶数入力なら偶数の5枚、奇数入力なら奇数の5枚を関連画像として返す
    """
    if digit % 2 == 0:
        return [2, 4, 6, 8, 0]
    return [1, 3, 5, 7, 9]


fixed_label_images = {d: label_images[d][0] for d in range(10)}


def get_fixed_related_images_batch(labels):
    batch_related = []
    for digit in labels.tolist():
        related_digits = get_related_digits(int(digit))
        imgs = torch.stack([fixed_label_images[d] for d in related_digits])
        batch_related.append(imgs)
    return torch.stack(batch_related, dim=0)


def sample_related_images(related_digits):
    return torch.stack([
        label_images[d][torch.randint(len(label_images[d]), (1,), device=device).item()]
        for d in related_digits
    ])


def sample_related_images_batch(labels):
    batch_related = []
    for digit in labels.tolist():
        related_digits = get_related_digits(int(digit))
        imgs = torch.stack([
            label_images[d][torch.randint(len(label_images[d]), (1,), device=device).item()]
            for d in related_digits
        ])
        batch_related.append(imgs)
    return torch.stack(batch_related, dim=0)