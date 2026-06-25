import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
latent_dim = 4
epochs = 200
lr = 1e-4
num_samples = 99  # N per reparameterize; must be divisible by T=3
beta = 0.1

TRAIN_PER_DIGIT = 50
VAL_PER_DIGIT = 10

transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Build per-digit index split (reproducible via seed)
train_indices, val_indices = [], []
rng = np.random.default_rng(seed)

for d in range(10):
    idx = (full_dataset.targets == d).nonzero(as_tuple=True)[0].numpy()
    perm = rng.permutation(len(idx))
    train_indices.extend(idx[perm[:TRAIN_PER_DIGIT]].tolist())
    val_indices.extend(idx[perm[TRAIN_PER_DIGIT:TRAIN_PER_DIGIT + VAL_PER_DIGIT]].tolist())

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# label_images: training images only (TRAIN_PER_DIGIT per digit) for related-image sampling
train_indices_t = torch.tensor(train_indices)
train_labels = full_dataset.targets[train_indices_t]
train_data = full_dataset.data[train_indices_t]

label_images = {
    d: train_data[train_labels == d].float().view(-1, 784).to(device) / 255.0
    for d in range(10)
}

# 固定監視サンプル: 数字ごと先頭 MONITOR_PER_DIGIT 枚をエポックをまたいで追跡
MONITOR_PER_DIGIT = 3
TRACK_DIGIT = 5   # latent_space / latent_teacher で追跡する数字

_monitor_imgs, _monitor_lbls = [], []
for d in range(10):
    imgs = train_data[train_labels == d][:MONITOR_PER_DIGIT].float().view(-1, 784) / 255.0
    _monitor_imgs.append(imgs)
    _monitor_lbls.extend([d] * imgs.size(0))

monitor_images = torch.cat(_monitor_imgs, dim=0)       # (10 * MONITOR_PER_DIGIT, 784)
monitor_labels = torch.tensor(_monitor_lbls)           # (10 * MONITOR_PER_DIGIT,)
