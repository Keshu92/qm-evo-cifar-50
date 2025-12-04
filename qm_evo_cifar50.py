#!/usr/bin/env python3
"""
qm_evo_cifar50.py

Minimal, reproducible demo of "ψ-evolution" vs standard baselines
on CIFAR-10 (animals vs vehicles) using only 50 labeled examples.

Pipeline:
  1. Download CIFAR-10
  2. Extract 512-d embeddings with ResNet-18 (ImageNet weights)
  3. Build binary task: animals vs vehicles
  4. Compare:
       - k-NN (all 20k labels)
       - Logistic regression (all 20k labels)
       - k-NN (50 labels)
       - Logistic regression (50 labels)
       - ψ-evo (50 hard seeds on a k-NN graph over train+test)

Dependencies:
  - torch, torchvision
  - numpy, scipy
  - scikit-learn
"""

import os
import numpy as np
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms

from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csgraph, csr_matrix


# ---------------------------
# 0. Utility: device + seed
# ---------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# 1. CIFAR-10 loading & embeddings
# ---------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Animals vs vehicles:
# animals  = {bird, cat, deer, dog, frog, horse}   -> label 1
# vehicles = {airplane, automobile, ship, truck}   -> label 0
ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}
VEHICLE_CLASSES = {0, 1, 8, 9}


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Return train/test dataloaders for CIFAR-10 with ImageNet normalization."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        )
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_ds, test_ds, train_loader, test_loader


def build_resnet18_encoder(device):
    """ResNet-18 feature extractor (512-d global avgpool output)."""
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # Drop the final fully-connected layer
    modules = list(model.children())[:-1]   # everything up to avgpool
    encoder = nn.Sequential(*modules).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def extract_embeddings(encoder, loader, device):
    """Extract 512-d embeddings for all images in given DataLoader."""
    all_embs = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = encoder(imgs)          # shape (B, 512, 1, 1)
        feats = feats.squeeze(-1).squeeze(-1)  # (B, 512)
        all_embs.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


# ---------------------------
# 2. Graph + ψ-evolution
# ---------------------------

def build_sym_laplacian(X, k=12):
    """
    Build symmetric normalized Laplacian L_sym from k-NN graph.

    X: (N, d) embeddings
    returns: sparse csr_matrix L_sym (N x N)
    """
    print(f"  Building k-NN graph (k={k}) on {X.shape[0]} points...")
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                         include_self=False, n_jobs=-1)
    A = 0.5 * (A + A.T)  # symmetrize
    L_sym = csgraph.laplacian(A, normed=True)
    return csr_matrix(L_sym)


def evolve_psi(L_sym, seed_inds, seed_labels, n_classes,
               alpha=0.2, eta=0.03, n_steps=60):
    """
    ψ-evo dynamics on graph Laplacian (toy version):

      psi <- psi - alpha * L_sym * psi + eta * psi * (1 - psi)

    with row-wise normalization and clamped seeds.

    L_sym: csr_matrix (N x N)
    seed_inds: (S,) int array
    seed_labels: (S,) int array in [0..n_classes-1]
    returns: psi (N x n_classes)
    """
    N = L_sym.shape[0]
    psi = np.zeros((N, n_classes), dtype=np.float32)
    psi[seed_inds, seed_labels] = 1.0

    for t in range(n_steps):
        lin = L_sym.dot(psi)                 # (N, C)
        nonlin = psi * (1.0 - psi)           # (N, C)
        psi = psi - alpha * lin + eta * nonlin

        # row-wise L2 normalization
        row_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-12
        psi = psi / row_norms

        # clamp seeds
        psi[seed_inds] = 0.0
        psi[seed_inds, seed_labels] = 1.0

    return psi


# ---------------------------
# 3. Main experiment
# ---------------------------

def main():
    set_seed(0)
    device = get_device()
    print("Device:", device)

    # 1) Load CIFAR-10 and extract embeddings
    print("Loading CIFAR-10 and building loaders...")
    train_ds, test_ds, train_loader, test_loader = get_cifar10_loaders(
        batch_size=128, num_workers=2
    )

    print("Building ResNet-18 encoder...")
    encoder = build_resnet18_encoder(device)

    print("Extracting train embeddings...")
    X_train_full, y_train_full = extract_embeddings(encoder, train_loader, device)
    print("Extracting test embeddings...")
    X_test_full, y_test_full = extract_embeddings(encoder, test_loader, device)

    print("Train embeddings:", X_train_full.shape,
          "Test embeddings:", X_test_full.shape)

    # For speed & direct comparison, take first 20k train, all 10k test
    N_train = 20000
    X_train = X_train_full[:N_train].astype(np.float32)
    y_train_raw = y_train_full[:N_train]
    X_test = X_test_full.astype(np.float32)
    y_test_raw = y_test_full

    # 2) Build animals vs vehicles binary labels
    def to_animals_vs_vehicles(y_raw):
        y_bin = np.zeros_like(y_raw, dtype=int)
        for i, c in enumerate(y_raw):
            if c in ANIMAL_CLASSES:
                y_bin[i] = 1
            elif c in VEHICLE_CLASSES:
                y_bin[i] = 0
            else:
                raise ValueError(f"Unexpected CIFAR class: {c}")
        return y_bin

    y_train = to_animals_vs_vehicles(y_train_raw)
    y_test = to_animals_vs_vehicles(y_test_raw)

    print("\nTask: animals vs vehicles")
    print("Train dist:", Counter(y_train))
    print("Test  dist:", Counter(y_test))

    # 3) Full-label baselines (20k labels)
    print("\nFULL-LABEL BASELINES (20k labels)")

    knn_full = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    knn_full.fit(X_train, y_train)
    acc_knn_full = knn_full.score(X_test, y_test)
    print(f"  k-NN (all labels) acc: {acc_knn_full:.4f}")

    # Logistic regression (can take a bit but 20k x 512 is manageable)
    lr_full = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1
    )
    lr_full.fit(X_train, y_train)
    acc_lr_full = lr_full.score(X_test, y_test)
    print(f"  Logistic (all labels) acc: {acc_lr_full:.4f}")

    # 4) 50-label regime (25 seeds per class)
    print("\n50-LABEL REGIME (25 labels per class)")

    rng = np.random.default_rng(0)
    seeds_per_class = 25
    seed_inds = []
    seed_labels = []

    for cls in [0, 1]:
        idxs = np.where(y_train == cls)[0]
        chosen = rng.choice(idxs, size=seeds_per_class, replace=False)
        seed_inds.extend(chosen.tolist())
        seed_labels.extend([cls] * seeds_per_class)

    seed_inds = np.array(seed_inds, dtype=int)
    seed_labels = np.array(seed_labels, dtype=int)
    print("Seeds per class:", Counter(seed_labels))

    # Subset for supervised baselines
    X_train_50 = X_train[seed_inds]
    y_train_50 = y_train[seed_inds]

    # k-NN with 50 labels
    knn_50 = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    knn_50.fit(X_train_50, y_train_50)
    acc_knn_50 = knn_50.score(X_test, y_test)
    print(f"  k-NN (50 labels) acc: {acc_knn_50:.4f}")

    # Logistic with 50 labels
    lr_50 = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1
    )
    lr_50.fit(X_train_50, y_train_50)
    acc_lr_50 = lr_50.score(X_test, y_test)
    print(f"  Logistic (50 labels) acc: {acc_lr_50:.4f}")

    # 5) ψ-evo on a shared graph over train+test
    print("\nBuilding shared k-NN graph over train+test for ψ-evo...")
    X_all = np.vstack([X_train, X_test])  # (30000, 512)
    N_all = X_all.shape[0]

    L_sym = build_sym_laplacian(X_all, k=12)
    print("  L_sym shape:", L_sym.shape)

    # Evolve ψ with the same 50 seeds (only on training indices)
    # We seed only on train nodes: seed_inds as indices in [0, N_train)
    # y_all_bin = [train labels, then test labels]
    y_all = np.concatenate([y_train, y_test])

    print("\nRunning ψ-evo with 50 hard seeds...")
    psi = evolve_psi(
        L_sym=L_sym,
        seed_inds=seed_inds,
        seed_labels=y_train[seed_inds],
        n_classes=2,
        alpha=0.2,
        eta=0.03,
        n_steps=60
    )

    preds_all = psi.argmax(axis=1)
    preds_test = preds_all[N_train:]
    acc_qm = (preds_test == y_test).mean()
    print(f"[ψ-evo] (50 hard seeds) test acc: {acc_qm:.4f}")

    # 6) Final summary
    print("\n===== SUMMARY (animals vs vehicles, 50 labels) =====")
    print(f"k-NN (all labels)     : {acc_knn_full:.4f}")
    print(f"Logistic (all labels) : {acc_lr_full:.4f}")
    print(f"k-NN (50 labels)      : {acc_knn_50:.4f}")
    print(f"Logistic (50 labels)  : {acc_lr_50:.4f}")
    print(f"ψ-evo (50 hard seeds) : {acc_qm:.4f}")


if __name__ == "__main__":
    main()
