from __future__ import annotations

from pathlib import Path

import torch


def make_synthetic_pca_data(
    n_agents: int = 40,
    samples_per_agent: int = 100,
    d: int = 100,
    *,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Generate the heterogeneous Gaussian data used in PCAStief_Syn_40.m.

    MATLAB scales client i's standard Gaussian samples by i/N, so the
    covariance scales as (i/N)^2 I.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    clients = []
    for i in range(1, n_agents + 1):
        samples = torch.randn(
            samples_per_agent, d, generator=generator, dtype=dtype
        )
        clients.append(samples * (i / n_agents))
    return clients


def partition_noniid_by_label(
    images: torch.Tensor,
    labels: torch.Tensor,
    n_agents: int = 50,
    shards_per_client: int = 2,
    *,
    seed: int = 0,
) -> list[torch.Tensor]:
    """FedAvg-style label-sorted shard partition used for non-IID CIFAR-10.

    Images must be row-wise flattened [num_images, d]. Each client receives
    shards_per_client consecutive label-sorted shards.
    """
    if images.ndim != 2 or labels.ndim != 1 or images.shape[0] != labels.shape[0]:
        raise ValueError("Expected images [n,d] and labels [n].")
    n_shards = n_agents * shards_per_client
    order = torch.argsort(labels.to(dtype=torch.long), stable=True)
    shard_size = images.shape[0] // n_shards
    if shard_size == 0:
        raise ValueError("Too many shards for the supplied dataset.")

    shards = [
        order[j * shard_size : (j + 1) * shard_size]
        for j in range(n_shards)
    ]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shard_order = torch.randperm(n_shards, generator=generator).tolist()

    clients = []
    for i in range(n_agents):
        chosen = shard_order[i * shards_per_client : (i + 1) * shards_per_client]
        indices = torch.cat([shards[j] for j in chosen])
        clients.append(images.index_select(0, indices).contiguous())

    # Include any remainder in the final client's local dataset.
    used = shard_size * n_shards
    if used < images.shape[0]:
        clients[-1] = torch.cat([clients[-1], images[order[used:]]], dim=0)
    return clients


def load_cifar10_noniid(
    root: str | Path = "./data",
    *,
    n_agents: int = 50,
    shards_per_client: int = 2,
    normalize: bool = True,
    download: bool = True,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Load CIFAR-10 training images and create non-IID client shards."""
    try:
        from torchvision.datasets import CIFAR10
    except Exception as exc:
        raise RuntimeError(
            "torchvision is required for CIFAR-10. Install the requirements first."
        ) from exc

    dataset = CIFAR10(root=str(root), train=True, download=download)
    # CIFAR10 stores uint8 NHWC images. Flattening order differs from MATLAB's
    # column-major reshape only by a fixed coordinate permutation, which leaves
    # the PCA objective and its spectrum unchanged.
    images = torch.as_tensor(dataset.data.copy(), dtype=torch.float32)
    images = images.reshape(images.shape[0], -1)
    if normalize:
        images = images / 255.0
    labels = torch.as_tensor(dataset.targets, dtype=torch.long)
    return partition_noniid_by_label(
        images,
        labels,
        n_agents=n_agents,
        shards_per_client=shards_per_client,
        seed=seed,
    )
