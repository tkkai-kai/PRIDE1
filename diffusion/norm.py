# Normalizers for diffusion.

from typing import List

import torch
from torch import nn


class BaseNormalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, dataset: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer('min', dataset.min(dim=0).values)
        self.register_buffer('max', dataset.max(dim=0).values + eps)
        print('Mins:', self.min)
        print('Maxs:', self.max)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_ = self.min.to(x.device)
        max_ = self.max.to(x.device)
        return (x - min_) / (max_ - min_) * 2 - 1

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        min_ = self.min.to(x.device)
        max_ = self.max.to(x.device)
        return (x + 1) / 2 * (max_ - min_) + min_

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        min_ = dataset.min(dim=0).values
        max_ = dataset.max(dim=0).values + eps
        self.register_buffer('min', min_)
        self.register_buffer('max', max_)
        print('Mins:', self.min)
        print('Maxs:', self.max)


class Normalizer(BaseNormalizer):
    def __init__(
            self,
            dataset: torch.Tensor,
            eps: float = 1e-5,
            skip_dims: List[int] = [],
            target_std: float = 1.0,
    ):
        super().__init__()
        self.register_buffer('mean', dataset.mean(dim=0))
        self.register_buffer('std', dataset.std(dim=0) + eps)
        self.skip_dims = skip_dims
        if skip_dims:
            self.mean[skip_dims] = 0.0
            self.std[skip_dims] = 1.0
        self.target_std = target_std
        print('Means:', self.mean)
        print('Stds:', self.std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std * self.target_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x / self.target_std * std + mean

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        mean = dataset.mean(dim=0)
        std = dataset.std(dim=0) + eps
        if self.skip_dims:
            mean = mean.clone()
            std = std.clone()
            mean[self.skip_dims] = 0.0
            std[self.skip_dims] = 1.0
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        print('Means:', self.mean)
        print('Stds:', self.std)


def normalizer_factory(
        normalizer_type: str,
        dataset: torch.Tensor,
        skip_dims: List[int] = [],
        **kwargs,
) -> BaseNormalizer:
    if normalizer_type == 'minmax':
        return MinMaxNormalizer(dataset, **kwargs)
    elif normalizer_type == 'standard':
        return Normalizer(dataset, skip_dims=skip_dims, **kwargs)
    else:
        raise ValueError(f'Unknown normalizer type: {normalizer_type}')
