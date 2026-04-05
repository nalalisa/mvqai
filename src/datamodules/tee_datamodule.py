from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightning import LightningDataModule
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    SpatialPadd,
)


class TEEDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_manifest: str,
        val_manifest: str,
        image_key: str = "image",
        target_key: str = "target",
        mask_key: str = "mask",
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 0.0,
        roi_size: tuple[int, int, int] = (128, 128, 128),
        spacing: tuple[float, float, float] | None = None,
        intensity_window: tuple[float, float] | None = None,
        sigma: float = 2.0,
        threshold: float = 0.3,
        augment: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.train_manifest = Path(train_manifest)
        self.val_manifest = Path(val_manifest)
        self.image_key = image_key
        self.target_key = target_key
        self.mask_key = mask_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.roi_size = tuple(roi_size)
        self.spacing = spacing
        self.intensity_window = intensity_window
        self.sigma = sigma
        self.threshold = threshold
        self.augment = augment or {}
        self.train_dataset: Dataset | CacheDataset | None = None
        self.val_dataset: Dataset | CacheDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train_items = self._read_manifest(self.train_manifest)
        val_items = self._read_manifest(self.val_manifest)

        dataset_cls = CacheDataset if self.cache_rate > 0 else Dataset
        self.train_dataset = dataset_cls(
            data=train_items,
            transform=self._build_transforms(train=True),
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        ) if dataset_cls is CacheDataset else dataset_cls(data=train_items, transform=self._build_transforms(train=True))
        self.val_dataset = dataset_cls(
            data=val_items,
            transform=self._build_transforms(train=False),
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        ) if dataset_cls is CacheDataset else dataset_cls(data=val_items, transform=self._build_transforms(train=False))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _read_manifest(self, manifest_path: Path) -> list[dict[str, Any]]:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. "
                "Create JSON lists for train/val under data/splits before launching training."
            )

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        items: list[dict[str, Any]] = []
        for sample in data:
            item = dict(sample)
            for key in (self.image_key, self.target_key, self.mask_key):
                if key in item and item[key] is not None:
                    item[key] = str((self.data_root / item[key]).resolve()) if not Path(item[key]).is_absolute() else item[key]
            items.append(item)
        return items

    def _build_transforms(self, train: bool) -> Compose:
        keys = [self.image_key, self.target_key]
        if self.mask_key:
            keys.append(self.mask_key)

        transforms = [
            LoadImaged(keys=keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
            SpatialPadd(keys=keys, spatial_size=self.roi_size, allow_missing_keys=True),
        ]

        if self.intensity_window is not None:
            transforms.append(
                ScaleIntensityRanged(
                    keys=[self.image_key],
                    a_min=self.intensity_window[0],
                    a_max=self.intensity_window[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            )

        if train:
            transforms.extend(
                [
                    RandFlipd(
                        keys=keys,
                        prob=float(self.augment.get("flip_prob", 0.0)),
                        spatial_axis=0,
                        allow_missing_keys=True,
                    ),
                    RandRotate90d(
                        keys=keys,
                        prob=float(self.augment.get("rotate90_prob", 0.0)),
                        max_k=3,
                        allow_missing_keys=True,
                    ),
                ]
            )

        transforms.append(EnsureTyped(keys=keys, allow_missing_keys=True))
        return Compose(transforms)

