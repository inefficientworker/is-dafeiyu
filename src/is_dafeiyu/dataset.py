"""Dataset and augmentations for Dafeiyu binary classification."""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

TRAIN_BG_COLORS = (
    (255, 255, 255),
    (245, 245, 245),
    (230, 235, 242),
    (32, 36, 44),
    (0, 0, 0),
    (250, 246, 236),
)


class RandomJPEGCompression:
    """Simulate chat app JPEG/WebP recompression artifacts."""

    def __init__(self, p: float = 0.3, quality_range: tuple[int, int] = (50, 95)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as recompressed:
            return recompressed.convert("RGB")


def load_rgb_image(path: str | Path, random_bg: bool = False) -> Image.Image:
    """Load an image and composite transparency onto either white or a random solid background."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg_color = random.choice(TRAIN_BG_COLORS) if random_bg else (255, 255, 255)
            bg = Image.new("RGB", rgba.size, bg_color)
            bg.paste(rgba, mask=rgba.getchannel("A"))
            return bg
        return img.convert("RGB")


def build_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    is_train: bool,
) -> transforms.Compose:
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.015),
                RandomJPEGCompression(p=0.3, quality_range=(55, 95)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.14)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class DafeiyuDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform: transforms.Compose,
        is_train: bool = False,
        teacher_logits: dict[str, float] | None = None,
    ) -> None:
        self.rows: list[dict] = []
        with Path(csv_path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["label"] = int(row["label"])
                self.rows.append(row)
        self.transform = transform
        self.is_train = is_train
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = load_rgb_image(row["path"], random_bg=self.is_train)
        tensor = self.transform(img)
        item = {
            "image": tensor,
            "label": float(row["label"]),
            "category": row["category"],
            "subcategory": row["subcategory"],
            "path": row["path"],
        }
        if self.teacher_logits is not None:
            item["teacher_logit"] = float(self.teacher_logits[row["path"]])
        return item
