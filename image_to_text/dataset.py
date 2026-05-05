import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import Config
from preprocessing import apply_crop_pipeline

class Im2LatexDataset(Dataset):
    def __init__(self, image_paths, formulas, tokenizer, max_len,
                 augment=False, cfg=None, do_crop=True):
        self.image_paths = list(image_paths)
        self.formulas = list(formulas)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.cfg = cfg if cfg is not None else Config
        self.do_crop = do_crop

    def __len__(self):
        return len(self.image_paths)

    def _augment(self, img):
        """Лёгкие аугментации (поворот, сдвиг, яркость)."""
        h, w = img.shape[:2]
        angle = random.uniform(-1, 1)
        scale = random.uniform(0.95, 1.05)
        dx = random.uniform(-0.02, 0.02) * w
        dy = random.uniform(-0.02, 0.02) * h
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        alpha = 1.0 + random.uniform(-0.1, 0.1)
        beta = random.uniform(-10, 10)
        img = alpha * img + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        img = img.astype(np.float32)

        if self.do_crop:
            img = apply_crop_pipeline(img, self.cfg)

        if self.augment:
            img = self._augment(img)

        img = cv2.resize(img, (self.cfg.img_width, self.cfg.img_height))
        img = (img / 255.0 - 0.5) / 0.5
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.from_numpy(img).float()

        tokens = self.tokenizer.encode(self.formulas[idx], self.max_len)
        tokens = torch.tensor(tokens, dtype=torch.long)
        return img_tensor, tokens


def load_data(csv_path, base_image_dir=None):
    df = pd.read_csv(csv_path)
    formula_col_candidates = ("equation", "latex_gt", "latex", "formula")
    formula_col = next((col for col in formula_col_candidates if col in df.columns), None)
    if "image_path" not in df.columns or formula_col is None:
        raise ValueError(
            f"CSV {csv_path} must contain 'image_path' and one of {formula_col_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[["image_path", formula_col]].dropna()
    image_paths = []
    formulas = []
    for _, row in df.iterrows():
        img_path = row['image_path']
        if base_image_dir:
            img_path = os.path.join(base_image_dir, img_path)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            formulas.append(str(row[formula_col]).strip())
    return image_paths, formulas
