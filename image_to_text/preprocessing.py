import cv2
import numpy as np
import torch

def fixed_crop(img, crop_left, crop_right, crop_top, crop_bottom):
    h, w = img.shape[:2]
    x1 = int(w * crop_left)
    x2 = int(w * (1 - crop_right))
    y1 = int(h * crop_top)
    y2 = int(h * (1 - crop_bottom))
    x1 = max(0, min(x1, w-1))
    x2 = max(x1+1, min(x2, w))
    y1 = max(0, min(y1, h-1))
    y2 = max(y1+1, min(y2, h))
    return img[y1:y2, x1:x2]

def auto_crop_to_content(img, pad_left, pad_right, pad_top, pad_bottom):
    # Приводим к uint8, если картинка float32
    if img.dtype == np.float32:
        img_for_thresh = img.astype(np.uint8)
    else:
        img_for_thresh = img

    _, thresh = cv2.threshold(img_for_thresh, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    y_coords, x_coords = np.where(thresh == 255)
    if len(y_coords) == 0:
        return img

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    h, w = img.shape[:2]
    x1 = max(0, x_min - pad_left)
    x2 = min(w, x_max + pad_right + 1)
    y1 = max(0, y_min - pad_top)
    y2 = min(h, y_max + pad_bottom + 1)
    return img[y1:y2, x1:x2]   # тип возвращаемого изображения остаётся исходным (float32)

def apply_crop_pipeline(img, cfg):
    img = fixed_crop(img, cfg.crop_left, cfg.crop_right, cfg.crop_top, cfg.crop_bottom)
    if cfg.auto_crop:
        img = auto_crop_to_content(img, cfg.crop_pad_left, cfg.crop_pad_right,
                                   cfg.crop_pad_top, cfg.crop_pad_bottom)
    return img

def prepare_image(image_path, cfg, do_crop=True):
    """
    Загружает изображение, (опционально) обрезает, ресайзит, нормализует.
    Возвращает тензор (1, 1, H, W).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    if do_crop:
        img = apply_crop_pipeline(img, cfg)
    img = cv2.resize(img, (cfg.img_width, cfg.img_height))
    img = (img / 255.0 - 0.5) / 0.5
    img = np.expand_dims(img, axis=0)  # (1, H, W)
    return torch.from_numpy(img).float().unsqueeze(0)  # (1,1,H,W)