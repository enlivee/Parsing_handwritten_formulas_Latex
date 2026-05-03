import numpy as np
import argparse
from pathlib import Path

import cv2 as c


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "formulas"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "preprocessed"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def imwrite_unicode(path: Path, image: np.ndarray):
    """Сохраняет изображение с поддержкой кириллических путей"""
    # Кодируем изображение в байты PNG
    success, encoded = c.imencode('.png', image)
    if success:
        # Записываем байты в файл (Path.write_bytes работает с Unicode)
        path.write_bytes(encoded.tobytes())
        return True
    return False


def preprocess_image(input_path: Path, save_steps_dir: Path | None = None):
    # Читаем файл через numpy в обход проблемы с кириллицей
    with open(str(input_path), 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = c.imdecode(file_bytes, c.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Ошибка чтения изображения: {input_path}")

    gray = c.cvtColor(image, c.COLOR_BGR2GRAY)

    binary = c.adaptiveThreshold(
        gray,
        255,
        c.ADAPTIVE_THRESH_GAUSSIAN_C,
        c.THRESH_BINARY,
        31,
        8,
    )

    if save_steps_dir is not None:
        save_steps_dir.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(save_steps_dir / f"{input_path.stem}_gray.png", gray)
        imwrite_unicode(save_steps_dir / f"{input_path.stem}_binary.png", binary)
        imwrite_unicode(save_steps_dir / f"{input_path.stem}_binary_inverted.png", 255 - binary)

    return binary


def preprocess_folder(input_dir: Path, output_dir: Path, save_steps: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = output_dir / "steps" if save_steps else None

    processed_count = 0

    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        processed = preprocess_image(image_path, save_steps_dir=steps_dir)

        output_path = output_dir / f"{image_path.stem}_preprocessed.png"
        imwrite_unicode(output_path, processed)
        processed_count += 1

        print(f"Saved: {output_path}")

    if processed_count == 0:
        print(f"Картинок в папке {input_dir} нет :D")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-steps", action="store_true")

    args = parser.parse_args()

    preprocess_folder(
        input_dir=args.input,
        output_dir=args.output,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()