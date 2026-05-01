import argparse
from pathlib import Path

import cv2 as c


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "formulas"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "preprocessed"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def preprocess_image(input_path: Path, save_steps_dir: Path | None = None):

    image = c.imread(str(input_path)) # Открываем картинку и переводим путь в строку

    if image is None:
        raise ValueError(f"Ошибка чтения изображения: {input_path}")

    gray = c.cvtColor(image, c.COLOR_BGR2GRAY) # Переводим изобрежение в серый цвет

    binary = c.adaptiveThreshold(
        gray, # Входное изображение
        255, # Макс значение белого пикселя
        c.ADAPTIVE_THRESH_GAUSSIAN_C, # Порог считается по соседним пикселям
        c.THRESH_BINARY, # Черный текст на белом фоне
        31, # Размер окна; 11 слишком сильно рвет мелкие символы
        8, # Константа, вычитаемая из порога
    )

    # Для этого датасета medianBlur(binary, 3) портит мелкие формулы:
    # съедает тонкие линии дробей, индексы и буквы.

    if save_steps_dir is not None:
        save_steps_dir.mkdir(parents=True, exist_ok=True)
        c.imwrite(str(save_steps_dir / f"{input_path.stem}_gray.png"), gray)
        c.imwrite(str(save_steps_dir / f"{input_path.stem}_binary.png"), binary)
        c.imwrite(str(save_steps_dir / f"{input_path.stem}_binary_inverted.png"), 255 - binary)

    return binary

def preprocess_folder(input_dir: Path, output_dir: Path, save_steps: bool = False):

    output_dir.mkdir(parents=True, exist_ok=True) # Обрабатываем папку
    steps_dir = output_dir / "steps" if save_steps else None

    processed_count = 0

    for image_path in sorted(input_dir.iterdir()):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS: # Пропуск некартинок
            continue

        processed = preprocess_image(image_path, save_steps_dir=steps_dir)

        output_path = output_dir / f"{image_path.stem}_preprocessed.png"
        c.imwrite(str(output_path), processed)
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
