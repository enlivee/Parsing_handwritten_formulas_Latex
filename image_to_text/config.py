import torch
import os

class Config:
    # ================== Данные ==================
    dataset_root = "/Users/emmat/.cache/kagglehub/datasets/mssjss/fastmathxassumption-2025-handwritten-math-to-latex/versions/1"
    csv_path = os.path.join(dataset_root, "latex_equation_groundtruth.csv")
    image_base_dir = os.path.join(dataset_root, "paper")
    model_save_path = "image_to_text/checkpoints/best_model.pth"

    # изображения
    img_height = 94
    img_width = 256
    channels = 1
    # ------------------- Обработка (crop) ------------------     
    #Фиксированная обрезка: доли от ширины/высоты, которые ОБРЕЗАЮТСЯ    
    crop_left   = 0.1    # обрезать слева 0% (0.0–1.0)     
    crop_right  = 0.35   # справа     
    crop_top    = 0.28   # сверху     
    crop_bottom = 0.05   # снизу

    auto_crop = True
    crop_pad_left   = 5
    crop_pad_right  = 5
    crop_pad_top    = 5
    crop_pad_bottom = 5

    # ================== Токены ==================
    sos_token = "<sos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    unk_token = "<unk>"
    special_tokens = [sos_token, eos_token, pad_token, unk_token]
    chars = list(
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]()\\^_+-*/=.,;:!?\"'`|&<>#%$@~"
    )

    # ================== Модель ==================
    enc_out_channels = 512
    enc_hidden = 256
    dec_hidden = 512
    emb_dim = 256
    num_layers_enc = 1
    num_layers_dec = 1
    dropout = 0.2

    # ================== Обучение ==================
    batch_size = 16
    epochs = 8
    lr = 1e-3
    teacher_forcing_ratio = 0.5
    max_seq_len = 200
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
