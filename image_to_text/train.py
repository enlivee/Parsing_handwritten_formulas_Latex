import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from config import Config
from tokenizer import LaTeXTokenizer
from dataset import Im2LatexDataset, load_data
from models.encoder import CRNNEncoder
from models.decoder import AttentionDecoder
from models.im2latex import Im2LaTeX
from utils import pad_collate, compute_metrics

def get_model(vocab_size):
    encoder = CRNNEncoder(out_channels=Config.enc_out_channels,
                          hidden_size=Config.enc_hidden,
                          num_layers=Config.num_layers_enc)
    decoder = AttentionDecoder(hidden_size=Config.dec_hidden,
                               enc_out_size=Config.enc_hidden,
                               vocab_size=vocab_size,
                               emb_dim=Config.emb_dim,
                               num_layers=Config.num_layers_dec,
                               dropout=Config.dropout)
    return Im2LaTeX(encoder, decoder).to(Config.device)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Чекпоинт для продолжения')
    args = parser.parse_args()

    tokenizer = LaTeXTokenizer()
    vocab_size = len(tokenizer)

    # Загрузка данных
    img_paths, formulas = load_data(Config.csv_path, Config.image_base_dir)
    split = int(len(img_paths) * 0.9)
    train_paths, train_forms = img_paths[:split], formulas[:split]
    val_paths, val_forms = img_paths[split:], formulas[split:]

    train_dataset = Im2LatexDataset(train_paths, train_forms, tokenizer,
                                    Config.max_seq_len, augment=True, do_crop=True)
    val_dataset   = Im2LatexDataset(val_paths, val_forms, tokenizer,
                                    Config.max_seq_len, augment=False, do_crop=True)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                              collate_fn=pad_collate, num_workers=Config.num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False,
                              collate_fn=pad_collate, num_workers=Config.num_workers)

    model = get_model(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    start_epoch = 1
    best_val_em = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"Загрузка чекпоинта {args.resume}")
        checkpoint = torch.load(args.resume, map_location=Config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_em = checkpoint.get('val_metrics', {}).get('exact_match', 0.0)
        print(f"Продолжаем с эпохи {start_epoch}, best EM: {best_val_em:.4f}")

    for epoch in range(start_epoch, Config.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(Config.device), targets.to(Config.device)
            optimizer.zero_grad()
            outputs = model(imgs, targets[:, :-1], Config.teacher_forcing_ratio)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)),
                             targets[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        train_loss /= len(train_loader)

        val_metrics = compute_metrics(model, val_loader, criterion, Config.device)
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_metrics['loss']:.4f}, "
              f"EM {val_metrics['exact_match']:.4f}, Edit {val_metrics['avg_edit_distance']:.4f}, "
              f"BLEU {val_metrics['bleu-4']:.4f}")

        if val_metrics['exact_match'] > best_val_em:
            best_val_em = val_metrics['exact_match']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'tokenizer_vocab': tokenizer.vocab
            }, Config.model_save_path)
            print(f"Checkpoint saved (EM: {best_val_em:.4f})")

if __name__ == "__main__":
    train()