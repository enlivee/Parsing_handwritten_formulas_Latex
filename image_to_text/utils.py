import torch
from torch.nn.utils.rnn import pad_sequence
import Levenshtein
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from config import Config
from tokenizer import LaTeXTokenizer

tokenizer = LaTeXTokenizer()

def pad_collate(batch):
    imgs, tokens = zip(*batch)
    imgs = torch.stack(imgs)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_id)
    if tokens_padded.size(1) > Config.max_seq_len:
        tokens_padded = tokens_padded[:, :Config.max_seq_len]
    return imgs, tokens_padded

@torch.no_grad()
def compute_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs, targets[:, :-1], teacher_forcing_ratio=0.0)
        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets[:, 1:].reshape(-1)
        )
        total_loss += loss.item()
        pred_ids = outputs.argmax(dim=-1)
        for b in range(imgs.size(0)):
            pred_str = tokenizer.decode(pred_ids[b].cpu().tolist())
            targ_str = tokenizer.decode(targets[b, 1:].cpu().tolist())
            all_preds.append(pred_str)
            all_targets.append(targ_str)

    loss = total_loss / len(loader)
    exact_matches = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    em = exact_matches / len(all_targets) if all_targets else 0.0

    edit_dists = []
    for p, t in zip(all_preds, all_targets):
        if len(t) == 0:
            edit_dists.append(1.0 if len(p)>0 else 0.0)
        else:
            edit_dists.append(Levenshtein.distance(p, t) / len(t))
    avg_edit = np.mean(edit_dists) if edit_dists else 1.0

    references = [[t.split()] for t in all_targets]
    hypotheses = [p.split() for p in all_preds]
    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    return {
        'loss': loss,
        'exact_match': em,
        'avg_edit_distance': avg_edit,
        'bleu-4': bleu
    }