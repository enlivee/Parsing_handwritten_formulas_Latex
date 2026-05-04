import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from tokenizer import LaTeXTokenizer

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, enc_out_size, vocab_size, emb_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.enc_out_size = enc_out_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim + enc_out_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.attn_energy = nn.Linear(hidden_size + enc_out_size, 1)
        self.out = nn.Linear(hidden_size, vocab_size)
        # SOS токен
        self.sos_id = LaTeXTokenizer().sos_id

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5, max_len=None):
        B, L_enc, _ = encoder_outputs.shape
        device = encoder_outputs.device
        h = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=device)
        input_token = torch.full((B,), self.sos_id, dtype=torch.long, device=device)
        decoder_outputs = []
        max_len = max_len if max_len else (targets.size(1) if targets is not None else Config.max_seq_len)

        for t in range(max_len):
            emb = self.embedding(input_token).unsqueeze(1)
            emb = self.dropout(emb)
            h_last = h[-1]
            h_expanded = h_last.unsqueeze(1).expand(-1, L_enc, -1)
            concat = torch.cat((h_expanded, encoder_outputs), dim=2)
            energy = self.attn_energy(concat).squeeze(2)
            attn_weights = F.softmax(energy, dim=1).unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs)
            lstm_input = torch.cat((emb, context), dim=2)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.out(lstm_out).squeeze(1)
            decoder_outputs.append(output.unsqueeze(1))

            if targets is not None and t < targets.size(1)-1:
                if random.random() < teacher_forcing_ratio:
                    input_token = targets[:, t+1]
                else:
                    input_token = output.argmax(1)
            else:
                input_token = output.argmax(1)

        return torch.cat(decoder_outputs, dim=1)