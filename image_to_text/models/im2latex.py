import torch.nn as nn

class Im2LaTeX(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, targets=None, teacher_forcing_ratio=0.5, max_len=None):
        enc_out = self.encoder(images)
        return self.decoder(enc_out, targets, teacher_forcing_ratio, max_len)