import torch.nn as nn
from config import Config

class CRNNEncoder(nn.Module):
    def __init__(self, out_channels=512, hidden_size=256, num_layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(Config.channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(256, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.MaxPool2d((2,1))
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 64))
        self.rnn = nn.LSTM(
            input_size=out_channels * 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        feat = self.cnn(x)
        feat = self.adaptive_pool(feat)   # (B, C, H', W')
        B, C, H, W = feat.shape
        feat = feat.permute(0, 3, 2, 1)    # (B, W, H, C)
        feat = feat.reshape(B, W, H * C)   # (B, W, C*H')
        rnn_out, _ = self.rnn(feat)        # (B, W, hidden*2)
        return self.proj(rnn_out)          # (B, W, hidden)