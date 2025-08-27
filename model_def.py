import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, kernel_size):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 3)  # 3 kelas: Negatif, Netral, Positif

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, channels, seq_len]
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)