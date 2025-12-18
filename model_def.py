import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        num_filters=64,
        dropout_rate=0.5,
        num_classes=3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=768,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # [batch, 768, seq_len]
        x = self.relu(self.conv(x))     # [batch, num_filters, seq_len]
        x = self.pool(x).squeeze(-1)    # [batch, num_filters]
        x = self.dropout(x)
        return self.fc(x)
