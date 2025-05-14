import torch
import torch.nn as nn
"""
此代码为传统CRNN网络结构代码
"""

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super().__init__()

        # -------------------- Basic CNN Backbone --------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, H/2, W/2]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 128, H/4, W/4]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # [B, 256, H/8, W/8]
        )

        # -------------------- Dimension Adaptation --------------------
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # Fix height to 1

        # -------------------- Bidirectional LSTM --------------------
        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=256,  # Match CNN output channels
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=False,
                bidirectional=True
            )
        )

        # -------------------- Output Layer --------------------
        self.fc = nn.Linear(hidden_size * 2, num_chars)  # Bidirectional LSTM outputs 2x hidden size

    def forward(self, x):
        # Feature extraction via CNN
        x = self.cnn(x)  # [B, 256, H/8, W/8]

        # Adaptive pooling to reduce height
        x = self.adaptive_pool(x)  # [B, 256, 1, W/8]
        x = x.squeeze(2)  # [B, 256, W/8]
        x = x.permute(2, 0, 1)  # [W/8, B, 256]

        # Sequence modeling with Bidirectional LSTM
        x, _ = self.rnn(x)  # [W/8, B, hidden_size * 2]

        # Character classification
        logits = self.fc(x)  # [W/8, B, num_chars]
        return logits


if __name__ == "__main__":
    # -------------------- Testing the Model --------------------
    dummy_input = torch.randn(4, 3, 32, 128)  # Batch size = 4, Image = 32x128 (H x W)
    model = BasicCRNN(num_chars=68)  # Assume 68 character classes

    print("Input shape:", dummy_input.shape)  # [4, 3, 32, 128]

    # Test CNN output
    cnn_out = model.cnn(dummy_input)
    print("CNN output shape:", cnn_out.shape)  # [4, 256, 4, 16]

    # Test Adaptive Pooling
    pooled = model.adaptive_pool(cnn_out)
    print("Pooled shape:", pooled.shape)  # [4, 256, 1, 16]

    # Test full forward pass
    logits = model(dummy_input)
    print("Final output shape:", logits.shape)  # [16, 4, 68] (Sequence length, Batch size, Character classes)