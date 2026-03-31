from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LSTMPretrainConfig:
    lstm_window_size: int = 128
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    feature_window: int = 20
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    print_every: int = 50


def _build_feature_matrix(df: pd.DataFrame, feature_window: int) -> np.ndarray:
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")
    if "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")

    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    n = len(df)

    log_return = np.zeros(n, dtype=np.float32)
    prev = close[:-1]
    curr = close[1:]
    valid = prev > 0
    log_return[1:][valid] = np.log(curr[valid] / prev[valid]).astype(np.float32)

    rolling_vol = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(1, i - feature_window + 1)
        window = log_return[start : i + 1]
        rolling_vol[i] = float(np.std(window)) if window.size > 1 else 0.0

    volume_norm = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - feature_window + 1)
        window = volume[start : i + 1]
        mean = float(window.mean()) if window.size > 0 else 0.0
        volume_norm[i] = float(window[-1] / mean) if mean > 0 else 0.0

    position = np.zeros(n, dtype=np.float32)
    features = np.stack([log_return, rolling_vol, volume_norm, position], axis=1)
    return features.astype(np.float32)


class _LSTMReturnDataset(Dataset):
    def __init__(self, features: np.ndarray, window_size: int):
        self.features = features
        self.window_size = int(window_size)

        if len(self.features) < self.window_size + 1:
            raise ValueError("Not enough rows to build LSTM windows")

        self.max_index = len(self.features) - 2

    def __len__(self) -> int:
        return self.max_index - (self.window_size - 1) + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = idx + self.window_size - 1
        start = end - self.window_size + 1
        window = self.features[start : end + 1]
        target = self.features[end + 1, 0]
        x = torch.from_numpy(window).float()
        y = torch.tensor([target], dtype=torch.float32)
        return x, y


def train_lstm_encoder(
    df: pd.DataFrame,
    save_path: str,
    config: Optional[LSTMPretrainConfig] = None,
) -> dict:
    if config is None:
        config = LSTMPretrainConfig()

    features = _build_feature_matrix(df, config.feature_window)
    dataset = _LSTMReturnDataset(features, config.lstm_window_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    device = torch.device(config.device)
    encoder = nn.LSTM(
        input_size=4,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_layers,
        batch_first=True,
    ).to(device)
    head = nn.Linear(config.lstm_hidden_size, 1).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    encoder.train()
    head.train()

    global_step = 0

    for epoch in range(config.epochs):
        epoch_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, (h_n, _) = encoder(batch_x)
            h_t = h_n[-1]
            pred = head(h_t)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

            if global_step % max(1, config.print_every) == 0:
                print(f"[epoch {epoch + 1}/{config.epochs}] step {global_step} loss {loss.item():.6f}")
            global_step += 1
        if epoch_losses:
            print(f"[epoch {epoch + 1}/{config.epochs}] mean_loss {float(np.mean(epoch_losses)):.6f}")

    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": head.state_dict(),
        "config": {
            "lstm_window_size": config.lstm_window_size,
            "lstm_hidden_size": config.lstm_hidden_size,
            "lstm_layers": config.lstm_layers,
            "feature_window": config.feature_window,
        },
    }

    torch.save(checkpoint, save_path)
    return checkpoint


def load_lstm_encoder(
    checkpoint_path: str,
    device: str = "cpu",
) -> nn.LSTM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})

    encoder = nn.LSTM(
        input_size=4,
        hidden_size=int(cfg.get("lstm_hidden_size", 64)),
        num_layers=int(cfg.get("lstm_layers", 2)),
        batch_first=True,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder
