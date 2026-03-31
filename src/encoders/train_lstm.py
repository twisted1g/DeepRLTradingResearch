from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoders.lstm_pretrain import LSTMPretrainConfig, train_lstm_encoder


def main() -> None:
    data_path = "data/raw/binance_BTCUSDT_1h_2018-2021.csv"
    output_path = "lstm_encoder.pt"

    config = LSTMPretrainConfig(
        lstm_window_size=128,
        lstm_hidden_size=64,
        lstm_layers=2,
        feature_window=20,
        epochs=10,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        print_every=50,
    )

    df = pd.read_csv(data_path)
    train_lstm_encoder(
        df,
        save_path=output_path,
        config=config,
    )


if __name__ == "__main__":
    main()
