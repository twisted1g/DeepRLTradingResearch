from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from pathlib import Path

from .trading_env_baseline import MyTradingEnv


class MyTradingEnvLSTM(MyTradingEnv):
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1000.0,
        window_size: int = 10,
        commission: float = 0.0001,
        slippage: float = 0.0005,
        max_holding_time: int = 72,
        max_drawdown_threshold: float = 0.08,
        max_steps: Optional[int] = None,
        lstm_window_size: int = 128,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        lstm_encoder: Optional[torch.nn.LSTM] = None,
        lstm_checkpoint_path: Optional[str] = None,
        lstm_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            commission=commission,
            slippage=slippage,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
            max_steps=max_steps,
            **kwargs,
        )

        self.lstm_window_size = int(lstm_window_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_layers = int(lstm_layers)
        self.lstm_device = str(lstm_device)

        if lstm_encoder is None and lstm_checkpoint_path is not None:
            from encoders.lstm_pretrain import load_lstm_encoder

            checkpoint_path = Path(lstm_checkpoint_path)
            if not checkpoint_path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                checkpoint_path = project_root / checkpoint_path

            self.lstm_encoder = load_lstm_encoder(
                str(checkpoint_path),
                device=self.lstm_device,
            )
        elif lstm_encoder is None:
            self.lstm_encoder = torch.nn.LSTM(
                input_size=4,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
            )
        else:
            self.lstm_encoder = lstm_encoder

        self.lstm_encoder.to(self.lstm_device)
        self.lstm_encoder.eval()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lstm_hidden_size,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        end = int(self.current_step)
        start = end - self.lstm_window_size + 1
        if start < 0:
            pad_len = -start
            start = 0
        else:
            pad_len = 0

        window_features = []
        if pad_len > 0:
            window_features.append(np.zeros((pad_len, 4), dtype=np.float32))

        for idx in range(start, end + 1):
            window_features.append(self._get_feature_vector_at(idx)[None, :])

        window = np.concatenate(window_features, axis=0)
        if window.shape[0] > self.lstm_window_size:
            window = window[-self.lstm_window_size :]

        seq = torch.as_tensor(
            window, dtype=torch.float32, device=self.lstm_device
        ).unsqueeze(0)
        with torch.no_grad():
            _, (h_n, _) = self.lstm_encoder(seq)
        h_t = h_n[-1, 0].detach().cpu().numpy().astype(np.float32)
        return h_t

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        if self.max_steps is None:
            start_max = len(self.df) - 1
        else:
            start_max = len(self.df) - self.max_steps

        min_start = max(1, self.lstm_window_size)
        self.current_step = np.random.randint(min_start, start_max)

        self.position = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.cash = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.position_value = 0.0
        self.current_holding_time = 0
        self.max_drawdown = 0.0
        self.trade_history = []
        self._steps_elapsed = 0

        self.prev_portfolio_value = float(self.initial_balance)
        self.last_exit_reason = None
        self.portfolio_history = [float(self.portfolio_value)]
        self.episode_id += 1

        obs = self._get_observation()
        return obs, {}
