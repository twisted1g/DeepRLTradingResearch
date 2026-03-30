from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

SRC_DIR = Path(__file__).resolve().parents[1]
if SRC_DIR.as_posix() not in sys.path:
    sys.path.append(SRC_DIR.as_posix())

from env.trading_env import MyTradingEnv


CSV_PATH = Path("data/raw/binance_BTCUSDT_1h_2020.csv")
ALGO = "dqn"
TIMESTEPS = 50_000
SEED = 42
SAVE_DIR = Path("models")


INITIAL_BALANCE = 1000.0
WINDOW_SIZE = 10
COMMISSION = 0.0001
SLIPPAGE = 0.0005
MAX_HOLDING_TIME = 72
MAX_DRAWDOWN_THRESHOLD = 0.08
MAX_STEPS: Optional[int] = None


def _load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "close" not in df.columns:
        raise ValueError("CSV must contain 'close' column")
    return df


def _make_env(
    df: pd.DataFrame,
    initial_balance: float,
    window_size: int,
    commission: float,
    slippage: float,
    max_holding_time: int,
    max_drawdown_threshold: float,
    max_steps: Optional[int],
):
    def _factory():
        return MyTradingEnv(
            df=df,
            initial_balance=initial_balance,
            window_size=window_size,
            commission=commission,
            slippage=slippage,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
            max_steps=max_steps,
        )

    return DummyVecEnv([_factory])


def train() -> None:
    df = _load_df(CSV_PATH)
    env = _make_env(
        df=df,
        initial_balance=INITIAL_BALANCE,
        window_size=WINDOW_SIZE,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        max_holding_time=MAX_HOLDING_TIME,
        max_drawdown_threshold=MAX_DRAWDOWN_THRESHOLD,
        max_steps=MAX_STEPS,
    )

    algo = ALGO.lower()
    if algo == "dqn":
        model = DQN("MlpPolicy", env, verbose=1, seed=SEED)
    elif algo == "a2c":
        model = A2C("MlpPolicy", env, verbose=1, seed=SEED)
    else:
        raise ValueError("ALGO must be 'dqn' or 'a2c'")

    model.learn(total_timesteps=TIMESTEPS)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVE_DIR / f"{algo}_trading_env"
    model.save(model_path.as_posix())
    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    train()
