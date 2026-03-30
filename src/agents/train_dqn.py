from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


@dataclass
class DQNConfig:
    total_timesteps: int = 50_000
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    seed: int = 42
    verbose: int = 1


def train_dqn(env, config: DQNConfig | None = None) -> Dict[str, object]:
    if config is None:
        config = DQNConfig()

    vec_env = DummyVecEnv([lambda: env])
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        seed=config.seed,
        verbose=config.verbose,
    )
    model.learn(total_timesteps=config.total_timesteps)

    return {
        "model": model,
        "config": config,
    }
