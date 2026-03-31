from __future__ import annotations

import numpy as np

from env.trading_env_baseline import MyTradingEnv
from env.trading_env_lstm import MyTradingEnvLSTM


class SharpeReward(MyTradingEnv):
    def __init__(self, *args, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = float(eps)

    def _calculate_reward(self, done: bool) -> float:
        history = list(self.portfolio_history) + [float(self.portfolio_value)]
        if len(history) < 3:
            return 0.0
        returns = []
        for i in range(1, len(history)):
            prev = history[i - 1]
            cur = history[i]
            if prev <= 0.0:
                returns.append(0.0)
            else:
                returns.append((cur / prev) - 1.0)
        if len(returns) < 2:
            return 0.0
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))
        if std_return <= self.eps:
            return 0.0
        return mean_return / std_return


class SharpeRewardLSTM(MyTradingEnvLSTM):
    def __init__(self, *args, eps: float = 1e-12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.eps = float(eps)

    def _calculate_reward(self, done: bool) -> float:
        history = list(self.portfolio_history) + [float(self.portfolio_value)]
        if len(history) < 3:
            return 0.0
        returns = []
        for i in range(1, len(history)):
            prev = history[i - 1]
            cur = history[i]
            if prev <= 0.0:
                returns.append(0.0)
            else:
                returns.append((cur / prev) - 1.0)
        if len(returns) < 2:
            return 0.0
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1))
        if std_return <= self.eps:
            return 0.0
        return mean_return / std_return
