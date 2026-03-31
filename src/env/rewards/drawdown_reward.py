from __future__ import annotations

from env.trading_env_baseline import MyTradingEnv
from env.trading_env_lstm import MyTradingEnvLSTM


class DrawdownAwareReward(MyTradingEnv):
    def __init__(self, *args, penalty_lambda: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.penalty_lambda = float(penalty_lambda)

    def _calculate_reward(self, done: bool) -> float:
        profit = self.portfolio_value - self.prev_portfolio_value
        return float(profit - self.penalty_lambda * self.max_drawdown)


class DrawdownAwareRewardLSTM(MyTradingEnvLSTM):
    def __init__(self, *args, penalty_lambda: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.penalty_lambda = float(penalty_lambda)

    def _calculate_reward(self, done: bool) -> float:
        profit = self.portfolio_value - self.prev_portfolio_value
        return float(profit - self.penalty_lambda * self.max_drawdown)
