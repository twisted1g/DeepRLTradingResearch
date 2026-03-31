from __future__ import annotations

from env.trading_env_baseline import MyTradingEnv
from env.trading_env_lstm import MyTradingEnvLSTM


class ReturnReward(MyTradingEnv):
    def _calculate_reward(self, done: bool) -> float:
        if self.prev_portfolio_value <= 0.0:
            return 0.0
        return (
            self.portfolio_value - self.prev_portfolio_value
        ) / self.prev_portfolio_value


class ReturnRewardLSTM(MyTradingEnvLSTM):
    def _calculate_reward(self, done: bool) -> float:
        if self.prev_portfolio_value <= 0.0:
            return 0.0
        return (
            self.portfolio_value - self.prev_portfolio_value
        ) / self.prev_portfolio_value
