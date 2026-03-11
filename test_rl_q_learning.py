"""一个可直接运行的强化学习测试示例（Q-learning）。

运行：
    python test_rl_q_learning.py
"""

from __future__ import annotations

import random
import unittest
from dataclasses import dataclass
from typing import List


@dataclass
class ChainEnv:
    """一个简化的确定性链式环境。

    状态: 0 -> 1 -> 2(终点)
    动作:
      0: 前进 (在 0/1 状态前进一格)
      1: 停留 (原地不动)

    奖励:
      到达终点状态 2 时奖励 +1，其余为 0。
    """

    state: int = 0
    goal_state: int = 2

    def reset(self) -> int:
        self.state = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        if action == 0 and self.state < self.goal_state:
            self.state += 1
        elif action == 1:
            self.state = self.state
        else:
            raise ValueError(f"不支持的动作: {action}")

        done = self.state == self.goal_state
        reward = 1.0 if done else 0.0
        return self.state, reward, done



def epsilon_greedy(q_table: List[List[float]], state: int, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, len(q_table[state]) - 1)
    return max(range(len(q_table[state])), key=lambda a: q_table[state][a])



def q_learning(
    env: ChainEnv,
    episodes: int = 200,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    seed: int = 42,
) -> List[List[float]]:
    random.seed(seed)
    n_states, n_actions = 3, 2
    q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    for _ in range(episodes):
        s = env.reset()
        done = False
        steps = 0

        while not done and steps < 10:
            a = epsilon_greedy(q, s, epsilon)
            s_next, r, done = env.step(a)
            best_next = max(q[s_next])
            td_target = r + gamma * best_next * (0 if done else 1)
            td_error = td_target - q[s][a]
            q[s][a] += alpha * td_error
            s = s_next
            steps += 1

    return q


class TestQLearning(unittest.TestCase):
    def test_learn_optimal_policy(self) -> None:
        env = ChainEnv()
        q = q_learning(env)

        # 在状态 0 和 1，最优动作都应为“前进”(0)
        self.assertGreater(q[0][0], q[0][1])
        self.assertGreater(q[1][0], q[1][1])

        # 状态 1 的前进动作价值接近 1
        self.assertGreater(q[1][0], 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
