"""一个从零实现的强化学习（Q-Learning）示例。

环境：4x4 GridWorld
目标：从起点 S 走到终点 G，避开陷阱 H。
"""

from __future__ import annotations

import random
from dataclasses import dataclass


Action = int
State = int


@dataclass
class GridWorld:
    size: int = 4

    def __post_init__(self) -> None:
        self.start_state: State = 0
        self.goal_state: State = self.size * self.size - 1
        # 固定几个陷阱位置（可自行修改）
        self.holes = {5, 7, 11, 12}
        self.state: State = self.start_state

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        # 0: 上, 1: 下, 2: 左, 3: 右
        return 4

    def reset(self) -> State:
        self.state = self.start_state
        return self.state

    def step(self, action: Action) -> tuple[State, float, bool]:
        row, col = divmod(self.state, self.size)

        if action == 0:  # up
            row = max(row - 1, 0)
        elif action == 1:  # down
            row = min(row + 1, self.size - 1)
        elif action == 2:  # left
            col = max(col - 1, 0)
        elif action == 3:  # right
            col = min(col + 1, self.size - 1)
        else:
            raise ValueError(f"未知动作: {action}")

        next_state = row * self.size + col
        self.state = next_state

        # 奖励设计：到达目标 +1；掉入陷阱 -1；其余每步 -0.01
        if next_state == self.goal_state:
            return next_state, 1.0, True
        if next_state in self.holes:
            return next_state, -1.0, True
        return next_state, -0.01, False


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def choose_action(self, state: State) -> Action:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(max(range(self.n_actions), key=lambda a: self.q_table[state][a]))

    def update(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else max(self.q_table[next_state])
        target = reward + self.gamma * max_next_q
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes: int = 1500, max_steps: int = 100, seed: int = 42) -> tuple[QLearningAgent, list[float]]:
    random.seed(seed)
    env = GridWorld(size=4)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    episode_rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

    return agent, episode_rewards


def evaluate(agent: QLearningAgent, episodes: int = 100, max_steps: int = 100, seed: int = 7) -> float:
    random.seed(seed)
    env = GridWorld(size=4)

    success = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 纯贪心策略评估

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.choose_action(state)
            state, _, done = env.step(action)
            if done:
                if state == env.goal_state:
                    success += 1
                break

    agent.epsilon = original_epsilon
    return success / episodes


def print_policy(agent: QLearningAgent, size: int = 4) -> None:
    arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    env = GridWorld(size=size)

    print("\n学习到的策略（S=起点, G=终点, H=陷阱）:")
    for s in range(env.n_states):
        if s == env.start_state:
            cell = "S"
        elif s == env.goal_state:
            cell = "G"
        elif s in env.holes:
            cell = "H"
        else:
            best_action = int(max(range(env.n_actions), key=lambda a: agent.q_table[s][a]))
            cell = arrows[best_action]

        end = "\n" if (s + 1) % size == 0 else " "
        print(f"{cell:>2}", end=end)


def main() -> None:
    agent, rewards = train()
    success_rate = evaluate(agent)

    print(f"训练完成，总回合数: {len(rewards)}")
    print(f"最后 100 回合平均奖励: {sum(rewards[-100:]) / 100:.3f}")
    print(f"评估成功率: {success_rate * 100:.1f}%")

    print_policy(agent)


if __name__ == "__main__":
    main()
