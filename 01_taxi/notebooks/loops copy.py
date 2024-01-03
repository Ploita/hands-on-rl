from typing import Tuple, List, Any
import random
from pdb import set_trace as stop
from multiprocessing import Process

import numpy as np
from tqdm import tqdm


def train(
    agent,
    env,
    n_episodes: int,
    epsilon_min: float,
    epsilon_decay: float
) -> Tuple[Any, List, List]:
    """
    Trains and agent and returns 3 things:
    - agent object
    - timesteps_per_episode
    - penalties_per_episode
    """
    # For plotting metrics
    timesteps_per_episode = []
    penalties_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()[0]
        epsilon = 1
        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:

            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                else:
                    epsilon = epsilon_min
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.update_parameters(state, action, reward, next_state)

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)

    return agent, timesteps_per_episode, penalties_per_episode


def evaluate(
    agent,
    env,
    n_episodes: int,
    epsilon: float,
    initial_state: int = None
) -> Tuple[List, List, List]:
    """
    Tests agent performance in random `n_episodes`.

    It returns:
    - timesteps_per_episode
    - penalties_per_episode
    """
    # For plotting metrics
    timesteps_per_episode = []
    penalties_per_episode = []
    frames_per_episode = []

    for i in tqdm(range(0, n_episodes)):

        if initial_state:
            # init the environment at 'initial_state'
            state = initial_state
            env.s = initial_state
        else:
            # random starting state
            state = env.reset()[0]

        epochs, penalties, reward, = 0, 0, 0
        frames = []
        done = False

        while not done:

            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values
                action = agent.get_action(state)

            next_state, reward, done, info = env.step(action)

            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        timesteps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)
        frames_per_episode.append(frames)

    return timesteps_per_episode, penalties_per_episode, frames_per_episode


def train_many_runs(
    agent,
    env,
    n_episodes: int,
    epsilon: float,
    n_runs: int,
) -> Tuple[List, List]:
    """
    Calls 'train' many times, stores results and averages them out.
    """
    timesteps = np.zeros(shape=(n_runs, n_episodes))
    penalties = np.zeros(shape=(n_runs, n_episodes))

    for i in range(0, n_runs):

        agent.reset()

        _, timesteps[i, :], penalties[i, :] = train(
            agent, env, n_episodes, epsilon_min=0.1, epsilon_decay = 0.99,
        )
    timesteps = np.mean(timesteps, axis=0).tolist()
    penalties = np.mean(penalties, axis=0).tolist()

    return timesteps, penalties

if __name__ == '__main__':

    import gym
    from q_agent import QAgent

    env = gym.make("Taxi-v3")
    alpha = 0.1
    gamma = 0.6
    agent = QAgent(env, alpha, gamma)

    agent, _, _ = train(
        agent, env, n_episodes=10000, epsilon_min=0.10, epsilon_decay= 0.99)

    timesteps_per_episode, penalties_per_episode, _ = evaluate(
        agent, env, n_episodes=100, epsilon=0.05
    )

    print(f'Avg steps to complete ride: {np.array(timesteps_per_episode).mean()}')
    print(f'Avg penalties to complete ride: {np.array(penalties_per_episode).mean()}')