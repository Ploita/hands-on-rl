import os
from typing import List, Optional, Tuple
from pathlib import Path
import json
from pdb import set_trace as stop

import tqdm
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import gym

from model_factory import get_model
from utils import (
    get_agent_id,
    set_seed,
    get_num_model_parameters,
    get_logger, get_model_path
)
from config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR

def reward_to_go(rews):

    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class VPGAgent:

    def __init__(
        self,
        env_name: str = 'LunarLander-v2',
        learning_rate: float = 3e-4,
        hidden_layers: List[int] = [32],
        gradient_weights: str = 'rewards'
    ):
        assert gradient_weights in {'rewards', 'rewards-to-go'}

        self.env_name = env_name
        self.env = gym.make(env_name, render_mode = 'rgb_array')
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        # stochastic policy network
        # the outputs of this network are un-normalized probabilities for each
        # action (aka logits)
        self.policy_net = get_model(input_dim=self.obs_dim,
                                    output_dim=self.act_dim,
                                    hidden_layers=hidden_layers)
        print(f'Policy network with {get_num_model_parameters(self.policy_net):,} parameters')
        print(self.policy_net)


        self.optimizer = Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gradient_weights = gradient_weights

        self.hparams = {
            'learning_rate': learning_rate,
            'hidden_layers': hidden_layers,
            'gradient_weights': gradient_weights,

        }

    def act(self, obs: torch.Tensor):
        """
        Action selection function (outputs int actions, sampled from policy)
        """
        return self._get_policy(obs).sample().item()

    def train(
        self,
        n_policy_updates: int = 1000,
        batch_size: int = 4000,
        logger: Optional[SummaryWriter] = None,
        model_path: Optional[Path] = None,
        seed: Optional[int] = 0,
        freq_eval: Optional[int] = 10,
    ):
        """
        """
        save_model = True if model_path is not None else False

        best_avg_reward = -np.inf

        # fix seeds to ensure reproducible training runs
        set_seed(self.env, seed)
        outter = tqdm.tqdm(range(n_policy_updates), desc="Training", unit="epoch", position=0, leave = True)


        for i in outter:

            # use current policy to collect trajectories
            states, actions, weights, rewards = self._collect_trajectories(n_samples=batch_size)

            # one step of gradient ascent to update policy parameters
            loss = self._update_parameters(states, actions, weights)

            # log epoch metrics
            '''print('epoch: %3d \t loss: %.3f \t reward: %.3f' % (i, loss, np.mean(rewards)))
            if logger is not None:
                # we use total_steps instead of epoch to render all plots in Tensorboard comparable
                # Agents wit different batch_size (aka steps_per_epoch) are fairly compared this way.
                total_steps += batch_size
                logger.add_scalar('train/loss', loss, total_steps)
                logger.add_scalar('train/episode_reward', np.mean(rewards), total_steps)'''
            
            info_str = f"epoch: {i:3d} \t loss: {loss:.3f} \t reward: {np.mean(rewards):.3f}"
            info_str = info_str.ljust(80)
            outter.set_postfix_str(info_str, refresh=True)


            # evaluate the agent on a fixed set of 100 episodes
            if (i + 1) % freq_eval == 0:
                rewards, success = self.evaluate(n_episodes=100)

                avg_reward = np.mean(rewards)
                avg_success_rate = np.mean(success)
                if save_model and (avg_reward > best_avg_reward):
                    self.save_to_disk(model_path)
                    print(f'New best model! Average reward = {avg_reward:.2f}, Success rate = {avg_success_rate:.2%}')
                    best_avg_reward = avg_reward

    def evaluate(self, n_episodes: int = 100, seed: Optional[int] = 1234) -> Tuple[List[float], List[float]]:
        """
        """
        # output metrics
        reward_per_episode = []
        success_per_episode = []

        # fix seed
        #self.env.seed(seed)
        self.env.action_space.seed(seed)

        
        for _ in range(0, n_episodes):

            state = self.env.reset()[0]
            rewards = 0
            done = False
            reward = None
            while not done:

                action = self.act(torch.as_tensor(state, dtype=torch.float32))

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                rewards += reward

                state = next_state

            reward_per_episode.append(rewards)
            success_per_episode.append(1 if reward > 0 else 0)

        return reward_per_episode, success_per_episode

    def _collect_trajectories(self, n_samples: int):

        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = self.env.reset()[0]       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            # act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            action = self.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(action)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                if self.gradient_weights == 'rewards':
                    # the weight for each logprob(a|s) is the total reward for the episode
                    batch_weights += [ep_ret] * ep_len
                elif self.gradient_weights == 'rewards-to-go':
                    # the weight for each logprob(a|s) is the total reward AFTER the action is taken
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    raise NotImplemented

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset()[0], False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > n_samples:
                    break

        return batch_obs, batch_acts, batch_weights, batch_rets

    def _update_parameters(self, states, actions, weights) -> float:
        """
        One step of policy gradient update
        """
        self.optimizer.zero_grad()

        loss = self._compute_loss(
            obs=torch.as_tensor(states, dtype=torch.float32),
            act=torch.as_tensor(actions, dtype=torch.int32),
            weights=torch.as_tensor(weights, dtype=torch.float32)
        )

        # compute gradients
        loss.backward()

        # update parameters with Adam
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, obs, act, weights):
        logp = self._get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def _get_policy(self, obs):
        """
        Get action distribution given the current policy
        """
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    @classmethod
    def load_from_disk(cls, env_name: str, path: Path):
        """
        We recover all necessary variables to be able to evaluate the agent.

        NOTE: training state is not stored, so it is not possible to resume
        an interrupted training run as it was.
        """
        # load hyper-params
        with open(path / 'hparams.json', 'r') as f:
            hparams = json.load(f)

        # generate Python object
        agent = cls(env_name, **hparams)

        agent.policy_net = torch.load(path / 'model')
        agent.policy_net.eval()

        return agent

    def save_to_disk(self, path: Path) -> None:
        """"""
        if not path.exists():
            os.makedirs(path)

        # save hyper-parameters in a json file
        with open(path / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f)

        # save main model
        torch.save(self.policy_net, path / 'model')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--n_policy_updates', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gradient_weights', type=str, default='rewards')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument("--hidden_layers", type=int, nargs="+",)
    parser.add_argument("--freq_eval", type=int)
    args = parser.parse_args()

    vpg_agent = VPGAgent(
        env_name=args.env,
        gradient_weights=args.gradient_weights,
        learning_rate=args.lr,
        hidden_layers=args.hidden_layers,
    )

    # generate a unique agent_id, that we later use to save results to disk, as
    # well as TensorBoard logs.
    agent_id = get_agent_id(args.env)
    print(f'agent_id = {agent_id}')

    # tensorboard logger to see training curves
    logger = get_logger(env_name=args.env, agent_id=agent_id)

    # path to save policy network weights
    model_path = get_model_path(env_name=args.env, agent_id=agent_id)

    # start training
    vpg_agent.train(
        n_policy_updates=args.n_policy_updates,
        batch_size=args.batch_size,
        logger=logger,
        model_path=model_path,
        freq_eval=args.freq_eval,
    )