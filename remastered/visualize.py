import gym
import torch
import argparse
import numpy as np


import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env

from sac import GaussianPolicy
from utils import get_arr_observation
from utils import policy_factory
from sac import SAC


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')
    parser.add_argument('--save', action="store_true",
                        help='save model parameters (default: False)')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--path', default='sac')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--hidden_size', type=int, default='256')
    parser.add_argument('--difficulty', type=int, default='1')
    parser.add_argument('--model', default='models/DKittyWalkFixed')
    args = parser.parse_args()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='models/DKittyWalkFixed')
    # parser.add_argument('--difficulty', type=int, default='1')
    # args = parser.parse_args()

    initializer = cube_env.RandomInitializer(difficulty=args.difficulty)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.TORQUE_AND_POSITION,
        frameskip=100,
        visualization=False,
    )

    agent = SAC(41, 9 + 9, args, env)
    agent.load_model('models/sac_actor_CubeEnv_diff_1_', 'models/sac_critic_CubeEnv_diff_1_')

    policy = policy_factory(args.model, env, 41, 18)

    is_done = False
    observation = get_arr_observation(env.reset())
    sum_reward = 0

    while not is_done:
        # print(observation)
        action = policy(get_arr_observation(observation))
        observation, _, is_done, reward = env.step(action)

        sum_reward += reward

    print(sum_reward)
