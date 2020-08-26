import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import json

import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env

from sac import SAC
from utils import action_to_dict
from replay_memory import ReplayMemory

from utils import get_arr_observation


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
    args = parser.parse_args()

    # Environment
    initializer = cube_env.RandomInitializer(difficulty=args.difficulty)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.TORQUE_AND_POSITION,
        frameskip=100,
        visualization=False,
    )

    obs_dim = 41
    agent = SAC(obs_dim, 9 + 9, args, env)

    if args.load_path is not None:
        agent.load_model(args.load_path + 'actor', args.load_path + 'critic')  # Load model params

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir='logs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_success = 0
        episode_steps = 0
        done = False
        state = get_arr_observation(env.reset())

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
                action = np.concatenate([action['position'], action['torque']])
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):  # Number of updates per step in environment
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    if args.tensorboard:
                        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar('loss/policy', policy_loss, updates)
                        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action_to_dict(action))
            next_state = get_arr_observation(next_state)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            memory.push(state, action, reward, next_state, done)  # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        if args.tensorboard:
            writer.add_scalar('reward/train', episode_reward, total_numsteps)
        if i_episode % 20 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, success: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), episode_success))

        if i_episode % 1 == 0:
            avg_reward = 0.
            avg_success = 0.
            episodes = 10
            for _ in range(episodes):
                env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = env.step(action_to_dict(action))
                    next_state = get_arr_observation(next_state)

                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward

            avg_reward /= episodes

            if args.tensorboard:
                writer.add_scalar('reward/test', avg_reward, total_numsteps)

            if args.save:
                agent.save_model('CubeEnv_diff_' + str(args.difficulty))

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()
