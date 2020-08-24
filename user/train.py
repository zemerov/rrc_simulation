import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import json

import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env


from sac import SACAgent
from sac import action_to_dict
from replay_memory import BasicBuffer


def get_arr_observation(observation):
    values = []

    for i in range(len(observation)):
        values.append(np.concatenate(list(list(observation.values())[i].values())))

    return np.concatenate(values)


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
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
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
    args = parser.parse_args()

    # Environment
    initializer = cube_env.RandomInitializer(difficulty=1)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.TORQUE_AND_POSITION,
        frameskip=100,
        visualization=False,
    )

    #cprint(np.concatenate([env.action_space['position'].high, env.action_space['torque'].high]))
    # print(env.action_space['torque'].high, env.action_space['torque'].low)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    obs_dim = 41
    agent = SACAgent(
        env, args.gamma, args.tau, args.alpha, args.lr, args.replay_size, obs_dim, 18, args.device
    )

    if args.path is not None:
        agent.load('models/' + args.load_path)  # Load model params

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir='logs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                                args.policy, "autotune" if args.automatic_entropy_tuning else ""))

    # Memory
    memory = BasicBuffer(args.replay_size)

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
                action = agent.get_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):  # Number of updates per step in environment
                    # Update parameters of all the networks
                    agent.update(memory, args.batch_size)
                    #critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    if args.tensorboard:
                        pass
                    #writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    #writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    #writer.add_scalar('loss/policy', policy_loss, updates)
                    #writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    #writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, info = env.step(action_to_dict(action))
            next_state = get_arr_observation(next_state)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            # mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, done)  # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        if args.tensorboard:
            writer.add_scalar('reward/train', episode_reward, total_numsteps)
        # writer.add_scalar('success/train', episode_success, total_numsteps)
        if i_episode % 20 == 0:
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, success: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), episode_success))

        if i_episode % 100 == 0:
            avg_reward = 0.
            avg_success = 0.
            episodes = 10
            for _ in range(episodes):
                env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action_to_dict(action))
                    next_state = get_arr_observation(next_state)

                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward

            avg_reward /= episodes

            if args.tensorboard:
                writer.add_scalar('reward/test', avg_reward, total_numsteps)

            if args.save:
                print('Save model to models')
                agent.save('models/' + args.path)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    env.close()
