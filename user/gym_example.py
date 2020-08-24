#!/usr/bin/env python3
"""Demo on how to run the simulation using the Gym environment

This demo creates a CubeEnv environment and runs one episode with random
initialization using a dummy policy which uses random actions.
"""
import gym
import numpy as np

from rrc_simulation.gym_wrapper.envs import cube_env


class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def get_arr_observation(observation):
    values = []

    for i in range(len(observation)):
        values.append(np.concatenate(list(list(observation.values())[i].values())))

    return np.concatenate(values)


def main():
    # Use a random initializer with difficulty 1
    initializer = cube_env.RandomInitializer(difficulty=1)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.TORQUE_AND_POSITION,
        frameskip=100,
        visualization=False
    )

    #print(env.action_space)

    policy = RandomPolicy(env.action_space)

    observation = env.reset()
    is_done = False
    reward = 0

    while not is_done:
        action = policy.predict(observation)
        observation, reward, is_done, info = env.step(action)
        is_done =True

    print('=' * 100)
    print(info, is_done)


if __name__ == "__main__":
    main()
