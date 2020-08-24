import robel
import gym
from robel.scripts import rollout
import torch
import argparse
import numpy as np

from sac import GaussianPolicy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/DKittyWalkFixed')
    parser.add_argument('--env', default='DKittyWalkFixed-v0')
    args = parser.parse_args()

    env = gym.make(args.env)#, torso_tracker_id=1)
    print('observation_space', env.observation_space)
    print('action_space', env.action_space)

    env.reset()

    def policy_factory(model_path):
        p = GaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            256,
            env.action_space
        ).to('cpu')
        p.load_state_dict(torch.load(model_path))

        def policy(obs):
            state = torch.FloatTensor(obs).unsqueeze(0)
            _, _, action = p.sample(state)
            return action.detach().cpu().numpy()[0]

        return policy

    for traj in rollout.do_rollouts(
        env,
        num_episodes=20,
        max_episode_length=100,
        action_fn=policy_factory(model_path=args.model),
        render_mode='human'
    ):
        env.seed(np.random.randint(0, 3))
        print(traj.infos['score/success'])

