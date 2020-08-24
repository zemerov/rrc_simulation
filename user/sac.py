import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from models import SoftQNetwork, PolicyNetwork
from replay_memory import BasicBuffer


def action_to_dict(action: np.array):
    dict_action = {'position': action[:9], 'torque': action[9:]}

    return dict_action


class SACAgent:
    def __init__(self,
                 env, gamma, tau, alpha, lr,
                 buffer_maxlen, obs_dim=None,
                 action_space_size=None, device='cpu'):
        q_lr = lr
        policy_lr = lr
        a_lr = lr

        self.device = torch.device(device)

        self.env = env
        self.action_range = [
            np.concatenate([env.action_space['position'].low, env.action_space['torque'].low]),
            np.concatenate([env.action_space['position'].high, env.action_space['torque'].high])
        ]

        if obs_dim is None:
            self.obs_dim = env.observation_space.shape[0]
        else:
            self.obs_dim = obs_dim

        if action_space_size is None:
            action_space_size = env.action_space.shape[0]

        self.action_dim = action_space_size  # env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2

        # initialize networks
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(action_space_size).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        action = self.rescale_action(action)

        return action

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
               (self.action_range[1] + self.action_range[0]) / 2.0

    def update(self, buffer, batch_size):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_actions = next_actions.cuda() #to(self.device)
        next_log_pi = next_log_pi.cuda()  # to(self.device)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1

    def save(self, path):
        torch.save(self.policy_net, path + 'policy_net')
        torch.save(self.q_net1, path + 'q_net1')
        torch.save(self.q_net2, path + 'q_net2')
        torch.save(self.target_q_net1, path + 'target_q_net1')
        torch.save(self.target_q_net2, path + 'target_q_net2')

    def load(self, path):
        self.policy_net = torch.load(path + 'policy_net')
        self.q_net1 = torch.load(path + 'q_net1')
        self.q_net2 = torch.load(path + 'q_net2')
        self.target_q_net1 = torch.load(path + 'target_q_net1')
        self.target_q_net2 = torch.load(path + 'target_q_net2')
