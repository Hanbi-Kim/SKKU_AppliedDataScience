import os
import gym
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1")
parser.add_argument('--load_model',type=str, default=None)
parser.add_argument("--save_path", default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--ent_coef', type=float, default=0.1)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=475)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')

args = parser.parse_args()
cuda = torch.device('cuda')

class Actor(nn.Module):
    def __init__(self, state_size, action_size, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        policies = torch.softmax(self.fc3(x), dim=1)
        return policies

class Critic(nn.Module):
    def __init__(self, state_size, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value



def train_model(actor, critic, actor_optimizer, critic_optimizer, transition, policies):
    state, action, reward, next_state, mask = transition

    #Update Critic
    criterion = torch.nn.MSELoss()
    value = critic(torch.Tensor(state).cuda()).squeeze(1)
    next_value = critic(torch.Tensor(next_state).cuda()).squeeze(1)
    target = reward + mask * args.gamma * next_value
    critic_loss = criterion(value, target.detach())
    #critic_loss = criterion(value, target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    #Update Actor
    categorical = Categorical(policies)
    log_policy = categorical.log_prob(torch.Tensor([action]).cuda())
    # entropy = categorical.entropy()
    advantage = target - value
    #actor_loss = -log_policy * advantage.item() + args.ent_coef * entropy
    actor_loss = -log_policy * advantage.item()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

def get_action(policies):
    categorical = Categorical(policies)
    action = categorical.sample()
    action = action.data.numpy()[0]
    return action

def main():
    env = gym.make(args.env_name)
    env.seed(100)
    torch.manual_seed(100)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    actor = Actor(state_size, action_size, args)
    critic = Critic(state_size, args)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)
    # writer = SummaryWriter(args.logdir)

    running_score = 0

    actor.cuda()
    critic.cuda()

    for episode in range(args.max_iter_num):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if args.render:
                env.render()

            policies = actor(torch.Tensor(state).cuda())
            action = get_action(policies.cpu())

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            mask = 0 if done else 1

            transition = [state, action, reward, next_state, mask]
            train_model(actor, critic, actor_optimizer, critic_optimizer,
                        transition, policies)

            state = next_state
            score += reward

        running_score = 0.99 * running_score + 0.01 * score

        if episode % args.log_interval == 0:
            print('{} episode | running_score: {:.2f}'.format(episode, running_score))

        if running_score > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            ckpt_path = args.save_path + 'model.pth.tar'
            torch.save(actor.state_dict(), ckpt_path)
            print('Running score exceeds 475, so ends')
            break

if __name__ == "__main__":
    main()


