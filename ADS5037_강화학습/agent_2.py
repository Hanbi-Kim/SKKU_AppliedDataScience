import gym
import collections
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import collections

def get_demo_traj():
    return np.load("demo_traj_2.npy", allow_pickle=True)

# Hyperparameters
learning_rate = 0.001
gamma = 0.98
buffer_limit = 50000
batch_size = 4


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def put_pretrain(self, transition):
        self.buffer.appendleft(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()

class DQfDAgent(object):
    def __init__(self, env, use_per, n_episode):
        self.n_EPISODES = n_episode
        self.env = env
        self.use_per = use_per

    def pretrain(self, q, q_target, memory, optimizer):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.mse_loss(q_a, target)
        je_loss = sum(q_out.max(1)[0].unsqueeze(1) + abs(q_out.max(1)[1].unsqueeze(1) - a) - q_a)
        loss = loss + 0.1 * je_loss[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    def train(self):
        elist = []
        rlist = []
        ###### 1. DO NOT MODIFY FOR TESTING ######
        test_mean_episode_reward = deque(maxlen=20)
        test_over_reward = False
        test_min_episode = np.inf
        ###### 1. DO NOT MODIFY FOR TESTING ######

        env = self.env

        q = Qnet()
        q_target = Qnet()
        q_target.load_state_dict(q.state_dict())
        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        optimizer1 = optim.Adam(q.parameters(), lr=learning_rate)

        memory = ReplayBuffer()
        demon = get_demo_traj()

        for i in range(len(demon)):
            for j in range(len(demon[i])):
                done_mask = 0.0 if demon[i][j][4] else 1.0
                memory.put_pretrain((demon[i][j][0], demon[i][j][1],
                                     demon[i][j][2], demon[i][j][3], done_mask ))

        # Do pretrain
        for k in range(1000):
            DQfDAgent.pretrain(self, q, q_target, memory, optimizer1)


        for e in range(self.n_EPISODES):
            ########### 2. DO NOT MODIFY FOR TESTING ###########2
            test_episode_reward = 0
            ########### 2. DO NOT MODIFY FOR TESTING  ###########

            s = env.reset()
            done = False

            while not done:
                a = q.sample_action(torch.from_numpy(s).float())
                s_prime, r, done, info = env.step(a)
                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r , s_prime, done_mask))
                s = s_prime

                ########### 3. DO NOT MODIFY FOR TESTING ###########
                test_episode_reward += r
                ########### 3. DO NOT MODIFY FOR TESTING  ###########


                ########### 4. DO NOT MODIFY FOR TESTING  ###########
                if done:
                    test_mean_episode_reward.append(test_episode_reward)
                    if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward) == 20):
                        test_over_reward = True
                        test_min_episode = e
                ########### 4. DO NOT MODIFY FOR TESTING  ###########

            if memory.size() > 400:
                for i in range(20):
                    s, a, r, s_prime, done_mask = memory.sample(batch_size)

                    q_out = q(s)
                    q_a = q_out.gather(1, a)
                    max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
                    target = r + gamma * max_q_prime * done_mask
                    loss = F.mse_loss(q_a, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if e % 20 == 0:
                q_target.load_state_dict(q.state_dict())
                elist.append(e)
                rlist.append(np.mean(test_mean_episode_reward))
                print("n_episode :{}, score : {:.1f}, n_buffer : {}".format(
                    e, test_episode_reward , memory.size()))

                ########### 5. DO NOT MODIFY FOR TESTING  ###########
                if test_over_reward:
                    print("END train function")
                    break
                ########### 5. DO NOT MODIFY FOR TESTING  ###########

                ## TODO



        ########### 6. DO NOT MODIFY FOR TESTING  ###########
        return test_min_episode, np.mean(test_mean_episode_reward)
        ########### 6. DO NOT MODIFY FOR TESTING  ###########


