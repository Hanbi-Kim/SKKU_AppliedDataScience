import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import collections
from torch import optim

def get_demo_traj():
    return np.load("demo_traj_2.npy", allow_pickle=True)


if __name__ == "__main__":
    b = np.load("demo_traj_2.npy", allow_pickle=True)
    print(b)






# # Hyperparameters
# learning_rate = 0.0005
# gamma = 0.99
# buffer_limit = 50000
# batch_size = 64
#
# ##########################################################################
# ############                                                  ############
# ############               Replay Buffer 구현                  ############
# ############                                                  ############
# ##########################################################################
# class ReplayBuffer():
#     def __init__(self):
#         self.buffer = collections.deque(maxlen=buffer_limit)
#
#     def put(self, transition):
#         self.buffer.append(transition)
#
#     def sample(self, n):
#         mini_batch = random.sample(self.buffer, n)
#         s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
#
#         for transition in mini_batch:
#             s, a, r, s_prime, done_mask = transition
#             s_lst.append(s)
#             a_lst.append([a])
#             r_lst.append([r])
#             s_prime_lst.append(s_prime)
#             done_mask_lst.append([done_mask])
#
#         return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
#                torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
#                torch.tensor(done_mask_lst)
#
#     def size(self):
#         return len(self.buffer)
#
# ##########################################################################
# ############                                                  ############
# ############                  DQfDNetwork 구현                 ############
# ############                                                  ############
# ##########################################################################
#
# class DQfDNetwork(nn.Module):
#     def __init__(self):
#         super(DQfDNetwork, self).__init__()
#         self.fc1 = nn.Linear(4, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 2)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def get_action(self, state, epsilon):
#         s = torch.from_numpy(state).float()
#         coin = random.random()
#         out = self.forward(s)
#         if coin < epsilon:
#             return random.randint(0, 1)
#         else:
#             return out.argmax().item()
#
# ##########################################################################
# ############                                                  ############
# ############                  DQfDagent 구현                   ############
# ############                                                  ############
# ##########################################################################
#
#
# class DQfDAgent(object):
#     def __init__(self, env, use_per, n_episode):
#         self.n_EPISODES = n_episode
#         self.env = env
#         self.use_per = use_per
#
#     def update(self):
#         pass
#
#     def pretrain(self, q, q_target, memory, optimizer):
#         for i in range(1):
#             s, a, r, s_prime, done_mask = memory.sample(batch_size)
#             q_out = q(s)
#             q_a = q_out.gather(1, a)
#             max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
#             target = r + gamma * max_q_prime * done_mask
#             loss = F.mse_loss(q_a, target)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#
#     ## Do pretrain for 1000 steps
#
#
#     def train(self):
#         ###### 1. DO NOT MODIFY FOR TESTING ######
#         test_mean_episode_reward = deque(maxlen=20)
#         test_over_reward = False
#         test_min_episode = np.inf
#         ###### 1. DO NOT MODIFY FOR TESTING ######
#
#         q = DQfDNetwork()
#         q_target = DQfDNetwork()
#         q_target.load_state_dict(q.state_dict())
#         optimizer = optim.Adam(q.parameters(), lr=learning_rate)
#         pretrained_memory = ReplayBuffer()
#         memory = ReplayBuffer()
#         demon = get_demo_traj()
#
#         env = self.env
#
#         for i in range(len(demon)):
#             for j in range(len(demon[i])):
#                 done_mask = 0.0 if demon[i][j][4] else 1.0
#                 pretrained_memory.put((demon[i][j][0], demon[i][j][1], demon[i][j][2], demon[i][j][3], done_mask ))
#         Do pretrain
#         DQfDAgent.pretrain(self, q, q_target, pretrained_memory, optimizer)
#
#         for e in range(self.n_EPISODES):
#             ########### 2. DO NOT MODIFY FOR TESTING ###########2
#             test_episode_reward = 0
#             ########### 2. DO NOT MODIFY FOR TESTING  ###########
#
#             done = False
#             state = env.reset()
#             epsilon = max(0.01, 0.08 - 0.01 * (e / 200))
#
#             while not done:
#                 action = q.get_action(state, epsilon)
#                 next_state, reward, done, _ = env.step(action)
#                 done_mask = 0.0 if done else 1.0
#                 memory.put((state, action, reward, next_state, done_mask))
#                 state = next_state
#
#                 ########### 3. DO NOT MODIFY FOR TESTING ###########
#                 test_episode_reward += reward
#                 ########### 3. DO NOT MODIFY FOR TESTING  ###########
#
#
#                 ########### 4. DO NOT MODIFY FOR TESTING  ###########
#                 if done:
#                     test_mean_episode_reward.append(test_episode_reward)
#                     if (np.mean(test_mean_episode_reward) > 475) and (len(test_mean_episode_reward) == 20):
#                         test_over_reward = True
#                         test_min_episode = e
#                 ########### 4. DO NOT MODIFY FOR TESTING  ###########
#             #
#             # print(e,"episodes","Reward =", test_episode_reward)
#
#             if memory.size() > 5000:
#                 for i in range(10):
#                     s, a, r, s_prime, done_mask = memory.sample(64)
#                     # s1, a1, r1, s_prime1, done_mask1 = pretrained_memory.sample(16)
#                     # s = torch.cat((s,s1),0)
#                     # a = torch.cat((a, a1), 0)
#                     # r = torch.cat((r, r1), 0)
#                     # s_prime = torch.cat((s_prime, s_prime1), 0)
#                     # done_mask = torch.cat((done_mask, done_mask1), 0)
#
#                     q_out = q(s)
#                     q_a = q_out.gather(1, a)
#                     max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
#                     target = r + gamma * max_q_prime * done_mask
#                     loss = F.smooth_l1_loss(q_a, target)
#
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#             if e != 0:
#                 q_target.load_state_dict(q.state_dict())
#                 print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
#                     e, test_episode_reward , memory.size(), 100))
#
#
#                 ########### 5. DO NOT MODIFY FOR TESTING  ###########
#                 if test_over_reward:
#                     print("END train function")
#                     break
#                 ########### 5. DO NOT MODIFY FOR TESTING  ###########
#
#             ## TODO
#
#             ########### 6. DO NOT MODIFY FOR TESTING  ###########
#             return test_min_episode, np.mean(test_mean_episode_reward)
#             ########### 6. DO NOT MODIFY FOR TESTING  ###########
#
