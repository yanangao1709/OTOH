# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Each agent is first equipped with DQN RL model  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp
from ResourceAllocation.reward_decomposition.decomposer import RewardDecomposer
from ResourceAllocation.reward_decomposition import decompose as decompose


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.set_seed(1)

        self.input_layer = nn.Linear(RLhp.NUM_STATES, 32)
        self.input_layer.weight.data.normal_(0, 0.1)

        self.hidden_layer1 = nn.Linear(32,64)
        self.hidden_layer1.weight.data.normal_(0, 0.1)
        self.hidden_layer2 = nn.Linear(64,32)
        self.hidden_layer2.weight.data.normal_(0, 0.1)

        self.req_layers = {}
        for r in range(tohp.request_num):
            r_layer = nn.Linear(32, 32)
            r_layer.weight.data.normal_(0, 0.1)
            r_candroute_layer = nn.Linear(32, RLhp.NUM_ACTIONS)
            r_candroute_layer.weight.data.normal_(0, 0.1)
            self.req_layers[r] = [r_layer, r_candroute_layer]

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        v = []
        for r in range(tohp.request_num):
            x = F.relu(self.req_layers[r][0](x))
            v.append(F.relu(self.req_layers[r][1](x)))
        return tuple(v)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

class DQN():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # storage data of state, action ,reward and next state
        self.memorys = {}
        for i in range(tohp.nodes_num):
            self.memorys[i] = np.zeros((RLhp.MEMORY_CAPACITY, 42))

        self.local_rewards = []

        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), RLhp.LR)
        self.loss = nn.MSELoss()

        self.reward_decomposer = RewardDecomposer()
        self.reward_optimiser = Adam(self.reward_decomposer.parameters(), lr=0.01)

    def store_trans(self, i, state, action, reward, next_state, memory_counter):
        index = memory_counter % RLhp.MEMORY_CAPACITY
        trans = np.hstack((state, action, [reward], next_state))#记录一条数据
        self.memorys[i][index,] = trans

    def choose_action(self, state_para):
        action = []
        # notation that the function return the action's index nor the real action
        state = torch.unsqueeze(torch.FloatTensor(state_para) ,0)
        if np.random.randn() <= RLhp.EPSILON:
            action_value = self.eval_net.forward(state)
            for av in action_value:
                a = torch.max(av, 1)[1].data.item()
                action.append(a)
        else:
            for i in range(tohp.request_num):
                a = np.random.randint(0,RLhp.NUM_ACTIONS)
                action.append(a)
        return action

    def build_rewards(self, batch_memorys):
        local_rewards = decompose.decompose(True, batch_memorys)
        return local_rewards


    def learn(self, m):
        # copy parameters to target each 100 episodes
        if self.learn_counter % RLhp.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1
        # sample data
        sample_index = np.random.choice(RLhp.MEMORY_CAPACITY, RLhp.BATCH_SIZE)

        # reward decomposition
        if m == 0:
            batch_memorys = {}
            for i in range(tohp.nodes_num):
                batch_memorys[i] = self.memories[i][sample_index, :]
                self.local_rewards = self.build_rewards(batch_memorys)
            # train the decomposer
            reward_sample = buffer.sample(args.reward_batch_size)
            decompose.train_decomposer(self.reward_decomposer, self.memories, reward_sample, self.reward_optimiser)

        # reward decomposition
        batch_reward = self.build_rewards(batch_memory)
        # batch_reward = torch.FloatTensor(batch_memory[:, RLhp.NUM_STATES+1: RLhp.NUM_STATES+2])

        # batch_memorys = []
        for i in range(tohp.nodes_num):
            # batch_memorys.append(self.memorys[i][sample_index, :])
            batch_memory = self.memorys[i][sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :RLhp.NUM_STATES])
            batch_action = torch.LongTensor(batch_memory[:, RLhp.NUM_STATES:RLhp.NUM_STATES + 1].astype(int))
            batch_next_state = torch.FloatTensor(batch_memory[:, -RLhp.NUM_STATES:])
            q_eval_total = []
            for bs in self.eval_net(batch_state):
                q_eval_total.append(bs.gather(1, batch_action))
            q_eval = sum(q_eval_total) / len(q_eval_total)
            # q_eval + next_action input F_J networks





        # q_eval = self.eval_net(batch_state).gather(1, batch_action) # 得到当前Q(s,a)

        q_next_total = []
        for bs in self.eval_net(batch_next_state):
            q_next_total.append(bs.gather(1, batch_action))
        q_next = sum(q_next_total) / len(q_next_total)
        # q_next = self.target_net(batch_next_state).detach() # 得到Q(s',a')，有三个值，下面选max
        q_target = batch_reward + RLhp.GAMMA*q_next.max(1)[0].view(RLhp.BATCH_SIZE, 1) # bellman公式：Q=R+折扣*Q‘

        loss = self.loss(q_eval, q_target) # 差异越小越好
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # 梯度更新

    # def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    #     actions = batch["actions"][:, :-1]
    #     terminated = batch["terminated"][:, :-1].float()
    #     mask = batch["filled"][:, :-1].float()
    #     mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
    #     avail_actions = batch["avail_actions"]
    #
    #     # Build the rewards based on the scheme (local/global, decompose or not ...)
    #     status, reward_mask, rewards = self.build_rewards(batch)
    #
    #     # Check if reward decomposition has failed
    #     if not status:
    #         # print("Decomposition failed for current batch, disregarding...")
    #         return
    #     # print("Successfully decomposed the reward function, training Q functions")
    #
    #     # if given a reward mask, use in order to not use on all data
    #     if reward_mask is not None:
    #         mask = th.logical_and(mask, reward_mask)
    #
    #     # Calculate estimated Q-Values
    #     mac_out = []
    #     hidden_states = []
    #     self.mac.init_hidden(batch.batch_size)
    #     for t in range(batch.max_seq_length):
    #         agent_outs = self.mac.forward(batch, t=t)
    #         mac_out.append(agent_outs)
    #         hidden_states.append(self.mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
    #
    #     mac_out = th.stack(mac_out, dim=1)  # Concat over time
    #     hidden_states = th.stack(hidden_states, dim=1)
    #
    #     # Pick the Q-Values for the actions taken by each agent
    #     chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
    #
    #     # Calculate the Q-Values necessary for the target
    #     target_mac_out = []
    #     target_hidden_states = []
    #     self.target_mac.init_hidden(batch.batch_size)
    #     for t in range(batch.max_seq_length):
    #         target_agent_outs = self.target_mac.forward(batch, t=t)
    #         target_mac_out.append(target_agent_outs)
    #         target_hidden_states.append(self.target_mac.hidden_states.view(batch.batch_size, self.args.n_agents, -1))
    #
    #     # We don't need the first timesteps Q-Value estimate for calculating targets
    #     target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
    #     target_hidden_states = th.stack(target_hidden_states[1:], dim=1)
    #
    #     # Mask out unavailable actions
    #     target_mac_out[avail_actions[:, 1:] == 0] = -9999999
    #
    #     # Max over target Q-Values
    #     if self.args.double_q:
    #         # Get actions that maximise live Q (for double q-learning)
    #         mac_out_detach = mac_out.clone().detach()
    #         mac_out_detach[avail_actions == 0] = -9999999
    #         cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
    #         target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
    #     else:
    #         target_max_qvals = target_mac_out.max(dim=3)[0]
    #
    #     # Record utility values
    #     utilities = chosen_action_qvals
    #
    #     # Mix
    #     if self.args.mixer == 'graphmix':
    #         chosen_action_qvals_peragent = chosen_action_qvals.clone()
    #         target_max_qvals_peragent = target_max_qvals.detach()
    #
    #         chosen_action_qvals, local_rewards, alive_agents_mask = self.mixer(chosen_action_qvals,
    #                                                                            batch["state"][:, :-1],
    #                                                                            agent_obs=batch["obs"][:, :-1],
    #                                                                            team_rewards=rewards,
    #                                                                            hidden_states=hidden_states[:, :-1]
    #                                                                            )
    #         chosen_output_qvals = chosen_action_qvals.clone()
    #
    #         target_max_qvals = self.target_mixer(target_max_qvals,
    #                                              batch["state"][:, 1:],
    #                                              agent_obs=batch["obs"][:, 1:],
    #                                              hidden_states=target_hidden_states
    #                                              )[0]
    #
    #         ## Global loss
    #         # Calculate 1-step Q-Learning targets
    #         targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
    #
    #         # Td-error
    #         td_error = (chosen_action_qvals - targets.detach())
    #
    #         mask = mask.expand_as(td_error)
    #
    #         # 0-out the targets that came from padded data
    #         masked_td_error = td_error * mask
    #
    #         # Normal L2 loss, take mean over actual data
    #         global_loss = (masked_td_error ** 2).sum() / mask.sum()
    #
    #         ## Local losses
    #         # Calculate 1-step Q-Learning targets
    #         local_targets = local_rewards + self.args.gamma * (1 - terminated).repeat(1, 1, self.args.n_agents) \
    #                         * target_max_qvals_peragent
    #
    #         # Td-error
    #         local_td_error = (chosen_action_qvals_peragent - local_targets)
    #         local_mask = mask.repeat(1, 1, self.args.n_agents) * alive_agents_mask
    #
    #         # 0-out the targets that came from padded data
    #         local_masked_td_error = local_td_error * local_mask
    #
    #         # Normal L2 loss, take mean over actual data
    #         local_loss = (local_masked_td_error ** 2).sum() / mask.sum()
    #
    #         # total loss
    #         lambda_local = self.args.lambda_local
    #         loss = global_loss + lambda_local * local_loss
    #
    #     else:
    #         if self.mixer is not None:
    #             # Since we want to optimize the bellman equation, and the target refers to the
    #             # next state, we do this 1: , :-1 trim to the state batch
    #             chosen_output_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], obs=batch["obs"][:, :-1])
    #             target_output_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], obs=batch["obs"][:, 1:])
    #         else:
    #             chosen_output_qvals = chosen_action_qvals
    #             target_output_qvals = target_max_qvals
    #
    #         # Shape debugging purposes
    #         # print(f"Target Max qvals: {target_max_qvals.shape}")
    #         # print(f"Chosen Action qvals: {chosen_action_qvals.shape}")
    #         # print(f"Rewards: {rewards.shape}")
    #         # print(f"Mask: {mask.shape}")
    #
    #         # Compute Loss
    #         if self.args.local_observer:
    #             loss_func = QLearner.compute_local_loss
    #         else:
    #             loss_func = QLearner.compute_global_loss
    #         loss = loss_func(self, rewards, terminated, mask, target_output_qvals, chosen_output_qvals)
    #
    #         # Add regularization if necessary
    #         if getattr(self.args, "monotonicity_method", "weights") == "regularization":
    #             # If sample is set to true, we don't use the given utilities, rather we sample from the bounding box that
    #             # they imply. This creates a more uniform punishment for the gradient of Q by U
    #             if getattr(self.args, "sample_utilities", False):
    #                 n_agents = utilities.shape[-1]
    #                 copied_utilities = utilities.cpu().detach().numpy()
    #                 flattened_utilities = np.reshape(copied_utilities, (-1, n_agents))
    #                 sampled_utilities = th.tensor(np.random.uniform(
    #                     low=[np.min(flattened_utilities[:, i]) for i in range(n_agents)],
    #                     high=[np.max(flattened_utilities[:, i]) for i in range(n_agents)],
    #                     size=copied_utilities.shape
    #                 ), requires_grad=True, device=self.args.device).float()
    #                 sampled_q_vals = self.mixer(sampled_utilities, batch["state"][:, :-1], obs=batch["obs"][:, :-1])
    #                 reg_loss = self.compute_regularization(sampled_utilities, sampled_q_vals, t_env)
    #             else:
    #                 reg_loss = self.compute_regularization(utilities, chosen_output_qvals, t_env)
    #
    #             coeff = self.args.monotonicity_coeff
    #             self.logger.log_stat("regularizing_loss", reg_loss.item(), t_env)
    #
    #             # Comupte the total loss
    #             loss = coeff * reg_loss + (1 - coeff) * loss
    #
    #     # Optimise
    #     self.optimiser.zero_grad()
    #     loss.backward()
    #     grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
    #     self.optimiser.step()
    #
    #     if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
    #         self._update_targets()
    #         self.last_target_update_episode = episode_num
    #
    #     if self.args.local_observer:
    #         self.update_l_params(t_env)
    #
    #     if t_env - self.log_stats_t >= self.args.learner_log_interval:
    #         self.logger.log_stat("loss", loss.item(), t_env)
    #         self.logger.log_stat("grad_norm", grad_norm.clone().cpu().detach().numpy(), t_env)
    #         mask_elems = mask.sum().item()
    #         # self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
    #         self.logger.log_stat("q_taken_mean",
    #                              (chosen_output_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
    #         # self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
    #         #                      t_env)
    #         self.log_stats_t = t_env
    #
    #         # Visualize Q values if necessary (mostly for the payoff matrix enviroment)
    #         if getattr(self.args, "display_q_values", False):
    #             print(f"Q Values for {self.args.run_name}")
    #             mac_output = mac_out[:1, :1]
    #             q_tot = []
    #             for agent1_act, agent2_act in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    #                 utilities = th.stack([mac_output[:, :, 0, agent1_act], mac_output[:, :, 1, agent2_act]], dim=2)
    #                 print(f"U1, U2: for a1={agent1_act}, a2={agent2_act}:\t{utilities}")
    #                 q_values = self.mixer(utilities, batch["state"][:1, :1], obs=None)
    #                 print(f"Q1, Q2: for a1={agent1_act}, a2={agent2_act}:\t{q_values}")
    #                 q_tot.append(th.sum(q_values))
    #               print(f"{q_tot[0]}\t{q_tot[1]}\n{q_tot[2]}\t{q_tot[3]}")
    #
    #         # My own local exp logger
    #         self.args.exp_logger.save_learner_data(
    #             learner_name=self.args.name,
    #             learner_data={
    #                 "t_env": t_env,
    #                 "loss": loss.item(),
    #                 # "td_error": masked_td_error.abs().sum().item() / mask_elems,
    #                 "grad_norm": grad_norm.clone().cpu().detach().numpy(),
    #                 "q_taken_mean": (chosen_output_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
    #                 # "target_mean": (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
    #             }
    #         )

# ------test------
if __name__=="__main__":
    a = [3,5,6]
    print(a.values)