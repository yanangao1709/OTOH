import torch
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from itertools import combinations
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from TOQN import TOQNHyperparameters as tohp
from ResourceAllocation import RLHyperparameters as RLhp


class RewardDecomposer:
    def __init__(self):
        self.n_agents = tohp.nodes_num
        self.input_shape_one_obs = tohp.nodes_num + tohp.request_num

        self.reward_groups = self.build_reward_groups()
        self.reward_networks = nn.ModuleList([reward_group.reward_network for reward_group in self.reward_groups])

    # Let the reward function observe the state and the last action
    def _get_input_shape(self, scheme):
        # observe the last state
        input_shape = scheme["obs"]["vshape"]

        # # observe the last action
        # input_shape += scheme["actions_onehot"]["vshape"][0]
        #
        # if self.args.obs_agent_id:
        #     input_shape += self.n_agents

        if self.args.reward_index_in_obs != -1:
            input_shape = 1

        return input_shape

    def build_team_inputs(self, batch, ep_idx, t_idx):
        team_inputs = list()
        for agent_idx in range(self.n_agents):
            team_inputs.append(self.build_input(batch, ep_idx, t_idx, agent_idx))
        return th.stack(team_inputs)

    def build_input(self, batch, ep_idx, t_idx, agent_idx):
        agent_input = list()

        # observe the last state
        agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx])
        # agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx][-1:])

        # observe the last action
        # agent_input.append(batch["actions_onehot"][ep_idx, t_idx, agent_idx])

        # if self.args.obs_agent_id:
        #     agent_input.append(F.one_hot(th.tensor([agent_idx])[0], num_classes=self.n_agents))

        agent_input = th.cat(agent_input)
        return agent_input

    def build_reward_groups(self):
        reward_groups = []
        for i in range(tohp.nodes_num):
            reward_groups.append(RewardGroup(
                self.input_shape_one_obs,
            ))
        return reward_groups

    def single_forward(self, reward_input, reward_group_idx=0):
        return self.reward_groups[reward_group_idx].single_forward(reward_input)

    def forward(self, reward_inputs):
        # Get the reward output for every reward group based on the output type
        reward_group_outputs = []
        for i in range(len(self.reward_groups)):
            reward_input = torch.unsqueeze(torch.FloatTensor(reward_inputs[i]), 0)
            reward_group_outputs.append(self.reward_groups[i].forward(reward_input))
        return reward_group_outputs

    def convert_raw_outputs(self, raw_outputs):
        converted_outputs = [
            self.reward_groups[idx].convert_raw_outputs(raw_output)
            for idx, raw_output in enumerate(raw_outputs)
        ]
        converted_outputs = th.cat(converted_outputs, dim=2)
        return converted_outputs

    def raw_to_pred(self, local_probs):
        if not len(local_probs):
            return local_probs

        num_classes = self.n_agents + 1
        global_prob_class_shape = local_probs[0].shape[:2]
        global_probs = th.zeros(*global_prob_class_shape, num_classes)

        for indices_list, class_num in self.reward_combos:
            # Compute probability for this indices group
            curr_probs = th.ones(*global_prob_class_shape)
            for reward_func_idx, class_idx in enumerate(indices_list):
                curr_probs *= local_probs[reward_func_idx][:, :, class_idx]

            # Add this combination to the total probability
            global_probs[:, :, class_num] += curr_probs

        return global_probs

    def compute_regularization(self, raw_outputs):
        reg_loss = 0
        for idx, raw_output in enumerate(raw_outputs):
            reg_loss += self.regularizing_weights[idx] * self.reward_groups[idx].compute_regularization(raw_output)
        return reg_loss

    def parameters(self):
        return self.reward_networks.parameters()

    def load_state(self, other_mac):
        self.reward_networks.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.reward_networks.cuda()



# This class represents all of the NN that are of the same size n_j
class RewardGroup(nn.Module):
    def __init__(self, input_shape_one_obs):
        super(RewardGroup, self).__init__()
        self.input_shape_one_obs = input_shape_one_obs
        self.num_reward_agents = tohp.nodes_num
        self.contrib_weight = 1 / self.num_reward_agents

        self.input_shape = RLhp.NUM_STATES
        self.output_shape, self.output_vals = self.compute_output_scheme()

        self.reward_network = self.build_reward_network()

    # If the reward network is a classifier, then outputs like classes otherwise 1
    def compute_output_scheme(self):
        output_vals = None
        output_shape = tohp.nodes_num
        output_vals = list(range(-self.num_reward_agents, self.num_reward_agents + 1))
        output_shape = len(output_vals)
        return output_shape, output_vals

    def build_reward_network(self):
        module_sub_list = RewardNetwork(self.input_shape)
        return module_sub_list

    def single_forward(self, reward_input):
        return self.reward_network(reward_input)

    def forward(self, reward_inputs):
        # a group has multiple agents
        reward_outputs = []
        for i in range(1):
            # reward_input = th.reshape(reward_inputs, shape=(*reward_inputs.shape[:-2], -1))
            # get outputs from the network for this combination
            reward_output = self.reward_network(reward_inputs)
            reward_outputs.append(reward_output)
        return reward_outputs

    def convert_raw_outputs(self, reward_outputs):
        reward_outputs = self.local_rewards_to_agent_rewards(reward_outputs)
        return reward_outputs

    @staticmethod
    def raw_to_classes(probs):
        return th.argmax(probs, dim=-1, keepdim=True)

    def classes_to_local_rewards(self, classes):
        return classes.apply_(lambda x: self.output_vals[x])

    def local_rewards_to_agent_rewards(self, local_rewards):
        # a group has multiple agents, but here a group just has one agent
        local_reward = local_rewards[0]
        agent_rewards = th.zeros(*local_reward.shape[:2], 1)
        local_reward = th.reshape(local_reward, shape=(*local_reward.shape[:2], -1))

        for i in range(1):
            weighted_reward = self.contrib_weight * local_reward[:, :, i]
            weighted_reward = weighted_reward.repeat(1, 1)
            agent_rewards[:, :, i] += weighted_reward

        return agent_rewards

    def compute_regularization(self, raw_outputs):
        # Gets as input a list of outputs representing every rnj output
        # a list of length num_rnj functions with tensors of shape [num_batches, ep_len, output_shape]
        # Now add the regularizing loss
        reg_loss = 0
        for output in raw_outputs:
            reg_loss += th.sum(th.abs(output))
        return reg_loss


class RewardNetwork(nn.Module):
    def __init__(self, input_shape):
        super(RewardNetwork, self).__init__()

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        y = F.leaky_relu(self.fc2(x))
        h = F.tanh(self.fc3(y))
        q = self.fc4(h)
        return q

