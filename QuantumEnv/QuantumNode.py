# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 01-08-2023                                      #
#      Goals: class definitions of QuantumNode,               #
#             MultiqubitsEntanglement,                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import random
import numpy as np
from QuantumEnv import HyperParameters as qs_hp
import TOQN.TOQNHyperparameters as tohp

class QuantumNode:
    def __init__(self, node_id, node_capacity):
        self.qubits = None
        self.create_qubits(node_capacity)
        self.used_qubit_flag = [0 for i in range(node_capacity)]
        self.multi_qubits_state_basises = None
        self.create_qubits(node_capacity)

    def get_multiqubits_bisises(self):
        return self.multi_qubits_state_basises

    def create_qubits(self, node_capacity):
        self.qubits = ns.qubits.create_qubits(node_capacity)
        self.multi_qubits_state_basises = ns.qubits.combine_qubits(self.qubits)

class MultiqubitsEntanglement:
    def __init__(self):
        self.entangled_state_basises = None
        quantum_node_tp = QuantumNodeTopology()
        self.nodes_capacties = quantum_node_tp.get_node_capacity()

    def redefine_assign_qstate_of_multiqubits(self, i, i_cp, j, j_cp):
        source = QuantumNode(i, i_cp)
        destination = QuantumNode(j, j_cp)
        source_basises = source.multi_qubits_state_basises
        destination_basises = destination.multi_qubits_state_basises
        self.entangled_state_basises = ns.qubits.combine_qubits(source_basises + destination_basises)
        test = ns.qubits.reduced_dm(self.entangled_state_basises)  # 密度矩阵的表示让状态表示为2的2n次方
        prob_num = ns.qubits.reduced_dm(self.entangled_state_basises).shape[0] * \
                   ns.qubits.reduced_dm(self.entangled_state_basises).shape[1]
        return np.random.dirichlet(np.ones(prob_num), size=1)[0]

class QuantumNodeTopology:
    def __init__(self):
        self.nodes_num = t_hp.topology_myself_nodes_num
        self.node_cap_exp = qs_hp.node_capacity_expectation
        self.node_cap_sigma = qs_hp.random_capacity_sigma
        self.node_capacities = [0 for i in range(self.nodes_num)]
        self.generate_node_capacity()

    def generate_node_capacity(self):
        for i in range(self.nodes_num):
            self.node_capacities[i] = int(random.normalvariate(self.node_cap_exp, self.node_cap_sigma))

    def get_node_capacity(self):
        return self.node_capacities


if __name__ == '__main__':
    multiqubit_entanglement = MultiqubitsEntanglement(3,11)

