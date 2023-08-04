# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: link fidelity calculation                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from QuantumState.QuantumNode import MultiqubitsEntanglement as MQE

GAMMA = 100
ETA = 10

class Fidelity:
    def __int__(self):
        test = 1

    def getLinkFidelity(self, i, j, X_i, X_j):
        multi_qe = MQE()
        alpha0_ij = multi_qe.redefine_assign_qstate_of_multiqubits(i, X_i, j, X_j)
        test = 1

if __name__ == '__main__':
    f = Fidelity()
    f.getLinkFidelity(3, 5, 1,1) # length = 4

