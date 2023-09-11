# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Calculate the fidelity, delay,                  #
#             capacity... constraints                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from QuantumState.Fidelity import Fidelity
class Constraints:
    def __init__(self):
        test = 1

    def obatin_fidelity(self, r, k, t):
        return Fidelity.obtain_route_fidelity(r, k, t)

    def obtain_delay(self, r, k, t):
        return