# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 28-07-2023                                      #
#      Goals: request and candidate route generation                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import HyperParameters
import numpy as np
import random

class RequestAndRouteGeneration:
    def __init__(self):
        self.nodes_num = HyperParameters.topology_myself_nodes_num
        self.requests = []
        self.volumn_upper = 10
        self.volumn_lower = 2

    def request_generation(self):
        source = random.randint(1,self.nodes_num)
        destination = random.randint(1,self.nodes_num)
        volumn = random.randint(self.volumn_lower, self.volumn_upper)
        print(1)






