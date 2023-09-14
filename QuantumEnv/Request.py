# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: request and its characteristics                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import random

class Request:
    def __init__(self):
        self.source = None
        self.destination = None
        self.volumn = None
        self.candidate_routes = []
        self.cand_route_hops = []

    def setSource(self, source):
        self.source = source

    def setDestination(self, destination):
        self.destination = destination

    def setVolumn(self, l, u):
        self.volumn = random.randint(l, u)

    def setCandidateRoutes(self, candidateRoutes):
        self.candidate_routes = candidateRoutes

    def setCandRouteHops(self, hops):
        self.cand_route_hops = hops

    def getSource(self):
        return self.source

    def getDestination(self):
        return self.destination

    def getVolumn(self):
        return self.volumn

    def getCandidateRoutes(self):
        return self.candidate_routes

    def getCandRouteHops(self):
        return self.cand_route_hops