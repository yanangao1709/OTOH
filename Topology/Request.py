# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 04-08-2023                                      #
#      Goals: request and its characteristics                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class Request:
    def __init__(self):
        self.source = None
        self.destination = None
        self.volumn = None
        self.candidate_routes = []

    def setSource(self, source):
        self.source = source

    def setDestination(self, destination):
        self.destination = None

    def setCandidateRoutes(self, candidateRoutes):
        self.candidate_routes = candidateRoutes