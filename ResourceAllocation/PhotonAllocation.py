# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL for the photon allocation                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from ResourceAllocation.RLmain import RLMain

class PhotonAllocation:
    def __int__(self):
        self.PApolicy = []
        self.selectedroute = None

    def setSelectedRoute(self, selectedroute):
        self.selectedroute = selectedroute

    def get_PApolicy(self):
        rl = RLMain(self.selectedroute)
        rl.run()
        return self.PApolicy


if __name__ == '__main__':
    test = 1