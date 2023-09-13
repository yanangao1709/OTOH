# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: RL for the photon allocation                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PhotonAllocation:
    def __int__(self):
        self.PApolicy = []
        self.selectedroute = None
        test = 1

    def setSelectedRoute(self, selectedroute):
        self.selectedroute = selectedroute

    def RLrun(self):
        test = 1



    def get_PApolicy(self):
        self.RLrun()
        return self.PApolicy


if __name__ == '__main__':
    test = 1