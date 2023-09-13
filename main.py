# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     Author: Yanan Gao                                       #
#       Date: 10-09-2023                                      #
#      Goals: Main                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from RouteSelection.OptimalRoute import OptimalRS
from Common.PolicyStorage import StoragePolicy
from ResourceAllocation import PhotonAllocation
T_thr = 100

if __name__ == '__main__':
    opr = OptimalRS()
    ps = StoragePolicy()
    pa = PhotonAllocation()

    # random photon allocation
    photonallocated = [
        [2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 8, 2]
    ]
    for t in range(T_thr):
        # route selection
        opr.set_photon_allocation(photonallocated)
        selected_route = opr.get_route_from_CRR(t, ps)
        ps.storage_policy(opr.get_Y(), photonallocated, t)

        # resource allocation
        pa.setSelectedRoute(selected_route)
        photonallocated.clear()
        photonallocated = pa.get_PApolicy()

        # calculate throughput




