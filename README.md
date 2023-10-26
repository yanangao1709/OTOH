OTOH: Online Transmission Optimization via High-Dimensional Entanglement
====
Yanan Gao, Song Yang, Fan Li, Liehuang Zhu, Stojan Trajanovski, Xiaoming Fu

This is the official implementation of our paper that is submitted to the journal of IEEE/ACM Transactions on Networking.

Installation
------
(1) Install Gurobipy according to [link](https://pypi.org/project/gurobipy/). It is free for a student or staff member of an academic institution.

(2) Python versions more than 3.5 are ok for the project.

(3) Download the project, and then run the ``main.py'', OTOH can return the optimization results by plotting a figure.

Topology:
------

(1) nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='y') Because the node position is undetermined in advance, "pos" can not limit the node position. The topology is different in each time generation. An example is as the following:
<p align="center">
  <img src="https://github.com/yanangao1709/OTOH/assets/43428644/77cba931-fc9d-41ad-bc57-a77b1036f821" alt="An topology example"/>
</p>

Results:
------
