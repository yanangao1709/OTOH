# OTOH: Online Transmission Optimization via High-Dimensional Entanglement

Yanan Gao, Song Yang, Fan Li, Youqi Li, Liehuang Zhu, Stojan Trajanovski, Xiaoming Fu

This is the official implementation of our paper that is submitted to the journal of IEEE/ACM Transactions on Networking.

## Installation and Run
(1) Install Gurobipy according to [link](https://pypi.org/project/gurobipy/). It is free for a student or staff member of an academic institution.

(2) Python versions more than 3.5 are feasible for the project.

(3) Installation of the relevant Python packages can be done by running
```c
cd OTOH/
pip install -r requirements.txt
```

(4) The way to return the code is by running OTOH/main.py.

## Topology:

(1) nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='y') Because the node position is undetermined in advance, "pos" can not limit the node position. The topology is different in each time generation. An example is as the following:
<p align="center">
  <img src="https://github.com/yanangao1709/OTOH/assets/43428644/77cba931-fc9d-41ad-bc57-a77b1036f821" alt="An topology example"/>
</p>


## Configuring parameters:
Our method is dependent on hyper-parameters that can be found under OTOH/config/. There are 3 different types of configuration files:

* **Default Configuration** - found under `OTOH/config/default.yaml`. This file depicts the default parameters
for all runs. It is overridden by any other configuration file. An example of a parameter here could be
  `batch_size`, which is generally similar for all algorithms. 

* **Enviroment Configuration** - found under `OTOH/config/envs/<env-name>.yaml`. This file depicts the parameters
that are relevant for running the environment. An example of this is `num_agents`. 
  
* **Algorithm Configuration** - found under `OTOH/config/algs/<alg-name>.yaml`. This file depicts the parameters
for the current algorithm. An example of this is `learning_rate`.

## Code Roadmap

The following file diagram depicts an outline of the code, with explanations
regarding key modules in our code. 

```
OTOH
└───documentation (includes some figures from the paper)   
└───results (where local results are stored)   
└───scripts (runnable scripts that are described above)  
└───src (main code folder)
│   └───config (configuration files described above)
│   └───envs (used environments, includes multi_cart (Coupled Multi Cart Pole), multi_particle (Bounded Cooperative Navigation), payoff_matrix....
│   └───reward_decomposition (includes the full implementation for our RD method)
│   └───learners (the main learning loop, bellman updates)
│   │   │   q_learner (a modified q_learner that supports local rewards and reward decomposition)
│   │   │   ...
│   └───modules (NN module specifications)
│   │   └───mixers (Mixing layers specifications)
│   │   │   │   gcn (a GCN implementation for LOMAQ, occasionally used)
│   │   │   │   lomaq.py (The main specification of our mixing networks)
│   │   │   │   ...
│   └───controllers (controls a set of agent utility networks)
│   │   │   hetro_controller.py (an agent controller that doesn't implement parameter sharing)
│   │   │   ...
│   └───components (general components for LOMAQ)
│   │   │   locality_graph.py (A module that efficiently represents the graph of agents)
│   │   │   ...
│   │   main.py (for running a certain env-alg pair with default parameters)
│   │   multi_main.py (for running a certain test with multiple runs)
│   │   single_main.py (for running arun within a test)
│   │   offline_plot.py (for plotting results)
│   │   ...
│   README.md (you are here)
│   requirements.txt (all the necessary packages for running the code)
│   ...
```
