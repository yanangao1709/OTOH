import Network
from ResourceAllocation.old import Agent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

T = 20
Nodes = [1, 2, 3, 4, 5]
Nodes_capacity = [5, 8, 14, 6, 3]
link_length= [
    [0,6,8,5,5],
    [6,0,10,0,7],
    [8,10,0,9,0],
    [5,0,9,0,3],
    [5,7,0,3,0]
]
gamma = 0.09
tau = 0.2

def initial():
    nodes = []
    for n, nc in zip(Nodes, Nodes_capacity):
        nodes.append(Network.Nodes(n, nc))
    return nodes

def run(nodes):
    agents = []
    for node in nodes:
        agents.append(Agent.agent(node))

    test = 1

def decoherence_qubits():
    fig = plt.figure(figsize=(10, 7))
    ax = Axes3D(fig)
    colors = ["red","green","blue","black","orange"]
    basis_num = 3
    # Xs = np.random.dirichlet(np.ones(basis_num),size=1)[0].tolist()
    # print(Xs)
    Xs = [0.3999298222182461, 0.3834619279782807, 0.21660824980347324]
    for b in range(basis_num):
        x = []
        y = []
        q = []
        for i in range(1, Nodes_capacity[1] + 1):
            for j in range(1, Nodes_capacity[2] + 1):
                length = link_length[1][2]
                if length == 0: continue
                x.append(i)
                y.append(j)
                sum = 0
                for xs in Xs:
                    sum += xs * gamma * ((i + j) / length)
                q.append(sum)
    ax.scatter3D(x, y, q, color="darkorange")
    ax.set_xlabel('qubit quantity')
    ax.set_ylabel('qubit quantity')
    ax.set_zlabel('quantum state decoherence')
    plt.title("quantum states")
    plt.show()

def decoherence_time():
    fig = plt.figure(figsize=(10, 7))
    ax = Axes3D(fig)
    colors = ["red", "green", "blue", "black", "orange"]
    basis_num = 3
    Xs = [0.3999298222182461, 0.3834619279782807, 0.21660824980347324]
    for b in range(basis_num):
        x = []
        y = []
        q = []
        for t in range(50):
            for j in range(1, Nodes_capacity[2] + 1):
                length = link_length[1][2]
                if length == 0: continue
                x.append(t)
                y.append(j)
                sum = 0
                for xs in Xs:
                    sum += math.sqrt(xs * math.exp(-1*tau*t)) * gamma * ((3 + j) / length)
                q.append(sum)
    ax.scatter3D(x, y, q, color="green")
    ax.set_xlabel('time series')
    ax.set_ylabel('qubit quantity')
    ax.set_zlabel('fidelity decoherence')
    plt.title("quantum states")
    plt.show()

def calculate_fidelity():
    X1 = [0.1381, 0.8523, 0.0096]
    X2 = [0.4842, 0.0192, 0.183, 0.3126, 0.001]

    for t in range(25):
        sum1 = 0
        for xs in X1:
            sum1 += math.sqrt(xs * math.exp(-1 * tau * t))
        sum1 +=  sum1 * gamma * ((2+4)/8 + (4+5)/15 + (5+3)/17)

        sum2 = 0
        for xss in X2:
            sum2 += math.sqrt(xss * math.exp(-1 * tau * t))
        sum2 += sum2 * gamma * ((2 + 2) / 7 + (2 + 2) / 21)

        print(str(t) + '---------' + str(sum1) + '----' + str(sum2))


if __name__ == '__main__':
    # nodes = initial()
    # run(nodes)
    # decoherence_qubits()
    # decoherence_time()
    calculate_fidelity()




