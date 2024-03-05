import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_node_capacity():
    x = [10, 12, 14, 16, 18]
    yQ_LEAP = [12, 12, 14,14, 15]
    ySEE = [13, 13, 14, 16, 17]
    yMulti_R = [14, 15, 15, 17, 18]
    yOTiM2R = [15, 15, 16, 18, 20]
    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Success request', fontsize=15)
    plt.xlabel('Node capacity', fontsize=15)
    plt.xticks([10, 12, 14, 16, 18])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def draw_requestnum_all():
    x = [20, 40, 60, 80, 100]
    yQ_LEAP = [175, 188.9, 197, 207, 216]
    ySEE = [180, 203, 212, 222, 225]
    yMulti_R = [190, 205.6, 218.2, 225, 229.9]
    yOTiM2R = [210, 220, 229, 237, 245]

    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Optimal-throughput', fontsize=15)
    plt.xlabel('Request number', fontsize=15)
    plt.xticks([20, 40, 60, 80, 100])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def throughput_fidelity():
    x = [140, 160, 180, 200, 220, 240, 260]
    yQ_LEAP = [0.45, 0.50, 0.52, 0.58, 0.67, 0.69, 0.74]
    ySEE = [0.45, 0.49, 0.53, 0.59, 0.67, 0.72, 0.77]
    yMulti_R = [0.50, 0.57, 0.63, 0.69, 0.72, 0.79, 0.85]
    yOTiM2R = [0.58, 0.63, 0.72, 0.80, 0.85, 0.92, 0.98]

    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Fidelity/|R|', fontsize=15)
    plt.xlabel('Throughput', fontsize=15)
    plt.xticks([140, 160, 180, 200, 220, 240, 260])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()

def throughput_delay():
    x = [140, 160, 180, 200, 220, 240, 260]
    yQ_LEAP = [21, 20.56, 19.456, 19.012, 18.56, 18.1, 17.99]
    ySEE = [21,22, 20.87, 20.81, 19.456, 19.012, 18.112]
    yMulti_R = [20.2987, 19.366, 18.501, 18.01, 17.2, 16.12, 15.23]
    yOTiM2R = [18, 17.2, 15.9, 15.14, 14.789, 14.0987,13.678]

    plt.plot(x, yQ_LEAP, color='#35903A', linestyle='-', marker='o', markevery=1, markersize=10, label='Q-LEAP')
    plt.plot(x, ySEE, color='#E47B26', linestyle='-', marker='^', markevery=1, markersize=10, label='SEE')
    plt.plot(x, yMulti_R, color='#BE2A2C', linestyle='-', marker='s', markevery=1, markersize=10, label='Multi-R')
    plt.plot(x, yOTiM2R, color='blue', linestyle='-', marker='*', markevery=1, markersize=10, label='OTOH')
    plt.ylabel('Delay/|R|', fontsize=15)
    plt.xlabel('Throughput', fontsize=15)
    plt.xticks([140, 160, 180, 200, 220, 240, 260])

    plt.tick_params(labelsize=12)
    plt.legend(fontsize=13)

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # fig1
    draw_node_capacity()
    # fig2
    draw_requestnum_all()
    # fig3
    throughput_fidelity()
    # fig4
    throughput_delay()