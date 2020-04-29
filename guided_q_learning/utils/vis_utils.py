import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(x_axis, mean_vecs, std_dev_vectors, labels, title, filename):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(5, 5)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average reward')
    ax1.set_title('Average reward')

    for i in range(len(mean_vecs)):
        plt.plot(x_axis, mean_vecs[i], label=labels[i])
        # plt.fill_between(x_axis, mean_vecs[i] - std_dev_vectors[i], mean_vecs[i] + std_dev_vectors[i],\
        #                 alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("{}.png".format(filename))