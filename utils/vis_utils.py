import matplotlib.pyplot as plt
import numpy as np

COLOR_CODES = {
    "Qlearning" : "#c0392b",
    "HalfBlindAgent": "#2980b9",
    "FullyInformedAgent": "#138d75",
    "Random": "#f39c12"
}

def plot_reward(mean, std_dev, name, _id=0):
    x_axis = [_ for _ in range(len(mean))]
    bin_size = len(mean) // 5
    plt.xticks(range(0, len(mean), bin_size))
    plt.plot(x_axis, mean, color=COLOR_CODES[name])
    plt.fill_between(x_axis, mean - std_dev, mean + std_dev,\
                     alpha=0.1, edgecolor=COLOR_CODES[name], facecolor=COLOR_CODES[name])
    plt.xlabel("# Rounds")
    plt.ylabel("Average Accumulative Reward")
    plt.savefig("figures/{}_{}.png".format(name, _id), bbox_inches='tight')
    plt.close()

def plot_rewards_comparison(results, rounds, labels, _id=0):
    x_axis = [_ for _ in range(rounds + 1)]
    bin_size = (rounds + 1) // 5
    plt.xticks(range(0, rounds + 1, bin_size))
    for i, measures in enumerate(results):
        mean = measures[0]
        std_dev = measures[1]
        plt.plot(x_axis, mean, label=labels[i], color=COLOR_CODES[labels[i]])
        plt.fill_between(x_axis, mean - std_dev, mean + std_dev,\
                         alpha=0.1, edgecolor=COLOR_CODES[labels[i]], facecolor=COLOR_CODES[labels[i]])
    plt.xlabel("# Rounds")
    plt.ylabel("Average Accumulative Reward")
    plt.legend(loc='best')
    plt.savefig("figures/comparison_{}.png".format(_id), bbox_inches='tight')
    plt.close()
