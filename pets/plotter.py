import os
import csv

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def plot_performance(log_dir, min_num_trials):
    returns = []
    for subdir in os.listdir(log_dir):
        data = loadmat(os.path.join(log_dir, subdir, "logs.mat"))
        if data["returns"].shape[1] >= min_num_trials:
            returns.append(data["returns"][0][:min_num_trials])

    returns = np.array(returns)
    # returns = np.maximum.accumulate(returns, axis=-1)
    mean = np.mean(returns, axis=0)

    # Plot result
    plt.figure()
    plt.plot(np.arange(1, min_num_trials + 1), mean)
    plt.title("Performance")
    plt.xlabel("Iteration number")
    plt.ylabel("Return")
    plt.savefig('performance_plot.png')  # save figure to a file
    # plt.show()  # this line can be commented out


if __name__ == "__main__":
    log_dir = './log/'  # Directory specified in script, not including date+time
    min_num_trials = 30  # Plots up to this many trials
    plot_performance(log_dir, min_num_trials)
