import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import pickle

from IPython import embed

log_file = '/local/home/zhqian/sp/data/calibration/train_log.pkl'

def plot_error():
    with open(log_file, 'rb') as f:
        log = pickle.load(f)
    plot_dict = {}
    for k, d in log.items():
        error = [x for x in d.values()]
        error = np.array(error)
        error = np.mean(error, axis=0)
        plot_dict[k]= error
    for k, v in plot_dict.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_error()