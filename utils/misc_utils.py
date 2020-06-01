import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_rewards_and_states(predictions, rewards, states, name):
    def plot(predictions, true, name):
        ys = predictions
        xs = np.arange(len(predictions))
        plt.plot(xs, ys, 'r-')

        ys = true
        xs = np.arange(len(true))
        plt.plot(xs, ys, 'k-')

        plt.savefig('%s.png' % name)
        plt.clf()

    rewards_predicted = [p[0, 0] for p in predictions]
    plot(rewards_predicted, rewards.flatten(), name + 'rewards')

    states_dim = predictions[0].shape[-1] - 1
    if states_dim > 0:
        for i in range(states_dim):
            states_predicted = [p[0, i + 1] for p in predictions]
            states_true = states[:, :, i].flatten()
            plot(states_predicted, states_true, name + 'states_%s' % i)


def plot_rewards(rewards, name):
    rews_seq = rewards.flatten()

    # rewards
    ys = rews_seq
    xs = np.arange(len(rews_seq))
    plt.plot(xs, ys, 'ro-')

    plt.savefig('rewards_%s.png' % name)
    plt.clf()


def autodir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def find_model_metadata(metadata_dir, config_name):
    metadata_paths = glob.glob(metadata_dir + '/%s-*/' % config_name)
    print(metadata_dir, config_name, metadata_paths)
    if not metadata_paths:
        raise ValueError('No metadata files for config %s' % config_name)
    elif len(metadata_paths) > 1:
        raise ValueError('Multiple metadata files for config %s' % config_name)
    return metadata_paths[0]


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
