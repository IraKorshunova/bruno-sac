import pickle

import numpy as np
import seaborn as sns

sns.set(style="darkgrid")
sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                             "lines.linewidth": 1.5})
from utils import misc_utils

if __name__ == "__main__":
    plot_mean = True
    color1 = sns.xkcd_rgb["pale red"]
    color2 = sns.xkcd_rgb["amber"]
    color3 = sns.xkcd_rgb["dark red"]
    color4 = sns.xkcd_rgb["brownish"]
    color5 = sns.xkcd_rgb["black"]
    color6 = sns.xkcd_rgb["red"]

    # ORACLE

    config_name_oracle = 'meta_cheetah_oracle'
    save_dir = misc_utils.find_model_metadata('metadata', config_name_oracle)

    with open(save_dir + '/test_rewards.pkl', 'rb') as f:
        d = pickle.load(f)

    env_ids = list(d.keys())

    if plot_mean:
        episode_ids = list(d[env_ids[0]].keys())
        n_steps = list(range(len(d[env_ids[0]][episode_ids[0]])))
        rewards = []
        lower_bound, upper_bound = [], []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            min = np.min([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            max = np.max([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            rewards.append(mean)
            upper_bound.append(max)
            lower_bound.append(min)

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='OracleSAC', color=color5)
        sns_plot.fill_between(n_steps, lower_bound, upper_bound, alpha=.3, facecolors=color5)
        print('average return oracle', sum(rewards))
    else:

        # backward
        env_id = 'HalfCheetahDirEnv_-1.0000-v0'
        episode_ids = list(d[env_id].keys())
        n_steps = list(range(len(d[env_id][episode_ids[0]])))
        rewards = []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for e in episode_ids])
            rewards.append(mean)

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='OracleSAC bwd', color=color3)

        print('average return oracle backward', sum(rewards))

        # forward
        env_id = 'HalfCheetahDirEnv_1.0000-v0'
        episode_ids = list(d[env_id].keys())
        n_steps = list(range(len(d[env_id][episode_ids[0]])))
        rewards = []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for e in episode_ids])
            rewards.append(mean)

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='OracleSAC fwd', color=color4)
        print('average return oracle forward', sum(rewards))

    # BRUNO

    config_name = 'meta_cheetah_dir'
    save_dir = misc_utils.find_model_metadata('metadata', config_name)
    with open(save_dir + '/test_rewards.pkl', 'rb') as f:
        d = pickle.load(f)

    env_ids = list(d.keys())

    if plot_mean:
        episode_ids = list(d[env_ids[0]].keys())
        n_steps = list(range(len(d[env_ids[0]][episode_ids[0]])))
        rewards = []
        lower_bound, upper_bound = [], []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            min = np.min([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            max = np.max([d[env_id][e][i] for env_id in env_ids for e in episode_ids])
            rewards.append(mean)
            upper_bound.append(max)
            lower_bound.append(min)

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='BrunoSAC', color=color6)
        sns_plot.fill_between(n_steps, lower_bound, upper_bound, alpha=.3, facecolors=color6)
        print('average return', sum(rewards))
    else:

        # backward
        env_id = 'HalfCheetahDirEnv_-1.0000-v0'
        episode_ids = list(d[env_id].keys())
        n_steps = list(range(len(d[env_id][episode_ids[0]])))
        rewards = []
        lower_bound, upper_bound = [], []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for e in episode_ids])
            rewards.append(mean)
            upper_bound.append(np.max([d[env_id][e][i] for e in episode_ids]))
            lower_bound.append(np.min([d[env_id][e][i] for e in episode_ids]))

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='BrunoSAC bwd', color=color1)
        sns_plot.fill_between(n_steps, lower_bound, upper_bound, alpha=.3, facecolors=color1)
        print('average return backward', sum(rewards))

        # forward
        env_id = 'HalfCheetahDirEnv_1.0000-v0'
        episode_ids = list(d[env_id].keys())
        n_steps = list(range(len(d[env_id][episode_ids[0]])))
        rewards = []
        lower_bound, upper_bound = [], []
        for i in n_steps:
            mean = np.mean([d[env_id][e][i] for e in episode_ids])
            rewards.append(mean)
            upper_bound.append(np.max([d[env_id][e][i] for e in episode_ids]))
            lower_bound.append(np.min([d[env_id][e][i] for e in episode_ids]))

        sns_plot = sns.lineplot(x=n_steps, y=rewards, label='BrunoSAC fwd', color=color2)
        sns_plot.fill_between(n_steps, lower_bound, upper_bound, alpha=.3, facecolors=color2)
        print('average return forward', sum(rewards))

    # common
    sns_plot.set(xlabel='environment steps', ylabel='average reward')
    sns_plot.tick_params(labelsize=18)
    sns_plot.legend(fontsize=18, loc=4)
    fig = sns_plot.get_figure()
    fig.savefig('test_%s.png' % config_name, bbox_inches='tight')
