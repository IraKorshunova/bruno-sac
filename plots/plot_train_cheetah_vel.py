import pickle

import numpy as np
import seaborn as sns

sns.set(style="darkgrid")
sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
                             "lines.linewidth": 2.5})

from utils import misc_utils

if __name__ == "__main__":
    color1 = sns.xkcd_rgb["pale red"]
    color2 = sns.xkcd_rgb["amber"]
    color3 = sns.xkcd_rgb["dark red"]
    color4 = sns.xkcd_rgb["brownish"]

    config_name = 'meta_cheetah_vel'
    save_dir = misc_utils.find_model_metadata('metadata', config_name)
    with open(save_dir + '/meta.pkl', 'rb') as f:
        d = pickle.load(f)

    n_steps = d.keys()
    n_steps = sorted(n_steps)
    returns = []
    lower_bound, upper_bound = [], []
    env_ids = d[n_steps[0]].keys()
    print(env_ids)
    for i in n_steps:
        mean = np.mean([d[i][env_id] for env_id in env_ids])
        min = np.min([d[i][env_id] for env_id in env_ids])
        max = np.max([d[i][env_id] for env_id in env_ids])
        returns.append(mean)
        upper_bound.append(max)
        lower_bound.append(min)

    sns_plot = sns.lineplot(x=n_steps, y=returns, label='BrunoSAC', color=color1)
    sns_plot.fill_between(n_steps, lower_bound, upper_bound, alpha=.3, facecolors=color1)

    # ORACLE
    config_name_oracle = 'meta_cheetah_vel_oracle'
    save_dir = misc_utils.find_model_metadata('metadata', config_name_oracle)
    with open(save_dir + '/meta.pkl', 'rb') as f:
        d = pickle.load(f)

    n_steps = d.keys()
    n_steps = sorted(n_steps)
    returns = []
    env_ids = d[n_steps[0]].keys()
    for i in n_steps:
        mean = np.mean([d[i][env_id] for env_id in env_ids])
        std = np.std([d[i][env_id] for env_id in env_ids])
        returns.append(mean)

    sns_plot = sns.lineplot(x=n_steps, y=returns, label='OracleSAC', color=color3)
 
    # common
    sns_plot.set(xlabel='environment steps', ylabel='average return')
    xlabels = []
    for x in sns_plot.get_xticks():
        if x < 1e6:
            xlabels.append('{:.0f}'.format(x / 1000) + 'K')
        else:
            xlabels.append('{:.0f}'.format(x / 1000000) + 'M')

    sns_plot.tick_params(labelsize=18)
    sns_plot.legend(fontsize=18, loc=4)
    sns_plot.set_xticklabels(xlabels)
    fig = sns_plot.get_figure()
    fig.savefig('train_%s.png' % config_name, bbox_inches='tight')
