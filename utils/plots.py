import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_functions(target_x, target_y, context_x, context_y, pred_y, std, plot_name,
                   y_limits=(-2, 2), x_limits=(-2, 2)):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape [B,num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [B,num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [B,num_contexts,1] that contains
          the x values of the context points.
      context_y: An array of shape [B,num_contexts,1] that contains
          the y values of the context points.
      pred_y: An array of shape [B,num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [B,num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    if y_limits is None:
        y_limits = [min(pred_y[0]), max(pred_y[0])]

    mse = np.mean((target_y[0] - pred_y[0]) ** 2)
    print('MSE:', mse)
    print('std', np.max(std), np.min(std))

    # Plot everything
    if int(y_limits[1]) == 3:
        f = plt.figure(figsize=(583. / 100, 442. * 1.5 / 100))
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=3)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=3)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=12)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - 2. * std[0, :, 0],
        pred_y[0, :, 0] + 2. * std[0, :, 0],
        alpha=0.7,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([np.rint(y_limits[0]), sum(y_limits) / 2., np.rint(y_limits[1])], fontsize=25)
    plt.xticks([x_limits[0], sum(x_limits) / 2., x_limits[1]], fontsize=25)
    plt.ylim(y_limits)
    plt.xlim(x_limits)

    ax = plt.gca()
    x_text = 0.22 if int(y_limits[1]) == 3 else 0.2
    y_text = 0.95 if int(y_limits[1]) == 3 else 0.93
    plt.text(x_text, y_text, "MSE:{:.3f}".format(round(mse, 3)), fontsize=25, ha='center', va='center',
             transform=ax.transAxes)

    plt.grid('off')
    plt.grid(False)
    if len(context_x[0]) == 1:
        plt.ylabel('x', fontsize=25)
    if int(y_limits[1]) >= 3:
        plt.xlabel('h', fontsize=25)

    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()

def plot_learning_curves(x, y, plot_name):
    plt.plot(x, y, 'b')
    plt.savefig(plot_name, bbox_inches='tight')
    plt.close()
