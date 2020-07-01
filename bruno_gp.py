import pickle

import numpy as np
import tensorflow as tf

from nn.bruno import BrunoNet
from utils.data_generator import GPCurvesReader
from utils.plots import plot_functions, plot_learning_curves

TRAIN = True
TRAINING_ITERATIONS = 100000
MAX_CONTEXT_POINTS = 50
PLOT_AFTER = 5000
SAVE_DIR = 'metadata_bruno_gp'
NUM_LATENTS = 2

#  ------------ some manuals for test plots
PLOT_Y_LIMITS = (-2.2, 2.2)
N_TEST_PLOTS = 10
TEST_CONTEXT = [1, 10, 100]

tf.reset_default_graph()
tf.set_random_seed(317070)

dataset_train = GPCurvesReader(batch_size=16,
                               max_num_context=MAX_CONTEXT_POINTS,
                               random_kernel_parameters=False)

dataset_test = GPCurvesReader(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True,
                              random_kernel_parameters=False)

data_train = dataset_train.generate_curves()
data_test = dataset_test.generate_curves()

# Define the model
bruno_net = BrunoNet(extra_dims=NUM_LATENTS - 1, maf_num_hidden=32, n_maf_layers=7,
                     maf_nonlinearity=tf.nn.leaky_relu, noise_distribution='gaussian',
                     bijection='maf', weight_norm=True,
                     process='gp')

# Define the loss
loss = bruno_net.loss(data_train.query, data_train.target_y)

# Get the predicted mean and variance at the target points for the testing set
median, _ = bruno_net.test_median(data_test.query, n_context=TEST_CONTEXT)
mean, std = bruno_net.test_sample_mean(data_test.query, n_context=TEST_CONTEXT)

# Set up the optimizer and train step
params = tf.trainable_variables(bruno_net.name)
print('bruno params', params)

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss, global_step=None, name='adam_step')
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Train and plot
with tf.Session() as sess:
    sess.run(init)

    if TRAIN:
        losses = []
        it2loss = {}

        for it in range(TRAINING_ITERATIONS):
            loss_value, _ = sess.run([loss, train_step])
            losses.append(loss_value)
            it2loss[it] = loss_value

            # Plot the predictions in `PLOT_AFTER` intervals
            if (it + 1) % PLOT_AFTER == 0 or it == 0:
                loss_value, pred_y, target_y, whole_query = sess.run(
                    [loss, median, data_test.target_y, data_test.query])

                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it + 1, np.mean(losses)))
                print('var, corr:', sess.run([bruno_net.gp_layer.var, bruno_net.gp_layer.corr]))
                losses = []

                # Plot the prediction and the context
                plot_functions(target_x, target_y, context_x, context_y, pred_y, pred_y * 0.,
                               plot_name='bruno_%s_%s.png' % (NUM_LATENTS, it + 1))

        print('test')
        losses = []
        for i in range(1000):
            loss_value, pred_y, target_y, whole_query = sess.run(
                [loss, median, data_test.target_y, data_test.query])
            losses.append(loss_value)

            if i < 10:
                print(i, loss_value)
                (context_x, context_y), target_x = whole_query
                plot_functions(target_x, target_y, context_x, context_y, pred_y, pred_y * 0.,
                               plot_name='bruno_%s_%s_test.png' % (NUM_LATENTS, i))

        print('mean loss', np.mean(losses))

        # save everything
        saver.save(sess, SAVE_DIR + '/params.ckpt')
        with open(SAVE_DIR + '/meta.pkl', 'wb') as f:
            pickle.dump(it2loss, f)

    # ------------------- TEST -------------------------
    else:
        ckpt_file = SAVE_DIR + '/params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

        for it in range(N_TEST_PLOTS):
            loss_value, pred_y, mean_y, std_y, target_y, whole_query = sess.run(
                [loss, median, mean, std, data_test.target_y, data_test.query])

            (context_x, context_y), target_x = whole_query

            print(it + 1, loss_value)

            for i, nc in enumerate(TEST_CONTEXT):
                plot_functions(target_x, target_y, context_x[:, :nc], context_y[:, :nc],
                               mean_y[i:i + 1], std_y[i:i + 1],
                               plot_name='test_bruno_%s_%s.png' % (it, nc))

        with open(SAVE_DIR + '/meta.pkl', 'rb') as f:
            d = pickle.load(f)
        n_steps, loss_v = [], []
        for k, v in d.items():
            n_steps.append(k)
            loss_v.append(v)
            # print(k, v)
        plot_learning_curves(x=n_steps, y=loss_v, plot_name='learning_curve_%s.png' % SAVE_DIR)
