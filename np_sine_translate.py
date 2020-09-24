import numpy as np
import tensorflow as tf

from nn.np_layers import LatentModel, Attention
from utils.data_generator import SineCurvesReader
from utils.plots import plot_functions

TRAIN = False
TRAINING_ITERATIONS = 100000
MAX_CONTEXT_POINTS = 50
PLOT_AFTER = 10000
HIDDEN_SIZE = 128
ATTENTION_TYPE = 'uniform'
SAVE_DIR = 'metadata_np_translate'

#  ------------ some manuals for test plots
TEST_OFFSET = 3.  # 0.7 , 3
PLOT_Y_LIMITS = (-0.2, 2.2) if TEST_OFFSET < 3 else (1.8, 4.2)
TEST_CONTEXT = 100
N_TEST_PLOTS = 1
N_TEST_SAMPLES = 2500

tf.reset_default_graph()
tf.set_random_seed(317070)

dataset_train = SineCurvesReader(batch_size=16,
                                 max_num_context=MAX_CONTEXT_POINTS,
                                 offset=(-2., 2.), scale=0.5)

dataset_test = SineCurvesReader(batch_size=1,
                                max_num_context=MAX_CONTEXT_POINTS,
                                testing=True,
                                offset=TEST_OFFSET,
                                scale=0.5)

data_train = dataset_train.generate_curves()
data_test = dataset_test.generate_curves()

# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
latent_encoder_output_sizes = [HIDDEN_SIZE] * 4
num_latents = 2
deterministic_encoder_output_sizes = [HIDDEN_SIZE] * 4
decoder_output_sizes = [HIDDEN_SIZE] * 2 + [2]
use_deterministic_path = False

attention = Attention(rep='identity', output_sizes=None, att_type='uniform')

# Define the model
model = LatentModel(latent_encoder_output_sizes, num_latents,
                    decoder_output_sizes, use_deterministic_path,
                    deterministic_encoder_output_sizes, attention=attention,
                    n_test_samples=N_TEST_SAMPLES)

# Define the loss
_, _, log_prob, _, loss = model(data_train.query, data_train.num_total_points,
                                data_train.target_y)

# Get the predicted mean and variance at the target points for the testing set
mu, sigma, _, _, _ = model(data_test.query, data_test.num_total_points, n_context=TEST_CONTEXT)

# Set up the optimizer and train step
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Train and plot
with tf.Session() as sess:
    sess.run(init)

    if TRAIN:

        for it in range(TRAINING_ITERATIONS):
            sess.run([train_step])

            # Plot the predictions in `PLOT_AFTER` intervals
            if (it + 1) % PLOT_AFTER == 0:
                loss_value, pred_y, std_y, target_y, whole_query = sess.run(
                    [loss, mu, sigma, data_test.target_y,
                     data_test.query])

                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, loss: {}'.format(it + 1, loss_value))

                # Plot the prediction and the context
                plot_functions(target_x, target_y, context_x, context_y, pred_y, std_y,
                               y_limits=None,
                               plot_name='np_%s_%s.png' % (num_latents, it + 1))

        print('test')
        print('num latents, num hidden', num_latents, HIDDEN_SIZE)
        losses = []
        for i in range(100):
            loss_value, pred_y, std_y, target_y, whole_query = sess.run(
                [loss, mu, sigma, data_test.target_y, data_test.query])
            print(i, loss_value)
            losses.append(loss_value)

            if i < 10:
                (context_x, context_y), target_x = whole_query
                plot_functions(target_x, target_y, context_x, context_y, pred_y, std_y,
                               y_limits=None,
                               plot_name='np_%s_%s_test.png' % (num_latents, i))

        print('mean loss', np.mean(losses))
        saver.save(sess, SAVE_DIR + '/params.ckpt')

    # ------------------- TEST -------------------------
    else:
        ckpt_file = SAVE_DIR + '/params.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))

        for it in range(N_TEST_PLOTS):
            loss_value, mean_y, std_y, target_y, whole_query = sess.run(
                [loss, mu, sigma, data_test.target_y, data_test.query])

            mean_preds_y, stds_y = [], []
            for i in range(mean_y.shape[1]):
                samples = []
                for j in range(mean_y.shape[0]):
                    samples.append(np.random.normal(loc=mean_y[j, i], scale=std_y[j, i]))
                samples = np.asarray(samples).flatten()
                mean_preds_y.append(np.mean(samples))
                stds_y.append(np.std(samples))

            pred_y = np.asarray(mean_preds_y)[None, :, None]
            std_y = np.asarray(stds_y)[None, :, None]

            (context_x, context_y), target_x = whole_query
            context_x = context_x[:, :TEST_CONTEXT]
            context_y = context_y[:, :TEST_CONTEXT]
            plot_functions(target_x, target_y, context_x, context_y, pred_y, std_y,
                           plot_name='test_np_%s_%s.png' % (TEST_OFFSET, TEST_CONTEXT),
                           y_limits=PLOT_Y_LIMITS)
