import numpy as np

# half cheetah vel
np.random.seed(1337)
half_cheetah_velocities = np.random.uniform(0.0, 3.0, size=130)
half_cheetah_velocities = [round(x, 4) for x in half_cheetah_velocities]
train_cheetah_velocities = half_cheetah_velocities[:100]
test_cheetah_velocities = half_cheetah_velocities[100:]

# cartpole: cart mass and pendulum length
cartpole_masses_train = [0.5, 1., 1.5, 2., 2.5]
cartpole_masses_test = [0.1, 0.25, 0.75, 1.75, 2.25, 3.]

cartpole_lengths_train = [0.25, 0.5, 0.75]
cartpole_lengths_test = [0.1, 0.2, 0.4, 0.6, 0.9, 0.98]

cartpole_params_train = [(np.float32(x), np.float32(y)) for x in cartpole_masses_train for y in cartpole_lengths_train]
cartpole_params_test = [(np.float32(x), np.float32(y)) for x in cartpole_masses_test for y in cartpole_lengths_test]
