import numpy as np

# half cheetah velocities
np.random.seed(1337)
half_cheetah_velocities = np.random.uniform(0.0, 3.0, size=130)
half_cheetah_velocities = [round(x, 4) for x in half_cheetah_velocities]
train_cheetah_velocities = half_cheetah_velocities[:100]
test_cheetah_velocities = half_cheetah_velocities[100:]
