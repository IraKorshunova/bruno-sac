from gym.envs.registration import register

import defaults
from envs import env_utils

""" Half Cheetah Direction """

for d in [-1., 1.]:
    d_str = env_utils.param_to_str(d)
    env_id = "HalfCheetahDirEnv_{}-v0".format(d_str)
    env_entry_point = "envs.half_cheetah_dir:HalfCheetahDirEnv"
    kwargs = {"goal_direction": d}
    register(id=env_id, entry_point=env_entry_point, kwargs=kwargs)

""" Half Cheetah Velocity """

for v in defaults.half_cheetah_velocities:
    v_str = env_utils.param_to_str(v)
    env_id = "HalfCheetahVelEnv_{}-v0".format(v_str)
    env_entry_point = "envs.half_cheetah_vel:HalfCheetahVelEnv"
    kwargs = {"goal_velocity": v}
    register(id=env_id, entry_point=env_entry_point, kwargs=kwargs)
