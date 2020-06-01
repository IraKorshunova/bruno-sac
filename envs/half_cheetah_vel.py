import numpy as np

from envs.half_cheetah import HalfCheetahEnv


class HalfCheetahVelEnv(HalfCheetahEnv):

    def __init__(self, goal_velocity):
        self.goal_velocity = goal_velocity
        self._goal = self.goal_velocity
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost)
        return (observation, reward, done, infos)

    def get_params(self):
        return np.asarray([self.goal_velocity])
