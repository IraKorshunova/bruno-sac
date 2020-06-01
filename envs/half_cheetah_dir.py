import numpy as np

from envs.half_cheetah import HalfCheetahEnv


class HalfCheetahDirEnv(HalfCheetahEnv):

    def __init__(self, goal_direction):
        self.goal_direction = np.float32(goal_direction)
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self.goal_direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self.goal_direction)
        return (observation, reward, done, infos)

    def get_params(self):
        return np.asarray([self.goal_direction], dtype=np.float32)
