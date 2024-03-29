import os
from gym import utils
from gym.envs.robotics import baxter_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('baxter', 'reach.xml')


class BaxterReachEnv(baxter_env.BaxterEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.0, #-----baxter
            'robot0:slide1': 0.0, #-----baxter
            'robot0:slide2': 0.0, #-----baxter
            'right_s0': -0.2286, #-----baxter
            'right_s1': -1.0044, #-----baxter
            'right_w0': -0.6535, #-----baxter
            'right_w1':  1.0028, #-----baxter
            'right_w2':  0.5196, #-----baxter
            'right_e0':  1.2598, #-----baxter
            'right_e1':  2.0003, #-----baxter
            'left_w0': 0.6477233869445801, #-----baxter
            'left_w1': 1.007825376489258, #-----baxter
            'left_w2': -0.48282045243530275, #-----baxter
            'left_e0': -1.1504855895996096, #-----baxter
            'left_e1': 1.9232284106140138, #-----baxter
            'left_s0': -0.07823302009277344, #-----baxter
            'left_s1': -0.9675583808532715, #-----baxter
        }
        baxter_env.BaxterEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)