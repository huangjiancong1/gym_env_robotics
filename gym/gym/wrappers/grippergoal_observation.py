import gym
import numpy as np


class GripperGoaltWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array and add the end-effector positions as the part of goal g = {g_object, g_gripper}.

    Output : 37 dimensions infomation
        0~6: achieved goal
        7~12: desired goal
        13~37: observation as before version
    """
    def __init__(self, env, dict_keys):
        super(GripperGoaltWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        # Figure out observation_space dimension.
        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        assert isinstance(observation, dict)

        observ = observation['observation']
        gripper_pos = observ[0:3] # grip_position
        
        gripper_goal = observation['desired_goal'].ravel()
        if self.env.env.spec.id == 'FetchSlide-v1':
            gripper_goal = observ[4:6].ravel() # object_position
        
        achieved_goals = np.append(observation['achieved_goal'].ravel(), gripper_pos)
        desired_goals = np.append(observation['desired_goal'].ravel(), gripper_goal)
        observations = observ.ravel()
        
        return np.concatenate([achieved_goals,desired_goals,observations])