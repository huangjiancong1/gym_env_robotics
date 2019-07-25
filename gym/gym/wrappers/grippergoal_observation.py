import gym
import numpy as np


class GripperGoaltWrapper(gym.ObservationWrapper):
    """Flattens selected keys of a Dict observation space into
    an array and add the end-effector positions as the part of goal g = {g_object, g_gripper}. 
    I follow here(https://github.com/openai/gym/blob/9dea81b48a2e1d8f7e7a81211c0f09f627ee61a9/gym/envs/robotics/fetch_env.py#L112)
    to select the gripper position and object position.

    Output : 37 dimensions infomation
        0~6: achieved goal
        7~12: desired goal
        13~37: observation as before version

    Usages : 
        1. Copy this scripts under the path in gym/gym/wrappers
        2. Add this in your code(follow the example https://blog.openai.com/ingredients-for-robotics-research):
            from gym.wrappers import grippergoal_observation
            env = grippergoal_observation.GripperGoaltWrapper(env, dict_keys=list(keys))       

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
