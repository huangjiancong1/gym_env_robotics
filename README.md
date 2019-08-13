# Robotics FetchEnv Goal Augmentation
This source is base on OpenAI-Gym robotics env

We will use the gripper goal as one of the reward contributors:
**object distance weight**: weight_o
**gripper distance weight**: weight_g
**object distance**: distance_o = ||ag_o - dg_o||
**gripper distance**: distance_g = ||ag_g - dg_g||

**distance function**:
$d = weight_o ||ag_o - dg_o|| + weight_g ||ag_g - dg_g||$

## Approach 1: define certain weight
```
weight_o = 0.7
weight_g = 0.3
```
## Approach 2: seperate the rewards and try to learn how to reach the object first
```
if distance_g <= distacne_threadhold_gripper:
  weight_o = 0
  weight_g = 1
  
 else:
  weight_o = 1
  weight_g = 0
```
