# Adding new datasets

In order to add a new dataset, you need to create a new class that inherits from `MappedIK` and implements the following methods:
1) Create a new class that inherits from `MappedIK`

```python
def get_dataset_position(self, body_name: str):
    """
    This method should return the position of the body in the world frame XYZ order
    """
    pass

def get_dataset_rotation(self, body_name: str):
    """
    This method should return the rotation matrix of the body in the world frame
    """
    pass

def __len__(self):
    """
    This method should return the number of frames in the motion
    """

def step_frame(self):
    """
    This method should move to the next frame in the motion.
    Accessing get_dataset_position and get_dataset_rotation 
    will now return the information for the next frame.
    """
    pass
```

2) Inherit from the `MappedIKConfig` and add any new paramets that you might want to use for your dataset
 
In order to understand how the `body_name` parameters are accessed, you can refer to the [`MappedIKConfig`](../recap/config/mapped_ik_config.py) and [`WBIKConfig`](../recap/config/wbik_config.py) classes:

```python
class MappedIKConfig(WBIKConfig):
    """
    MappedIK will try to access body_to_data_map[{key}] during the IK solving process.
    the goal of this mapping is to map the body names in the dataset to their aliases
    """
    body_to_data_map = {
        "root": "dataset_root_name",
        "left_knee": "dataset_left_knee_name",
        "right_knee": "dataset_right_knee_name",
        "left_foot": "dataset_left_foot_name",
        "right_foot": "dataset_right_foot_name",
        "left_elbow": "dataset_left_elbow_name",
        "right_elbow": "dataset_right_elbow_name",
        "left_hand": "dataset_left_hand_name",
        "right_hand": "dataset_right_hand_name",
    }
```
```python
class WBIKConfig:
    .....
    task_weights = {
        """
        The first task determines relative weights 
        of the joint velocity
        """
        "joint_velocity": 2.0,

        """
        The last three limit maximum linear, angular root and joint accelerations
        """
        "max_joint_acceleration": np.inf,
        "max_root_lin_acceleration": np.inf,
        "max_root_ang_acceleration": np.inf,

        "position": {
            """
            The position task determines relative weights of the root, foot, hand, knee, elbow and COM tasks.
            You can freely remove any of the body names from the dictionary 
            to disable the task for that body or zero out the weight to make it have no effect.
            """
            "root": 2.0,
            "foot": 10.0,
            "hand": 4.0,
            "knee": 4.0,
            "elbow": 4.0,
        },
        "rotation": {
            """
            The same rule applies to the rotation task. But note that the center of mass has no rotation task.
            """
            "root": 5.0,
            "foot": 1.0,
        },
    }
    """
    The idea behind the body_to_model_map is the same as the body_to_data_map 
    but for the MJCF model of the robot. Note that there are extra bodies in this dictionary. 
    This can be used for instance to rescale the skeleton according 
    to the robot model parameters procedurally as in the [`CMURetarget`](../recap/cmu/retarget.py) class.
    """
    body_to_model_map = {
        "root": "",
        "left_hip": "",
        "right_hip": "",
        "left_knee": "",
        "right_knee": "",
        "left_foot": "",
        "right_foot": "",
        "left_hand": "",
        "right_hand": "",
        "left_elbow": "",
        "right_elbow": "",
        "left_shoulder": "",
        "right_shoulder": "",
    }
```
Any intermediate data handling can be done in the inherited class constructor.
Make sure to check out the [`CMURetarget`](../recap/cmu/retarget.py) class for an example of how to procedurally rescale the skeleton according to the robot model parameters.

3) You're all set! Now you can use your new class to retarget motions from the dataset. But you might want to use it on the actual robot model [Adding new robot models](adding_new_robot_models.md)