# Solving IK without the dataset
You can solve IK for any arbitrary pose by setting the desired transforms manually.
First you have to initialize the [`WBIKConfig`](../recap/config/wbik_config.py) object and set the following parameters:

1) `mjcf_path` - Path to the MJCF model of the robot
2) `body_to_model_map` - Map of body aliases to mjcf names. For details on how to define this mapping see [Adding new datasets](adding_new_datasets.md)

```python
wbik_params = WBIKConfig()
wbik_params.mjcf_path = "path/to/your/model.xml"
wbik_params.body_to_model_map = {
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
# .... You can edit task weights here
    
wbik = WBIKSolver(wbik_params=wbik_params)
wbik.set_target_transform("root", root_position, root_rotation)
wbik.set_target_transform("right_foot", right_foot_position, right_foot_rotation)
wbik.set_target_transform("left_foot", left_foot_position, left_foot_rotation)
wbik.set_target_transform("left_knee", left_knee_position)
wbik.set_target_transform("right_knee", right_knee_position)
wbik.set_target_transform("left_elbow", left_elbow_position)
wbik.set_target_transform("right_elbow", right_elbow_position)
wbik.set_target_transform("left_hand", left_hand_position)
wbik.set_target_transform("right_hand", right_hand_position)
```

No you can solve the IK. `wbik.solve()` will return a [`PoseData`](./trajectory.md) object

```python
pose_data = wbik.solve() 
pose_data.transforms # list of BodyTransform objects
pose_data.q # np.ndarray of the minimal coordinates of the robot model
pose_data.body_aliases # map of body names to mjcf names
pose_data.model_root # name of the root body in mjcf model
pose_data.contacts # Contact states are update on every consecutive solve call
pose_data.position_error # absolute position error of all links wrt to the reference
pose_data.dt # integration step duration in seconds
pose_data.joint_order # order of the joints in the q vector
```

# Solving IK with the dataset

Check out the [CMU](../examples/retarget_cmu.py) and [AMASS](../examples/retarget_amass.py) examples for more details on how to use the dataset to solve IK.