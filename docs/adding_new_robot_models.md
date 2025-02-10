# Adding new robot models

In order to add a new robot model, you need to follow these steps:

1) Inherit from the `Yournewclass(MappedIK)` class and add the following required class parameters:

1) `mjcf_path`: Path to the MJCF model of the robot
2) `body_to_model_map`
3) `body_to_data_map`

Don't know what the last two are for? Check out the [Adding new datasets](adding_new_datasets.md).

# Adding new frames to the robot model

Sometimes your robot model might not have the frames that align well with the dataset frames. In that case, you can add new frames to the robot model:

```python
from robot_descriptions.my_awesome_robot import MJCF_PATH
editor = MJCFModelEditor.from_path(MJCF_PATH)
# Add a new body to the torso link to serve as the center of the torso
# The position and quaternion are defined in the parent link frame
editor.add_body(
    body_name="torso_center",
    parent_name="torso_link",
    position=np.array([0, 0, 0.25]),
    quaternion=np.array([1, 0, 0, 0]),
)
editor.compile()

# Save the edited model
if not os.path.exists("my_awesome_robot/"):
    os.makedirs("my_awesome_robot/")
editor.save("my_awesome_robot/my_awesome_robot.xml")
```
You can now define `mjcf_path` as `"my_awesome_robot/my_awesome_robot.xml"` and use the new robot model. New frames can be placed in the `body_to_model_map` dictionary.