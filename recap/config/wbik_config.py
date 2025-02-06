class WBIKConfig:
    """
    mjcf_path (str): Path to the MJCF model of the robot
    termination_velocity (float): Terminates integration when joints reach specified velocity
    max_iters (float): Max integration steps
    skip_iters (float): Max integration steps for initial position, throws exception when reached
    contact_velocity (float): foot velocity contact threshold
    height_offset (float): global z coordinate offset for the motion data
    yaw_only_feet (bool): wether to ignore pitch and roll orientation of the foot task
    joint_vel_weight (float): joint velocity constraints weight
    joint_limit_scale (float): scales min and max limits of the joints
    contact_target_lerp (float): alpha for exponential filtering of the desired foot positions and rotations when not in contact with the ground
    com_pos_weight (float): weight for keeping the com close to the foot support line
    body_to_model_map (dict): MJCF to task name mapping of the corresponding frames
    step_dt (float): integration step size
    """

    mjcf_path: str
    termination_velocity: float = 5e-1
    max_iters: int = 1
    skip_iters: int = 200
    contact_velocity: float = 0.6
    height_offset: float = 0.06
    yaw_only_feet: bool = False
    joint_limit_scale: float = 0.7
    contact_target_lerp: float = 0.3
    step_dt: float = 1 / 60
    task_weights = {
        "joint_velocity": 2.0,
        "root_linear_velocity": 2.0,
        "root_angular_velocity": 2.0,
        "position": {
            "root": 2.0,
            "foot": 10.0,
            "hand": 4.0,
            "knee": 4.0,
            "elbow": 4.0,
            "com": 1.0,
        },
        "rotation": {
            "root": 5.0,
            "foot": 1.0,
        },
    }
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
