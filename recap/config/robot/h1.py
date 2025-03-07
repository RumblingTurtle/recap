from recap.config.amass_retarget_config import AMASSRetargetConfig
from recap.config.cmu_retarget_config import CMURetargetConfig
from recap.config.lafan_retarget_config import LAFANRetargetConfig
import numpy as np
from robot_descriptions.h1_mj_description import MJCF_PATH
from recap.mujoco.model_editor import MJCFModelEditor
import os

# Add a new body to the torso link to serve as the center of the torso
editor = MJCFModelEditor.from_path(MJCF_PATH)
editor.add_body("torso_center", "torso_link", np.array([0.0, 0, 0.4]), np.array([1, 0, 0, 0]))
editor.add_body("spine", "torso_link", np.array([0.0, 0, 0.17]), np.array([1, 0, 0, 0]))
# Moving the elbow slightly backward to avoid singular configurations
editor.add_body(
    "right_elbow_center",
    "right_shoulder_yaw_link",
    np.array([-0.05, 0, -0.2]),
    np.array([1, 0, 0, 0]),
)
editor.add_body(
    "left_elbow_center",
    "left_shoulder_yaw_link",
    np.array([-0.05, 0, -0.2]),
    np.array([1, 0, 0, 0]),
)

editor.add_body(
    "right_hand",
    "right_elbow_link",
    np.array([0.26, 0, -0.025]),
    np.array([1, 0, 0, 0]),
)
editor.add_body("left_hand", "left_elbow_link", np.array([0.26, 0, -0.025]), np.array([1, 0, 0, 0]))

editor.compile()

save_path = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")

if not os.path.exists(save_path):
    os.makedirs(save_path)
editor.save(MODEL_PATH)


class H1_AMASS_CONFIG(AMASSRetargetConfig):
    mjcf_path = MODEL_PATH
    joint_limit_scale = 0.9
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_pitch_link",
        "right_hip": "right_hip_pitch_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_link",
        "right_foot": "right_ankle_link",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_elbow": "left_elbow_center",
        "right_elbow": "right_elbow_center",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    beta = np.array(
        [
            -3.5657,
            -2.1251,
            4.6726,
            10.7957,
            8.8308,
            9.1279,
            9.8109,
            -3.4021,
            2.5436,
            -7.7658,
            6.8073,
            -6.5959,
            -11.9745,
            -10.1198,
            -8.4182,
            10.9544,
        ],
    )
    template_scale = 1.0128
    extra_bodies = ["imu"]


class H1_CMU_CONFIG(CMURetargetConfig):
    spine_scale = 1
    shoulder_scale = 1
    hip_length_scale = 1
    hip_width_scale = 1
    tibia_length_scale = 1
    elbow_length_scale = 1
    forearm_length_scale = 1
    height_offset = 0
    movement_scale = 1
    output_fps = 60
    dataset_path: str = "./data"
    mjcf_path = MODEL_PATH
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_pitch_link",
        "right_hip": "right_hip_pitch_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_link",
        "right_foot": "right_ankle_link",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_elbow": "left_elbow_center",
        "right_elbow": "right_elbow_center",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }


class H1_LAFAN_CONFIG(LAFANRetargetConfig):
    mjcf_path = MODEL_PATH
    joint_limit_scale = 0.9
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_pitch_link",
        "right_hip": "right_hip_pitch_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_link",
        "right_foot": "right_ankle_link",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_elbow": "left_elbow_center",
        "right_elbow": "right_elbow_center",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    template_scale = 1.0128
    body_to_model_map = {
        "root": "spine",
        "left_hip": "left_hip_pitch_link",
        "right_hip": "right_hip_pitch_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_link",
        "right_foot": "right_ankle_link",
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "left_elbow": "left_elbow_center",
        "right_elbow": "right_elbow_center",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    extra_bodies = ["imu"]
