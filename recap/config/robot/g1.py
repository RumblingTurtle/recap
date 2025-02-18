from recap.config.cmu_retarget_config import CMURetargetConfig
from recap.config.amass_retarget_config import AMASSRetargetConfig
from recap.config.lafan_retarget_config import LAFANRetargetConfig

from robot_descriptions.g1_mj_description import MJCF_PATH
from recap.mujoco.model_editor import MJCFModelEditor
import numpy as np
import os

# Add a new body to the torso link to serve as the center of the torso
editor = MJCFModelEditor.from_path(MJCF_PATH)
editor.add_body("torso_center", "torso_link", np.array([0, 0, 0.25]), np.array([1, 0, 0, 0]))
editor.compile()

save_path = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(save_path, f"{os.path.basename(MJCF_PATH).split('.')[0]}_edited.xml")

if not os.path.exists(save_path):
    os.makedirs(save_path)
editor.save(MODEL_PATH)


class G1_AMASS_CONFIG(AMASSRetargetConfig):
    template_scale = 0.8013491034507751
    mjcf_path = MODEL_PATH
    joint_limit_scale: float = 0.9
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_roll_link",
        "right_hip": "right_hip_roll_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
        "left_hand": "left_zero_link",
        "right_hand": "right_zero_link",
        "left_elbow": "left_elbow_pitch_link",
        "right_elbow": "right_elbow_pitch_link",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
    beta = np.array(
        [
            1.5236,
            -2.0745,
            0.5834,
            1.0016,
            0.9520,
            1.4606,
            -1.9969,
            1.4413,
            -0.2042,
            1.4710,
            0.6569,
            -1.3871,
            -1.4304,
            -1.3851,
            -1.5396,
            -0.9727,
        ],
    )


class G1_CMU_CONFIG(CMURetargetConfig):
    spine_scale = 1.2
    shoulder_scale = 1.5
    hip_length_scale = 1.0
    hip_width_scale = 1.5
    tibia_length_scale = 1.0
    elbow_length_scale = 1.0
    forearm_length_scale = 0.7
    height_offset = 0
    movement_scale = 1
    output_fps = 60
    dataset_path: str = "./data"
    mjcf_path = MODEL_PATH
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_roll_link",
        "right_hip": "right_hip_roll_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
        "left_hand": "left_zero_link",
        "right_hand": "right_zero_link",
        "left_elbow": "left_elbow_pitch_link",
        "right_elbow": "right_elbow_pitch_link",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }


class G1_LAFAN_CONFIG(LAFANRetargetConfig):
    template_scale = 0.8013491034507751
    mjcf_path = MODEL_PATH
    joint_limit_scale: float = 0.9
    body_to_model_map = {
        "root": "torso_center",
        "left_hip": "left_hip_roll_link",
        "right_hip": "right_hip_roll_link",
        "left_knee": "left_knee_link",
        "right_knee": "right_knee_link",
        "left_foot": "left_ankle_roll_link",
        "right_foot": "right_ankle_roll_link",
        "left_hand": "left_zero_link",
        "right_hand": "right_zero_link",
        "left_elbow": "left_elbow_pitch_link",
        "right_elbow": "right_elbow_pitch_link",
        "left_shoulder": "left_shoulder_roll_link",
        "right_shoulder": "right_shoulder_roll_link",
    }
