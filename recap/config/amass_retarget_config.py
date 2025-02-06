from recap.config.mapped_ik_config import MappedIKConfig
import numpy as np


class AMASSRetargetConfig(MappedIKConfig):
    template_scale: float = 1.0
    body_to_data_map: dict = {
        "root": "spine3",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_foot": "left_foot",
        "right_foot": "right_foot",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "left_hand": "left_wrist",
        "right_hand": "right_wrist",
    }
    beta: np.ndarray
