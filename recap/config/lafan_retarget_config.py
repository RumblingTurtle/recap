from recap.config.mapped_ik_config import MappedIKConfig


class LAFANRetargetConfig(MappedIKConfig):
    body_to_data_map: dict = {
        "root": "Spine2",
        "left_knee": "LeftLeg",
        "right_knee": "RightLeg",
        "left_foot": "LeftFoot",
        "right_foot": "RightFoot",
        "left_elbow": "LeftForeArm",
        "right_elbow": "RightForeArm",
        "left_hand": "LeftHand",
        "right_hand": "RightHand",
    }
