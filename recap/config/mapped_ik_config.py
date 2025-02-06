from recap.config.wbik_config import WBIKConfig


class MappedIKConfig(WBIKConfig):
    body_to_data_map = {
        "root": "",
        "left_knee": "",
        "right_knee": "",
        "left_foot": "",
        "right_foot": "",
        "left_elbow": "",
        "right_elbow": "",
        "left_hand": "",
        "right_hand": "",
    }
