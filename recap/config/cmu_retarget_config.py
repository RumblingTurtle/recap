from recap.config.mapped_ik_config import MappedIKConfig


class CMURetargetConfig(MappedIKConfig):
    """
    NOTE: All scaling factors are strictly positive
    """

    # Pelvis -> head
    spine_scale: float
    # Thorax -> shoulder
    shoulder_scale: float
    # Hip -> knee
    hip_length_scale: float
    # Hip -> pelvis
    hip_width_scale: float
    # Knee -> ankle
    tibia_length_scale: float
    # Shoulder -> elbow
    elbow_length_scale: float
    # Elbow -> hand
    forearm_length_scale: float

    # Root height
    height_offset: float

    # XY direction
    movement_scale: float

    # Value in range[1,120]
    output_fps: int

    # Path to ./all_asfamc folder
    dataset_path: str

    body_to_data_map = {
        "root": "thorax",
        "left_knee": "lfemur",
        "right_knee": "rfemur",
        "left_foot": "lfoot",
        "right_foot": "rfoot",
        "left_elbow": "lhumerus",
        "right_elbow": "rhumerus",
        "left_hand": "lradius",
        "right_hand": "rradius",
    }
