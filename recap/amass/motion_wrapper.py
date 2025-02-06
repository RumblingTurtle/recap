import torch


class AMASSMotionWrapper:
    """
    Wrapper for AMASS motion data tensor for easy access to body positions and quaternions
    of the SMPLH model.
    """

    def __init__(self, positions: torch.Tensor, rotations: torch.Tensor):
        self.positions = positions.float()
        self.rotations = rotations.float()

    def get_body_positions(self, body_name):
        return self.positions[:, SMPLH_JOINT_NAMES_TO_IDX[body_name]]

    def get_body_rotations(self, body_name):
        """
        Returns the global frame rotation of the body
        """
        return self.rotations[:, SMPLH_JOINT_NAMES_TO_IDX[body_name]]

    def to(self, device: torch.device):
        self.positions = self.positions.to(device)
        self.rotations = self.rotations.to(device)

    def __len__(self):
        return self.positions.shape[0]

    def __iadd__(self, other):
        self.positions = torch.cat([self.positions, other.positions], dim=0)
        self.rotations = torch.cat([self.rotations, other.rotations], dim=0)
        return self


SMPLH_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

SMPLH_PARENT_INDICES = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
]

SMPLH_JOINT_NAMES_TO_IDX = {name: i for i, name in enumerate(SMPLH_JOINT_NAMES)}
