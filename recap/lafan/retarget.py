from recap.mapped_ik import MappedIK
from recap.quat_utils import quat2mat, yaw_matrix
import sys, os
import glob
import numpy as np

DATA_MODULE_PATH = os.path.join(os.path.dirname(__file__), "../../data/")
LAFAN_DATA_PATH = os.path.join(DATA_MODULE_PATH, "lafan1/data")
MOTION_PATHS = glob.glob(os.path.join(LAFAN_DATA_PATH, "*.bvh"))

# Can't use the code directly
sys.path.append(DATA_MODULE_PATH)
if os.path.exists(LAFAN_DATA_PATH):
    from lafan1.extract import get_lafan1_set, read_bvh
    from lafan1.utils import quat_fk


class LAFANRetarget(MappedIK):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def set_motion(self, path: str):
        anim = read_bvh(path, order="zyx")
        quats, self.positions = quat_fk(anim.quats, anim.pos, anim.parents)
        self.positions = self.positions[:, :, [2, 0, 1]]
        swap_transpose = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.mats = ((quat2mat(quats.reshape(-1, 4)))[:, [2, 0, 1], :] @ swap_transpose).reshape(-1, 22, 3, 3)

        # Remove global XY offset
        self.positions[:, :, :2] -= self.positions[0, 0, :2]

        # Realign feet with the torso along the z axis
        feet_indices = [FRAME_2_IDX[name] for name in ["LeftFoot", "RightFoot"]]
        feet_orientations = self.mats[0, feet_indices]
        root_orientation = self.mats[0, 0]
        rel_rotation = feet_orientations @ yaw_matrix(root_orientation.T)
        self.mats[:, feet_indices[0]] = self.mats[:, feet_indices[0]] @ rel_rotation[0]
        self.mats[:, feet_indices[1]] = self.mats[:, feet_indices[1]] @ rel_rotation[1]

        self.positions /= 100.0
        self.frame_idx = 0
        self.reset()

    def get_dataset_position(self, body_name: str):
        return self.positions[self.frame_idx, FRAME_2_IDX[body_name]]

    def get_dataset_rotation(self, body_name: str):
        return self.mats[self.frame_idx, FRAME_2_IDX[body_name]]

    def __len__(self):
        return self.mats.shape[0]

    def step_frame(self):
        self.frame_idx += 1


FRAME_NAMES = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

FRAME_2_IDX = {name: i for i, name in enumerate(FRAME_NAMES)}
