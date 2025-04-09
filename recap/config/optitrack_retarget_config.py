from recap.config.mapped_ik_config import MappedIKConfig
import os
import glob


class OptitrackRetargetConfig(MappedIKConfig):
    body_to_data_map: dict = {
        "root": "Skeleton:Chest",
        "left_knee": "Skeleton:LShin",
        "right_knee": "Skeleton:RShin",
        "left_foot": "Skeleton:LFoot",
        "right_foot": "Skeleton:RFoot",
        "left_elbow": "Skeleton:LFArm",
        "right_elbow": "Skeleton:RFArm",
        "left_hand": "Skeleton:LHand",
        "right_hand": "Skeleton:RHand",
    }


DATA_MODULE_PATH = os.path.join(os.path.dirname(__file__), "../../data/optitrack")
MOTION_PATHS = glob.glob(os.path.join(DATA_MODULE_PATH, "*.csv"))

OPTITRACK_BODY_NAMES = [
    "Skeleton:Skeleton",
    "Skeleton:Ab",
    "Skeleton:Chest",
    "Skeleton:Neck",
    "Skeleton:Head",
    "Skeleton:LShoulder",
    "Skeleton:LUArm",
    "Skeleton:LFArm",
    "Skeleton:LHand",
    "Skeleton:LThumb1",
    "Skeleton:LThumb2",
    "Skeleton:LThumb3",
    "Skeleton:LIndex1",
    "Skeleton:LIndex2",
    "Skeleton:LIndex3",
    "Skeleton:LMiddle1",
    "Skeleton:LMiddle2",
    "Skeleton:LMiddle3",
    "Skeleton:LRing1",
    "Skeleton:LRing2",
    "Skeleton:LRing3",
    "Skeleton:LPinky1",
    "Skeleton:LPinky2",
    "Skeleton:LPinky3",
    "Skeleton:RShoulder",
    "Skeleton:RUArm",
    "Skeleton:RFArm",
    "Skeleton:RHand",
    "Skeleton:RThumb1",
    "Skeleton:RThumb2",
    "Skeleton:RThumb3",
    "Skeleton:RIndex1",
    "Skeleton:RIndex2",
    "Skeleton:RIndex3",
    "Skeleton:RMiddle1",
    "Skeleton:RMiddle2",
    "Skeleton:RMiddle3",
    "Skeleton:RRing1",
    "Skeleton:RRing2",
    "Skeleton:RRing3",
    "Skeleton:RPinky1",
    "Skeleton:RPinky2",
    "Skeleton:RPinky3",
    "Skeleton:LThigh",
    "Skeleton:LShin",
    "Skeleton:LFoot",
    "Skeleton:LToe",
    "Skeleton:RThigh",
    "Skeleton:RShin",
    "Skeleton:RFoot",
    "Skeleton:RToe",
]
