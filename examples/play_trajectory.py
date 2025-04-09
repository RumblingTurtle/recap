import pickle
import numpy as np
import glob
import time
import os
from recap.mujoco.renderer import MujocoRenderer
from recap.config.robot.g1 import MODEL_PATH as G1_MODEL_PATH
from recap.config.robot.h1 import MODEL_PATH as H1_MODEL_PATH
from recap.trajectory import to_numpy

dataset = ["amass", "cmu"]
robots = {
    "h1": H1_MODEL_PATH,
    "g1": G1_MODEL_PATH,
}

ROBOT_NAME = "h1"
motion_files = glob.glob(os.path.join(os.path.dirname(__file__), f"./motions/{dataset[0]}/{ROBOT_NAME}/*.pkl"))
with MujocoRenderer(robots[ROBOT_NAME]) as renderer:
    for motion_file in motion_files:
        with open(motion_file, "rb") as f:
            motion = pickle.load(f)
        motion = to_numpy(motion)
        data_dt = motion["dt"]
        root_link_pos = motion["transforms"][motion["model_root"]]["position"]
        root_link_quat = motion["transforms"][motion["model_root"]]["quaternion"]
        joint_pos = motion["joint_positions"]
        qs = np.hstack([root_link_pos, root_link_quat, joint_pos])
        for frame_idx in range(qs.shape[0]):
            renderer.set_configuration(qs[frame_idx], pin_notation=False)
            renderer.step()
            time.sleep(data_dt)
