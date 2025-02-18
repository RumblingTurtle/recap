import os
from tqdm import tqdm
from recap.wbik_solver import NoSolutionException
from recap.trajectory import Trajectory
from recap.lafan.retarget import LAFANRetarget, MOTION_PATHS
import time
import numpy as np
from recap.config.robot.g1 import G1_LAFAN_CONFIG
from recap.config.robot.h1 import H1_LAFAN_CONFIG

from recap.mujoco.renderer import MujocoRenderer

robots = {
    "g1": G1_LAFAN_CONFIG,
    "h1": H1_LAFAN_CONFIG,
}

ROBOT_NAME = "h1"
RENDER = True

wbik_params = robots[ROBOT_NAME]

retargetee = LAFANRetarget(wbik_params=wbik_params)

output_path = os.path.join(os.path.dirname(__file__), f"./motions/lafan/{ROBOT_NAME}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

with MujocoRenderer(robots[ROBOT_NAME].mjcf_path) as renderer:
    for path in MOTION_PATHS:
        retargetee.set_motion(path)
        motion_name = os.path.basename(path).split(".")[0]
        trajectory = Trajectory(1 / 30)
        try:
            for pose_data in retargetee:
                if RENDER:
                    retargetee.render_solution(renderer)
                    renderer.step()
                trajectory.add_sample(pose_data)
        except NoSolutionException:
            print(f"Skipping {path}")
            continue
        out_filename = f"lafan_{ROBOT_NAME}_{motion_name}.npy"
        trajectory.save(os.path.join(output_path, out_filename))
