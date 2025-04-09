import os
from recap.wbik_solver import NoSolutionException
from recap.trajectory import Trajectory
from recap.config.robot.h1 import H1_OPTITRACK_CONFIG
from recap.optitrack.retarget import OptitrackRetarget
from recap.config.optitrack_retarget_config import MOTION_PATHS
from recap.mujoco.renderer import MujocoRenderer


ROBOT_NAME = "h1"
RENDER = True

retargetee = OptitrackRetarget(wbik_params=H1_OPTITRACK_CONFIG)

output_path = os.path.join(os.path.dirname(__file__), f"./motions/optitrack/{ROBOT_NAME}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

with MujocoRenderer(retargetee.wbik_params.mjcf_path, "optitrack.mp4") as renderer:
    for path in MOTION_PATHS:
        retargetee.set_motion(path)
        motion_name = os.path.basename(path).split(".")[0]
        trajectory = Trajectory(sample_dt=1 / retargetee.fps)
        try:
            for pose_data in retargetee:
                if RENDER:
                    retargetee.render_solution(renderer)
                    renderer.step()
                trajectory.add_sample(pose_data)
        except NoSolutionException:
            print(f"Skipping {path}")
            continue
        out_filename = f"optitrack_{ROBOT_NAME}_{motion_name}"
        renderer.flush_frames(retargetee.fps)
        trajectory.save(os.path.join(output_path, out_filename))
