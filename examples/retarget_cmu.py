import os
from tqdm import tqdm
from recap.wbik_solver import NoSolutionException
from recap.trajectory import Trajectory
from recap.cmu.retarget import (
    CMURetarget,
    CMU_WALKING,
    CMU_RUNNING,
    CMU_STAND,
    CMU_DANCE,
    CMU_DATASET_FPS,
)
from recap.mujoco.renderer import MujocoRenderer
from recap.config.robot.g1 import G1_CMU_CONFIG
from recap.config.robot.h1 import H1_CMU_CONFIG

robots = {
    "g1": (G1_CMU_CONFIG),
    "h1": (H1_CMU_CONFIG),
}

ROBOT_NAME = "g1"
RENDER = True

cfg = robots[ROBOT_NAME]

retargetee = CMURetarget(wbik_params=cfg)

output_path = os.path.join(os.path.dirname(__file__), f"./motions/cmu/{ROBOT_NAME}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

skills = [
    ("dance", CMU_DANCE),
    ("walking", CMU_WALKING),
    ("running", CMU_RUNNING),
    ("stand", CMU_STAND),
]
with MujocoRenderer(robots[ROBOT_NAME].mjcf_path) as renderer:
    for skill_name, skill in skills:
        counter = 0
        for sample in skill:
            subject = sample[0]
            indices = sample[1]
            print(f"Subject: {subject}")
            retargetee.set_skeleton(subject_id=subject)
            for motion_idx in tqdm(indices):
                retargetee.set_motion(subject_id=subject, motion_id=motion_idx)
                trajectory = Trajectory(dt=1.0 / CMU_DATASET_FPS)
                try:
                    for pose_data in retargetee:
                        if RENDER:
                            retargetee.render_solution(renderer)
                            renderer.step()
                        trajectory.add_sample(pose_data)
                except NoSolutionException:
                    print(f"Skipping {subject} {motion_idx}")
                    continue
                out_filename = f"cmu_{ROBOT_NAME}_{skill_name}_{subject}_{motion_idx}.npy"
                trajectory.save(os.path.join(output_path, out_filename))
