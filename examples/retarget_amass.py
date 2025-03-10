import os
from tqdm import tqdm
from recap.wbik_solver import NoSolutionException
from recap.trajectory import Trajectory
from recap.amass.retarget import AMASSRetarget
from recap.amass.data_loader import AMASSMotionLoader

from recap.config.robot.g1 import G1_AMASS_CONFIG
from recap.config.robot.h1 import H1_AMASS_CONFIG

from recap.mujoco.renderer import MujocoRenderer

robots = {
    "g1": G1_AMASS_CONFIG,
    "h1": H1_AMASS_CONFIG,
}

ROBOT_NAME = "h1"
RENDER = True

wbik_params = robots[ROBOT_NAME]

data_loader = AMASSMotionLoader(
    datasets_path=os.path.join(os.path.dirname(__file__), "../data"),
    beta=wbik_params.beta,
    name_blacklist=["handrail", "jump", "box", "hop", "push", "kick", "dance", "punch", "sit"],
    name_whitelist=["run", "walk", "stand"],
    target_fps=30,
    template_scale=wbik_params.template_scale,
)
retargetee = AMASSRetarget(wbik_params=wbik_params)

output_path = os.path.join(os.path.dirname(__file__), f"./motions/amass/{ROBOT_NAME}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

with MujocoRenderer(robots[ROBOT_NAME].mjcf_path, "amass.mp4") as renderer:
    for data in tqdm(data_loader):
        if data is None:
            continue

        motion_name, motion_data = data
        print(motion_name)
        retargetee.set_motion(motion_data)
        trajectory = Trajectory(1 / data_loader.target_fps)
        try:
            for pose_data in retargetee:
                trajectory.add_sample(pose_data)
                if RENDER:
                    retargetee.render_solution(renderer)
                    renderer.step()
        except NoSolutionException:
            print(f"Skipping {motion_name}")
            continue
        out_filename = f"amass_{ROBOT_NAME}_{motion_name}"
        trajectory.save(os.path.join(output_path, out_filename))
        renderer.flush_frames(fps=data_loader.target_fps)
