import os
from recap.amass.retarget import AMASSRetarget
from recap.amass.data_loader import AMASSMotionLoader
from recap.amass.motion_wrapper import (
    AMASSMotionWrapper,
    SMPLH_JOINT_NAMES_TO_IDX,
)
from recap.mujoco.renderer import MujocoRenderer
from recap.config.robot.h1 import H1_AMASS_CONFIG
from recap.config.robot.g1 import G1_AMASS_CONFIG
from recap.config.robot.h1 import MODEL_PATH as H1_MODEL_PATH
from recap.config.robot.g1 import MODEL_PATH as G1_MODEL_PATH

from recap.torch_fk import TorchForwardKinematics
from recap.trajectory import Trajectory
from recap.amass.transforms import calculate_body_transforms
import torch
from tqdm import tqdm

robot_name = "g1"
robots = {
    "h1": (H1_AMASS_CONFIG, H1_MODEL_PATH, 0.01),
    "g1": (G1_AMASS_CONFIG, G1_MODEL_PATH, 0.01),
}
robot_config, model_path, lr = robots[robot_name]
# At this point we're only interested in the link alignment, so we disable all velocity tasks
robot_config.task_weights["joint_velocity"] = 0.0
robot_config.task_weights["root_linear_velocity"] = 0.0
robot_config.task_weights["root_angular_velocity"] = 0.0

# Disable contact velocity to avoid foot sticking to the ground
robot_config.contact_velocity = -1

# Initialize the beta variable and optimizer
beta = torch.autograd.Variable(
    torch.zeros([robot_config.beta.shape[0]], device="cuda"),
    requires_grad=True,
)
# AMASS model uses a base template to compute vertex offsets and joints positions
# in case if the target model is not 1:1 with the AMASS template, we need to scale the template
# by a constant factor
template_scale = torch.autograd.Variable(
    torch.ones(1, device="cuda"),
    requires_grad=True,
)
optimizer = torch.optim.Adam([beta, template_scale], lr=lr)

data_loader = AMASSMotionLoader(
    motion_filename="912_3_01_poses.npz",
    datasets_path=os.path.join(os.path.dirname(__file__), "../data"),
    beta=beta.to("cpu").detach().numpy(),
    device="cuda",
)

retargetee = AMASSRetarget(wbik_params=robot_config)

# Initialize the forward kinematics model
fk = TorchForwardKinematics(model_path, torch.device("cuda"))


def solve_ik(motion, renderer: MujocoRenderer = None):
    with torch.no_grad():
        retargetee.set_motion(motion)
        trajectory = Trajectory()
        for pose_data in retargetee:
            trajectory.add_sample(pose_data)
            if renderer is not None:
                retargetee.render_solution(renderer)
                renderer.step()
        # Compute the forward kinematics
        body_names, body_positions, body_quaternions = fk.forward_kinematics(
            torch.from_numpy(trajectory.qs).to("cuda"), pin_notation=True
        )
        body_positions.to("cuda")
        return body_names, body_positions, body_quaternions


# Solve IK for the whole motion
name, motion_data = next(data_loader)
# Get the arguments for the AMASS model calculation
transform_args = data_loader.get_transform_args()

body_names, body_positions, body_quaternions = solve_ik(motion_data)
# Get the root index of the models to compute local frame errors
fk_root_idx = body_names.index(robot_config.body_to_model_map["root"])
data_root_idx = SMPLH_JOINT_NAMES_TO_IDX[robot_config.body_to_data_map["root"]]

# Center the FK positions at the root
body_positions[:, :, :] -= body_positions[:, fk_root_idx].unsqueeze(1)

# Get the indices of the body parts for both dataset and FK
fk_idxs = []
data_idxs = []
for name in robot_config.body_to_data_map.keys():
    fk_idx = body_names.index(robot_config.body_to_model_map[name])
    data_idx = SMPLH_JOINT_NAMES_TO_IDX[robot_config.body_to_data_map[name]]
    fk_idxs.append(fk_idx)
    data_idxs.append(data_idx)

# Optimize the beta parameters
num_iterations = 10
num_beta_steps = 200

with MujocoRenderer(model_path) as renderer:
    for i in range(num_iterations):
        pbar = tqdm(range(num_beta_steps))
        for b in pbar:
            positions, rotations = calculate_body_transforms(
                **transform_args, betas=beta, template_scale=template_scale
            )
            # Convert positions to the same frame as the FK
            # NOTE: the frames are not aligned, so we need to subtract the root rotation
            # But since we've done the IK for the torso position and orientation task, we're kinda aligned :)
            local_dataset_positions = positions[:, data_idxs] - positions[:, data_root_idx].unsqueeze(1)
            local_fk_positions = body_positions[:, fk_idxs] - body_positions[:, fk_root_idx].unsqueeze(1)
            loss = (local_fk_positions - local_dataset_positions).square().sum(-1).sum(-1).mean()
            loss += (1 - template_scale).square().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"loss: {loss.item()}")

        print(f"Iteration {i} loss: {loss.item()}")
        print(f"beta: {beta}")
        print(f"template_scale: {template_scale.item()}")
        # Solve IK for new beta and start over
        motion_data = AMASSMotionWrapper(positions=positions, rotations=rotations)
        with torch.no_grad():
            _, body_positions, body_quaternions = solve_ik(motion_data, renderer)

print(f"Done: {os.linesep}beta: {beta}{os.linesep}template_scale: {template_scale}")
print(beta)
