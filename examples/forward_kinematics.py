# Standard library imports
import time

# Third-party imports
import numpy as np
import torch

# Local imports
from recap.config.robot.g1 import MODEL_PATH
from recap.mujoco.renderer import MujocoRenderer
from recap.torch_fk import TorchForwardKinematics
from recap.quat_utils import quat2mat

# Constants
TIME_STEP = 0.001
SIMULATION_DURATION = 10.0

# Define robot links to track
ROBOT_LINK_NAMES = [
    "pelvis",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_roll_link",
    "right_elbow_roll_link",
    "left_six_link",
    "right_six_link",
]

# Initialize forward kinematics
fk = TorchForwardKinematics(MODEL_PATH, torch.device("cpu"))

# Generate random DOF indices for animation
random_dof_indices = torch.randint(7, fk.nq, (10,))

with MujocoRenderer(MODEL_PATH) as renderer:
    step = 0
    while TIME_STEP * step < SIMULATION_DURATION:
        # Create joint configuration
        joint_config = np.zeros((1, fk.nq))
        joint_config[:, 3] = 1  # Unit quaternion
        joint_config[:, 2] = 0.73
        joint_config[:, random_dof_indices] = np.sin(np.pi * step / SIMULATION_DURATION)

        # Compute forward kinematics
        body_names, body_positions, body_quaternions = fk.forward_kinematics(torch.from_numpy(joint_config))

        # Extract relevant body information
        body_indices = [body_names.index(name) for name in ROBOT_LINK_NAMES]
        quats = body_quaternions[0, body_indices]
        positions = body_positions[0, body_indices]

        positions = positions.squeeze().numpy()
        rotations = quat2mat(quats.squeeze().numpy())

        # Render the current frame
        renderer.render_frames(
            positions,
            rotations,
        )
        # Update renderer and step simulation
        renderer.set_configuration(joint_config)
        renderer.step()
        step += 1
        time.sleep(TIME_STEP)
