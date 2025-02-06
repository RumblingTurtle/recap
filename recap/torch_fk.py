import torch
import numpy as np
import mujoco
from recap.quat_utils import quat_multiply, rotate, axis_angle_to_quat


class TorchForwardKinematics:
    """
    Torch based batched forward kinematics for MuJoCo robot model
    """

    def __init__(self, mjcf_file: str, device: torch.device):
        self.device = device
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
        self.parents = {}
        self.local_position = {}
        self.local_rotation = {}
        self.joint_offsets = {}
        self.joint_axes = {}
        self.dof_ids = {}
        self.body_names = {}

        self.lower_limit = (
            torch.from_numpy(np.concatenate([[-100] * 3, [-1] * 4, self.mj_model.jnt_range[1:, 0]]))
            .float()
            .to(self.device)
        )
        self.upper_limit = (
            torch.from_numpy(np.concatenate([[100] * 3, [1] * 4, self.mj_model.jnt_range[1:, 1]]))
            .float()
            .to(self.device)
        )

        for body_id in range(self.mj_model.nbody):
            body = self.mj_model.body(body_id)
            body_name = body.name
            self.body_names[body_id] = body_name
            parent_id = int(body.parentid[0])
            self.parents[body_id] = parent_id
            self.local_position[body_id] = torch.from_numpy(np.array(body.pos, dtype=np.float32)).to(self.device)
            self.local_rotation[body_id] = torch.from_numpy(np.array(body.quat, dtype=np.float32)).to(self.device)

        for joint_id in range(self.mj_model.njnt):
            joint = self.mj_model.joint(joint_id)
            child_id = int(joint.bodyid[0])
            self.dof_ids[child_id] = int(joint.qposadr[0])
            self.joint_offsets[child_id] = torch.from_numpy(np.array(joint.pos, dtype=np.float32)).to(self.device)
            self.joint_axes[child_id] = torch.from_numpy(np.array(joint.axis, dtype=np.float32)).to(self.device)

    @property
    def nq(
        self,
    ):
        return self.mj_model.nq

    @property
    def nbodies(
        self,
    ):
        return self.mj_model.nbody

    def random_configuration(self, batch_size):
        """
        Returns a random configuration for the robot within the joint limits
        """
        samples = self.lower_limit + torch.rand(batch_size, self.nq, device=self.device) * (
            self.upper_limit - self.lower_limit
        )
        samples = torch.clip(samples, self.lower_limit, self.upper_limit)
        samples[:, 3:7] = samples[:, 3:7] / (torch.norm(samples[:, 3:7], dim=1).unsqueeze(1) + 1e-4).float()
        return samples

    def forward_kinematics(self, q: torch.Tensor, pin_notation: bool = False):
        """
        Computes the forward kinematics of the robot given a configuration in minimal coordinates q
            q tensor of shape (batch_size, nq)
        Returns:
            body_names: list of body names
            body_positions: tensor of shape (batch_size, nbodies, 3)
            body_quaternions: tensor of shape (batch_size, nbodies, 4) scalar first
        """
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        if pin_notation:
            q[:, 3:7] = q[:, [6, 3, 4, 5]]

        batch_size = q.shape[0]

        if q.shape[1] != self.nq:
            raise ValueError(f"Expected q to have shape (batch_size, {self.nq}), got {q.shape}")

        if (torch.norm(q[:, 3:7], dim=1) == 0).any():
            raise ValueError("Zero norm quaternion in forward_kinematics call")

        body_positions = torch.zeros(q.shape[0], self.nbodies, 3, device=self.device)
        body_quaternions = torch.zeros(q.shape[0], self.nbodies, 4, device=self.device)
        body_quaternions[:, :, 0] = 1

        root_id = 0
        for id, dofid in self.dof_ids.items():
            if dofid == 0:
                root_id = id

        body_positions[:, root_id, :] = q[:, :3]
        body_quaternions[:, root_id, :] = q[:, 3:7]

        body_names = []
        for body_id, parent_id in self.parents.items():
            body_names.append(self.body_names[body_id])

            parent_pos = body_positions[:, parent_id]
            parent_quat = body_quaternions[:, parent_id]

            local_quat = self.local_rotation[body_id].unsqueeze(0).repeat((batch_size, 1))
            local_pos = self.local_position[body_id].unsqueeze(0).repeat((batch_size, 1))

            pos = parent_pos + rotate(parent_quat, local_pos)
            quat = quat_multiply(parent_quat, local_quat)

            if body_id not in self.dof_ids:
                body_positions[:, body_id] = pos.clone()
                body_quaternions[:, body_id] = quat.clone()
                continue

            dof_adr = self.dof_ids[body_id]
            if dof_adr == 0:
                continue
            joint_position = self.joint_offsets[body_id].unsqueeze(0).repeat((batch_size, 1))
            joint_axes = self.joint_axes[body_id].unsqueeze(0).repeat((batch_size, 1))

            anchor = rotate(quat, joint_position) + pos

            joint_rotation = axis_angle_to_quat(joint_axes, q[:, dof_adr].unsqueeze(1))

            body_positions[:, body_id] = anchor - rotate(quat, joint_position)
            body_quaternions[:, body_id] = quat_multiply(quat, joint_rotation)

        return body_names, body_positions, body_quaternions
