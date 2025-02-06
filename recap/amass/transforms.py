import torch
import torch.nn.functional as F

"""
Utility functions for AMASS dataset
"""


def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : np.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: np.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    np.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    return torch.einsum("bl,mkl->bmk", betas, shape_disps)


def vertices2joints(J_regressor, vertices):
    """Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : np.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : np.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    np.tensor BxJx3
        The location of the joints
    """

    return torch.einsum("bik,ji->bjk", vertices, J_regressor)


def batch_rodrigues(rot_vecs):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: np.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: np.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]

    angle = torch.norm(rot_vecs + 1e-8, dim=1).unsqueeze(1)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)

    # Bx1 arrays
    rx, ry, rz = rot_dir[:, 0], rot_dir[:, 1], rot_dir[:, 2]
    K = torch.zeros((batch_size, 3, 3), device=rot_vecs.device)

    zeros = torch.zeros((batch_size), device=rot_vecs.device)
    K = torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros]).T.reshape((batch_size, 3, 3))

    return torch.eye(3, device=rot_vecs.device).unsqueeze(0) + sin * K + (1 - cos) * K @ K


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """

    T = F.pad(R, (0, 1, 0, 1), "constant", 0.0)
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    return T


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].squeeze()

    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3)).reshape(
        -1, joints.shape[1], 4, 4
    )

    transform_chain = [transforms_mat[:, 0]]  # Root
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    return torch.stack(transform_chain, dim=1)


@torch.jit.script
def calculate_body_transforms(
    poses: torch.Tensor,
    root_pos: torch.Tensor,
    blendshapes: torch.Tensor,
    betas: torch.Tensor,
    shapedirs: torch.Tensor,
    eigvec: torch.Tensor,
    v_template: torch.Tensor,
    J_regressor: torch.Tensor,
    kintree_table: torch.Tensor,
    template_scale: torch.Tensor,
):
    """Calculate body transforms from processed motion data"""
    batch_size = poses.shape[0]

    # Combine betas with blendshapes
    beta = torch.concatenate([betas.expand(batch_size, 16), blendshapes], dim=-1)

    # Calculate shaped vertices and joints
    shapedirs = torch.cat(
        [
            shapedirs,
            eigvec,
        ],
        dim=2,
    )
    v_shaped = v_template * template_scale + blend_shapes(beta, shapedirs)

    J = vertices2joints(J_regressor, v_shaped)

    # Calculate motion transforms
    rot_mats = batch_rodrigues(poses.reshape((-1, 3))).reshape([batch_size, -1, 3, 3])

    motion_transforms = batch_rigid_transform(rot_mats, J, kintree_table[0])
    positions = motion_transforms[:, :, :3, 3]
    rotations = motion_transforms[:, :, :3, [2, 0, 1]]

    # Set the ground level to the lowest point in the motion
    # Note: detach() is used to prevent loops in the backward gradient pass
    ground_level = torch.min(positions[:, :, 2] + root_pos[:, 2].unsqueeze(1)).detach()
    root_pos[:, 2] -= ground_level

    positions += root_pos.unsqueeze(1)

    return positions, rotations
