import torch
import numpy as np
import quaternion


def normalize(quaternions: torch.Tensor):
    return positive_quat(quaternions / torch.norm(quaternions, dim=1).unsqueeze(1))


def positive_quat(quat: torch.Tensor):
    quat[quat[:, 0] < 0] *= -1
    return quat


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """
    s, c = torch.sin(angle * 0.5), torch.cos(angle * 0.5)
    return normalize(torch.cat([c, axis * s], dim=1))


def quat_multiply(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    quaternion = torch.stack(
        [
            u[:, 0] * v[:, 0] - u[:, 1] * v[:, 1] - u[:, 2] * v[:, 2] - u[:, 3] * v[:, 3],
            u[:, 0] * v[:, 1] + u[:, 1] * v[:, 0] + u[:, 2] * v[:, 3] - u[:, 3] * v[:, 2],
            u[:, 0] * v[:, 2] - u[:, 1] * v[:, 3] + u[:, 2] * v[:, 0] + u[:, 3] * v[:, 1],
            u[:, 0] * v[:, 3] + u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1] + u[:, 3] * v[:, 0],
        ]
    ).T
    return normalize(quaternion)


def rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    quat = normalize(quat)
    s, u = quat[:, 0].unsqueeze(1), quat[:, 1:]
    u_dot_vec = (u * vec).sum(1).unsqueeze(1)
    u_dot_u = (u * u).sum(1).unsqueeze(1)
    r = 2 * (u_dot_vec * u) + ((s * s) - u_dot_u) * vec
    r = r + 2 * s * torch.cross(u, vec, dim=1)
    return r


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculates the inverse of quaternion q.

    Args:
      q: (4,) quaternion [w, x, y, z]

    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return normalize(q * torch.tensor([[1, -1, -1, -1]], device=q.device))


def yaw_matrix(rotation_matrix: np.ndarray):
    """
    Returns a quaternion that keeps only the yaw rotation of the given rotation matrix
    """
    quat_yaw = quaternion.from_rotation_matrix(rotation_matrix)
    quat_yaw.x = 0.0
    quat_yaw.y = 0.0
    return quaternion.as_rotation_matrix(np.normalized(quat_yaw))


def quat2mat(q: np.ndarray, scalar_last=False, eps=np.finfo(np.float64).eps):
    """
    Convert quaternions to rotation matrices.

    Args:
        q: Quaternions to convert, shape (..., 4).
        scalar_last: If True, the scalar (w) is the last element of q.
    """
    if len(q.shape) == 1:
        q = q.reshape(1, -1)

    batch_size = q.shape[0]
    if scalar_last:
        x, y, z, w = np.split(q, 4, axis=-1)
    else:
        w, x, y, z = np.split(q, 4, axis=-1)

    Nq = w * w + x * x + y * y + z * z
    out = np.zeros((batch_size, 3, 3))
    out[Nq.reshape(batch_size) < eps] = np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    out[:, 0, 0] = (1.0 - (yY + zZ)).reshape(-1)
    out[:, 0, 1] = (xY - wZ).reshape(-1)
    out[:, 0, 2] = (xZ + wY).reshape(-1)
    out[:, 1, 0] = (xY + wZ).reshape(-1)
    out[:, 1, 1] = (1.0 - (xX + zZ)).reshape(-1)
    out[:, 1, 2] = (yZ - wX).reshape(-1)
    out[:, 2, 0] = (xZ - wY).reshape(-1)
    out[:, 2, 1] = (yZ + wX).reshape(-1)
    out[:, 2, 2] = (1.0 - (xX + yY)).reshape(-1)
    return out


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions
