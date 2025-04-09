import numpy as np
import quaternion
from dataclasses import dataclass
import pickle
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, RotationSpline


def derivative(data, sample_dt):
    return (
        np.diff(
            data,
            prepend=[data[0]],
            axis=0,
        )
        / sample_dt
    )


def angular_velocity(quaternions, sample_dt):
    q_dot_savgol = savgol_filter(x=quaternions, polyorder=3, window_length=5, axis=0, deriv=1, delta=sample_dt)
    angular_velocities = np.zeros((len(q_dot_savgol), 3))
    for i in range(len(q_dot_savgol)):
        q_dot_quat = quaternion.quaternion(*q_dot_savgol[i])
        q_quat = quaternion.quaternion(*quaternions[i])
        q_conj_quat = q_quat.conjugate()
        result = 2 * q_dot_quat * q_conj_quat
        # Extract vector part (x, y, z) as angular velocity
        angular_velocities[i, 0] = result.x
        angular_velocities[i, 1] = result.y
        angular_velocities[i, 2] = result.z
    return angular_velocities


@dataclass
class BodyTransform:
    """
    Transform info for a single body
    """

    name: str
    position: np.ndarray
    quaternion: np.ndarray


@dataclass
class PoseData:
    """
    Frame data for a single time step
        transforms: list[BodyTransform] # list of body transforms
        q: np.ndarray # minimal coordinates of the robot model
        body_aliases: dict # map of body names to mjcf names
        model_root: str # name of the root body in mjcf model
        contacts: np.ndarray # contact states
        position_error: float # absolute position error of all links wrt to the reference
        dt: float # sample time
        joint_order: list # order of the joints in the q vector
    """

    transforms: list[BodyTransform]
    q: np.ndarray
    body_aliases: dict
    model_root: str
    contacts: np.ndarray
    position_error: float
    dt: float
    joint_order: list


class BodyTimeSeries:
    """
    Data structure for storing body transform timeseries
    """

    def __init__(self, name: str, sample_dt: float):
        self.name = name
        self.sample_dt = sample_dt
        self.positions = None
        self.quaternions = None
        self._linear_velocities = None
        self._angular_velocities = None

    def add_sample(self, position: np.ndarray, quat: np.ndarray):
        """
        Add a sample to the body data

        Args:
            position (np.ndarray): position of the body in the world frame
            quat (np.ndarray): quaternion of the body in the world frame [w, x, y, z] order
        """
        np_quat = quaternion.as_float_array(quat)
        if self.positions is None:
            self.positions = position.copy()
            self.quaternions = np_quat.copy()
        else:
            self.positions = np.vstack((self.positions, position.copy()))
            self.quaternions = np.vstack((self.quaternions, np_quat.copy()))

    def to_dict(self, scalar_first: bool = True, out_dt: float = None):
        if out_dt is None:
            out_dt = self.sample_dt

        quaternions = quaternion.as_float_array(quaternion.unflip_rotors(quaternion.from_float_array(self.quaternions)))

        old_t = np.arange(self.positions.shape[0]) * self.sample_dt
        new_t = np.arange(old_t[0], old_t[-1], out_dt)

        positions_spline = make_interp_spline(old_t, self.positions, k=1, bc_type="not-a-knot", axis=0)
        sci_quats = Rotation.from_quat(quaternions)
        quat_spline = RotationSpline(old_t, sci_quats)

        positions = positions_spline(new_t)
        quaternions = quat_spline(new_t).as_quat()
        return {
            "position": positions,
            "quaternion": (quaternions if scalar_first else quaternions[:, [1, 2, 3, 0]]),
            "linear_velocity": derivative(positions, out_dt),
            "angular_velocity": angular_velocity(quaternions, out_dt),
        }


class Trajectory:
    def __init__(self, sample_dt: float = 0.001):
        self.contacts = None
        self.bodies: dict[str, BodyTimeSeries] = {}
        self.qs = None
        self._joint_velocities = None
        self.sample_dt = sample_dt

    def add_sample(
        self,
        pose_data: PoseData,
    ):
        """
        Add a pose data to the trajectory.

        Args:
            pose_data: PoseData object
        """
        contacts = pose_data.contacts.copy()

        if self.contacts is None:
            self.contacts = contacts
            self.qs = pose_data.q
            self.body_aliases = pose_data.body_aliases
            self.model_root = pose_data.model_root
            self.joint_order = pose_data.joint_order
        else:
            self.contacts = np.vstack([self.contacts, contacts])
            self.qs = np.vstack([self.qs, pose_data.q])

        for transform in pose_data.transforms:
            if transform.name in ["world", "universe"]:
                continue
            if transform.name not in self.bodies.keys():
                self.bodies[transform.name] = BodyTimeSeries(transform.name, self.sample_dt)
            self.bodies[transform.name].add_sample(transform.position, transform.quaternion)

    def to_dict(self, out_dt: float = None):
        if out_dt is None:
            out_dt = self.sample_dt

        old_t = np.arange(self.qs.shape[0]) * self.sample_dt
        new_t = np.arange(old_t[0], old_t[-1], out_dt)

        joint_position_spline = make_interp_spline(old_t, self.qs[:, 7:], k=1, bc_type="not-a-knot", axis=0)
        joint_positions = joint_position_spline(new_t)
        joint_velocities = joint_position_spline(new_t, 1)

        out_dict = {
            "dt": out_dt,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_order": self.joint_order,
            "body_aliases": self.body_aliases,
            "model_root": self.model_root,
            "contacts": self.contacts[np.abs(old_t - new_t[:, None]).argmin(1)],
            "transforms": {},
        }

        for name, body in self.bodies.items():
            body_dict = body.to_dict(out_dt=out_dt)

            out_dict["transforms"][name] = {}
            for k, v in body_dict.items():
                out_dict["transforms"][name][k] = v
        return out_dict

    def convert_ndarrays(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = v.tolist()
            if isinstance(v, dict):
                data_dict[k] = self.convert_ndarrays(v)
        return data_dict

    def save(self, path, out_dt: float = 0.02):
        # Have to remove numpy arrays entirely
        # to preserve compatibility between versions
        trajectory_dict = self.convert_ndarrays(self.to_dict(out_dt=out_dt))
        with open(path + ".pkl", "wb") as f:
            pickle.dump(trajectory_dict, f)


def to_numpy(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, list):
            data_dict[k] = np.array(v)
        if isinstance(v, dict):
            data_dict[k] = to_numpy(v)
    return data_dict
