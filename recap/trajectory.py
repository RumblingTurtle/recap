import numpy as np
import quaternion
from scipy.signal import savgol_filter
from dataclasses import dataclass


def derivative(data, sample_dt):
    return savgol_filter(x=data, polyorder=3, deriv=1, window_length=9, delta=sample_dt, axis=0)


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

    @property
    def linear_velocities(self):
        if self._linear_velocities is None:
            self._linear_velocities = derivative(self.positions, self.sample_dt)
        return self._linear_velocities

    @property
    def angular_velocities(self):
        if self._angular_velocities is None:
            quaternions = quaternion.from_float_array(self.quaternions)
            self._angular_velocities = quaternion.angular_velocity(
                quaternions, np.arange(quaternions.shape[0]) * self.sample_dt
            )

        return self._angular_velocities

    def to_dict(self, scalar_first: bool = True):
        return {
            "position": self.positions,
            "quaternion": (self.quaternions if scalar_first else self.quaternions[:, [1, 2, 3, 0]]),
            "linear_velocity": self.linear_velocities,
            "angular_velocity": self.angular_velocities,
        }


class Trajectory:
    def __init__(self, dt: float = 0.001):
        self.contacts = None
        self.bodies = {}
        self.qs = None
        self._joint_velocities = None
        self.dt = dt

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
            if "world" in transform.name:
                continue
            if transform.name not in self.bodies.keys():
                self.bodies[transform.name] = BodyTimeSeries(transform.name, pose_data.dt)
            self.bodies[transform.name].add_sample(transform.position, transform.quaternion)

    @property
    def joint_positions(self):
        return self.qs[:, 7:]

    @property
    def joint_velocities(self):
        if self._joint_velocities is None:
            self._joint_velocities = derivative(self.joint_positions, self.dt)
        return self._joint_velocities

    def to_dict(self):
        out_dict = {
            "q": self.qs.copy(),
            "dt": self.dt,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_order": self.joint_order,
            "body_aliases": self.body_aliases,
            "model_root": self.model_root,
            "contacts": self.contacts,
            "transforms": {},
        }
        for name, body in self.bodies.items():
            body_dict = body.to_dict()

            out_dict["transforms"][name] = {}
            for k, v in body_dict.items():
                out_dict["transforms"][name][k] = v
        return out_dict

    def save(self, path):
        trajectory_dict = self.to_dict()
        np.save(path, trajectory_dict, allow_pickle=True)
