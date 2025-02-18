from enum import Enum
from functools import partial
import numpy as np
import mujoco


class MarkerType(Enum):
    PLANE = mujoco.mjtGeom.mjGEOM_PLANE
    SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
    CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
    ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    ARROW = mujoco.mjtGeom.mjGEOM_ARROW
    ARROW1 = mujoco.mjtGeom.mjGEOM_ARROW1
    ARROW2 = mujoco.mjtGeom.mjGEOM_ARROW2
    LINE = mujoco.mjtGeom.mjGEOM_LINE


class MarkerSet:
    """Represents spherical marker parameter storage for simulation visualization

    Attributes:
        max_markers (int): Maximum number of markers
        dirty_markers (bool): True if the marker state was altered
        active_markers (numpy.ndarray): Bitmask indicating if the marker should be rendered
        marker_positions (numpy.ndarray): World space positions of the markers with stride 3
        marker_sizes (numpy.ndarray): Sizes of the markers in cm
        marker_colors (numpy.ndarray): RGBA[0-1] colors of the markers with stride 4
    Args:
        max_markers (int): Maximum number of markers
    """

    def __init__(self, max_markers: int = 200) -> None:
        """_summary_

        Args:
            max_markers (int, optional): _description_. Defaults to 100.
        """
        self.max_markers = max_markers
        self.dirty_markers = True

        self.active_markers = np.zeros(max_markers, dtype=bool)
        self.marker_positions = np.zeros(
            max_markers * 3,
            dtype=np.float32,
        )
        self.marker_rotations = np.zeros(
            max_markers * 9,
            dtype=np.float32,
        )
        self.marker_sizes = np.zeros(
            max_markers * 3,
            dtype=np.float32,
        )
        self.marker_colors = np.zeros(
            max_markers * 4,
            dtype=np.float32,
        )
        self.marker_types = np.zeros(
            max_markers,
            dtype=np.int32,
        )

        self.marker_types[:] = MarkerType.SPHERE.value
        self.marker_sizes[:] = 0.05
        self.marker_colors[:] = 1.0
        self.marker_positions[:] = 0
        self.marker_rotations.reshape((max_markers, 3, 3))[:] = np.eye(3, dtype=np.float32)
        self.active_markers[:] = False

    def set_marker(
        self,
        marker_id: int,
        marker_type: MarkerType = MarkerType.SPHERE,
        position: np.ndarray = None,
        rotation: np.ndarray = None,
        size: np.ndarray = None,
        color: np.ndarray = None,
        enabled: bool = True,
    ) -> None:
        """Sets the marker's parameters

        Args:
            marker_id (int): index of the marker [0,max_markers)
            marker_type (MarkerType): marker type. Defaults to sphere.
            position (np.ndarray, optional): XYZ position of the marker in world space. Defaults to None.
            rotation: (np.ndarray, optional): 3x3 rotation matrix of the marker in world space. Defaults to None.
            size (np.ndarray, optional): Size of the marker in cm. Defaults to None.
            color (np.ndarray, optional): RGBA[0-1] color of the marker. Defaults to None.
            enabled (bool, optional): Enables or the marker from rendering. Defaults to True.
        """
        assert marker_id >= 0 and marker_id < self.max_markers
        if not enabled:
            self.active_markers[marker_id] = False
            return
        self.marker_types[marker_id] = marker_type.value
        self.active_markers[marker_id] = True
        if position is not None:
            self.marker_positions[3 * marker_id : 3 * marker_id + 3] = position
        if size is not None:
            if isinstance(size, (int, float)):
                self.marker_sizes[3 * marker_id : 3 * marker_id + 3] = np.ones(3) * size
            else:
                self.marker_sizes[3 * marker_id : 3 * marker_id + 3] = size
        if color is not None:
            self.marker_colors[4 * marker_id : 4 * marker_id + 4] = color
        if rotation is not None:
            self.marker_rotations[9 * marker_id : 9 * marker_id + 9] = rotation.flatten()
        self.dirty_markers = True

    def __getitem__(self, marker_id: int) -> callable:
        """
        Args:
            marker_id (int): index of the marker [0,max_markers)

        Returns:
            self.set_marker(marker_id,...) function object
        """
        return partial(self.set_marker, marker_id)
