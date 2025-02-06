from recap.config.robot.g1 import MODEL_PATH
from recap.mujoco.renderer import MujocoRenderer
from recap.mujoco.marker_set import MarkerType
import numpy as np
import time
from quaternion import from_spherical_coords, as_rotation_matrix

# Animation parameters
SIMULATION_DURATION = 10.0  # seconds
TIME_STEP = 0.001  # seconds

# Marker animation frequencies
DISABLE_MARKER_FREQ = 500
ENABLE_MARKER_FREQ = 1000

step = 0

# Initialize renderer
with MujocoRenderer(MODEL_PATH) as renderer:
    while TIME_STEP * step < SIMULATION_DURATION:
        current_time = TIME_STEP * step

        renderer.step()
        # Animate rotating plane marker
        plane_time = current_time
        plane_position = np.array(
            [
                np.sin(plane_time),
                np.cos(plane_time),
                0.1 * np.sin(plane_time * 10) + 0.5,
            ]
        )
        renderer.markers[0](
            marker_type=MarkerType.PLANE,
            position=plane_position,
            color=[0, 0, 0, 1],
            size=0.3,
            rotation=as_rotation_matrix(from_spherical_coords(np.sin(plane_time), phi=np.cos(plane_time))),
        )

        # Animate arrow with varying size
        arrow_time = current_time + np.pi
        arrow_position = np.array(
            [
                np.sin(arrow_time),
                np.cos(arrow_time),
                0.1 * np.sin(arrow_time * 10) + 0.5,
            ]
        )
        renderer.markers[1](
            marker_type=MarkerType.ARROW1,
            position=arrow_position,
            size=0.3 * np.abs(np.sin(arrow_time)) + 0.05,
        )

        # Animate ellipsoid with color changes
        ellipsoid_time = current_time + np.pi / 2
        ellipsoid_position = np.array(
            [
                np.sin(ellipsoid_time),
                np.cos(ellipsoid_time),
                0.1 * np.sin(ellipsoid_time * 10) + 0.5,
            ]
        )
        ellipsoid_color = np.array([np.abs(np.sin(ellipsoid_time)), np.abs(np.cos(ellipsoid_time * 2)), 0, 1])
        renderer.markers[2](
            marker_type=MarkerType.ELLIPSOID,
            position=ellipsoid_position,
            size=[0.1, 0.05, 0.1],
            color=ellipsoid_color,
        )

        # Animate line.
        renderer.render_line(
            point_a=np.array([np.cos(current_time), 0, 1.3]),
            point_b=np.array([0, np.sin(current_time), 1.3]),
            color=[1, 0, 0, 1],
            width=0.002,
        )

        # Animate rotating arrow
        arrow2_time = (ellipsoid_time + np.pi / 3) * 10
        arrow2_position = np.array([np.sin(arrow2_time), np.cos(arrow2_time), 0.5 + 0.5])
        renderer.markers[4](
            marker_type=MarkerType.ARROW,
            position=arrow2_position,
            size=[0.05, 0.05, 1],
            rotation=as_rotation_matrix(from_spherical_coords(np.sin(arrow2_time), phi=np.cos(arrow2_time))),
        )

        # Toggle marker visibility periodically
        if step % DISABLE_MARKER_FREQ == 0:
            renderer.markers[3](enabled=False)
        if step % ENABLE_MARKER_FREQ == 0:
            renderer.markers[3](enabled=True)

        step += 1
        time.sleep(TIME_STEP)
