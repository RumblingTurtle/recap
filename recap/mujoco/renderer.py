import numpy as np
import mujoco
from recap.mujoco.marker_set import MarkerSet, MarkerType
from mujoco import viewer
from recap.quat_utils import quat2mat
import traceback
import os

try:
    import mediapy

    VIDEO_ENABLED = True
except ImportError:
    VIDEO_ENABLED = False
    print(
        f"mediapy package not found. Video export is disabled{os.linesep}You can install it by running: pip install mediapy"
    )

AXIS_LENGTH = 0.3
AXIS_RADIUS = 0.01
DEFAULT_MARKER_COLOR = [0, 1, 0, 1]

ARROW_OFFSETS = [
    np.array([0, 0.7071068, 0, 0.7071068]),
    np.array([-0.7071068, 0, 0, 0.7071068]),
    np.array([0, 0, 0.7071068, 0.7071068]),
]

ARROW_ROTATIONS = [quat2mat(quat, True) for quat in ARROW_OFFSETS]


class MujocoRenderer:
    def __init__(self, mjcf_path: str, video_path: str = None):
        self.setup_scene(mjcf_path)
        self.reset()
        self.camera_tracking = True

        self.markers = MarkerSet()
        self.marker_offset = 0
        self.video_path = video_path
        if video_path is not None and VIDEO_ENABLED:
            self.frames = []
            self.renderer = mujoco.Renderer(self.model)
            self.camera = mujoco.MjvCamera()
        else:
            self.frames = None

    @property
    def export_video(self):
        return self.frames is not None

    def __enter__(self):
        self.viewer = viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback,
        )
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        traceback.print_tb(tb)

    def flush_frames(self, fps=None):
        if not VIDEO_ENABLED:
            return

        if fps is None:
            fps = 1 / self.model.opt.timestep

        if self.export_video and len(self.frames) != 0:
            mediapy.write_video(
                self.video_path,
                self.frames,
                fps=fps,
            )
            self.frames = []

    def setup_scene(self, mjcf_path: str):
        self.spec = mujoco.MjSpec.from_file(mjcf_path)
        self.spec.visual.quality.shadowsize = 8192
        ground = self.spec.worldbody.add_geom()

        ground.name = "ground"
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground.size = [0, 0, 0.05]
        ground.pos = [0, 0, 0]
        ground.material = "floor_material"

        floor_texture = self.spec.add_texture()
        floor_texture.name = "groundplane"
        floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
        floor_texture.width = 320
        floor_texture.height = 320
        floor_texture.rgb1 = np.array([1.0, 1.0, 1.0])
        floor_texture.rgb2 = np.array([1.0, 1.0, 1.0])
        floor_texture.builtin = 2
        floor_texture.nchannel = 3
        floor_texture.mark = 1
        floor_texture.markrgb = [0, 0, 0]

        floor = self.spec.add_material()
        floor.name = "floor_material"
        raw = ["" for _ in range(10)]
        raw[1] = "groundplane"

        floor.textures = raw
        floor.texuniform = 1
        floor.texrepeat = np.array([5, 5])
        floor.reflectance = 0.1
        floor.shininess = 0.6

        skybox = self.spec.add_texture()
        skybox.name = "skybox"
        skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        skybox.width = 512
        skybox.height = 512
        skybox.rgb1 = np.zeros(3) * 1.0
        skybox.rgb2 = np.ones(3)
        skybox.builtin = 1
        skybox.nchannel = 3

        light = self.spec.worldbody.add_light()
        light.active = 1
        light.pos = [0, 0, 10]
        light.dir = [0, 0, -1]

        light.ambient = np.ones(3) * 0.01
        light.specular = np.ones(3) * 0.0
        light.diffuse = np.ones(3) * 0.25
        light.directional = 0

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

    def key_callback(self, keycode):
        if chr(keycode) == "R":  # If 'R' is pressed, reset the simulation
            self.reset()
        elif chr(keycode) == "F":  # If 'F' is pressed, flip the camera follow mode
            self.camera_tracking = not self.camera_tracking

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def close(self) -> None:
        self.viewer.close()

    def step(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        with self.viewer.lock():
            if self.export_video:
                self.renderer.update_scene(self.data, self.camera)
            self.update_markers()

            if self.camera_tracking:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.trackbodyid = 0
            else:
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            if self.export_video:
                self.camera.elevation = self.viewer.cam.elevation
                self.camera.azimuth = self.viewer.cam.azimuth
                self.camera.lookat = self.viewer.cam.lookat
                self.camera.trackbodyid = self.viewer.cam.trackbodyid
                self.camera.fixedcamid = self.viewer.cam.fixedcamid
                self.camera.type = self.viewer.cam.type

            if self.export_video:
                self.frames.append(self.renderer.render())

        self.viewer.sync()
        self.marker_offset = 0

    def set_configuration(self, qpos: np.ndarray, pin_notation: bool = False):
        self.data.qpos[:] = qpos
        if pin_notation:
            self.data.qpos[3:7] = qpos[[6, 3, 4, 5]]
            self.data.qvel[:] = 0

    def update_markers(self):
        """Updates recently altered shared markers using user_scn"""
        if self.markers.dirty_markers:
            ngeoms = 0
            for geom_id in range(self.markers.max_markers):
                if self.markers.active_markers[geom_id]:
                    mujoco.mjv_initGeom(
                        self.viewer.user_scn.geoms[ngeoms],
                        self.markers.marker_types[geom_id],
                        self.markers.marker_sizes[3 * geom_id : 3 * geom_id + 3].copy(),
                        self.markers.marker_positions[3 * geom_id : 3 * geom_id + 3].copy(),
                        self.markers.marker_rotations[9 * geom_id : 9 * geom_id + 9].copy(),
                        self.markers.marker_colors[4 * geom_id : 4 * geom_id + 4].copy(),
                    )
                    if self.export_video:
                        mujoco.mjv_initGeom(
                            self.renderer.scene.geoms[self.renderer.scene.ngeom + ngeoms],
                            self.markers.marker_types[geom_id],
                            self.markers.marker_sizes[3 * geom_id : 3 * geom_id + 3].copy(),
                            self.markers.marker_positions[3 * geom_id : 3 * geom_id + 3].copy(),
                            self.markers.marker_rotations[9 * geom_id : 9 * geom_id + 9].copy(),
                            self.markers.marker_colors[4 * geom_id : 4 * geom_id + 4].copy(),
                        )
                    ngeoms += 1

            if self.export_video:
                self.renderer.scene.ngeom = self.renderer.scene.ngeom + ngeoms
            self.viewer.user_scn.ngeom = ngeoms
            self.markers.dirty_markers = False

    def render_line(self, point_a, point_b, width=0.005, color=DEFAULT_MARKER_COLOR):
        """
        Render a line between two points.

        Since mjt_connector seems to be resetting all of the geoms in the scene
        we are left with primitive geoms for line rendering.
        Note that cylinders are expensive to render.
        So take that into account when making many line segments

        Args:
            positions: Tensor of frame positions
            color: RGBA color for the line
        """
        midpoint = (point_a + point_b) / 2
        z_axis = point_b - point_a
        length = np.linalg.norm(z_axis)
        z_axis = z_axis / length

        # Make sure that the cross product vecors are unaligned
        unaligned_vector = z_axis.copy()
        unaligned_vector[np.argmin(z_axis)] += 2
        unaligned_vector = unaligned_vector / np.linalg.norm(unaligned_vector)
        x_axis = np.cross(z_axis, np.array(unaligned_vector))
        y_axis = np.cross(z_axis, x_axis)

        rotation = np.column_stack([x_axis, y_axis, z_axis])

        self.markers[self.marker_offset](
            marker_type=MarkerType.CYLINDER,
            position=midpoint,
            size=np.array([width, length / 2, width]),
            rotation=rotation,
            color=color,
        )
        self.marker_offset += 1

    def render_points(self, positions, size=0.03, colors=None):
        for i in range(len(positions)):
            if colors is None:
                color = DEFAULT_MARKER_COLOR
            else:
                color = colors[i]
            self.markers[self.marker_offset](position=positions[i], size=size, color=color)
            self.marker_offset += 1

    def render_frames(
        self,
        positions,
        rotations,
        axis_length=AXIS_LENGTH,
        axis_radius=AXIS_RADIUS,
    ):
        """
        Render desired and target link positions and orientations in Mujoco.

        Args:
            frame_positions: Tensor of frame positions
            frame_rotations: Tensor of frame scalar first quaternions
            axis_length: Length of the coordinate axes arrows
            marker_color: RGBA color for the markers
        """
        for frame_idx in range(len(positions)):
            target_pos = positions[frame_idx].copy()
            target_rot = rotations[frame_idx].copy()

            for axis_idx in range(3):
                color = np.zeros(4)
                color[axis_idx] = 1
                color[3] = 1  # Alpha channel

                self.markers[self.marker_offset](
                    marker_type=MarkerType.ARROW,
                    position=target_pos,
                    size=np.array([axis_radius, axis_radius, axis_length]),
                    rotation=target_rot @ ARROW_ROTATIONS[axis_idx],
                    color=color,
                )
                self.marker_offset += 1
