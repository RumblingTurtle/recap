import mujoco
import numpy as np
import os


class MJCFModelEditor:
    """
    A class to edit a MJCF model.
    Used primarily to add new
    """

    def __init__(self, mjspec: mujoco.MjSpec):
        self.spec = mjspec

    @staticmethod
    def from_string(mjcf_string: str):
        return MJCFModelEditor(mujoco.MjSpec.from_string(mjcf_string))

    @staticmethod
    def from_path(mjcf_path: str):
        new_spec = mujoco.MjSpec.from_file(mjcf_path)
        # Adjust meshdir to be relative to absolute path
        new_spec.meshdir = os.path.join(os.path.dirname(mjcf_path), new_spec.meshdir)
        return MJCFModelEditor(new_spec)

    @staticmethod
    def empty():
        return MJCFModelEditor(mujoco.MjSpec())

    def add_body(self, body_name: str, parent_name: str, position: np.ndarray, quaternion: np.ndarray):
        parent_body = self.spec.find_body(parent_name)
        child_body = parent_body.add_body()
        child_body.name = body_name
        child_body.pos = position
        child_body.quat = quaternion

    def compile(self):
        self.model = self.spec.compile()

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.spec.to_xml())
