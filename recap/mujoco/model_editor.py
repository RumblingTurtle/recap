import mujoco
import numpy as np
import os


class MJCFModelEditor:
    """
    A class to edit a MJCF model.
    Used primarily to add new
    """

    def __init__(self, mjcf_path: str):
        self.spec = mujoco.MjSpec.from_file(mjcf_path)
        self.spec.meshdir = os.path.join(os.path.dirname(mjcf_path), self.spec.meshdir)

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
