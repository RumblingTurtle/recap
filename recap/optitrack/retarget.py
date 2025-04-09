from recap.mapped_ik import MappedIK
import numpy as np
from recap.optitrack.csv_parser import Take
from recap.quat_utils import quat2mat


class OptitrackRetarget(MappedIK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_motion(self, csv_path: str):
        self.frame_idx = 0
        self.reset()
        self.take = Take()
        self.take.readCSV(csv_path)
        self.fps = self.take.frame_rate
        self.total_frames = int(self.take._raw_info["Total Exported Frames"]) - 1
        self.pos_multiplier = 1e-3 if self.take.units == "Millimeters" else 1.0

    def get_dataset_position(self, body_name: str):
        return (np.array(self.take.rigid_bodies[body_name].positions[self.frame_idx]) * self.pos_multiplier)[[2, 0, 1]]

    def get_dataset_rotation(self, body_name: str):
        return quat2mat(
            np.array(self.take.rigid_bodies[body_name].rotations[self.frame_idx])[[2, 0, 1, 3]], scalar_last=True
        )[0]

    def __len__(self):
        return self.total_frames

    def step_frame(self):
        self.frame_idx += 1
        # If the data is missing in one bone, then there rest are also missing
        first_body_positions = list(self.take.rigid_bodies.values())[0].positions
        while self.frame_idx < self.total_frames and first_body_positions[self.frame_idx] == None:
            self.frame_idx += 1
        if self.frame_idx >= len(first_body_positions):
            raise StopIteration()
