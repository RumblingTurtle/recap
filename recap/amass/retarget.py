from recap.amass.motion_wrapper import AMASSMotionWrapper
from recap.mapped_ik import MappedIK


class AMASSRetarget(MappedIK):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def set_motion(self, motion: AMASSMotionWrapper):
        """Sets AMASS motion data as

        Args:
            motion (np.array): A tensor [frames,len(SMPLH_JOINT_NAMES),7] containing [quat,pos]
            of the AMASS skeleton bodies in the world frame
        """
        motion.to("cpu")
        self.motion = motion
        self.frame_idx = 0
        self.reset()

    def get_dataset_position(self, body_name: str):
        return self.motion.get_body_positions(body_name)[self.frame_idx].numpy() + [
            0,
            0,
            self.wbik_params.height_offset,
        ]

    def get_dataset_rotation(self, body_name: str):
        return self.motion.get_body_rotations(body_name)[self.frame_idx].numpy()

    def __len__(self):
        return len(self.motion)

    def step_frame(self):
        self.frame_idx += 1
