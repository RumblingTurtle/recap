from recap.cmu.amc_parser import parse_asf, parse_amc

from recap.mapped_ik import MappedIK
import numpy as np
import os

CMU_DATASET_FPS = 120


class CMURetarget(MappedIK):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rescale_skeleton(self, asf_path):
        root_to_shoulder = np.linalg.norm(self.body("left_shoulder").translation - self.body("root").translation)
        shoulder_to_elbow = np.linalg.norm(self.body("left_elbow").translation - self.body("left_shoulder").translation)

        elbow_to_hand = np.linalg.norm(self.body("left_hand").translation - self.body("left_elbow").translation)

        hip_to_knee_len = np.linalg.norm(self.body("right_hip").translation - self.body("right_knee").translation)

        knee_to_foot_len = np.linalg.norm(self.body("right_knee").translation - self.body("right_foot").translation)

        root_to_hip_y = np.abs(self.body("root").translation[1] - self.body("left_hip").translation[1])
        root_to_hip_height = np.linalg.norm(self.body("root").translation[1] - self.body("left_hip").translation[1])
        self.skeleton = parse_asf(
            asf_path,
            reassign_lengths={
                "ltibia": knee_to_foot_len * self.wbik_params.tibia_length_scale,
                "rtibia": knee_to_foot_len * self.wbik_params.tibia_length_scale,
                "lhipjoint": root_to_hip_y * self.wbik_params.hip_width_scale,
                "rhipjoint": root_to_hip_y * self.wbik_params.hip_width_scale,
                "lfemur": hip_to_knee_len * self.wbik_params.hip_length_scale,
                "rfemur": hip_to_knee_len * self.wbik_params.hip_length_scale,
                "lowerback": root_to_hip_height / (3 * self.wbik_params.spine_scale),
                "upperback": root_to_hip_height / (3 * self.wbik_params.spine_scale),
                "thorax": root_to_hip_height / (3 * self.wbik_params.spine_scale),
                "rhumerus": shoulder_to_elbow * self.wbik_params.elbow_length_scale,
                "lhumerus": shoulder_to_elbow * self.wbik_params.elbow_length_scale,
                "rclavicle": root_to_shoulder * self.wbik_params.shoulder_scale,
                "lclavicle": root_to_shoulder * self.wbik_params.shoulder_scale,
                "lradius": elbow_to_hand * self.wbik_params.forearm_length_scale,
                "rradius": elbow_to_hand * self.wbik_params.forearm_length_scale,
            },
        )
        self.skeleton["root"].set_zero()

    def set_skeleton(self, subject_id: int):
        asf_path = os.path.join(
            os.path.abspath(self.wbik_params.dataset_path),
            f"all_asfamc/subjects/{subject_id:02d}/{subject_id:02d}.asf",
        )
        self.rescale_skeleton(asf_path)

    def set_motion(self, subject_id: int, motion_id: int):
        self.frame_idx = 0
        amc_path = os.path.join(
            os.path.abspath(self.wbik_params.dataset_path),
            f"all_asfamc/subjects/{subject_id:02d}/{subject_id:02d}_{motion_id:02d}.amc",
        )
        motions = parse_amc(amc_path)

        frame_skip = np.clip(
            int(CMU_DATASET_FPS / self.wbik_params.output_fps),
            a_min=1,
            a_max=len(motions),
        )
        self.frames = motions[::frame_skip]

        height_offset = self.get_height_offset(self.frames)
        for i in range(len(self.frames)):
            self.frames[i]["root"][1] = self.frames[i]["root"][1] - height_offset + self.wbik_params.height_offset

        self.skeleton["root"].set_motion(self.frames[self.frame_idx])
        self.reset()

    def get_height_offset(self, frames):
        min_height = np.inf
        for motion in frames:
            self.skeleton["root"].set_motion(motion)
            min_height = min(
                min_height,
                min(
                    self.get_dataset_position(self.wbik_params.body_to_data_map["left_foot"])[2],
                    self.get_dataset_position(self.wbik_params.body_to_data_map["right_foot"])[2],
                ),
            )
        self.skeleton["root"].set_zero()
        return min_height

    def get_dataset_position(self, body_name: str):
        return self.skeleton[body_name].position

    def get_dataset_rotation(self, body_name: str):
        return self.skeleton[body_name].R

    def __len__(self):
        return len(self.frames)

    def step_frame(self):
        self.frame_idx += 1
        if self.frame_idx >= len(self):
            self.frame_idx = len(self)
        else:
            self.skeleton["root"].set_motion(self.frames[self.frame_idx])


"""
These are the indices used from the official CMU listing
http://mocap.cs.cmu.edu/motcat.php?maincat=3

Each entry in the dataset list corresponds to:
(subject_name,[motion1, motion2, ......])
"""

# fmt: off
CMU_RUNNING = [
    (2, [3]),
    (9,[1,2,3,4,5,6,7,8,9,10,11,]),
    (16,[8,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,57]),
    (35, [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
    (38, [3]),
]

CMU_WALKING = [
    (2, [1, 2]),
    (5, [1]),
    (6, [1]),
    (7, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    (8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    (9, [12]),
    (10, [4]),
    (12, [1, 2, 3]),
    (15, [1, 3, 9, 14]),
    (16, [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,47,58]),
    (17, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (26, [1]),
    (27, [1]),
    (29, [1]),
    (32, [1, 2]),
    (35, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,28,29,30,31,32,33,34]),
    (36, [2, 3, 9]),
    (37, [1]),
    (38, [1, 2, 4]),
    (39, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    (40, [2, 3, 4, 5]),
    (41, [2, 3, 4, 5, 6]),
    (43, [1]),
    (45, [1]),
    (46, [1]),
    (47, [1]),
    (49, [1]),
    (55, [1]),
    (56, [1]),
]

CMU_JUMPING = [
    (13, [11, 13, 19, 32, 39, 40, 41, 42]),
    (16, [1, 2, 3, 4, 5, 6, 7, 9, 10]),
    (49, [2, 3, 4, 5]),
]

CMU_STAND = [(113, [21]), (111, [28])]

CMU_DANCE = [(55,[1]), (5,[7,2])]

CMU_BACKFLIP = [(88,[1])]
# fmt: on
