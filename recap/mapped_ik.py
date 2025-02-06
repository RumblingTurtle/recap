from recap.wbik_solver import WBIKSolver

import numpy as np
from abc import ABC, abstractmethod


class MappedIK(WBIKSolver, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_idx = 0

    @abstractmethod
    def get_dataset_position(self, body_name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_dataset_rotation(self, body_name: str) -> np.ndarray:
        pass

    @abstractmethod
    def step_frame(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __next__(
        self,
    ):
        if self.frame_idx >= len(self):
            raise StopIteration()
        solution = self.solve()
        self.step_frame()
        return solution

    def set_targets(self):
        body_to_data_map = self.wbik_params.body_to_data_map
        root_d = self.get_dataset_position(body_to_data_map["root"])
        # fmt: off
        lk_d = self.get_dataset_position(body_to_data_map["left_knee"])
        rk_d = self.get_dataset_position(body_to_data_map["right_knee"])

        lf_d = self.get_dataset_position(body_to_data_map["left_foot"])
        rf_d = self.get_dataset_position(body_to_data_map["right_foot"])

        le_d = self.get_dataset_position(body_to_data_map["left_elbow"])
        re_d = self.get_dataset_position(body_to_data_map["right_elbow"])

        lh_d = self.get_dataset_position(body_to_data_map["left_hand"])
        rh_d = self.get_dataset_position(body_to_data_map["right_hand"])

        lf_R = self.get_dataset_rotation(body_to_data_map["left_foot"])
        rf_R = self.get_dataset_rotation(body_to_data_map["right_foot"])
        root_R = self.get_dataset_rotation(body_to_data_map["root"])

        self.set_target_transform("root", root_d, root_R)
        self.set_target_transform("right_foot", rf_d, rf_R)
        self.set_target_transform("left_foot", lf_d, lf_R)

        self.set_target_transform("left_knee", lk_d)
        self.set_target_transform("right_knee", rk_d)

        self.set_target_transform("left_elbow", le_d)
        self.set_target_transform("right_elbow", re_d)

        self.set_target_transform("left_hand", lh_d)
        self.set_target_transform("right_hand", rh_d)
