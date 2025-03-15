from recap.wbik_solver import WBIKSolver

import numpy as np
from abc import ABC, abstractmethod


class MappedIK(WBIKSolver, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_idx = 0
        self.renderer = None

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

    def __iter__(
        self,
    ):
        return self

    def renderable_iterator(self, renderer):
        self.renderer = renderer
        return self

    def __next__(
        self,
    ):
        if self.frame_idx >= len(self):
            raise StopIteration()
        solution = self.solve(self.renderer)
        self.step_frame()
        return solution

    def set_targets(self):
        body_to_data_map = self.wbik_params.body_to_data_map

        for k, v in body_to_data_map.items():
            position = self.get_dataset_position(body_to_data_map[k])
            rotation = self.get_dataset_rotation(body_to_data_map[k])
            self.set_target_transform(k, position, rotation)
