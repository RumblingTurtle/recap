import os
from abc import abstractmethod
import numpy as np
import qpsolvers
import pinocchio as pin
import quaternion

from recap.velocity_estimator import VelocityEstimator
from recap.quat_utils import yaw_matrix
from recap.mujoco.renderer import MujocoRenderer
from recap.trajectory import PoseData, BodyTransform
from recap.config.wbik_config import WBIKConfig

import pink
from pink import solve_ik
from pink.limits import ConfigurationLimit, AccelerationLimit
from pink.tasks import FrameTask, PostureTask, ComTask
from pink.barriers import PositionBarrier

# The cons of using manual limits.
# Pink would spam warnings if we violiate the limits
import logging

logging.disable("WARN")
EPS = 0.0


class NoSolutionException(Exception):
    pass


class WBIKSolver:
    """Wholebody inverse kinematics problem definition and solver for bipedal robots"""

    def __init__(
        self,
        wbik_params: WBIKConfig,
    ):
        """_summary_

        Args:
            wbik_params (WBIKConfig): Task configuration
            data_dt (float): Integration step of the solver

        """
        self.wbik_params = wbik_params
        self.body_to_model_map = self.wbik_params.body_to_model_map
        self.model = pin.buildModelFromMJCF(self.wbik_params.mjcf_path)
        # Mujoco and pinocchio handle root body offsets differently
        # if the floating base link has pos= attribute assigned in the MJCF file.
        # Mujoco ignores this attribute and pinocchio actually offsets the body.
        # So we preemptively set the root offset to 0.
        self.model.jointPlacements[1].translation[:] = 0.0
        self.data = self.model.createData()

        self.step_dt = self.wbik_params.step_dt

        self.mjcf_to_alias_map = {v: k for k, v in self.body_to_model_map.items()}
        if len(self.mjcf_to_alias_map) != len(self.body_to_model_map):
            raise ValueError(f"Body map and contains duplicate definitions: {self.body_to_model_map}")

        self.body_to_pin_id = {}
        self.available_frames = []
        for i in range(self.model.nframes):
            frame_name = self.model.frames[i].name
            self.available_frames.append(frame_name)
            if frame_name in self.mjcf_to_alias_map.keys():
                self.body_to_pin_id[self.mjcf_to_alias_map[frame_name]] = i

        for alias, name in self.body_to_model_map.items():
            if alias not in self.body_to_pin_id.keys():
                raise ValueError(
                    f"Frame {alias}:{name} not found in the robot model{os.linesep}\
                                 Available frames: {self.available_frames}"
                )

        self.joint_order = list(self.model.names)

        self.foot_vel_estimators = [
            VelocityEstimator(in_shape=3, window_size=5, sample_dt=self.wbik_params.step_dt) for i in range(2)
        ]

        self.__setup_nik()

    def reset(self):
        """
        Method to reset the solution to the initial state
        Needs to be called at the start of every consecutive motion clip
        """
        for i in range(2):
            self.foot_vel_estimators[i].reset()

        self.contacts = [False, False]
        self.first_solution = True

    def set_target_transform(
        self,
        body_name: str,
        position: np.ndarray = None,
        rotation_matrix: np.ndarray = None,
    ):
        """Sets the desried frame task position and translation in the world frame


        Args:
            body_name (str): body frame in the body_to_model_map dict
            position (np.ndarray, optional): desired frame position in world frame
            rotation_matrix (np.ndarray, optional):  desired frame rotation in world frame
        """
        if position is None:
            position = np.zeros(3)
        if rotation_matrix is None:
            rotation_matrix = np.eye(3)
        else:
            if "foot" in body_name and self.wbik_params.yaw_only_feet:
                rotation_matrix = yaw_matrix(rotation_matrix)

        self.targets[body_name] = pin.SE3(rotation_matrix, position)

    @abstractmethod
    def set_targets(
        self,
    ):
        pass

    def __setup_nik(self):
        self.tasks = {}
        self.targets = {}
        self.prev_targets = {}

        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

        self.reset()

        # Create a frame task for each frame in the mapping
        for body_name in self.body_to_model_map.keys():
            subname = "root" if body_name == "root" else body_name.split("_")[1]
            pos_weight = self.wbik_params.task_weights["position"].get(subname, EPS)
            rot_weight = self.wbik_params.task_weights["rotation"].get(subname, EPS)
            if pos_weight == EPS and rot_weight == EPS:
                continue

            self.tasks[body_name] = FrameTask(
                self.body_to_model_map[body_name],
                position_cost=pos_weight,
                orientation_cost=rot_weight,
            )
        # Add joint velocity task
        self.joint_pos_task = PostureTask(cost=self.wbik_params.task_weights["joint_velocity"])

        # Add barriers and limits
        self.barriers = []
        ## Prevent the feet from penetrating the ground
        for side in ["left", "right"]:
            self.barriers.append(
                PositionBarrier(
                    self.body_to_model_map[f"{side}_foot"],
                    indices=[2],  # Z axis
                    p_min=np.array([0]),
                    p_max=np.array([np.inf]),
                    gain=np.array([100.0]),
                    safe_displacement_gain=1.0,
                )
            )
        half_limit = (
            np.abs(self.model.lowerPositionLimit - self.model.upperPositionLimit)
            * (1 - self.wbik_params.joint_limit_scale)
            * 0.5
        )
        self.model.lowerPositionLimit += half_limit
        self.model.upperPositionLimit -= half_limit
        self.dof_limits = ConfigurationLimit(self.model, config_limit_gain=1.0)
        acceleration_limit = np.full(self.model.nv, 0.0)
        acceleration_limit[:3] = self.wbik_params.task_weights["max_root_lin_acceleration"]
        acceleration_limit[3:6] = self.wbik_params.task_weights["max_root_ang_acceleration"]
        acceleration_limit[6:] = self.wbik_params.task_weights["max_joint_acceleration"]

        self.acceleration_limit = AccelerationLimit(self.model, acceleration_limit)

    def solve(
        self,
        renderer=None,
    ) -> PoseData:
        """
        Solves IK using the desired positions of the limbs
        specified in the set_targets method

        Returns:
            tuple(np.ndarray):
                -position of the root link in the world frame
                -world frame root link quaternion
                -joint positions
                -sum of translation task errors
        """
        self.set_targets()

        if self.first_solution:
            # Init configuration at target root pos
            q_init = np.zeros(self.model.nq)
            q_init[:3] = self.targets["root"].translation
            q_init[3:7] = quaternion.as_float_array(quaternion.from_rotation_matrix(self.targets["root"].rotation))[
                [1, 2, 3, 0]
            ]
            self.prev_solution = q_init.copy()
            self.configuration = pink.Configuration(self.model, self.data, q_init)

        self.__process_targets()

        self.joint_pos_task.set_target(self.prev_solution)
        for name in self.targets.keys():
            self.tasks[name].set_target(self.targets[name])

        velocity_norm = np.inf
        iter_count = 0
        while velocity_norm > self.wbik_params.termination_velocity:
            if not self.first_solution:
                if iter_count >= self.wbik_params.max_iters:
                    break
            else:
                if iter_count >= self.wbik_params.skip_iters:
                    raise NoSolutionException("Too many iterations")

            tasks = list(self.tasks.values())
            limits = [self.dof_limits]
            if not self.first_solution:
                tasks += [self.joint_pos_task]
                limits += [self.acceleration_limit]

            velocity = solve_ik(
                self.configuration,
                tasks,
                self.wbik_params.step_dt,
                solver=self.solver,
                barriers=self.barriers,
                safety_break=False,
                limits=limits,
            )
            self.configuration.integrate_inplace(velocity, self.wbik_params.step_dt)
            velocity_norm = np.linalg.norm(velocity[6:])
            if renderer is not None:
                self.render_solution(renderer)
                renderer.step()

            iter_count += 1

        self.prev_solution = self.configuration.q.copy()
        self.first_solution = False
        return self.build_pose_data()

    def adjust_feet_contacts(self):
        # If the feet are in contact the target transform is fixed
        # otherwise previous solution is interpolated to the desired position
        self.contacts = [False, False]
        for i, side in enumerate(["left", "right"]):
            # Estimate foot velocity from input data
            foot_velocity = self.foot_vel_estimators[i](self.targets[f"{side}_foot"].translation)
            velocity_norm = np.linalg.norm(foot_velocity)
            self.contacts[i] = velocity_norm < self.wbik_params.contact_velocity
            if self.contacts[i]:
                # Keeping the yaw angle so that the feet won't rotate into the ground
                # when in contact
                target_matrix = yaw_matrix(self.prev_targets[f"{side}_foot"].rotation)
                self.targets[f"{side}_foot"].translation = self.prev_targets[f"{side}_foot"].translation
            else:
                target_matrix = self.targets[f"{side}_foot"].rotation

            current_target = quaternion.from_rotation_matrix(self.prev_targets[f"{side}_foot"].rotation)
            prev_target = quaternion.from_rotation_matrix(target_matrix)

            self.targets[f"{side}_foot"].rotation = quaternion.as_rotation_matrix(
                quaternion.slerp(current_target, prev_target, 0, 1, self.wbik_params.contact_target_lerp)
            )

    def __process_targets(self):
        """
        Process the targets and set the desired positions and orientations
        """
        if self.first_solution:
            self.prev_targets = {k: v.copy() for k, v in self.targets.items()}

        self.adjust_feet_contacts()

        self.prev_targets = {k: v.copy() for k, v in self.targets.items()}

    def body(self, body_name):
        """
        Returns the transform of the body in the world frame
        """
        if body_name not in self.body_to_pin_id.keys() and body_name:
            raise KeyError(
                f"""Invalid frame name: {body_name}{os.linesep}\
                    Avalable frames:{os.linesep}\
                    {list(self.body_to_pin_id.keys())}"""
            )
        return self.configuration.data.oMf[self.body_to_pin_id[body_name]]

    def compute_position_errors(
        self,
    ):
        """
        Computes the position errors relative to the targets
        """
        errors = {}
        for frame_name in self.body_to_pin_id:
            if frame_name in self.targets.keys():
                errors[frame_name] = np.linalg.norm(
                    self.body(frame_name).translation - self.targets[frame_name].translation
                )
        return errors

    def build_pose_data(
        self,
    ) -> PoseData:
        """
        Builds dictionary with the frame data including positions and quaternions
        of all tracked frames
        """
        transforms = []
        added_bodies = []
        for frame_name in self.body_to_pin_id:
            mjcf_name = self.body_to_model_map[frame_name]
            added_bodies.append(mjcf_name)
            body_transform = self.body(frame_name)
            transforms.append(
                BodyTransform(
                    name=mjcf_name,
                    position=body_transform.translation.copy(),
                    quaternion=quaternion.from_rotation_matrix(body_transform.rotation.copy()),
                )
            )

        for i, frame in enumerate(self.model.frames):
            if frame.name not in added_bodies and frame.name in self.wbik_params.extra_bodies:
                body_transform = self.configuration.data.oMf[i]
                transforms.append(
                    BodyTransform(
                        name=frame.name,
                        position=body_transform.translation.copy(),
                        quaternion=quaternion.from_rotation_matrix(body_transform.rotation.copy()),
                    )
                )

        root_name = self.model.frames[list(self.model.parents).index(1)].name

        pose_data = PoseData(
            transforms=transforms,
            q=self.prev_solution.copy(),
            body_aliases=self.body_to_model_map,
            model_root=root_name,
            contacts=np.array(self.contacts),
            position_error=sum(self.compute_position_errors().values()),
            dt=self.wbik_params.step_dt,
            joint_order=self.joint_order,
        )
        return pose_data

    def render_solution(self, renderer: MujocoRenderer):
        renderer.set_configuration(self.prev_solution.copy(), pin_notation=True)
        # Render desired and target link positions and orientations
        frame_color = np.array([1, 0, 0, 1])
        for frame_name in self.body_to_pin_id.keys():
            renderer.render_points(
                positions=[self.body(frame_name).translation],
                size=0.03,
                colors=[frame_color],
            )

        # Render cached frame targets
        for name, target in self.targets.items():
            if np.any(self.tasks[name].cost[:3] > EPS):
                marker_color = np.array([0, 1, 0, 1])
                if "foot" in name:
                    foot_idx = 0
                    if "right" in name:
                        foot_idx = 1
                    if self.contacts[foot_idx]:
                        marker_color = np.array([0, 0, 0, 1])

                renderer.render_points(
                    positions=[target.translation],
                    size=0.03,
                    colors=[marker_color],
                )

            if np.any(self.tasks[name].cost[3:] > EPS):
                # Draw unit axes
                if np.linalg.norm(self.tasks[name].cost[3:6]) != 0:
                    renderer.render_frames(
                        positions=[target.translation],
                        rotations=[target.rotation],
                        axis_length=0.3,
                        axis_radius=0.01,
                    )
