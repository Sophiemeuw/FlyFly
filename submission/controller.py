import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple
from collections import deque
from enum import IntEnum

class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float

    def get_drive(self):
        return np.array([self.left_descending_signal, self.right_descending_signal])

class EscapeDirection(IntEnum):
    LEFT = -1
    RIGHT = 1

class ControllerState(IntEnum):
    SEEKING_ODOR = 0
    TURNING = 1
    RETURNING_HOME = 2

class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from .preprogrammed_steps import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0

        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = EscapeDirection.LEFT
        self.ESCAPE_DURATION = 1000
        self.TURN_DURATION = 400

        self.POS_HIST_LEN = 5000
        self.heading = 0.0
        self.integrated_position = np.zeros(2)
        self.integrated_position_history = deque([], self.POS_HIST_LEN)
        self.pos_inhibit_cooldown = 0

        self.controller_state = ControllerState.SEEKING_ODOR
        self.odor_counter = 0
        self.ODOR_CONFIRM_STEPS = 30

        self.turning_steps = 0
        self.max_turning_steps = 100

        self.last_drive = np.zeros(2)

    def get_integrated_position(self) -> np.ndarray:
        return self.integrated_position.copy()

    def get_last_drive(self) -> np.ndarray:
        return self.last_drive.copy()

    def get_odor_taxis(self, obs: Observation, velocity: np.ndarray) -> CommandWithImportance:
        ODOR_GAIN = -600
        DELTA_MIN = 0.2
        DELTA_MAX = 0.8
        BASE_IMPORTANCE = 0.4
        MAX_IMPORTANCE = 0.9

        I_right = (obs["odor_intensity"][0][1] + obs["odor_intensity"][0][3]) / 2
        I_left = (obs["odor_intensity"][0][0] + obs["odor_intensity"][0][2]) / 2
        I_total = I_left + I_right
        I_norm = min(I_total / 0.0015, 1.0)
        importance = BASE_IMPORTANCE + (MAX_IMPORTANCE - BASE_IMPORTANCE) * I_norm

        asymmetry = (I_left - I_right) / ((I_left + I_right + 1e-6) / 2)
        s = asymmetry * ODOR_GAIN
        turning_bias = np.clip(np.abs(np.tanh(s)), 0, 0.8)

        if s > 0:
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else:
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        velocity_mag = np.linalg.norm(velocity[:2])
        VELOCITY_THRESHOLD = 0.4
        MAX_DAMPING = 0.5
        damping_factor = 1.0
        if I_total > 0.001:
            damping_factor = 1.0 - min(velocity_mag / VELOCITY_THRESHOLD, 1.0) * MAX_DAMPING

        left_descending_signal = max(left_descending_signal * damping_factor, 0.1)
        right_descending_signal = max(right_descending_signal * damping_factor, 0.1)

        return CommandWithImportance(left_descending_signal, right_descending_signal, importance)

    def path_integration(self, obs: Observation):
        heading = -obs["heading"].copy()
        vel = obs["velocity"].copy()

        self.integrated_position_history.appendleft(self.integrated_position.copy())
        rot_matrix = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )
        world_frame_vel = vel.ravel() @ rot_matrix
        self.integrated_position += world_frame_vel * self.timestep

    def return_to_home(self) -> np.ndarray:
        match self.controller_state:
            case ControllerState.TURNING:
                if self.turning_steps < self.max_turning_steps:
                    self.turning_steps += 1
                else:
                    self.controller_state = ControllerState.RETURNING_HOME
                return np.array([0.0, 1.0])

            case ControllerState.RETURNING_HOME:
                to_home = -self.integrated_position
                dist_to_home = np.linalg.norm(to_home)
                if dist_to_home < 4:
                    self.quit = True
                    print("Homing done, quitting simulation.")
                desired_heading = np.arctan2(to_home[1], to_home[0])
                heading_error = np.arctan2(np.sin(desired_heading - self.heading), np.cos(desired_heading - self.heading))
                if abs(heading_error) > 0.1:
                    return np.array([0.0, 1.0]) if heading_error > 0 else np.array([1.0, 0.0])
                else:
                    return np.array([1.0, 1.0])

            case _:
                print("Unexpected state in return_to_home")
                return np.array([0.5, 0.5])

    def get_actions(self, obs: Observation) -> Action:
        self.time += self.timestep
        self.heading = obs["heading"].copy()
        self.path_integration(obs)

        odor_intensity = np.mean(obs["odor_intensity"])

        if self.controller_state == ControllerState.SEEKING_ODOR:
            if odor_intensity > 0.2:
                self.odor_counter += 1
                if self.odor_counter >= self.ODOR_CONFIRM_STEPS:
                    self.controller_state = ControllerState.TURNING
                    self.odor_counter = 0
            else:
                self.odor_counter = 0

        if self.controller_state == ControllerState.SEEKING_ODOR:
            odor_taxis_command = self.get_odor_taxis(obs, obs["velocity"])
            drive = odor_taxis_command.get_drive()
        else:
            drive = self.return_to_home()

        drive = np.clip(drive, 0.1, 0.6)
        self.last_drive = drive

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=drive,
        )
        return {"joints": joint_angles, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = EscapeDirection.LEFT
        self.heading = 0.0
        self.controller_state = ControllerState.SEEKING_ODOR
        self.odor_counter = 0
        self.integrated_position = np.zeros(2)
        self.integrated_position_history.clear()
        self.turning_steps = 0
        self.last_drive = np.zeros(2)
        self.quit = False