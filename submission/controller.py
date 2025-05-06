import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple


class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float  # 0 -> 1


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.timestep = timestep
        self.time = 0

        # Path integration state
        self.start_pos = np.array([0.0, 0.0])  # Save starting position
        self.internal_pos = np.array([0.0, 0.0])
        self.odor_source_pos = None
        self.odor_found = False
        self.current_heading = 0.0

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        ODOR_GAIN = -500
        DELTA_MIN = 0.4
        DELTA_MAX = 1.6
        IMPORTANCE = 0.5

        I_right = (obs["odor_intensity"][0][1] + obs["odor_intensity"][0][3]) / 2
        I_left = (obs["odor_intensity"][0][0] + obs["odor_intensity"][0][2]) / 2
        asymmetry = (I_left - I_right) / ((I_left + I_right + 1e-6) / 2)
        s = asymmetry * ODOR_GAIN
        turning_bias = np.abs(np.tanh(s))

        if s > 0:
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else:
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)

    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        MAX_DELTA = 1.4
        MIN_DELTA = 0.2
        IMPORTANCE = 0.7
        GAIN = 15500

        vision = obs["vision"]
        brightness = np.mean(vision, axis=2)
        center = 360
        std = 120
        weights = np.exp(-((np.arange(721) - center) ** 2) / (2 * std**2))

        left_weighted = np.sum(brightness[0] * weights)
        right_weighted = np.sum(brightness[1] * weights)

        left_front = brightness[0, 620:]
        right_front = brightness[1, :100]
        front_overlap_brightness = np.mean(np.concatenate([left_front, right_front]))

        left_side_brightness = np.mean(brightness[0, 500:620])
        right_side_brightness = np.mean(brightness[1, 100:220])

        if (
            front_overlap_brightness < 5
            or left_side_brightness < 3
            or right_side_brightness < 3
        ) and np.linalg.norm(obs["velocity"][:2]) < 0.2:

            if left_side_brightness < right_side_brightness:
                left_signal = 0.3
                right_signal = 1.0
            elif right_side_brightness < left_side_brightness:
                left_signal = 1.0
                right_signal = 0.3
            else:
                if random.random() < 0.5:
                    left_signal = 1.0
                    right_signal = 0.3
                else:
                    left_signal = 0.3
                    right_signal = 1.0

            return CommandWithImportance(
                IMPORTANCE * left_signal + (1 - IMPORTANCE) * odor_taxis_command.left_descending_signal,
                IMPORTANCE * right_signal + (1 - IMPORTANCE) * odor_taxis_command.right_descending_signal,
                IMPORTANCE
            )

        diff = left_weighted - right_weighted
        turn_signal = np.tanh(GAIN * diff)
        turn_signal += np.random.uniform(-0.05, 0.05)

        left_signal = MAX_DELTA
        right_signal = MAX_DELTA
        if turn_signal > 0:
            left_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)
        else:
            right_signal = MAX_DELTA - (MAX_DELTA - MIN_DELTA) * abs(turn_signal)

        left_descending_signal = (
            IMPORTANCE * left_signal + odor_taxis_command.importance * odor_taxis_command.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal + odor_taxis_command.importance * odor_taxis_command.right_descending_signal
        )

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)

    def path_integration_return(self) -> CommandWithImportance:
        if not self.odor_found:
            return CommandWithImportance(0, 0, 0)

        # Go back to starting position instead of odor source
        vector_to_start = self.start_pos - self.internal_pos
        print(f"[DEBUG] Returning to start. Vector: {vector_to_start}, Current pos: {self.internal_pos}, Start pos: {self.start_pos}")
        angle_to_start = np.arctan2(vector_to_start[1], vector_to_start[0])
        angle_diff = angle_to_start - self.current_heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        TURN_GAIN = 5
        s = np.tanh(angle_diff * TURN_GAIN)

        DELTA_MIN = 0.2
        DELTA_MAX = 1.4
        if s > 0:
            right_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * s
            left_signal = DELTA_MAX
        else:
            left_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * -s
            right_signal = DELTA_MAX

        return CommandWithImportance(left_signal, right_signal, 0.6)

    def get_actions(self, obs: Observation) -> Action:
        self.time += self.timestep

        # Update heading and estimate position
        if self.time == self.timestep:
            self.start_pos = self.internal_pos.copy()  # Save actual starting position once
        self.current_heading = obs["heading"]
        velocity = np.array(obs["velocity"]).flatten()
        print(f"[DEBUG] velocity: {velocity}")
        if velocity.shape[0] >= 2:
            vx, vy = velocity[0], velocity[1]
        else:
            vx, vy = 0.0, 0.0
        vx_global = vx * np.cos(self.current_heading) - vy * np.sin(self.current_heading)
        vy_global = vx * np.sin(self.current_heading) + vy * np.cos(self.current_heading)
        self.internal_pos += np.array([vx_global, vy_global]) * self.timestep
        if "fly" in obs.keys():
            print(f"[DEBUG] fly: {obs['fly'][0, :]}")
        print(f"[DEBUG] internal_pos: {self.internal_pos}, heading: {self.current_heading}")

        # Store odor source location when found
        if obs.get("reached_odour", False) and not self.odor_found:
            print(f"[DEBUG] Odor source reached at position: {self.internal_pos}")
            self.odor_source_pos = self.internal_pos.copy()
            self.odor_found = True

        # Get behavior commands
        odor_taxis_command = self.get_odor_taxis(obs)
        combined_command = self.pillar_avoidance(obs, odor_taxis_command)
        return_command = self.path_integration_return()

        # Blend final command with weighted importance based on command sources
        # Increase return_command's influence only when it is active
        if self.odor_found:
            weight_return = 0.8
            weight_combined = 0.6
        else:
            weight_return = 0.0
            weight_combined = 1.0

        left_signal = (
            weight_combined * combined_command.left_descending_signal + weight_return * return_command.left_descending_signal
        )
        right_signal = (
            weight_combined * combined_command.right_descending_signal + weight_return * return_command.right_descending_signal
        )

        action = np.array([left_signal, right_signal])

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.internal_pos = np.array([0.0, 0.0])
        self.odor_source_pos = None
        self.odor_found = False
        self.time = 0
