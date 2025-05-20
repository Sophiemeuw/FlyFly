import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple


class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float  # 0->1


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.timestep = timestep
        self.time = 0.0

        # Odor taxis & CPG setup
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()

        # Position tracking
        self.position = np.zeros(2)
        self.position_trace = []
        self.initial_position = np.zeros(2)

        # Path integration state
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.g = 0.3  # empirical gain

        # Homing logic
        self.reached_target = False
        self.turning_timer = 0
        self.turning_duration = int(1.0 / timestep)  # turn for ~1 second
        self.begin_return = False
        self.odor_threshold = 200
        self.go_home = False

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
        ODOR_GAIN, DELTA_MIN, DELTA_MAX, IMPORTANCE = -500, 0.2, 1.0, 0.5
        I = obs["odor_intensity"][0]
        I_left = (I[0] + I[2]) / 2
        I_right = (I[1] + I[3]) / 2
        asym = (I_left - I_right) / ((I_left + I_right + 1e-6) / 2)
        s = asym * ODOR_GAIN
        bias = abs(np.tanh(s))
        if s > 0:
            right = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * bias
            left = DELTA_MAX
        else:
            left = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * bias
            right = DELTA_MAX
        return CommandWithImportance(left, right, IMPORTANCE)

    def get_return_command(self, obs: Observation) -> CommandWithImportance:
        DELTA_MIN, DELTA_MAX, IMPORTANCE = 0.2, 1.0, 1.0
        vector = self.initial_position - self.position
        angle_to_target = np.arctan2(vector[1], vector[0])
        fly_heading = obs.get("heading", 0.0)
        angle_diff = angle_to_target - fly_heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        turn = np.tanh(2.0 * angle_diff)

        aligned_threshold = 0.2
        if abs(angle_diff) < aligned_threshold:
            return CommandWithImportance(DELTA_MAX, DELTA_MAX, IMPORTANCE)
        else:
            if turn > 0:
                return CommandWithImportance(DELTA_MIN, DELTA_MAX, IMPORTANCE)
            else:
                return CommandWithImportance(DELTA_MAX, DELTA_MIN, IMPORTANCE)

    def pillar_avoidance(self, obs: Observation, base_cmd: CommandWithImportance) -> CommandWithImportance:
        MAX_DELTA = 1.0
        MIN_DELTA = 0.2
        IMPORTANCE = 0.7
        GAIN = 22500

        vision = obs["vision"]
        brightness = np.mean(vision, axis=2)

        center = 360
        std = 120
        weights = np.exp(-((np.arange(721) - center) ** 2) / (2 * std**2))

        left_weighted = np.sum(brightness[0] * weights)
        right_weighted = np.sum(brightness[1] * weights)

        left_front = brightness[0, 620:]
        right_front = brightness[1, :100]
        front_brightness = np.mean(np.concatenate([left_front, right_front]))

        left_side_brightness = np.mean(brightness[0, 500:620])
        right_side_brightness = np.mean(brightness[1, 100:220])

        if (
            front_brightness < 5
            or left_side_brightness < 3
            or right_side_brightness < 3
        ) and np.linalg.norm(obs["velocity"][:2]) < 0.5:

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
                IMPORTANCE * left_signal + (1 - IMPORTANCE) * base_cmd.left_descending_signal,
                IMPORTANCE * right_signal + (1 - IMPORTANCE) * base_cmd.right_descending_signal,
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
            IMPORTANCE * left_signal + base_cmd.importance * base_cmd.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal + base_cmd.importance * base_cmd.right_descending_signal
        )

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)

    def integrate_displacement(self, obs: Observation):
        legs = obs["end_effectors"]
        forces = obs["contact_forces"]
        stance = [np.linalg.norm(f) > 4 for f in forces]
        for i in range(6):
            if stance[i] and not self.leg_stance_history[i]:
                self.leg_stance_start_positions[i] = np.array(legs[i])
            if not stance[i] and self.leg_stance_history[i]:
                start = self.leg_stance_start_positions[i]
                if start is not None:
                    disp3 = np.array(legs[i]) - start
                    disp2 = np.clip(disp3[:2], -5, 5)
                    self.leg_displacements[i].append(disp2)
                self.leg_stance_start_positions[i] = None
            self.leg_stance_history[i] = stance[i]
        total, cnt = np.zeros(2), 0
        for dlist in self.leg_displacements:
            for d in dlist:
                total += d
                cnt += 1
        if cnt == 0:
            return
        avg = total / cnt
        if "heading" in obs:
            yaw = obs["heading"]
            c, s = np.cos(yaw), np.sin(yaw)
            avg = np.array([[c, -s], [s, c]]).dot(avg)
        self.position += -self.g * avg
        self.leg_displacements = [[] for _ in range(6)]

    def get_actions(self, obs: Observation) -> Action:
        # Update heading and position from external state
        self.heading = obs["heading"]
        self.position = np.array(obs["fly"][0][:2])

        self.time += self.timestep
        self.integrate_displacement(obs)
        self.position_trace.append(self.position.copy())

        if self.time < self.timestep * 2:
            self.initial_position = self.position.copy()

        # --- Odor source reached: stop and prepare to turn ---
        if not self.reached_target and np.mean(obs["odor_intensity"]) > self.odor_threshold:
            self.reached_target = True
            self.turning_timer = self.turning_duration
            self.arrival_heading = obs["heading"]  # Store heading at arrival

        # --- Turning in place 180 degrees ---
        if self.reached_target and not self.go_home:
            # Calculate desired heading (180 deg from arrival)
            desired_heading = (self.arrival_heading + np.pi) % (2 * np.pi)
            current_heading = obs.get("heading", 0.0)
            angle_diff = desired_heading - current_heading
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            # If not yet facing away, keep turning in place
            if abs(angle_diff) > 0.1:
                # Turn in place: left=0, right=1 (or vice versa)
                turn_action = np.array([0.0, 1.0]) if angle_diff > 0 else np.array([1.0, 0.0])
                joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, turn_action)
                return {
                    "joints": joint_angles,
                    "adhesion": adhesion
                }
            else:
                # Finished turning, start homing
                self.go_home = True

        # --- Homing phase ---
        if self.go_home:
            base_cmd = self.get_return_command(obs)
        elif not self.reached_target:
            base_cmd = self.get_odor_taxis(obs)
        else:
            base_cmd = CommandWithImportance(0.0, 1.0, 1.0)  # default turning while waiting

        combined_cmd = self.pillar_avoidance(obs, base_cmd)
        action = np.array([combined_cmd.left_descending_signal, combined_cmd.right_descending_signal])
        joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, action)
        return {"joints": joint_angles, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return self.go_home and np.linalg.norm(self.position - self.initial_position) < 0.02

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.position[:] = 0
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.position_trace = []
        self.reached_target = False
        self.turning_timer = 0
        self.begin_return = False
        self.go_home = False
        self.initial_position = np.zeros(2)
