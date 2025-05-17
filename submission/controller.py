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

        # Path integration state
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.g = 0.3  # empirical gain

        # Homing logic
        self.reached_target = False
        self.odor_threshold = 200  # adjust as needed
        self.retrace_index = None

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

    def get_retrace_command(self, obs: Observation) -> CommandWithImportance:
        IMPORTANCE = 1.0
        DELTA_MIN, DELTA_MAX = 0.2, 1.0
        GAIN = 2.0

        if self.retrace_index is None or self.retrace_index < 0:
            return CommandWithImportance(1.0, 1.0, IMPORTANCE)

        target_pos = self.position_trace[self.retrace_index]
        vector = target_pos - self.position
        angle_to_target = np.arctan2(vector[1], vector[0])
        fly_heading = obs.get("heading", 0.0)
        angle_diff = angle_to_target - fly_heading
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        turn = np.tanh(GAIN * angle_diff)

        left = DELTA_MAX
        right = DELTA_MAX
        if turn > 0:
            left = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * abs(turn)
        else:
            right = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * abs(turn)

        if np.linalg.norm(vector) < 0.01 and self.retrace_index > 0:
            self.retrace_index -= 1

        return CommandWithImportance(left, right, IMPORTANCE)

    def pillar_avoidance(self, obs: Observation, base_cmd: CommandWithImportance) -> CommandWithImportance:
        MAX_DELTA = 1.0
        MIN_DELTA = 0.2
        IMPORTANCE = 0.7
        GAIN = 20500  # Increased sensitivity

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
        self.time += self.timestep
        self.integrate_displacement(obs)
        self.position_trace.append(self.position.copy())

        if not self.reached_target and np.mean(obs["odor_intensity"]) > self.odor_threshold:
            self.reached_target = True
            self.retrace_index = len(self.position_trace) - 1

        if not self.reached_target:
            base_cmd = self.get_odor_taxis(obs)
        else:
            base_cmd = self.get_retrace_command(obs)

        combined_cmd = self.pillar_avoidance(obs, base_cmd)
        action = np.array([combined_cmd.left_descending_signal, combined_cmd.right_descending_signal])
        joint_angles, adhesion = step_cpg(self.cpg_network, self.preprogrammed_steps, action)
        return {"joints": joint_angles, "adhesion": adhesion}

    def done_level(self, obs: Observation):
        return False

    def reset(self, **kwargs):
        self.cpg_network.reset()
        self.position[:] = 0
        self.leg_stance_history = [False] * 6
        self.leg_stance_start_positions = [None] * 6
        self.leg_displacements = [[] for _ in range(6)]
        self.position_trace = []
        self.reached_target = False
        self.retrace_index = None
