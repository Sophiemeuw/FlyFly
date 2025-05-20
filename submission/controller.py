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
        self.timestep = timestep
        self.time = 0

        # Escape state
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = 0.0  # left - right signal
        self.ESCAPE_DURATION = 1000
        self.TURN_DURATION = 80

    def get_odor_taxis(self, obs: Observation) -> CommandWithImportance:
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
        turning_bias = np.abs(np.tanh(s))

        if s > 0:
            right_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            left_descending_signal = DELTA_MAX
        else:
            left_descending_signal = DELTA_MAX - (DELTA_MAX - DELTA_MIN) * turning_bias
            right_descending_signal = DELTA_MAX

        return CommandWithImportance(left_descending_signal, right_descending_signal, importance)

    def pillar_avoidance(self, obs: Observation, odor_taxis_command: CommandWithImportance) -> CommandWithImportance:
        MAX_DELTA = 0.8
        MIN_DELTA = 0.2
        IMPORTANCE = 0.8
        GAIN = 30000

        vision = obs["vision"]
        brightness = np.mean(vision, axis=2)
        left_weighted = np.sum(brightness[0])
        right_weighted = np.sum(brightness[1])
        left_front = brightness[0, 620:]
        right_front = brightness[1, :100]
        front_overlap_brightness = np.mean(np.concatenate([left_front, right_front]))
        left_side_brightness = np.mean(brightness[0, 500:620])
        right_side_brightness = np.mean(brightness[1, 100:220])

        velocity_mag = np.linalg.norm(obs["velocity"][:2])
        contact = obs["contact_forces"]

        # Contact forces for front and middle legs
        front_forces = contact[0][:2] + contact[3][:2]
        middle_forces = contact[1][:2] + contact[4][:2]
        total_force = front_forces + middle_forces
        force_mag = np.linalg.norm(total_force)

        # -------- ESCAPE MODE --------
        if self.escape_timer > 0:
            self.escape_timer -= 1
            # Turn slightly away from the contact force vector
            turn_bias = np.clip(self.escape_direction, -1, 1)
            left = 0.1 + 0.5 * (-turn_bias)
            right = 0.1 + 0.5 * (turn_bias)
            return CommandWithImportance(-left, -right, 1.0)

        if self.turn_timer > 0:
            self.turn_timer -= 1
            return CommandWithImportance(0.1, 1.0, 1.0)

        # Trigger escape if strong force and not moving
        if force_mag > 0.3 and velocity_mag < 0.4:
            self.escape_timer = self.ESCAPE_DURATION
            self.turn_timer = self.TURN_DURATION
            # Direction to escape: +1 = left leg more contact → escape right
            #                     -1 = right leg more contact → escape left
            escape_vector = total_force
            self.escape_direction = np.sign(escape_vector[0])  # x-axis as directional hint
            return CommandWithImportance(-0.6, -0.6, 1.0)

        # Visual emergency
        if (
            front_overlap_brightness < 10
            or left_side_brightness < 3
            or right_side_brightness < 3
        ) and velocity_mag < 0.2:
            if left_side_brightness < right_side_brightness:
                left_signal = 0.1
                right_signal = 1.0
            elif right_side_brightness < left_side_brightness:
                left_signal = 1.0
                right_signal = 0.1
            else:
                left_signal, right_signal = (1.0, 0.1) if random.random() < 0.5 else (0.1, 1.0)

            return CommandWithImportance(
                IMPORTANCE * left_signal + (1 - IMPORTANCE) * base_cmd.left_descending_signal,
                IMPORTANCE * right_signal + (1 - IMPORTANCE) * base_cmd.right_descending_signal,
                IMPORTANCE
            )

        # Regular vision-based turning
        diff = left_weighted - right_weighted
        turn_signal = np.tanh(GAIN * diff) + np.random.uniform(-0.05, 0.05)

        left_signal = MAX_DELTA
        right_signal = MAX_DELTA
        if turn_signal > 0:
            left_signal -= (MAX_DELTA - MIN_DELTA) * abs(turn_signal)
        else:
            right_signal -= (MAX_DELTA - MIN_DELTA) * abs(turn_signal)

        left_descending_signal = (
            IMPORTANCE * left_signal + base_cmd.importance * base_cmd.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal + base_cmd.importance * base_cmd.right_descending_signal
        )

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)
    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        return CommandWithImportance(0, 0, 0)
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

        odor_taxis_command = self.get_odor_taxis(obs)
        combined_command = self.pillar_avoidance(obs, odor_taxis_command)

        action = np.array([
            combined_command.left_descending_signal,
            combined_command.right_descending_signal,
        ])

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
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = 0.0
