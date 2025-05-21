import numpy as np
import random
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from typing import NamedTuple
import numpy as np 
from .looming_detector import LoomDetector





class CommandWithImportance(NamedTuple):
    left_descending_signal: float
    right_descending_signal: float
    importance: float  # 0->1


class Controller(BaseController):
    def __init__(self, timestep=1e-4, seed=0):
        from .preprogrammed_steps import PreprogrammedSteps

        super().__init__()
        self.quit = False
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

        # Path integration variables
        self.heading = 0.0
        self.integrated_position = np.zeros(2)
        self.reached_target = False
        self.go_home = False
        self.odor_threshold = 0.2  # Set as appropriate for your environment
        self.turning = False
        self.turning_steps = 0
        self.max_turning_steps = 100  # Number of steps to turn 180°
        self.homing_done = False
        
        self.min_home = 100

        self.loom_detector = LoomDetector(debug=True)
        self.ball_escape_timer = 0

    def get_integrated_position(self) -> np.ndarray:
        return self.integrated_position.copy()

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
        IMPORTANCE = 0.9
        GAIN = 40000

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
                IMPORTANCE * left_signal + (1 - IMPORTANCE) * odor_taxis_command.left_descending_signal,
                IMPORTANCE * right_signal + (1 - IMPORTANCE) * odor_taxis_command.right_descending_signal,
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
            IMPORTANCE * left_signal + odor_taxis_command.importance * odor_taxis_command.left_descending_signal
        )
        right_descending_signal = (
            IMPORTANCE * right_signal + odor_taxis_command.importance * odor_taxis_command.right_descending_signal
        )

        return CommandWithImportance(left_descending_signal, right_descending_signal, IMPORTANCE)

    def path_integration(self, obs: Observation): 
        heading = obs["heading"].copy()
        vel = obs["velocity"].copy()
        heading = -heading

        rot_matrix = np.array(
            [
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]
            ]
        )

        world_frame_vel = vel.ravel() @ rot_matrix
        self.integrated_position += world_frame_vel * self.timestep

        

    def ball_avoidance(self, obs: Observation) -> CommandWithImportance:
        # Process current vision
        detected = self.loom_detector.process(obs)
        if detected: 
            # reset ball escape timer 
            print(f"Triggered escape")
            self.ball_escape_timer = 1500

        if self.ball_escape_timer > 0 and self.ball_escape_timer % 500 == 0:
            print(f"Escaping... {self.ball_escape_timer}/1500")
        
        if self.ball_escape_timer > 0:
            self.ball_escape_timer -= 1
            return CommandWithImportance(1, 1, 1)
        else: 
            return CommandWithImportance(0, 0, 0)

    def get_actions(self, obs: Observation, suppress_motion=False) -> Action:
        # End the level if done_level is True
        if self.done_level(obs):
            joint_angles = obs["joints"][0] if "joints" in obs else self.preprogrammed_steps.default_pose
            adhesion = np.ones(6)
            return {
                "joints": joint_angles,
                "adhesion": adhesion,
            }

        self.time += self.timestep        

        # Update heading and position from observation
        self.heading = obs["heading"].copy()
        self.path_integration(obs)

        ball_cmd = self.ball_avoidance(obs)

        # Check if odor source is reached
        odor_intensity = np.mean(obs["odor_intensity"])
        if not self.reached_target and odor_intensity > self.odor_threshold:
            self.reached_target = True
            self.turning = True
            self.turning_steps = 0

        # Stop and turn 180° at odor source
        if self.reached_target and self.turning and self.turning_steps < self.max_turning_steps:
            self.turning_steps += 1
            # Turn in place: left=0, right=1 (or vice versa)
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=np.array([0.0, 1.0])
            )
            if self.turning_steps >= self.max_turning_steps:
                self.turning = False
                self.go_home = True
            return {
                "joints": joint_angles,
                "adhesion": adhesion,
            }

        # Homing: go back to initial position
        if self.go_home and not self.homing_done:
            # Compute vector to home
            to_home =  -self.integrated_position
            dist_to_home = np.linalg.norm(to_home)

            if dist_to_home < 4:  # Close enough, stop
                # End the simulation when the fly returns to the drop position after reaching the odor source
                self.homing_done = True
                self.quit = True
                joint_angles = obs["joints"] if "joints" in obs else np.zeros_like(self.preprogrammed_steps.default_joint_angles)
                adhesion = np.ones(6)
                print("Homing done, quitting simulation.")

            # Compute desired heading
            desired_heading = np.arctan2(to_home[1], to_home[0])
            heading_error = desired_heading - self.heading
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            # Simple proportional controller for turning
            if abs(heading_error) > 0.1:
                # Turn towards home
                action = np.array([0.0, 1.0]) if heading_error > 0 else np.array([1.0, 0.0])
            else:
                # Move forward
                action = np.array([1.0, 1.0])
            joint_angles, adhesion = step_cpg(
                cpg_network=self.cpg_network,
                preprogrammed_steps=self.preprogrammed_steps,
                action=action,
            )
            return {
                "joints": joint_angles,
                "adhesion": adhesion,
            }

        # Normal behavior (odor taxis + pillar avoidance)
        odor_taxis_command = self.get_odor_taxis(obs)
        combined_command = self.pillar_avoidance(obs, odor_taxis_command)
        action = np.array([
            combined_command.left_descending_signal,
            combined_command.right_descending_signal,
        ])
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
        return self.quit or self.homing_done

    def reset(self, **kwargs):
        """Clean reset of all state"""
        self.cpg_network.reset()
        self.escape_timer = 0
        self.turn_timer = 0
        self.escape_direction = 0.0
        self.heading = 0.0
        self.initial_position = None
        self.reached_target = False
        self.go_home = False
        self.turning = False
        self.turning_steps = 0
        self.homing_done = False