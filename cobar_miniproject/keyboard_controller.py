import numpy as np
from pynput import keyboard
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
import threading

from cobar_miniproject.base_controller import BaseController

# Initialize CPG network
intrinsic_freqs = np.ones(6) * 12 * 1.5
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20


class KeyBoardController(BaseController):
    def __init__(
        self,
        timestep: float,
        seed: int = 0,
        leg_step_time=0.025,
    ):
        """Controller that listens to your keypresses and uses these to
        modulate CPGs that control fly walking and turning.

        Parameters
        ----------
        timestep : float
            Timestep of the simulation.
        seed : int
            Random seed.
        leg_step_time : float, optional
            Duration of each step, by default 0.025.
        """
        self.timestep = timestep
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep,
            intrinsic_freqs,
            intrinsic_amps,
            coupling_weights,
            phase_biases,
            convergence_coefs,
            np.random.rand(6),
            np.random.rand(6),
            seed,
        )

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)
        self.phase_increment = self.timestep / leg_step_time * 2 * np.pi

        self.turning = 0  # 1 is left 0 is stationary -1 is right
        self.forward = 0  # 1 is forward 0 is stationary -1 is backward
        self.gain_right = 0.0
        self.gain_left = 0.0

        # Shared lists to store key presses
        self.lock = threading.Lock()

        print("Starting key listener")
        # Start the keyboard listener thread
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

        self.quit = False

    def on_press(self, key):
        # check if key is w, a, s, d
        if key == keyboard.KeyCode.from_char("w"):
            with self.lock:
                self.forward = 1
        if key == keyboard.KeyCode.from_char("a"):
            with self.lock:
                self.turning = 1
        if key == keyboard.KeyCode.from_char("s"):
            with self.lock:
                self.forward = -1
        if key == keyboard.KeyCode.from_char("d"):
            with self.lock:
                self.turning = -1
        if key == keyboard.Key.esc:
            self.listener.stop()
            self.quit = True

    def on_release(self, key):
        # check if key is w, a, s, d
        if key == keyboard.KeyCode.from_char("w"):
            with self.lock:
                self.forward = 0
        if key == keyboard.KeyCode.from_char("a"):
            with self.lock:
                self.turning = 0
        if key == keyboard.KeyCode.from_char("s"):
            with self.lock:
                self.forward = 0
        if key == keyboard.KeyCode.from_char("d"):
            with self.lock:
                self.turning = 0

    def set_cpg_bias(self):
        if np.abs(self.forward) == 1.0:
            if self.turning == 0:
                self.gain_left = 1.0 * self.forward
                self.gain_right = 1.0 * self.forward
            else:
                left_gain_increment = 0.6 * self.forward if self.turning == 1 else 0.0
                right_gain_increment = 0.6 * self.forward if self.turning == -1 else 0.0
                self.gain_left = 1.2 * self.forward - left_gain_increment
                self.gain_right = 1.2 * self.forward - right_gain_increment
        else:
            self.gain_left = -1.0 * self.turning
            self.gain_right = 1.0 * self.turning

    def get_cpg_joint_angles(self):
        action = np.array([self.gain_left, self.gain_right])

        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

    def get_actions(self, obs):
        self.set_cpg_bias()
        return self.get_cpg_joint_angles()

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)

    def done_level(self, obs):
        # check if quit is set to true
        if self.quit:
            return True
        # check if the simulation is done
        return False
